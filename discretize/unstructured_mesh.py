"""Module containing unstructured meshes for discretize."""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree, ConvexHull
from discretize.utils import Identity, invert_blocks, spzeros, cross2d
from discretize.base import BaseMesh
from discretize._extensions.simplex_helpers import (
    _build_faces_edges,
    _build_adjacency,
    _directed_search,
    _interp_cc,
)
from discretize.mixins import InterfaceMixins, SimplexMeshIO


def _morton_order(locs, lower, upper):
    """Order points along a Morton (Z-order) curve over the box [lower, upper].

    Consecutive points in this order are usually close together. Points outside
    the box are clamped to it.

    Each coordinate is quantized to an integer (21 bits in 3D, 31 bits in 2D,
    so ``dim`` of them fit in 64 bits), and the Morton key interleaves their
    bits: bit ``b`` of coordinate ``d`` becomes bit ``b * dim + d`` of the key.
    Sorting by the key orders the points along the Z-order curve. Written out
    directly, the key is::

        key = 0
        for b in range(bits):
            for d in range(dim):
                key |= ((q[d] >> b) & 1) << (b * dim + d)

    which is what ``test_morton_order_matches_bit_by_bit`` in
    ``tests/simplex/test_point_location.py`` checks this against.

    Rather than looping over every bit, each coordinate's bits are spread apart
    with a few ``x = (x | (x << shift)) & mask`` steps (the "magic bits"
    method), using the 64-bit shifts and masks from libmorton:

    * 3D: https://github.com/Forceflow/libmorton/blob/1ac6131ad153f64c93a8dcb76dc5bbdc515ba7ee/include/libmorton/morton3D.h#L101-L114
    * 2D: https://github.com/Forceflow/libmorton/blob/1ac6131ad153f64c93a8dcb76dc5bbdc515ba7ee/include/libmorton/morton2D.h#L95-L109

    (Their first step is left out here: it only clears bits above the 21 or 31
    bits that the clipping already guarantees.)
    """
    dim = locs.shape[1]
    if dim == 3:
        bits = 21
        steps = [
            (32, 0x1F00000000FFFF),
            (16, 0x1F0000FF0000FF),
            (8, 0x100F00F00F00F00F),
            (4, 0x10C30C30C30C30C3),
            (2, 0x1249249249249249),
        ]
    else:
        bits = 31
        steps = [
            (16, 0x0000FFFF0000FFFF),
            (8, 0x00FF00FF00FF00FF),
            (4, 0x0F0F0F0F0F0F0F0F),
            (2, 0x3333333333333333),
            (1, 0x5555555555555555),
        ]
    # Quantize each coordinate to an integer in [0, 2**bits - 1].
    n_bins = (1 << bits) - 1
    width = np.where(upper > lower, upper - lower, 1.0)
    q = np.clip((locs - lower) / width * n_bins, 0, n_bins).astype(np.uint64)
    key = np.zeros(len(locs), dtype=np.uint64)
    for d in range(dim):
        # Spread the bits of coordinate d out to every dim-th bit...
        x = q[:, d]
        for shift, mask in steps:
            x = (x | (x << np.uint64(shift))) & np.uint64(mask)
        # ...and interleave it with the others at offset d.
        key |= x << np.uint64(d)
    return np.argsort(key, kind="stable")


class SimplexMesh(BaseMesh, SimplexMeshIO, InterfaceMixins):
    """Class for traingular (2D) and tetrahedral (3D) meshes.

    Simplex is the abstract term for triangular like elements in an arbitrary dimension.
    Simplex meshes are subdivided into trianglular (in 2D) or tetrahedral (in 3D)
    elements. They are capable of representing abstract geometric surfaces, with widely
    variable element sizes.

    Parameters
    ----------
    nodes : (n_nodes, dim) array_like of float
        Defines every node of the mesh.

    simplices : (n_cells, dim+1) array_like of int
        This array defines the connectivity of nodes to form cells. Each element
        indexes into the `nodes` array. Each row defines which nodes make a given cell.
        This array is sorted along each row and then stored on the mesh.

    Notes
    -----
    Only rudimentary checking of the input nodes and simplices is performed, only
    checking for degenerate simplices who have zero volume. There are no checks for
    overlapping cells, or for the quality of the mesh.

    Examples
    --------
    Here we generate a basic 2D triangular mesh, by triangulating a rectangular domain.

    >>> from discretize import SimplexMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.tri as tri

    First we define the nodes of our mesh

    >>> X, Y = np.mgrid[-20:20:21j, -10:10:11j]
    >>> nodes = np.c_[X.reshape(-1), Y.reshape(-1)]

    Then we triangulate the nodes, here we use matplotlib, but you could also use
    scipy's Delaunay, or any other triangular mesh generator. Essentailly we are
    creating every node in the mesh, and a list of triangles/tetrahedrals defining which
    nodes make of each cell.

    >>> triang = tri.Triangulation(nodes[:, 0], nodes[:, 1])
    >>> simplices = triang.triangles

    Finally we can assemble them into a SimplexMesh

    >>> mesh = SimplexMesh(nodes, simplices)
    >>> mesh.plot_grid()
    >>> plt.show()
    """

    _meshType = "simplex"
    _items = {"nodes", "simplices"}

    def __init__(self, nodes, simplices):
        # grab copies of the nodes and simplices for protection
        nodes = np.asarray(nodes)
        simplices = np.asarray(simplices)
        dim = nodes.shape[1]
        if dim not in [3, 2]:
            raise ValueError(
                f"Mesh must be either 2 or 3 dimensions, nodes has {dim} dimension."
            )
        if simplices.shape[1] != dim + 1:
            raise ValueError(
                "simplices second dimension is not compatible with the mesh dimension. "
                f"Saw {simplices.shape[1]}, and expected {dim + 1}."
            )

        self._nodes = nodes.copy()
        self._nodes.setflags(write="false")
        self._simplices = simplices.copy()
        # sort the simplices by node index to simplify further functions...
        self._simplices.sort(axis=1)
        self._simplices.setflags(write="false")

        if self.cell_volumes.min() == 0.0:
            raise ValueError("Triangulation contains degenerate simplices")

        self._number()

    def _number(self):
        items = _build_faces_edges(self.simplices)
        self._simplex_faces = np.array(items[0])
        self._faces = np.array(items[1])
        self._simplex_edges = np.array(items[2])
        self._edges = np.array(items[3])
        self._n_faces = self._faces.shape[0]
        self._n_edges = self._edges.shape[0]
        if self.dim == 3:
            self._face_edges = np.array(items[4])

    @property
    def simplices(self):
        """The node indices for all simplexes of the mesh.

        This array defines the connectivity of the mesh. For each simplex this array is
        sorted by the node index.

        Returns
        -------
        (n_cells, dim + 1) numpy.ndarray of int
        """
        return self._simplices

    @property
    def neighbors(self):
        """
        The adjacancy graph of the mesh.

        This array contains the adjacent cell index for each cell of the mesh. For each
        cell the i'th neighbor is opposite the i'th node, across the i'th face. If a
        cell has no neighbor in a particular direction, then it is listed as having -1.
        This also implies that this is a boundary face.

        Returns
        -------
        (n_cells, dim + 1) numpy.ndarray of int
        """
        if getattr(self, "_neighbors", None) is None:
            self._neighbors = np.array(
                _build_adjacency(self._simplex_faces, self.n_faces)
            )
        return self._neighbors

    @property
    def transform_and_shift(self):
        """
        The barycentric transformation matrix and shift.

        Returns
        -------
        transform : (n_cells, dim, dim) numpy.ndarray
        shift : (n_cells) numpy.ndarray
        """
        if getattr(self, "_transform", None) is None:
            # compute the barycentric transforms
            points = self.nodes
            simplices = self.simplices

            shift = points[self.simplices[:, -1]]

            T = (points[simplices[:, :-1]] - shift[:, None, :]).transpose((0, 2, 1))

            self._transform = invert_blocks(T)
            self._shift = shift
        return self._transform, self._shift

    @property
    def dim(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self.nodes.shape[-1]

    @property
    def n_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self._nodes.shape[0]

    @property
    def nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self._nodes

    @property
    def n_cells(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self.simplices.shape[0]

    @property
    def cell_centers(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_cell_centers", None) is None:
            self._cell_centers = np.mean(self.nodes[self.simplices], axis=1)
        return self._cell_centers

    @property
    def cell_volumes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_cell_volumes", None) is None:
            simplex_nodes = self._nodes[self.simplices]
            mats = np.pad(simplex_nodes, ((0, 0), (0, 0), (0, 1)), constant_values=1)
            V1 = np.abs(np.linalg.det(mats))
            V1 /= 6 if self.dim == 3 else 2
            self._cell_volumes = V1
        return self._cell_volumes

    @property
    def n_edges(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self._n_edges

    @property
    def edges(self):  # NOQA D102
        if getattr(self, "_edge_locs", None) is None:
            self._edge_locs = np.mean(self.nodes[self._edges], axis=1)
        # Documentation inherited from discretize.base.BaseMesh
        return self._edge_locs

    @property
    def edge_tangents(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_edge_tangents", None) is None:
            tangents = np.diff(self.nodes[self._edges], axis=1).squeeze()
            tangents /= np.linalg.norm(tangents, axis=-1)[:, None]
            self._edge_tangents = tangents
        return self._edge_tangents

    @property
    def edge_lengths(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_edge_lengths", None) is None:
            self._edge_lengths = np.linalg.norm(
                np.diff(self.nodes[self._edges], axis=1).squeeze(), axis=-1
            )
        return self._edge_lengths

    @property
    def n_faces(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self._n_faces

    @property
    def faces(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_face_locs", None) is None:
            self._face_locs = np.mean(self.nodes[self._faces], axis=1)
        return self._face_locs

    @property
    def face_areas(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 2:
            return self.edge_lengths
        else:
            if getattr(self, "_face_areas", None) is None:
                face_nodes = self._nodes[self._faces]
                v01 = face_nodes[:, 1] - face_nodes[:, 0]
                v02 = face_nodes[:, 2] - face_nodes[:, 0]
                self._face_areas = np.linalg.norm(np.cross(v01, v02), axis=1) / 2
            return self._face_areas

    @property
    def face_normals(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_face_normals", None) is None:
            if self.dim == 2:
                # Take the normal as being the cross product of edge_tangents
                # and a unit vector in a "3rd" dimension.
                normal = np.c_[self.edge_tangents[:, 1], -self.edge_tangents[:, 0]]
            else:
                # define normal as |01 x 02|
                # therefore clockwise path about the normal is 0->1->2->0
                face_nodes = self._nodes[self._faces]
                v01 = face_nodes[:, 1] - face_nodes[:, 0]
                v02 = face_nodes[:, 2] - face_nodes[:, 0]
                normal = np.cross(v01, v02)
                normal /= np.linalg.norm(normal, axis=1)[:, None]
            self._face_normals = normal
        return self._face_normals

    @property
    def face_divergence(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_face_divergence", None) is None:
            areas = self.face_areas
            normals = self.face_normals

            # approx outward normals (to figure out if face normal is opposite the outward direction)
            test = self.faces[self._simplex_faces] - self.cell_centers[:, None, :]
            dirs = np.einsum("ijk,ijk->ij", normals[self._simplex_faces], test)

            Aijs = areas[self._simplex_faces] / self.cell_volumes[:, None]
            Aijs[dirs < 0] *= -1

            Aijs = Aijs.reshape(-1)
            ind_ptr = (self.dim + 1) * np.arange(self.n_cells + 1)
            col_inds = self._simplex_faces.reshape(-1)
            self._face_divergence = sp.csr_matrix(
                (Aijs, col_inds, ind_ptr), shape=(self.n_cells, self.n_faces)
            )
        return self._face_divergence

    @property
    def nodal_gradient(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_nodal_gradient", None) is None:
            ind_ptr = 2 * np.arange(self.n_edges + 1)
            col_inds = self._edges.reshape(-1)
            Aijs = ((1.0 / self.edge_lengths[:, None]) * [-1, 1]).reshape(-1)
            self._nodal_gradient = sp.csr_matrix(
                (Aijs, col_inds, ind_ptr), shape=(self.n_edges, self.n_nodes)
            )
        return self._nodal_gradient

    @property
    def edge_curl(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_edge_curl", None) is None:
            dim = self.dim
            n_edges = self.n_edges
            if dim == 2:
                face_edges = self._simplex_edges
                face_nodes = self.nodes[self.simplices]
                face_areas = self.cell_volumes
                n_faces = self.n_cells
            else:
                face_edges = self._face_edges
                face_nodes = self.nodes[self._faces]
                face_areas = self.face_areas
                n_faces = self.n_faces

            ind_ptr = 3 * np.arange(n_faces + 1)
            col_inds = face_edges.reshape(-1)

            # edge tangents point from lower node to higher node
            # clockwise about the face normal on face is path from 0 -> 1 -> 2 -> 0
            # so for each face can take the dot product of the path with the edge tangent

            l01 = face_nodes[:, 1] - face_nodes[:, 0]  # path opposite node 2
            l12 = face_nodes[:, 2] - face_nodes[:, 1]  # path opposite node 0
            l20 = face_nodes[:, 0] - face_nodes[:, 2]  # path opposite node 1

            face_path_tangents = self.edge_tangents[face_edges]

            if dim == 2:
                # need to do an adjustment in 2D for the places where the simplices
                # are not oriented counter clockwise about the +z axis
                # cp = np.cross(l01, -l20)
                # cp is a bunch of 1s (where simplices are CCW) and -1s (where simplices are CW)
                # (but we take the sign here to guard against numerical precision)
                cp = np.sign(cross2d(l20, l01))
                face_areas = face_areas * cp
                # don't due *= here

            Aijs = (
                np.c_[
                    np.einsum("ij,ij->i", face_path_tangents[:, 0], l12),
                    np.einsum("ij,ij->i", face_path_tangents[:, 1], l20),
                    np.einsum("ij,ij->i", face_path_tangents[:, 2], l01),
                ]
                / face_areas[:, None]
            ).reshape(-1)

            self._edge_curl = sp.csr_matrix(
                (Aijs, col_inds, ind_ptr), shape=(n_faces, n_edges)
            )
        return self._edge_curl

    def __validate_model(self, model, invert_model=False):
        n_cells = self.n_cells
        dim = self.dim
        # determines the tensor type of the model and reshapes it properly
        model = np.atleast_1d(model)
        n_aniso = ((dim + 1) * dim) // 2
        if model.size == n_aniso * n_cells:
            # model is fully anisotropic
            # reshape it into a stack of dim x dim matrices
            if model.ndim == 1:
                model = model.reshape((-1, n_aniso), order="F")
            vals = model
            if self.dim == 2:
                model = np.stack(
                    [[vals[:, 0], vals[:, 2]], [vals[:, 2], vals[:, 1]]]
                ).transpose((2, 0, 1))
            else:
                model = np.stack(
                    [
                        [vals[:, 0], vals[:, 3], vals[:, 4]],
                        [vals[:, 3], vals[:, 1], vals[:, 5]],
                        [vals[:, 4], vals[:, 5], vals[:, 2]],
                    ]
                ).transpose((2, 0, 1))
        elif model.size == dim * n_cells:
            if model.ndim == 1:
                model = model.reshape((n_cells, dim), order="F")
            model = model.reshape(-1)

        if invert_model:
            if model.ndim == 1:
                model = 1.0 / model
            else:
                model = invert_blocks(model)
        return model

    def __get_inner_product_projection_matrices(
        self, i_type, with_volume=True, return_pointers=True
    ):
        if getattr(self, "_proj_stash", None) is None:
            self._proj_stash = {}
        key = (i_type, with_volume)
        if key not in self._proj_stash:
            dim = self.dim
            n_cells = self.n_cells
            if i_type == "F":
                vecs = self.face_normals
                n_items = self.n_faces
                simplex_items = self._simplex_faces
                if dim == 2:
                    node_items = np.array([[1, 2], [0, 2], [0, 1]])
                else:
                    node_items = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
            elif i_type == "E":
                vecs = self.edge_tangents
                n_items = self.n_edges
                simplex_items = self._simplex_edges
                if dim == 2:
                    node_items = np.array([[1, 2], [0, 2], [0, 1]])
                elif dim == 3:
                    node_items = np.array([[1, 2, 3], [0, 2, 4], [0, 1, 5], [3, 4, 5]])

            Ps = []
            # Precalc indptr and values for the projection matrix
            P_indptr = np.arange(dim * n_cells + 1)
            ones = np.ones(dim * n_cells)
            if with_volume:
                V = np.sqrt(self.cell_volumes / (dim + 1))

            # precalculate indices for the block diagonal matrix
            d = np.ones(dim, dtype=int)[:, None] * np.arange(dim)
            t = np.arange(n_cells)
            T_col_inds = (d + t[:, None, None] * dim).reshape(-1)
            T_ind_ptr = dim * np.arange(dim * n_cells + 1)

            for i in range(dim + 1):
                # array which selects the items associated with node i of each simplex...
                item_inds = np.take(simplex_items, node_items[i], axis=1)
                P_col_inds = item_inds.reshape(-1)
                P = sp.csr_matrix(
                    (ones, P_col_inds, P_indptr), shape=(dim * n_cells, n_items)
                )

                item_vectors = vecs[item_inds]
                trans_inv = invert_blocks(item_vectors)
                if with_volume:
                    trans_inv *= V[:, None, None]
                T = sp.csr_matrix(
                    (trans_inv.reshape(-1), T_col_inds, T_ind_ptr),
                    shape=(dim * n_cells, dim * n_cells),
                )
                Ps.append(T @ P)

            self._proj_stash[key] = (Ps, T_col_inds, T_ind_ptr)
        Ps, T_col_inds, T_ind_ptr = self._proj_stash[key]
        if return_pointers:
            return Ps, (T_col_inds, T_ind_ptr)
        else:
            return Ps

    def __get_inner_product(self, i_type, model, invert_model):
        Ps, (T_col_inds, T_ind_ptr) = self.__get_inner_product_projection_matrices(
            i_type
        )
        n_cells = self.n_cells
        dim = self.dim
        if model is None:
            Mu = Identity()
        else:
            model = self.__validate_model(model, invert_model)
            if model.size == 1:
                Mu = sp.diags((model,), (0,), shape=(dim * n_cells, dim * n_cells))
            elif model.size == n_cells:
                Mu = sp.diags(np.repeat(model, dim))
            elif model.size == dim * n_cells:
                # diagonally anisotropic model
                Mu = sp.diags(model)
            elif model.size == (dim * dim) * n_cells:
                Mu = sp.csr_matrix(
                    (model.reshape(-1), T_col_inds, T_ind_ptr),
                    shape=(dim * n_cells, dim * n_cells),
                )
            else:
                raise ValueError("Unrecognized size of model vector")

        A = np.sum([P.T @ Mu @ P for P in Ps])
        return A

    def get_face_inner_product(  # NOQA D102
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
    ):
        # Documentation inherited from discretize.base.BaseMesh
        if invert_matrix:
            raise NotImplementedError(
                "The inverse of the inner product matrix with a tetrahedral mesh is not supported."
            )
        return self.__get_inner_product("F", model, invert_model)

    def get_edge_inner_product(  # NOQA D102
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
    ):
        # Documentation inherited from discretize.base.BaseMesh
        if invert_matrix:
            raise NotImplementedError(
                "The inverse of the inner product matrix with a tetrahedral mesh is not supported."
            )
        return self.__get_inner_product("E", model, invert_model)

    def get_edge_inner_product_surface(  # NOQA D102
        self, model=None, invert_model=False, invert_matrix=False
    ):
        # Documentation inherited from discretize.base.BaseMesh
        dim = self.dim
        if dim == 2:
            return self.get_face_inner_product_surface(
                model=model, invert_model=invert_model, invert_matrix=invert_matrix
            )

        if invert_matrix:
            raise NotImplementedError(
                "The inverse of the inner product matrix with a tetrahedral mesh is not supported."
            )

        # Edge inner product surface projection matrices
        n_faces = self.n_faces
        face_areas = self.face_areas

        Ps = self._get_edge_surf_int_proj_mats(with_area=False)

        if model is None:
            Mu = sp.diags(np.tile(face_areas, dim))  # Number of edges per face
        else:
            if invert_model:
                model = 1.0 / model

            if (model.size == 1) | (model.size == n_faces):
                Mu = sp.diags(
                    np.tile(model * face_areas, dim)
                )  # Number of edges per face
            else:
                raise ValueError(
                    "Unrecognized size of model vector.",
                    "Must be scalar or have length equal to total number of faces.",
                )

        A = np.sum([P.T @ Mu @ P for P in Ps])

        return A

    def __get_inner_product_deriv_func(self, i_type, model):
        Ps, _ = self.__get_inner_product_projection_matrices(i_type)
        dim = self.dim
        n_cells = self.n_cells
        if model.size == 1:
            tensor_type = 0
        elif model.size == n_cells:
            tensor_type = 1
            col_inds = np.repeat(np.arange(n_cells), dim)
            ind_ptr = np.arange(n_cells * dim + 1)
        elif model.size == dim * n_cells:
            tensor_type = 2
            col_inds = np.arange(dim * n_cells).reshape((n_cells, dim), order="F")
            col_inds = col_inds.reshape(-1)
            ind_ptr = np.arange(n_cells * dim + 1)
        elif model.size == (((dim + 1) * dim) // 2) * n_cells:
            tensor_type = 3
            # create a stencil that goes from the model vector ordering
            # into the anisotropy tensor
            if dim == 2:
                stencil = np.array([[0, 2], [2, 1]])
            elif dim == 3:
                stencil = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])
            col_inds = n_cells * stencil + np.arange(n_cells)[:, None, None]
            col_inds = col_inds.reshape(-1)
            ind_ptr = dim * np.arange(n_cells * dim + 1)
        else:
            raise ValueError("Unrecognized size of model vector")
        inv_items = model.size

        if i_type == "F":
            n_items = self.n_faces
        elif i_type == "E":
            n_items = self.n_edges

        def func(v):
            dMdm = spzeros(n_items, inv_items)
            if tensor_type == 0:
                for P in Ps:
                    dMdm = dMdm + sp.csr_matrix(
                        (P.T * (P * v), (range(n_items), np.zeros(n_items))),
                        shape=(n_items, inv_items),
                    )
            elif tensor_type == 1:
                for P in Ps:
                    ys = P @ v
                    dMdm = dMdm + P.T @ sp.csr_matrix(
                        (ys, col_inds, ind_ptr), shape=(n_cells * dim, inv_items)
                    )
            elif tensor_type == 2:
                for P in Ps:
                    ys = P @ v
                    dMdm = dMdm + P.T @ sp.csr_matrix(
                        (ys, col_inds, ind_ptr), shape=(n_cells * dim, inv_items)
                    )
            elif tensor_type == 3:
                for P in Ps:
                    ys = P @ v
                    ys = np.repeat(ys, dim).reshape((-1, dim, dim))
                    ys = ys.transpose((0, 2, 1)).reshape(-1)
                    dMdm = dMdm + P.T @ sp.csr_matrix(
                        (ys, col_inds, ind_ptr), shape=(n_cells * dim, inv_items)
                    )
            return dMdm

        return func

    def get_face_inner_product_deriv(  # NOQA D102
        self, model, do_fast=True, invert_model=False, invert_matrix=False
    ):
        # Documentation inherited from discretize.base.BaseMesh
        if invert_model:
            raise NotImplementedError(
                "Inverted model derivatives are not supported here"
            )
        if invert_matrix:
            raise NotImplementedError("Inverted matrix derivatives are not supported")
        return self.__get_inner_product_deriv_func("F", model)

    def get_edge_inner_product_deriv(  # NOQA D102
        self, model, do_fast=True, invert_model=False, invert_matrix=False
    ):
        # Documentation inherited from discretize.base.BaseMesh
        if invert_model:
            raise NotImplementedError(
                "Inverted model derivatives are not supported here"
            )
        if invert_matrix:
            raise NotImplementedError("Inverted matrix derivatives are not supported")
        return self.__get_inner_product_deriv_func("E", model)

    def get_edge_inner_product_surface_deriv(  # NOQA D102
        self,
        model,
        invert_model=False,
        invert_matrix=False,
    ):
        # Documentation inherited from discretize.base.BaseMesh
        dim = self.dim
        if dim == 2:
            return super().get_face_inner_product_surface_deriv(
                model=model, invert_model=invert_model, invert_matrix=invert_matrix
            )

        if invert_model:
            raise NotImplementedError(
                "Inverted model derivatives are not supported here"
            )
        if invert_matrix:
            raise NotImplementedError(
                "The inverse of the inner product matrix with a tetrahedral mesh is not supported."
            )
        model = np.asarray(model)
        # Edge inner product surface projection matrices
        n_faces = self.n_faces
        n_edges = self.n_edges
        face_areas = self.face_areas

        Ps = self._get_edge_surf_int_proj_mats(with_area=False)
        area = sp.diags(np.tile(np.sqrt(face_areas), dim))
        Ps = list([area @ P for P in Ps])

        if model.size == 1:

            def func(v):
                dMdm = spzeros(n_edges, 1)
                for P in Ps:
                    dMdm = dMdm + sp.csr_matrix(
                        (P.T @ (P @ v), (range(n_edges), np.zeros(n_edges))),
                        shape=(n_edges, 1),
                    )
                return dMdm

        elif model.size == n_faces:
            col_inds = np.tile(np.arange(n_faces), dim)
            ind_ptr = np.arange(n_faces * dim + 1)

            def func(v):
                dMdm = spzeros(n_edges, n_faces)
                for P in Ps:
                    ys = P @ v
                    dMdm = dMdm + P.T @ sp.csr_matrix(
                        (ys, col_inds, ind_ptr), shape=(n_faces * dim, n_faces)
                    )
                return dMdm

        else:
            raise ValueError(
                "Unrecognized size of model vector.",
                "Must be scalar or have length equal to total number of faces.",
            )
        return func

    def _get_edge_surf_int_proj_mats(self, only_boundary=False, with_area=True):
        """Return the projection operators for integrating edges on each face.

        Parameters
        ----------
        only_boundary : bool, optional
            Whether to only operate on the boundary faces or not.
        with_area : bool, optional
            Whether to include the face area.

        Returns
        -------
        list of (3 * n_faces, n_edges) scipy.sparse.csr_matrix
        """
        # edges associated with each face... can just get the indices of the curl...
        face_edges = self._face_edges
        face_areas = self.face_areas
        if only_boundary:
            bf_inds = self.boundary_face_list
            face_edges = face_edges[bf_inds]
            face_areas = face_areas[bf_inds]
            face_normals = self.boundary_face_outward_normals
        else:
            face_normals = self.face_normals

        # face_edges for these are:
        edge_inds = [[1, 2], [0, 2], [0, 1]]

        n_f = face_edges.shape[0]

        ones = np.ones(n_f * 2)
        P_indptr = np.arange(2 * n_f + 1)

        d = np.ones(3, dtype=int)[:, None] * np.arange(2)
        t = np.arange(n_f)
        T_col_inds = (d + t[:, None, None] * 2).reshape(-1)
        T_ind_ptr = 2 * np.arange(3 * n_f + 1)

        Ps = []

        # translate c to fortran ordering
        C2F_col_inds = np.arange(n_f * 3).reshape((-1, 3), order="C").reshape(-1)
        C2F_row_inds = np.arange(n_f * 3).reshape((-1, 3), order="F").reshape(-1)
        C2F = sp.csr_matrix(
            (np.ones(n_f * 3), (C2F_row_inds, C2F_col_inds)), shape=(n_f * 3, n_f * 3)
        )

        for i in range(3):
            # matrix which selects the edges associate with each of the nodes of each boundary face
            node_edges = face_edges[:, edge_inds[i]]
            P = sp.csr_matrix(
                (ones, node_edges.reshape(-1), P_indptr), shape=(2 * n_f, self.n_edges)
            )

            edge_dirs = self.edge_tangents[node_edges]
            t_for = np.concatenate((edge_dirs, face_normals[:, None, :]), axis=1)
            t_inv = invert_blocks(t_for)
            t_inv = t_inv[:, :, :-1]

            if with_area:
                t_inv *= face_areas[:, None, None]
                t_inv /= 3  # n_edges_per_thing
            else:
                t_inv /= np.sqrt(3)  # sqrt n_edges_per_thing

            T = C2F @ sp.csr_matrix(
                (t_inv.reshape(-1), T_col_inds, T_ind_ptr),
                shape=(3 * n_f, 2 * n_f),
            )
            Ps.append((T @ P))
        return Ps

    @property
    def cell_centers_tree(self):
        """A KDTree object built from the cell centers.

        Returns
        -------
        scipy.spatial.KDTree
        """
        if getattr(self, "_cc_tree", None) is None:
            self._cc_tree = KDTree(self.cell_centers)
        return self._cc_tree

    @property
    def _is_convex(self):
        """Whether the mesh fills its convex hull, with no internal boundaries.

        On such a mesh, a walk that reaches the boundary with the point beyond
        it proves the point is outside. Otherwise (concave boundaries, interior
        voids, cracks) that needs checking.

        The mesh lies inside its hull, so equal volumes rule out concavities
        and voids, and equal boundary areas also rule out cracks, which have no
        volume but add boundary on both of their sides.
        """
        if getattr(self, "_is_convex_mesh", None) is None:
            facets = self._boundary_facets_nodes
            hull = ConvexHull(self.nodes[np.unique(facets)])
            v = self.nodes[facets]
            if self.dim == 2:
                boundary_area = np.linalg.norm(v[:, 1] - v[:, 0], axis=-1).sum()
            else:
                boundary_area = (
                    0.5
                    * np.linalg.norm(
                        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1
                    ).sum()
                )
            self._is_convex_mesh = bool(
                np.isclose(hull.volume, self.cell_volumes.sum(), rtol=1e-10, atol=0)
                and np.isclose(hull.area, boundary_area, rtol=1e-10, atol=0)
            )
            self._hull_equations = np.ascontiguousarray(hull.equations)
        return self._is_convex_mesh

    @property
    def _boundary_hull_equations(self):
        """Unit outward normals and offsets of the convex hull's faces."""
        self._is_convex
        return self._hull_equations

    @property
    def _boundary_facet_grid(self):
        """A uniform grid over the boundary faces.

        Lets a segment be tested against only the boundary faces near it.

        Returns
        -------
        origin : (dim) numpy.ndarray of float
            The grid's lower corner.
        h : float
            Width of its (square) cells.
        shape : (dim) numpy.ndarray of int
            Number of cells along each axis.
        ptr, facets : numpy.ndarray of int
            In CSR form, the boundary faces whose bounding box overlaps each
            cell (in C order).
        """
        if getattr(self, "_facet_grid", None) is None:
            dim = self.dim
            v = self.nodes[self._boundary_facets_nodes]
            f_lo = v.min(axis=1)
            f_hi = v.max(axis=1)
            origin = f_lo.min(axis=0)
            extent = f_hi.max(axis=0) - origin
            n_facets = len(v)
            # Cells about the size of a typical face, but no more than a few
            # cells per face.
            h = max(np.median((f_hi - f_lo).max(axis=1)), 1e-6 * extent.max())
            while np.prod(np.ceil(extent / h)) > 8 * n_facets:
                h *= 1.5
            shape = np.maximum(np.ceil(extent / h).astype(np.int64), 1)
            # Pad the faces' boxes so round off can't miss a cell they touch.
            pad = 1e-6 * h
            i0 = np.clip(((f_lo - pad - origin) / h).astype(np.int64), 0, shape - 1)
            i1 = np.clip(((f_hi + pad - origin) / h).astype(np.int64), 0, shape - 1)
            ext = i1 - i0 + 1
            counts = ext.prod(axis=1)
            facet = np.repeat(np.arange(n_facets, dtype=np.int64), counts)
            off = np.arange(counts.sum()) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            index = np.empty((len(facet), dim), dtype=np.int64)
            for d in range(dim - 1, -1, -1):
                e = ext[facet, d]
                index[:, d] = i0[facet, d] + off % e
                off //= e
            cell = np.ravel_multi_index(index.T, shape)
            order = np.argsort(cell, kind="stable")
            ptr = np.zeros(np.prod(shape) + 1, dtype=np.int64)
            ptr[1:] = np.cumsum(np.bincount(cell, minlength=np.prod(shape)))
            self._facet_grid = (origin, float(h), shape, ptr, facet[order])
        return self._facet_grid

    @property
    def _boundary_facets_nodes(self):
        """Node indices of each boundary face, ordered so its normal points out."""
        return self._boundary_facets_and_cells[0]

    @property
    def _boundary_facet_cells(self):
        """The cell each boundary face belongs to."""
        return self._boundary_facets_and_cells[1]

    @property
    def _boundary_facets_and_cells(self):
        """Outward oriented boundary faces (node indices) and their cells."""
        if getattr(self, "_boundary_facets", None) is None:
            cells, opposite = np.nonzero(self.neighbors == -1)
            simplices = self.simplices[cells]
            # The face opposite local node j holds the other nodes, in order.
            keep = np.arange(self.dim + 1)[None, :] != opposite[:, None]
            facets = simplices[keep].reshape(len(cells), self.dim)
            apex = self.nodes[simplices[np.arange(len(cells)), opposite]]
            v = self.nodes[facets]
            if self.dim == 2:
                d = v[:, 1] - v[:, 0]
                normal = np.c_[d[:, 1], -d[:, 0]]
            else:
                normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            inward = np.einsum("ij,ij->i", normal, v[:, 0] - apex) < 0
            facets[inward, -2:] = facets[inward, -2:][:, ::-1]
            self._boundary_facets = (facets, cells.astype(np.int64))
        return self._boundary_facets

    def _locate(self, locs, spatial_sort=True, nearest_outside=True):
        """Find the simplex containing each point.

        Parameters
        ----------
        locs : (n, dim) array_like of float
            Points to locate.
        spatial_sort : bool, optional
            Walk the points in Morton (Z-order) curve order, so each walk starts
            near its point. If False, walk them in the given order, which is
            as good for points that are already ordered, e.g. along lines.
        nearest_outside : bool, optional
            Give points outside the mesh the cell behind the boundary face
            nearest to them. If False, they get whichever boundary cell their
            walk ended in (for when that cell won't be used).

        Returns
        -------
        inds : (n) numpy.ndarray of int
            The simplex containing each point. For points outside the mesh,
            the simplex behind the nearest boundary face, and -1 for points
            with non-finite coordinates.
        barys : (n, dim + 1) numpy.ndarray of float
            Each point's barycentric coordinates in ``inds`` (zero for points
            with non-finite coordinates).
        outside : (n) numpy.ndarray of bool
            Which points are outside the mesh (including those with non-finite
            coordinates).
        """
        locs = np.atleast_2d(np.asarray(locs, dtype=np.float64))
        if locs.ndim != 2 or locs.shape[1] != self.dim:
            raise ValueError(
                f"locations must have shape (n, {self.dim}), got {locs.shape}"
            )
        n = len(locs)
        if n == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, self.dim + 1)),
                np.empty(0, dtype=bool),
            )
        transform, shift = self.transform_and_shift

        # A NaN would compare as inside the first simplex the walk tests.
        finite = np.flatnonzero(np.isfinite(locs).all(axis=1))
        inds = np.full(n, -1, dtype=np.int64)
        out_barys = np.zeros((n, self.dim + 1))
        outside = np.ones(n, dtype=bool)
        if len(finite) == 0:
            return inds, out_barys, outside

        lower = self.nodes.min(axis=0)
        upper = self.nodes.max(axis=0)
        if spatial_sort and len(finite) > 1:
            order = finite[_morton_order(locs[finite], lower, upper)]
        else:
            order = finite
        # On a convex mesh with no internal boundaries, a walk stopping at the
        # boundary proves the point is outside. Otherwise it could have stopped
        # at a concavity, a void or a crack short of a point inside the mesh,
        # and the walk continues across the gap along a ray to the point.
        # A point outside the bounding box or convex hull is outside without
        # that, and a grid over the boundary faces keeps the ray test local.
        verify = not self._is_convex
        facets = self.nodes[self._boundary_facets_nodes]
        facet_cells = self._boundary_facet_cells
        if verify:
            hull_equations = self._boundary_hull_equations
        else:
            hull_equations = np.empty((0, self.dim + 1))
        if verify or nearest_outside:
            grid = self._boundary_facet_grid
        else:
            grid = (
                np.zeros(self.dim),
                1.0,
                np.ones(self.dim, dtype=np.int64),
                np.zeros(2, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        # Same step limit as scipy's walk, so a walk that gives up costs less
        # than the exhaustive search it falls back to:
        # https://github.com/scipy/scipy/blob/2dd345a7824adb2af902d939a7ba9385c4de4351/scipy/spatial/_qhull.pyx#L1423-L1428
        max_steps = 1 + self.n_cells // 4
        status, last, barys, _, _ = _directed_search(
            np.ascontiguousarray(locs[order]),
            0,
            self.neighbors,
            transform,
            shift,
            max_steps,
            self.cell_centers,
            facets,
            facet_cells,
            verify,
            lower,
            upper,
            hull_equations,
            *grid,
            nearest_outside,
        )
        inds[order] = last
        out_barys[order] = barys
        outside[order] = status == -1
        return inds, out_barys, outside

    def point2index(self, locs, spatial_sort=True):
        """Find cells that contain the given points.

        Parameters
        ----------
        locs : (n, dim) array_like of float
            Points to locate.
        spatial_sort : bool, optional
            The search walks from simplex to simplex, starting each point's walk
            where the previous point's ended. By default the points are first
            sorted along a Morton (Z-order) curve so consecutive points are
            close. Set to False to walk them in the given order instead, which
            is as fast for points that are already ordered (e.g. along lines).

        Returns
        -------
        (n) numpy.ndarray of int
            Index of the cell containing each point. For points outside the
            mesh, the index of the cell behind the boundary face nearest to
            it, and -1 for points with non-finite coordinates.
        """
        return self._locate(locs, spatial_sort=spatial_sort)[0]

    def get_interpolation_matrix(
        self,
        loc,
        location_type="cell_centers",
        zeros_outside=False,
        spatial_sort=True,
        **kwargs,
    ):
        """Construct a linear interpolation matrix from mesh.

        This method constructs a linear interpolation matrix from tensor locations
        (nodes, cell-centers, faces, etc...) on the mesh to a set of arbitrary
        locations.

        Parameters
        ----------
        loc : (n_pts, dim) numpy.ndarray
            Location of points to interpolate to. Must have same dimensions as
            the mesh.
        location_type : str, optional
            Tensor locations on the mesh being interpolated from. *location_type*
            must be one of:

            - 'Ex', 'edges_x'           -> x-component of field defined on x edges
            - 'Ey', 'edges_y'           -> y-component of field defined on y edges
            - 'Ez', 'edges_z'           -> z-component of field defined on z edges
            - 'Fx', 'faces_x'           -> x-component of field defined on x faces
            - 'Fy', 'faces_y'           -> y-component of field defined on y faces
            - 'Fz', 'faces_z'           -> z-component of field defined on z faces
            - 'N', 'nodes'              -> scalar field defined on nodes
            - 'CC', 'cell_centers'      -> scalar field defined on cell centers

            On a 2D mesh, 'Ez' and 'Fz' interpolate the component along its second
            axis, the same as 'Ey' and 'Fy'.
        zeros_outside : bool, optional
            If *False*, values at locations outside the mesh are extrapolated
            from the cell behind the boundary face nearest to them. If *True*,
            values at locations outside the mesh will be zero. Values at
            locations with non-finite coordinates are always zero.
        spatial_sort : bool, optional
            Locating the points walks from cell to cell, starting each point's
            walk where the previous point's ended. By default the points are
            first sorted along a Morton (Z-order) curve so consecutive points are
            close. Set to False to walk them in the given order instead, which is
            as fast for points that are already ordered (e.g. along lines).

        Returns
        -------
        (n_pts, n_loc_type) scipy.sparse.csr_matrix
            A sparse matrix which interpolates the specified tensor quantity on
            mesh to the set of specified locations.
        """
        location_type = self._parse_location_type(location_type)
        loc = np.atleast_2d(loc)
        simplex_nodes = self.simplices
        transform, shift = self.transform_and_shift

        inds, barys, outside = self._locate(
            loc, spatial_sort=spatial_sort, nearest_outside=not zeros_outside
        )
        # Points with non-finite coordinates have zero weights and index -1.
        not_found = inds == -1

        if zeros_outside:
            barys[outside] = 0.0

        n_loc = len(loc)
        location_type = self._parse_location_type(location_type)
        if location_type == "nodes":
            nodes_per_cell = self.dim + 1
            ind_ptr = nodes_per_cell * np.arange(n_loc + 1)
            col_inds = simplex_nodes[inds].reshape(-1)
            Aij = barys.reshape(-1)
            n_items = self.n_nodes
        elif location_type == "cell_centers":
            # detemine which node each point is closest to.
            which_node = simplex_nodes[inds, np.argmax(barys, axis=-1)]
            # this matrix can be used to lookup which cells a given node touch,
            # which will also be the cells used to interpolate from.
            mat = self.average_node_to_cell.T[which_node].tocsr()
            # this will overwrite the "mat" matrices data to create the interpolation
            if np.any(not_found):
                # stand in a finite location, the row is zeroed below
                loc = loc.astype(np.float64, copy=True)
                loc[not_found] = self.nodes[which_node[not_found]]
            _interp_cc(loc, self.cell_centers, mat.data, mat.indices, mat.indptr)
            if zeros_outside or np.any(not_found):
                e = np.ones(n_loc)
                if zeros_outside:
                    e[outside] = 0.0
                e[not_found] = 0.0
                mat = sp.diags(e, format="csr") @ mat
            return mat
        else:
            component = location_type[-1]
            if component == "x":
                i_dir = 0
            elif component == "y":
                i_dir = 1
            else:
                # z, which on a 2D mesh is the last (second) axis
                i_dir = self.dim - 1
            if location_type[:-2] == "edges":
                # grab the barycentric transforms associated with each simplex:
                ts = transform[inds, :, i_dir]
                ts = np.hstack((ts, -ts.sum(axis=1)[:, None]))

                # edges_x, edges_y, edges_z
                # grab edge lengths
                lengths = self.edge_lengths

                # use 1-form Whitney basis functions for (edges)
                edges = self._simplex_edges[inds]

                # (1, 2), (0, 2), (0, 1)
                e0 = (barys[:, 1] * ts[:, 2] - barys[:, 2] * ts[:, 1]) * lengths[
                    edges[:, 0]
                ]
                e1 = (barys[:, 0] * ts[:, 2] - barys[:, 2] * ts[:, 0]) * lengths[
                    edges[:, 1]
                ]
                e2 = (barys[:, 0] * ts[:, 1] - barys[:, 1] * ts[:, 0]) * lengths[
                    edges[:, 2]
                ]
                if self.dim == 3:
                    # (0, 3), (1, 3), (2, 3)
                    e3 = (barys[:, 0] * ts[:, 3] - barys[:, 3] * ts[:, 0]) * lengths[
                        edges[:, 3]
                    ]
                    e4 = (barys[:, 1] * ts[:, 3] - barys[:, 3] * ts[:, 1]) * lengths[
                        edges[:, 4]
                    ]
                    e5 = (barys[:, 2] * ts[:, 3] - barys[:, 3] * ts[:, 2]) * lengths[
                        edges[:, 5]
                    ]
                    Aij = np.c_[e0, e1, e2, e3, e4, e5].reshape(-1)
                    ind_ptr = 6 * np.arange(n_loc + 1)
                else:
                    Aij = np.c_[e0, e1, e2].reshape(-1)
                    ind_ptr = 3 * np.arange(n_loc + 1)
                col_inds = edges.reshape(-1)
                n_items = self.n_edges
            elif location_type[:-2] == "faces":
                # grab the barycentric transforms associated with each simplex:
                ts = transform[inds, :]
                ts = np.hstack((ts, -ts.sum(axis=1)[:, None]))
                # use  Whitney 2 - form basis functions for face vector interp
                faces = self._simplex_faces[inds]
                areas = self.face_areas

                # [1, 2], [0, 2], [0, 1]
                if self.dim == 2:
                    # i j k
                    # t0 t1 t2
                    # 0 0 1
                    # t1 * i - t0 * j

                    f0 = (
                        cross2d(barys[:, 1:], ts[:, 1:, 1 - i_dir]) * areas[faces[:, 0]]
                    )
                    f1 = (
                        cross2d(barys[:, [0, 2]], ts[:, [0, 2], 1 - i_dir])
                        * areas[faces[:, 1]]
                    )
                    f2 = (
                        cross2d(barys[:, :-1], ts[:, :-1, 1 - i_dir])
                        * areas[faces[:, 2]]
                    )
                    if i_dir == 1:
                        f0 *= -1
                        f1 *= -1
                        f2 *= -1
                    Aij = np.c_[f0, f1, f2].reshape(-1)
                    ind_ptr = 3 * np.arange(n_loc + 1)
                else:
                    # [1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]
                    # f123 = (L1 * G_2 x G_3 + L2 * G_3 x G_1 + L3 * G_1 x G_2)
                    # f023 = (L0 * G_2 x G_3 + L2 * G_3 x G_0 + L3 * G_0 x G_2)
                    # f013 = (L0 * G_1 x G_3 + L1 * G_3 x G_0 + L3 * G_0 x G_1)
                    # f012 = (L0 * G_1 x G_2 + L1 * G_2 x G_0 + L2 * G_0 x G_1)
                    f0 = (
                        2
                        * (
                            barys[:, 1] * (np.cross(ts[:, 2], ts[:, 3])[:, i_dir])
                            + barys[:, 2] * (np.cross(ts[:, 3], ts[:, 1])[:, i_dir])
                            + barys[:, 3] * (np.cross(ts[:, 1], ts[:, 2])[:, i_dir])
                        )
                        * areas[faces[:, 0]]
                    )
                    f1 = (
                        2
                        * (
                            barys[:, 0] * (np.cross(ts[:, 2], ts[:, 3])[:, i_dir])
                            + barys[:, 2] * (np.cross(ts[:, 3], ts[:, 0])[:, i_dir])
                            + barys[:, 3] * (np.cross(ts[:, 0], ts[:, 2])[:, i_dir])
                        )
                        * areas[faces[:, 1]]
                    )
                    f2 = (
                        2
                        * (
                            barys[:, 0] * (np.cross(ts[:, 1], ts[:, 3])[:, i_dir])
                            + barys[:, 1] * (np.cross(ts[:, 3], ts[:, 0])[:, i_dir])
                            + barys[:, 3] * (np.cross(ts[:, 0], ts[:, 1])[:, i_dir])
                        )
                        * areas[faces[:, 2]]
                    )
                    f3 = (
                        2
                        * (
                            barys[:, 0] * (np.cross(ts[:, 1], ts[:, 2])[:, i_dir])
                            + barys[:, 1] * (np.cross(ts[:, 2], ts[:, 0])[:, i_dir])
                            + barys[:, 2] * (np.cross(ts[:, 0], ts[:, 1])[:, i_dir])
                        )
                        * areas[faces[:, 3]]
                    )
                    Aij = np.c_[f0, f1, f2, f3].reshape(-1)
                    ind_ptr = 4 * np.arange(n_loc + 1)
                col_inds = faces.reshape(-1)
                n_items = self.n_faces
        return sp.csr_matrix((Aij, col_inds, ind_ptr), shape=(n_loc, n_items))

    @property
    def average_node_to_cell(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_average_node_to_cell", None) is None:
            nodes_per_cell = self.dim + 1
            n_cells = self.n_cells

            ind_ptr = nodes_per_cell * np.arange(n_cells + 1)
            col_inds = self.simplices.reshape(-1)
            Aij = np.full(nodes_per_cell * n_cells, 1 / nodes_per_cell)
            self._average_node_to_cell = sp.csr_matrix(
                (Aij, col_inds, ind_ptr), shape=(n_cells, self.n_nodes)
            )
        return self._average_node_to_cell

    @property
    def average_node_to_face(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_average_node_to_face", None) is None:
            nodes_per_face = self.dim
            n_faces = self.n_faces

            ind_ptr = nodes_per_face * np.arange(n_faces + 1)
            col_inds = self._faces.reshape(-1)
            Aij = np.full(nodes_per_face * n_faces, 1 / nodes_per_face)
            self._average_node_to_face = sp.csr_matrix(
                (Aij, col_inds, ind_ptr), shape=(n_faces, self.n_nodes)
            )
        return self._average_node_to_face

    @property
    def average_node_to_edge(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_average_node_to_edge", None) is None:
            n_edges = self.n_edges

            ind_ptr = 2 * np.arange(n_edges + 1)
            col_inds = self._edges.reshape(-1)
            Aij = np.full(2 * n_edges, 0.5)
            self._average_node_to_edge = sp.csr_matrix(
                (Aij, col_inds, ind_ptr), shape=(n_edges, self.n_nodes)
            )
        return self._average_node_to_edge

    @property
    def average_cell_to_node(self):
        """Averaging matrix from cell centers to nodes.

        The averaging operation uses a volume weighting scheme.

        Returns
        -------
        (n_nodes, n_cells) scipy.sparse.csr_matrix
        """
        # this reproduces linear functions everywhere except on the boundary nodes
        if getattr(self, "_average_cell_to_node", None) is None:
            simps = self.simplices
            cells = np.broadcast_to(
                np.arange(self.n_cells)[:, None], simps.shape
            ).reshape(-1)
            weights = np.broadcast_to(self.cell_volumes[:, None], simps.shape).reshape(
                -1
            )
            simps = simps.reshape(-1)

            A = sp.csr_matrix(
                (weights, (simps, cells)), shape=(self.n_nodes, self.n_cells)
            )
            norm = sp.diags(1.0 / np.asarray(A.sum(axis=1))[:, 0])
            self._average_cell_to_node = norm @ A
        return self._average_cell_to_node

    @property
    def average_cell_to_edge(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        # Simple averaging of all cells with a common edge
        if getattr(self, "_average_cell_to_edge", None) is None:
            simps = self._simplex_edges
            cells = np.broadcast_to(
                np.arange(self.n_cells)[:, None], simps.shape
            ).reshape(-1)
            weights = np.broadcast_to(self.cell_volumes[:, None], simps.shape).reshape(
                -1
            )
            simps = simps.reshape(-1)

            A = sp.csr_matrix(
                (weights, (simps, cells)), shape=(self.n_edges, self.n_cells)
            )
            norm = sp.diags(1.0 / np.asarray(A.sum(axis=1))[:, 0])
            self._average_cell_to_edge = norm @ A
        return self._average_cell_to_edge

    @property
    def average_face_to_cell_vector(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_average_face_to_cell_vector", None) is None:
            dim = self.dim
            n_cells = self.n_cells
            n_faces = self.n_faces

            nodes_per_cell = dim + 1

            Av = sp.csr_matrix((dim * n_cells, n_faces))
            Ps = self.__get_inner_product_projection_matrices(
                "F", with_volume=False, return_pointers=False
            )
            for P in Ps:
                Av = Av + 1 / (nodes_per_cell) * P
            # Av needs to be re-ordered to comply with discretize standard
            ind = np.arange(Av.shape[0]).reshape(n_cells, -1).flatten(order="F")
            P = sp.eye(Av.shape[0], format="csr")[ind]
            self._average_face_to_cell_vector = P @ Av

        return self._average_face_to_cell_vector

    @property
    def average_edge_to_cell_vector(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_average_edge_to_cell_vector", None) is None:
            dim = self.dim
            n_cells = self.n_cells
            n_edges = self.n_edges
            nodes_per_cell = dim + 1

            Av = sp.csr_matrix((dim * n_cells, n_edges))
            # Precalc indptr and values for the projection matrix
            Ps = self.__get_inner_product_projection_matrices(
                "E", with_volume=False, return_pointers=False
            )
            for P in Ps:
                Av = Av + 1 / (nodes_per_cell) * P
            # Av needs to be re-ordered to comply with discretize standard
            ind = np.arange(Av.shape[0]).reshape(n_cells, -1).flatten(order="F")
            P = sp.eye(Av.shape[0], format="csr")[ind]
            self._average_edge_to_cell_vector = P @ Av
        return self._average_edge_to_cell_vector

    @property
    def average_face_to_cell(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_average_face_to_cell", None) is None:
            n_cells = self.n_cells
            n_faces = self.n_faces
            col_inds = self._simplex_faces
            n_face_per_cell = col_inds.shape[1]
            Aij = np.full((n_cells, n_face_per_cell), 1.0 / n_face_per_cell)
            row_ptr = np.arange(n_cells + 1) * (n_face_per_cell)

            self._average_face_to_cell = sp.csr_matrix(
                (Aij.reshape(-1), col_inds.reshape(-1), row_ptr),
                shape=(n_cells, n_faces),
            )
        return self._average_face_to_cell

    @property
    def average_edge_to_cell(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_average_edge_to_cell", None) is None:
            n_cells = self.n_cells
            n_edges = self.n_edges
            col_inds = self._simplex_edges
            n_edge_per_cell = col_inds.shape[1]
            Aij = np.full((n_cells, n_edge_per_cell), 1.0 / (n_edge_per_cell))
            row_ptr = np.arange(n_cells + 1) * (n_edge_per_cell)

            self._average_edge_to_cell = sp.csr_matrix(
                (Aij.reshape(-1), col_inds.reshape(-1), row_ptr),
                shape=(n_cells, n_edges),
            )
        return self._average_edge_to_cell

    @property
    def average_edge_to_face(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 2:
            # in 2D edges are at the same location as faces
            return sp.eye(self.n_edges, self.n_edges)

        if getattr(self, "_average_edge_to_face", None) is None:
            # in 3D there are three edges per face
            n_faces = self.n_faces
            n_edges = self.n_edges
            face_edges = self._face_edges

            ind_ptr = 3 * np.arange(n_faces + 1)
            col_inds = face_edges.reshape(-1)
            Aijs = np.full(3 * self.n_faces, 1 / 3)
            self._average_edge_to_face = sp.csr_matrix(
                (Aijs, col_inds, ind_ptr), shape=(n_faces, n_edges)
            )
        return self._average_edge_to_face

    @property
    def average_cell_to_face(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh

        if getattr(self, "_average_cell_to_face", None) is None:
            A = self.average_face_to_cell.T
            row_sum = np.asarray(A.sum(axis=-1))[:, 0]
            row_sum[row_sum == 0.0] = 1.0
            self._average_cell_to_face = sp.diags(1.0 / row_sum) @ A
        return self._average_cell_to_face

    @property
    def stencil_cell_gradient(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        # An operator that differences cells on each side of a face
        # in the direction of the face normal
        if getattr(self, "_stencil_cell_gradient", None) is None:
            tests = self.cell_centers[:, None, :] - self.faces[self._simplex_faces]
            Aij = np.sign(
                np.einsum(
                    "ijk, ijk -> ij", tests, self.face_normals[self._simplex_faces]
                )
            ).reshape(-1)
            ind_ptr = 3 * np.arange(self.n_cells + 1)
            col_inds = self._simplex_faces.reshape(-1)

            self._stencil_cell_gradient = sp.csr_matrix(
                (Aij, col_inds, ind_ptr), shape=(self.n_cells, self.n_faces)
            ).T
        return self._stencil_cell_gradient

    @property
    def boundary_face_list(self):
        """Boolean array of faces that lie on the boundary of the mesh.

        Returns
        -------
        (n_faces) numpy.ndarray of bool
        """
        if getattr(self, "_boundary_face_list", None) is None:
            ind_dir = np.where(self.neighbors == -1)
            self._is_boundary_face = self._simplex_faces[ind_dir]
        return self._is_boundary_face

    @property
    def project_face_to_boundary_face(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return sp.eye(self.n_faces, format="csr")[self.boundary_face_list]

    @property
    def project_edge_to_boundary_edge(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 2:
            return self.project_face_to_boundary_face
        bound_edges = np.unique(self._face_edges[self.boundary_face_list])
        return sp.eye(self.n_edges, format="csr")[bound_edges]

    @property
    def project_node_to_boundary_node(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        bound_nodes = np.unique(self._faces[self.boundary_face_list])
        return sp.eye(self.n_nodes, format="csr")[bound_nodes]

    @property
    def boundary_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        bound_nodes = np.unique(self._faces[self.boundary_face_list])
        return self.nodes[bound_nodes]

    @property
    def boundary_edges(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 2:
            return self.boundary_faces
        bound_nodes = np.unique(self._face_edges[self.boundary_face_list])
        return self.edges[bound_nodes]

    @property
    def boundary_faces(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self.faces[self.boundary_face_list]

    @property
    def boundary_face_outward_normals(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        bound_cells, which_face = np.where(self.neighbors == -1)
        bound_faces = self._simplex_faces[(bound_cells, which_face)]
        bound_face_normals = self.face_normals[bound_faces]

        out_ish = self.faces[bound_faces] - self.cell_centers[bound_cells]
        direc = np.sign(np.einsum("ij,ij->i", bound_face_normals, out_ish))
        boundary_face_outward_normals = direc[:, None] * bound_face_normals

        return boundary_face_outward_normals

    @property
    def boundary_face_scalar_integral(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        P = self.project_face_to_boundary_face

        w_h_dot_normal = np.sum(
            (P @ self.face_normals) * self.boundary_face_outward_normals, axis=-1
        )
        A = sp.diags(self.face_areas) @ P.T @ sp.diags(w_h_dot_normal)
        return A

    @property
    def boundary_node_vector_integral(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        Pn = self.project_node_to_boundary_node
        Pf = self.project_face_to_boundary_face
        n_boundary_nodes = Pn.shape[0]

        dA = self.boundary_face_outward_normals * (Pf @ self.face_areas)[:, None]

        Av = Pf @ self.average_node_to_face @ Pn.T

        u_dot_ds = Av.T @ dA
        diags = u_dot_ds.T
        offsets = n_boundary_nodes * np.arange(self.dim)

        return Pn.T @ sp.diags(
            diags, offsets, shape=(n_boundary_nodes, self.dim * n_boundary_nodes)
        )

    @property
    def boundary_edge_vector_integral(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        boundary_faces = self.boundary_face_list
        if self.dim == 2:
            boundary_face_edges = boundary_faces
            dA = (
                self.boundary_face_outward_normals
                * self.face_areas[boundary_faces][:, None]
            )

            # projection matrices
            # for each edge on boundary faces
            Pe = self.project_edge_to_boundary_edge
            n_boundary_edges, n_edges = Pe.shape
            index = boundary_face_edges
            w_cross_n = cross2d(dA, self.edge_tangents[index])
            M_be = (
                sp.csr_matrix((w_cross_n, (index, index)), shape=(n_edges, n_edges))
                @ Pe.T
            )
        else:
            Ps = self._get_edge_surf_int_proj_mats(only_boundary=True, with_area=True)
            # the cross product of a vector defined on each face with the face outward normal...
            # so V cross n = -n cross V
            cx = sp.diags(self.boundary_face_outward_normals[:, 0])
            cy = sp.diags(self.boundary_face_outward_normals[:, 1])
            cz = sp.diags(self.boundary_face_outward_normals[:, 2])
            # the negative cross mat
            cross_mat = sp.bmat([[None, cz, -cy], [-cz, None, cx], [cy, -cx, None]])

            Pf = self.project_face_to_boundary_face
            Pe = self.project_edge_to_boundary_edge
            Av = (Pf @ self.average_edge_to_face) @ Pe.T
            Av = cross_mat @ sp.block_diag((Av, Av, Av))

            M_be = np.sum(Ps).T @ Av
        return M_be

    def __reduce__(self):
        """Return the class and attributes necessary to reconstruct the mesh."""
        return self.__class__, (
            self.nodes,
            self.simplices,
        )
