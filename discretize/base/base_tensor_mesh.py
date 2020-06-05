"""
Base class for tensor-product style meshes
"""

import numpy as np
import scipy.sparse as sp
from typing import List
from .pydantic.typing import Array
from pydantic import validator

from .base_mesh import BaseMesh
from .. import utils

class BaseTensorMesh(BaseMesh):
    """Base class for tensor-product style meshes

    This class contains properites and methods that are common to cartesian
    and cylindrical meshes defined by tensor-produts of vectors describing
    cell spacings.

    Do not use this class directly, instead, inherit it if you plan to develop
    a tensor-style mesh (e.g. a spherical mesh) or use the
    :meth:`discretize.TensorMesh` class to create a cartesian tensor mesh.

    Attributes
    ----------

    h : list of numpy.ndarray
    shape : numpy.ndarray
    x0 : numpy.ndarray
    reference_system : str
    axis_u : numpy.ndarray
    axis_v : numpy.ndarray
    axis_w : numpy.ndarray
    dim
    nC
    vnC
    vectorCCx
    vectorCCy
    vectorCCz
    gridCC
    hx
    hy
    hz
    h_gridded
    nN
    vectorNx
    vectorNy
    vectorNz
    gridN
    nE
    vnE
    nEx
    gridEx
    nEy
    gridEy
    nEz
    gridEz
    nF
    vnF
    nFx
    gridFx
    nFy
    gridFy
    nFz
    gridFz
    normals
    tangents
    reference_is_rotated
    rotation_matrix
    """

    _meshType = 'BASETENSOR'

    _unitDimensions = [1, 1, 1]

    # properties
    h: List[Array[float, -1]]

    def __init__(self, h, x0, **kwargs):
        """
        Parameters
        ----------
        h : list of numpy.ndarray, list of list of (int, float) tuples, or list of int
            Describes the cell widths in each dimension. The length of the list is the
            dimensionality of the mesh. For more information please see `utils.meshTensor`.
        x0 : array_like of str or float, or str
            Origin of the mesh. If the elements of the array are strings, '0', 'C', or 'N'
            are valid, and correspond to [0, 0, 0], [-hi.sum()/2 for hi in h], or
            [-hi.sum() for hi in h], respectively.
        reference_system: ['cartesian', 'cylindrical', 'spherical'], optional
            The type of coordinate reference frame. Can take on the values
            cartesian, cylindrical, or spherical. Abbreviations of these are allowed.
        axis_u : array_like, optional
            length three array of floats for orientation direction of axis 1, defaults to [1, 0, 0]
        axis_v : array_like, optional
            length three array of floats for orientation direction of axis 2, defaults to [0, 1, 0]
        axis_w : array_like, optional
            length three array of floats for orientation direction of axis 2, defaults to [0, 0, 1]
        """
        h = self._easy_validate('h', h)
        if 'shape' not in kwargs:
            shape = [len(hi) for hi in h]
        else:
            shape = kwargs.pop('shape')

        x0_in = x0
        x0 = self._easy_validate('x0', x0_in)
        super().__init__(h=h, shape=shape, x0=x0, **kwargs)

        # now self.h is set so must call this again to revalidate the string chars
        self.x0 = self._easy_validate('x0', x0_in)

    @validator('h', pre=True)
    def _prevalid_h(cls, v):
        # validate that it is a list before other things
        if not isinstance(v, (list, tuple)):
            raise ValueError(f'h must be a list or tuple not {type(v)}')
        if len(v) > 3:
            raise ValueError(f'h must be of dimension less than 3, not {len(v)}')

        # First try to convert it to
        h = []
        for i, v_i in enumerate(v):
            try:
                v_i = np.linspace(0, cls._unitDimensions[i], int(v_i))
            except TypeError:
                try:
                    v_i = utils.meshTensor(v_i)
                except:
                    pass
            h.append(v_i)
        return h

    @validator('x0', pre=True)
    def _prevalid_x0(cls, v, values, **kwargs):
        v_out = []
        if 'h' in values:
            h = values['h']
            if len(v) != len(h):
                raise ValueError('Dimension mismatch. len(x0) != len(h)')
        else:
            h = [np.array[unit] for unit in cls._unitDimension]
        for i, vi in enumerate(v):
            if isinstance(vi, str):
                if vi.lower() == '0':
                    vi = 0
                elif vi.lower() == 'c':
                    vi = -h[i].sum()/2
                elif vi.lower() == 'n':
                    vi = -h[i].sum
                else:
                    raise ValueError(f"x0 string character {vi} not recognized, must be either '0', 'N', or 'C'.")
            v_out.append(vi)
        return v_out

    @property
    def hx(self):
        """Width of cells in the x direction"""
        return self.h[0]

    @property
    def hy(self):
        "Width of cells in the y direction"
        return None if self.dim < 2 else self.h[1]

    @property
    def hz(self):
        "Width of cells in the z direction"
        return None if self.dim < 3 else self.h[2]

    @property
    def vectorNx(self):
        """Nodal grid vector (1D) in the x direction."""
        return np.r_[0., self.hx.cumsum()] + self.x0[0]

    @property
    def vectorNy(self):
        """Nodal grid vector (1D) in the y direction."""
        return (
            None if self.dim < 2 else np.r_[0., self.hy.cumsum()] + self.x0[1]
        )

    @property
    def vectorNz(self):
        """Nodal grid vector (1D) in the z direction."""
        return (
            None if self.dim < 3 else np.r_[0., self.hz.cumsum()] + self.x0[2]
        )

    @property
    def vectorCCx(self):
        """Cell-centered grid vector (1D) in the x direction."""
        return np.r_[0, self.hx[:-1].cumsum()] + self.hx*0.5 + self.x0[0]

    @property
    def vectorCCy(self):
        """Cell-centered grid vector (1D) in the y direction."""
        return (
            None if self.dim < 2 else
            np.r_[0, self.hy[:-1].cumsum()] + self.hy*0.5 + self.x0[1]
        )

    @property
    def vectorCCz(self):
        """Cell-centered grid vector (1D) in the z direction."""
        return (
            None if self.dim < 3 else
            np.r_[0, self.hz[:-1].cumsum()] + self.hz*0.5 + self.x0[2]
        )

    @property
    def gridCC(self):
        """Cell-centered grid."""
        return self._getTensorGrid('CC')

    @property
    def gridN(self):
        """Nodal grid."""
        return self._getTensorGrid('N')

    @property
    def h_gridded(self):
        """
        Returns an (nC, dim) numpy array with the widths of all cells in order
        """

        if self.dim == 1:
            return np.reshape(self.h, (self.nC, 1))
        elif self.dim == 2:
            hx = np.kron(np.ones(self.nCy), self.h[0])
            hy = np.kron(self.h[1], np.ones(self.nCx))
            return np.c_[hx, hy]
        elif self.dim == 3:
            hx = np.kron(np.ones(self.nCy*self.nCz), self.h[0])
            hy = np.kron(np.ones(self.nCz), np.kron(self.h[1], np.ones(self.nCx)))
            hz = np.kron(self.h[2], np.ones(self.nCx*self.nCy))
            return np.c_[hx, hy, hz]

    @property
    def gridFx(self):
        """Face staggered grid in the x direction."""
        if self.nFx == 0:
            return
        return self._getTensorGrid('Fx')

    @property
    def gridFy(self):
        """Face staggered grid in the y direction."""
        if self.nFy == 0 or self.dim < 2:
            return
        return self._getTensorGrid('Fy')

    @property
    def gridFz(self):
        """Face staggered grid in the z direction."""
        if self.nFz == 0 or self.dim < 3:
            return
        return self._getTensorGrid('Fz')

    @property
    def gridEx(self):
        """Edge staggered grid in the x direction."""
        if self.nEx == 0:
            return
        return self._getTensorGrid('Ex')

    @property
    def gridEy(self):
        """Edge staggered grid in the y direction."""
        if self.nEy == 0 or self.dim < 2:
            return
        return self._getTensorGrid('Ey')

    @property
    def gridEz(self):
        """Edge staggered grid in the z direction."""
        if self.nEz == 0 or self.dim < 3:
            return
        return self._getTensorGrid('Ez')

    def _getTensorGrid(self, key):
        if getattr(self, '_grid' + key, None) is None:
            setattr(self, '_grid' + key, utils.ndgrid(self.getTensor(key)))
        return getattr(self, '_grid' + key)

    def getTensor(self, key):
        """ Returns a tensor list.

        Parameters
        ----------

        key : str
            Which tensor (see below)

            key can be::

                'CC'    -> scalar field defined on cell centers
                'N'     -> scalar field defined on nodes
                'Fx'    -> x-component of field defined on faces
                'Fy'    -> y-component of field defined on faces
                'Fz'    -> z-component of field defined on faces
                'Ex'    -> x-component of field defined on edges
                'Ey'    -> y-component of field defined on edges
                'Ez'    -> z-component of field defined on edges

        Returns
        -------

        list
            list of the tensors that make up the mesh.

        """

        if key == 'Fx':
            ten = [self.vectorNx, self.vectorCCy, self.vectorCCz]
        elif key == 'Fy':
            ten = [self.vectorCCx, self.vectorNy, self.vectorCCz]
        elif key == 'Fz':
            ten = [self.vectorCCx, self.vectorCCy, self.vectorNz]
        elif key == 'Ex':
            ten = [self.vectorCCx, self.vectorNy, self.vectorNz]
        elif key == 'Ey':
            ten = [self.vectorNx, self.vectorCCy, self.vectorNz]
        elif key == 'Ez':
            ten = [self.vectorNx, self.vectorNy, self.vectorCCz]
        elif key == 'CC':
            ten = [self.vectorCCx, self.vectorCCy, self.vectorCCz]
        elif key == 'N':
            ten = [self.vectorNx, self.vectorNy, self.vectorNz]

        return [t for t in ten if t is not None]

    # --------------- Methods ---------------------

    def isInside(self, pts, locType='N'):
        """
        Determines if a set of points are inside a mesh.

        :param numpy.ndarray pts: Location of points to test
        :rtype: numpy.ndarray
        :return: inside, numpy array of booleans
        """
        pts = utils.asArray_N_x_Dim(pts, self.dim)

        tensors = self.getTensor(locType)

        if locType == 'N' and self._meshType == 'CYL':
            # NOTE: for a CYL mesh we add a node to check if we are inside in
            # the radial direction!
            tensors[0] = np.r_[0., tensors[0]]
            tensors[1] = np.r_[tensors[1], 2.0*np.pi]

        inside = np.ones(pts.shape[0], dtype=bool)
        for i, tensor in enumerate(tensors):
            TOL = np.diff(tensor).min() * 1.0e-10
            inside = (
                inside &
                (pts[:, i] >= tensor.min()-TOL) &
                (pts[:, i] <= tensor.max()+TOL)
            )
        return inside

    def _getInterpolationMat(self, loc, locType='CC', zerosOutside=False):
        """ Produces interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        locType: stc
            What to interpolate

            locType can be::

                'Ex'    -> x-component of field defined on edges
                'Ey'    -> y-component of field defined on edges
                'Ez'    -> z-component of field defined on edges
                'Fx'    -> x-component of field defined on faces
                'Fy'    -> y-component of field defined on faces
                'Fz'    -> z-component of field defined on faces
                'N'     -> scalar field defined on nodes
                'CC'    -> scalar field defined on cell centers
                'CCVx'  -> x-component of vector field defined on cell centers
                'CCVy'  -> y-component of vector field defined on cell centers
                'CCVz'  -> z-component of vector field defined on cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the interpolation matrix

        """

        loc = utils.asArray_N_x_Dim(loc, self.dim)

        if zerosOutside is False:
            assert np.all(self.isInside(loc)), "Points outside of mesh"
        else:
            indZeros = np.logical_not(self.isInside(loc))
            loc[indZeros, :] = np.array([
                v.mean() for v in self.getTensor('CC')
            ])

        if locType in ['Fx', 'Fy', 'Fz', 'Ex', 'Ey', 'Ez']:
            ind = {'x': 0, 'y': 1, 'z': 2}[locType[1]]
            assert self.dim >= ind, 'mesh is not high enough dimension.'
            nF_nE = self.vnF if 'F' in locType else self.vnE
            components = [utils.spzeros(loc.shape[0], n) for n in nF_nE]
            components[ind] = utils.interpmat(loc, *self.getTensor(locType))
            # remove any zero blocks (hstack complains)
            components = [comp for comp in components if comp.shape[1] > 0]
            Q = sp.hstack(components)

        elif locType in ['CC', 'N']:
            Q = utils.interpmat(loc, *self.getTensor(locType))

        elif locType in ['CCVx', 'CCVy', 'CCVz']:
            Q = utils.interpmat(loc, *self.getTensor('CC'))
            Z = utils.spzeros(loc.shape[0], self.nC)
            if locType == 'CCVx':
                Q = sp.hstack([Q, Z, Z])
            elif locType == 'CCVy':
                Q = sp.hstack([Z, Q, Z])
            elif locType == 'CCVz':
                Q = sp.hstack([Z, Z, Q])

        else:
            raise NotImplementedError(
                'getInterpolationMat: locType==' + locType +
                ' and mesh.dim==' + str(self.dim)
            )

        if zerosOutside:
            Q[indZeros, :] = 0

        return Q.tocsr()

    def getInterpolationMat(self, loc, locType='CC', zerosOutside=False):
        """ Produces interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        locType : str
            What to interpolate (see below)

            locType can be::

                'Ex'    -> x-component of field defined on edges
                'Ey'    -> y-component of field defined on edges
                'Ez'    -> z-component of field defined on edges
                'Fx'    -> x-component of field defined on faces
                'Fy'    -> y-component of field defined on faces
                'Fz'    -> z-component of field defined on faces
                'N'     -> scalar field defined on nodes
                'CC'    -> scalar field defined on cell centers
                'CCVx'  -> x-component of vector field defined on cell centers
                'CCVy'  -> y-component of vector field defined on cell centers
                'CCVz'  -> z-component of vector field defined on cell centers

        Returns
        -------

        scipy.sparse.csr_matrix
            M, the interpolation matrix

        """
        return self._getInterpolationMat(loc, locType, zerosOutside)

    def _fastInnerProduct(
        self, projType, prop=None, invProp=False, invMat=False
    ):
        """ Fast version of getFaceInnerProduct.
            This does not handle the case of a full tensor prop.

        Parameters
        ----------

        prop : numpy.array
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))

        projType : str
            'E' or 'F'

        returnP : bool
            returns the projection matrices

        invProp : bool
            inverts the material property

        invMat : bool
            inverts the matrix

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the inner product matrix (nF, nF)

        """
        assert projType in ['F', 'E'], (
            "projType must be 'F' for faces or 'E' for edges"
        )

        if prop is None:
            prop = np.ones(self.nC)

        if invProp:
            prop = 1./prop

        if utils.isScalar(prop):
            prop = prop*np.ones(self.nC)

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == 'CYL':
            n_elements = np.sum(getattr(self, 'vn'+projType).nonzero())
        else:
            n_elements = self.dim

        # Isotropic? or anisotropic?
        if prop.size == self.nC:
            Av = getattr(self, 'ave'+projType+'2CC')
            Vprop = self.vol * utils.mkvc(prop)
            M = n_elements * utils.sdiag(Av.T * Vprop)

        elif prop.size == self.nC*self.dim:
            Av = getattr(self, 'ave'+projType+'2CCV')

            # if cyl, then only certain components are relevant due to symmetry
            # for faces, x, z matters, for edges, y (which is theta) matters
            if self._meshType == 'CYL':
                if projType == 'E':
                    prop = prop[:, 1]  # this is the action of a projection mat
                elif projType == 'F':
                    prop = prop[:, [0, 2]]

            V = sp.kron(sp.identity(n_elements), utils.sdiag(self.vol))
            M = utils.sdiag(Av.T * V * utils.mkvc(prop))
        else:
            return None

        if invMat:
            return utils.sdInv(M)
        else:
            return M

    def _fastInnerProductDeriv(self, projType, prop, invProp=False,
                               invMat=False):
        """

        Parameters
        ----------

        projType : str
            'E' or 'F'

        tensorType : TensorType
            type of the tensor

        invProp : bool
            inverts the material property

        invMat : bool
            inverts the matrix


        Returns
        -------
        function
            dMdmu, the derivative of the inner product matrix

        """
        assert projType in ['F', 'E'], ("projType must be 'F' for faces or 'E'"
                                        " for edges")

        tensorType = utils.TensorType(self, prop)

        dMdprop = None

        if invMat or invProp:
            MI = self._fastInnerProduct(projType, prop, invProp=invProp,
                                        invMat=invMat)

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == 'CYL':
            n_elements = np.sum(getattr(self, 'vn'+projType).nonzero())
        else:
            n_elements = self.dim

        if tensorType == 0:  # isotropic, constant
            Av = getattr(self, 'ave'+projType+'2CC')
            V = utils.sdiag(self.vol)
            ones = sp.csr_matrix(
                (np.ones(self.nC), (range(self.nC), np.zeros(self.nC))),
                shape=(self.nC, 1)
            )
            if not invMat and not invProp:
                dMdprop = n_elements * Av.T * V * ones
            elif invMat and invProp:
                dMdprop =  n_elements * (
                    utils.sdiag(MI.diagonal()**2) * Av.T * V * ones *
                    utils.sdiag(1./prop**2)
                )
            elif invProp:
                dMdprop = n_elements * Av.T * V * utils.sdiag(- 1./prop**2)
            elif invMat:
                dMdprop = n_elements * (
                    utils.sdiag(- MI.diagonal()**2) * Av.T * V
                )

        elif tensorType == 1:  # isotropic, variable in space
            Av = getattr(self, 'ave'+projType+'2CC')
            V = utils.sdiag(self.vol)
            if not invMat and not invProp:
                dMdprop = n_elements * Av.T * V
            elif invMat and invProp:
                dMdprop =  n_elements * (
                    utils.sdiag(MI.diagonal()**2) * Av.T * V *
                    utils.sdiag(1./prop**2)
                )
            elif invProp:
                dMdprop = n_elements * Av.T * V * utils.sdiag(-1./prop**2)
            elif invMat:
                dMdprop = n_elements * (
                    utils.sdiag(- MI.diagonal()**2) * Av.T * V
                )

        elif tensorType == 2: # anisotropic
            Av = getattr(self, 'ave'+projType+'2CCV')
            V = sp.kron(sp.identity(self.dim), utils.sdiag(self.vol))

            if self._meshType == 'CYL':
                Zero = sp.csr_matrix((self.nC, self.nC))
                Eye = sp.eye(self.nC)
                if projType == 'E':
                    P = sp.hstack([Zero, Eye, Zero])
                    # print(P.todense())
                elif projType == 'F':
                    P = sp.vstack([sp.hstack([Eye, Zero, Zero]),
                                   sp.hstack([Zero, Zero, Eye])])
                    # print(P.todense())
            else:
                P = sp.eye(self.nC*self.dim)

            if not invMat and not invProp:
                dMdprop = Av.T * P * V
            elif invMat and invProp:
                dMdprop = (utils.sdiag(MI.diagonal()**2) * Av.T * P * V *
                           utils.sdiag(1./prop**2))
            elif invProp:
                dMdprop = Av.T * P * V * utils.sdiag(-1./prop**2)
            elif invMat:
                dMdprop = utils.sdiag(- MI.diagonal()**2) * Av.T * P * V

        if dMdprop is not None:
            def innerProductDeriv(v=None):
                if v is None:
                    warnings.warn(
                        "Depreciation Warning: TensorMesh.innerProductDeriv."
                        " You should be supplying a vector. "
                        "Use: sdiag(u)*dMdprop", FutureWarning
                    )
                    return dMdprop
                return utils.sdiag(v) * dMdprop
            return innerProductDeriv
        else:
            return None
