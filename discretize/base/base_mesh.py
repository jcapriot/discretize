"""
Base classes for all discretize meshes
"""

import numpy as np
import os
import json

from ..pydantic.base_object import BaseDiscretize
from ..pydantic.typing import Array, UnitaryArray
try:
    from typing import Instance
except ImportError:
    from typing_extensions import Instance
from pydantic import validator, root_validator
from ..utils import mkvc
from ..mixins import InterfaceMixins


class BaseMesh(BaseDiscretize, InterfaceMixins):
    """The base type for regular structured meshes

    `BaseMesh` does all the counting you don't want to do.
    `BaseMesh` should be inherited by meshes with a regular structure.

    Attributes
    ----------
    shape : numpy.ndarray
    x0 : numpy.ndarray
    reference_system : str
    axis_u : numpy.ndarray
    axis_v : numpy.ndarray
    axis_w : numpy.ndarray
    dim
    nC
    nN
    nE
    vnE
    nEx
    nEy
    nEz
    nF
    vnF
    nFx
    nFy
    nFz
    normals
    tangents
    reference_is_rotated
    rotation_matrix
    """

    # Properties
    shape: Array[int, -1]
    x0: Array[float, -1]
    # Optional properties
    axis_u: UnitaryArray[float, 3] = [1.0, 0.0, 0.0]
    axis_v: UnitaryArray[float, 3] = [0.0, 1.0, 0.0]
    axis_w: UnitaryArray[float, 3] = [0.0, 0.0, 1.0]

    reference_system: Instance['cartesian', 'cylindrical', 'spherical'] = 'cartesian'

    # Instantiate the class
    def __init__(self, shape, x0, **kwargs):
        """BaseMesh(shape, x0, reference_system='cartesian', axis_u=[1, 0, 0], axis_v=[0, 1, 0], axis_w=[0, 0, 1])

        Parameters
        ----------
        shape : array_like
            list of integers, must be have length 3 or less
        x0 : array_like
            list of floats defining the origin of the mesh. Must also have length 3 or less
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
        super().__init__(shape=shape, x0=x0)

    # Validators
    @validator('shape')
    def _check_shape(cls, v, values, **kwargs):
        if len(v) > 3:
            raise ValueError(f"Dimensions of {v}, which is higher than 3, are "
                             "not supported")
        if 'x0' in values and len(values['x0']) != len(v):
            # can't change dimension of the mesh
            raise ValueError(
                "Cannot change dimensionality of the mesh. Expected {} "
                "dimensions, got {} dimensions".format(
                    len(change['previous']), len(change['value'])
                )
            )
        if 'nodes' in values:
            # check that if nodes have been set for curvi mesh, sizes still
            # agree
            nodes = values['nodes']
            node_shape = np.array(nodes[0].shape)-1
            if np.any(v != node_shape):
                raise TypeError(
                    f"Mismatched shape of n. Expected {node_shape}, "
                    f"got {v}"
                )

    @validator('x0')
    def _check_x0(cls, v, values, **kwargs):
        if 'shape' in values and len(v) != len(values['shape']):
            raise ValueError(
                f"Dimension mismatch. x0 has length {len(v)} != len(shape) which is "
                f"{len(values['shape'])}"
            )

    @root_validator
    def _check_ortho(cls, values):
        axis_u = values.get('axis_u')
        axis_v = values.get('axis_v')
        axis_w = values.get('axis_w')
        if axis_u is not None and axis_v is not None and axis_w is not None and (
                np.abs(axis_u.dot(axis_w)) > 1E-6 or
                np.abs(axis_v.dot(axis_w)) > 1E-6 or
                np.abs(axis_u.dot(axis_v)) > 1E-6
            ):
            raise ValueError('axis_u, axis_v, and axis_w must be orthogonal')

    @validator('reference_system', pre=True)
    def _validate_reference_system(cls, v):
        v = v.lower()
        abrevs = {
            'car': 'cartesian',
            'cart': 'cartesian',
            'cy': 'cylindrical',
            'cyl': 'cylindrical',
            'sph': 'spherical',
        }
        if v in abrevs:
            v = abrevs[v]
        return v

    @property
    def dim(self):
        """The dimension of the mesh (1, 2, or 3).

        Returns
        -------
        int
            dimension of the mesh
        """
        return len(self.shape)

    @property
    def nC(self):
        """Total number of cells in the mesh.

        Returns
        -------
        int
            number of cells in the mesh

        Examples
        --------
        .. plot::
            :include-source:

            import discretize
            import numpy as np
            mesh = discretize.TensorMesh([np.ones(n) for n in [2,3]])
            mesh.plotGrid(centers=True, showIt=True)

            print(mesh.nC)
        """
        return int(self.shape.prod())

    def __len__(self):
        return self.nC

    @property
    def nN(self):
        """Total number of nodes

        Returns
        -------
        int
            number of nodes in the mesh

        Examples
        --------
        .. plot::
            :include-source:

            import discretize
            import numpy as np
            mesh = discretize.TensorMesh([np.ones(n) for n in [2,3]])
            mesh.plotGrid(nodes=True, showIt=True)

            print(mesh.nN)
        """
        return int((self.shape+1).prod())

    @property
    def nEx(self):
        """Number of x-edges

        Returns
        -------
        nEx : int

        """
        return int((self.shape + np.r_[0, 1, 1][:self.dim]).prod())

    @property
    def nEy(self):
        """Number of y-edges

        Returns
        -------
        nEy : int or None
            None if dimension < 2
        """
        if self.dim < 2:
            return None
        return int((self.shape + np.r_[1, 0, 1][:self.dim]).prod())

    @property
    def nEz(self):
        """Number of z-edges

        Returns
        -------
        nEz : int or None
            None if dimension < 3
        """
        if self.dim < 3:
            return None
        return int((self.shape + np.r_[1, 1, 0][:self.dim]).prod())

    @property
    def vnE(self):
        """Total number of edges in each direction

        Returns
        -------
        vnE : numpy.ndarray
            [nEx, ], [nEx, nEy, ], or [nEx, nEy, nEz] (depending on dimensions)

        .. plot::
            :include-source:

            import discretize
            import numpy as np
            M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
            M.plotGrid(edges=True, showIt=True)
        """
        return np.array(
            [x for x in [self.nEx, self.nEy, self.nEz] if x is not None],
            dtype=int
        )

    @property
    def nE(self):
        """Total number of edges.

        Returns
        -------
        nE : int
            sum([nEx, nEy, nEz])

        """
        return int(self.vnE.sum())

    @property
    def nFx(self):
        """Number of x-faces

        Returns
        -------
        nFx : int
        """
        return int((self.shape + np.r_[1, 0, 0][:self.dim]).prod())

    @property
    def nFy(self):
        """Number of y-faces

        Returns
        -------
        nFy : int or None
            None if dimension < 2
        """
        if self.dim < 2:
            return None
        return int((self.shape + np.r_[0, 1, 0][:self.dim]).prod())

    @property
    def nFz(self):
        """Number of z-faces

        Returns
        -------
        nFz : int or None
            None if dimension <3
        """
        if self.dim < 3:
            return None
        return int((self.shape + np.r_[0, 0, 1][:self.dim]).prod())

    @property
    def vnF(self):
        """Total number of faces in each direction

        Returns
        -------
        vnF : numpy.ndarray
            [nFx, ], [nFx, nFy], or [nFx, nFy, nFz] (depending on dimension)

        .. plot::
            :include-source:

            import discretize
            import numpy as np
            M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
            M.plotGrid(faces=True, showIt=True)
        """
        return np.array(
            [x for x in [self.nFx, self.nFy, self.nFz] if x is not None],
            dtype=int
        )

    @property
    def nF(self):
        """Total number of faces.

        Returns
        -------
        nF : int
            sum([nFx, nFy, nFz])

        """
        return int(self.vnF.sum())

    @property
    def normals(self):
        """Vectors normal to each Face

        Returns
        -------
        normals : numpy.ndarray
            float array of shape (nF, dim)
        """
        if self.dim == 2:
            nX = np.c_[
                np.ones(self.nFx), np.zeros(self.nFx)
            ]
            nY = np.c_[
                np.zeros(self.nFy), np.ones(self.nFy)
            ]
            return np.r_[nX, nY]
        elif self.dim == 3:
            nX = np.c_[
                np.ones(self.nFx), np.zeros(self.nFx), np.zeros(self.nFx)
            ]
            nY = np.c_[
                np.zeros(self.nFy), np.ones(self.nFy), np.zeros(self.nFy)
            ]
            nZ = np.c_[
                np.zeros(self.nFz), np.zeros(self.nFz), np.ones(self.nFz)
            ]
            return np.r_[nX, nY, nZ]

    @property
    def tangents(self):
        """Vectors tangent to each Edge

        Returns
        -------
        tangents : numpy.ndarray
            float array of shape (nE, dim)
        """
        if self.dim == 2:
            tX = np.c_[
                np.ones(self.nEx), np.zeros(self.nEx)
            ]
            tY = np.c_[
                np.zeros(self.nEy), np.ones(self.nEy)
            ]
            return np.r_[tX, tY]
        elif self.dim == 3:
            tX = np.c_[
                np.ones(self.nEx), np.zeros(self.nEx), np.zeros(self.nEx)
            ]
            tY = np.c_[
                np.zeros(self.nEy), np.ones(self.nEy), np.zeros(self.nEy)
            ]
            tZ = np.c_[
                np.zeros(self.nEz), np.zeros(self.nEz), np.ones(self.nEz)
            ]
            return np.r_[tX, tY, tZ]

    def projectFaceVector(self, fV):
        """Project a vector onto the face normals

        Given a vector, fV, in cartesian coordinates, this will project
        it onto the mesh using the normals.

        Parameters
        ----------
        fV : numpy.ndarray
            face vector with shape (nF, dim)

        Returns
        -------
        numpy.ndarray
            projected face vector, (nF, )
        """
        if not isinstance(fV, np.ndarray):
            raise Exception('fV must be an ndarray')
        if not (
            len(fV.shape) == 2 and
            fV.shape[0] == self.nF and
            fV.shape[1] == self.dim
        ):
            raise Exception('fV must be an ndarray of shape (nF x dim)')
        return np.sum(fV*self.normals, 1)

    def projectEdgeVector(self, eV):
        """Project a vector onto the edge tangents

        Given a vector, eV, in cartesian coordinates, this will project
        it onto the mesh using the tangents

        Parameters
        ----------
        eV : numpy.ndarray
            edge vector with shape (nE, dim)

        Returns
        -------
        numpy.ndarray
            projected edge vector, (nE, )
        """
        if not isinstance(eV, np.ndarray):
            raise Exception('eV must be an ndarray')
        if not (
            len(eV.shape) == 2 and
            eV.shape[0] == self.nE and
            eV.shape[1] == self.dim
        ):
            raise Exception('eV must be an ndarray of shape (nE x dim)')
        return np.sum(eV*self.tangents, 1)

    def save(self, filename='mesh.json', verbose=False):
        """
        Save the mesh to json

        Parameters
        ----------
        filename : str
            filename for saving the casing properties
        verbose : bool, optional
            verbose mode printout flag
        """

        f = os.path.abspath(filename)  # make sure we are working with abs path
        with open(f, 'w') as outfile:
            json.dump(self, outfile)

        if verbose is True:
            print('Saved {}'.format(f))

        return f

    @property
    def reference_is_rotated(self):
        """True if the axes are rotated from the traditional <X,Y,Z> system
        with vectors of :math:`(1,0,0)`, :math:`(0,1,0)`, and :math:`(0,0,1)`

        Returns
        -------
        bool
        """
        if (    np.allclose(self.axis_u, (1, 0, 0)) and
                np.allclose(self.axis_v, (0, 1, 0)) and
                np.allclose(self.axis_w, (0, 0, 1)) ):
            return False
        return True

    @property
    def rotation_matrix(self):
        """Rotation matrix of the mesh if local coordinates are rotated.

        Builds a rotation matrix to transform coordinates from their coordinate
        system into a conventional cartesian system. This is built off of the
        three `axis_u`, `axis_v`, and `axis_w` properties; these mapping
        coordinates use the letters U, V, and W (the three letters preceding X,
        Y, and Z in the alphabet) to define the projection of the X, Y, and Z
        durections. These UVW vectors describe the placement and transformation
        of the mesh's coordinate sytem assuming at most 3 directions.

        Why would you want to use these UVW mapping vectors the this
        `rotation_matrix` property? They allow us to define the realationship
        between local and global coordinate systems and provide a tool for
        switching between the two while still maintaing the connectivity of the
        mesh's cells. For a visual example of this, please see the figure in the
        docs for the :class:`~discretize.mixins.vtkModule.InterfaceVTK`.

        Returns
        -------
        numpy.ndarray
            array of shape (3, 3)
        """
        return np.array([self.axis_u, self.axis_v, self.axis_w])


class BaseRectangularMesh(BaseMesh):
    """The base type for Rectangular meshes

    `BaseRectangularMesh` adds a few attributes to the `BaseMesh` that are useful
    for rectangularly structured meshes, as well as a useful reshaping command.
    `BaseRectangularMesh` should be inherited by meshes with a rectangular structure.

    Parameters
    ----------
    shape : array_like
        list of integers, must be have length 3 or less
    x0 : array_like
        list of floats defining the origin of the mesh. Must also have length 3 or less
    reference_system: ['cartesian', 'cylindrical', 'spherical']
        The type of coordinate reference frame. Can take on the values
        cartesian, cylindrical, or spherical. Abbreviations of these are allowed.
    axis_u : array_like, optional
        length three array of floats for orientation direction of axis 1, defaults to [1, 0, 0]
    axis_v : array_like, optional
        length three array of floats for orientation direction of axis 2, defaults to [0, 1, 0]
    axis_w : array_like, optional
        length three array of floats for orientation direction of axis 2, defaults to [0, 0, 1]

    Attributes
    ----------
    shape : numpy.ndarray
    x0 : numpy.ndarray
    dim
    nC
    nCx
    nCy
    nCz
    vnC
    nN
    nNx
    nNy
    nNz
    vnN
    nE
    vnE
    nEx
    nEy
    nEz
    vnEx
    vnEy
    vnEz
    nF
    vnF
    nFx
    nFy
    nFz
    vnFx
    vnFy
    vnFz
    normals
    tangents
    reference_is_rotated
    rotation_matrix
    """

    @property
    def nCx(self):
        """Number of cells in the x direction

        Returns
        -------
        nCx : int
        """
        return int(self.shape[0])

    @property
    def nCy(self):
        """Number of cells in the y direction

        Returns
        -------
        nCy : int or None
            returns None if dimension < 2
        """
        if self.dim < 2:
            return None
        return int(self.shape[1])

    @property
    def nCz(self):
        """Number of cells in the z direction

        Returns
        -------
        nCz : int or None
            returns None if dimension < 3
        """
        if self.dim < 3:
            return None
        return int(self.shape[2])

    @property
    def vnC(self):
        """Total number of cells in each direction

        Returns
        -------
        vnC : numpy.ndarray
            [nCx, ], [nCx, nCy, ], or [nCx, nCy, nCz] (depending on dimension)
        """
        return self.shape

    @property
    def nNx(self):
        """Number of nodes in the x-direction

        Returns
        -------
        nNx : int
        """
        return self.nCx + 1

    @property
    def nNy(self):
        """Number of nodes in the y-direction

        Returns
        -------
        nNy : int or None
            returns None if dimension <2
        """
        if self.dim < 2:
            return None
        return self.nCy + 1

    @property
    def nNz(self):
        """Number of nodes in the z-direction

        Returns
        -------
        nNz : int or None
            returns None if dimension <3
        """
        if self.dim < 3:
            return None
        return self.nCz + 1

    @property
    def vnN(self):
        """Total number of nodes in each direction

        Returns
        -------
        vnC : numpy.ndarray
            [nNx, ], [nNx, nNy, ], or [nNx, nNy, nNz] (depending on dimension)
        """
        return self.shape+1

    @property
    def vnEx(self):
        """Number of x-edges in each direction

        Returns
        -------
        vnEx : numpy.ndarray
            [nCx, ], [nCx, nNy], or [nCx, nNy, nNz] (depending on dimension)
        """
        return np.array(
            [x for x in [self.nCx, self.nNy, self.nNz] if x is not None],
            dtype=int
        )

    @property
    def vnEy(self):
        """Number of y-edges in each direction

        Returns
        -------
        vnEy : numpy.ndarray or None
            None, [nNx, nCy], or [nNx, nCy, nNz] (depending on dimension)
        """
        if self.dim < 2:
            return None
        return np.array(
            [x for x in [self.nNx, self.nCy, self.nNz] if x is not None],
            dtype=int
        )

    @property
    def vnEz(self):
        """Number of z-edges in each direction

        Returns
        -------
        vnEz : numpy.ndarray or None
            None or [nNx, nNy, nCz] (depending on dimension)
        """
        if self.dim < 3:
            return None
        return np.array(
            [x for x in [self.nNx, self.nNy, self.nCz] if x is not None],
            dtype=int
        )

    @property
    def vnFx(self):
        """Number of x-faces in each direction

        Returns
        -------
        vnFx : numpy.ndarray
            [nNx, ], [nNx, nCy], or [nNx, nCy, nCz] (depending on dimension)
        """
        return np.array(
            [x for x in [self.nNx, self.nCy, self.nCz] if x is not None],
            dtype=int
        )

    @property
    def vnFy(self):
        """Number of y-faces in each direction

        Returns
        -------
        vnFy : numpy.ndarray or None
            None, [nCx, nNy], or [nCx, nNy, nCz] (depending on dimension)
        """
        if self.dim < 2:
            return None
        return np.array(
            [x for x in [self.nCx, self.nNy, self.nCz] if x is not None],
            dtype=int
        )

    @property
    def vnFz(self):
        """Number of z-faces in each direction

        Returns
        -------
        vnFz : numpy.ndarray or None
            None or [nCx, nCy, nNz] (depending on dimension)
        """
        if self.dim < 3:
            return None
        return np.array(
            [x for x in [self.nCx, self.nCy, self.nNz] if x is not None],
            dtype=int
        )

    def r(self, x, xType='CC', outType='CC', format='V'):
        """A quick reshape command

        `r` is a quick reshape command that will do the best it
        can at giving you what you want.

        For example, you have a face variable, and you want the x
        component of it reshaped to a 3D matrix.

        `r` can fulfil your dreams.

        Parameters
        ----------
        x :  numpy.ndarray or list of numpy.ndarrays
            The input arrays to reshape
        xType : {'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'}
            Where the values of the input array are located.
        outType : {'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'}
            Where the values of the output array(s) are located.
        format : {'V', 'M'}
            How to shape the output as either a vector 'V' of shape (n, dim), or
            as an ndgrid style matrix

        Notes
        -----

        You cannot change types while reshaping, i.e. if xType is 'Fx', outType must be 'Fx',
        or if xType is 'F' outType can be 'Fx', 'Fy', or 'Fz'.


        Examples
        --------

        Separates each component of the Ex grid into 3 matrices
        >>> Xex, Yex, Zex = r(mesh.gridEx, 'Ex', 'Ex', 'M')

        Given an edge vector, return just the x edges as a vector
        >>> XedgeVector = r(edgeVector, 'E', 'Ex', 'V')

        Separates each component of the edgeVector into 3 vectors
        >>> eX, eY, eZ = r(edgeVector, 'E', 'E', 'V')
        """

        allowed_xType = [
            'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'
        ]
        if not (
            isinstance(x, list) or isinstance(x, np.ndarray)
        ):
            raise Exception("x must be either a list or a ndarray")
        if xType not in allowed_xType:
            raise Exception (
                "xType must be either "
                "'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez'"
            )
        if outType not in allowed_xType:
            raise Exception(
                "outType must be either "
                "'CC', 'N', 'F', Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez'"
            )
        if format not in ['M', 'V']:
            raise Exception("format must be either 'M' or 'V'")
        if outType[:len(xType)] != xType:
            raise Exception(
                "You cannot change types when reshaping."
            )
        if xType not in outType:
            raise Exception("You cannot change type of components.")

        if isinstance(x, list):
            for i, xi in enumerate(x):
                if not isinstance(x, np.ndarray):
                    raise Exception(
                        "x[{0:d}] must be a numpy array".format(i)
                    )
                if xi.size != x[0].size:
                    raise Exception(
                        "Number of elements in list must not change."
                    )

            x_array = np.ones((x.size, len(x)))
            # Unwrap it and put it in a np array
            for i, xi in enumerate(x):
                x_array[:, i] = mkvc(xi)
            x = x_array

        if not isinstance(x, np.ndarray):
            raise Exception("x must be a numpy array")

        x = x[:]  # make a copy.
        xTypeIsFExyz = (
            len(xType) > 1 and
            xType[0] in ['F', 'E'] and
            xType[1] in ['x', 'y', 'z']
        )

        def outKernal(xx, nn):
            """Returns xx as either a matrix (shape == nn) or a vector."""
            if format == 'M':
                return xx.reshape(nn, order='F')
            elif format == 'V':
                return mkvc(xx)

        def switchKernal(xx):
            """Switches over the different options."""
            if xType in ['CC', 'N']:
                nn = (self.shape) if xType == 'CC' else (self.shape+1)
                if xx.size != np.prod(nn):
                    raise Exception(
                        "Number of elements must not change."
                    )
                return outKernal(xx, nn)
            elif xType in ['F', 'E']:
                # This will only deal with components of fields,
                # not full 'F' or 'E'
                xx = mkvc(xx)  # unwrap it in case it is a matrix
                nn = self.vnF if xType == 'F' else self.vnE
                nn = np.r_[0, nn]

                nx = [0, 0, 0]
                nx[0] = self.vnFx if xType == 'F' else self.vnEx
                nx[1] = self.vnFy if xType == 'F' else self.vnEy
                nx[2] = self.vnFz if xType == 'F' else self.vnEz

                for dim, dimName in enumerate(['x', 'y', 'z']):
                    if dimName in outType:
                        if self.dim <= dim:
                            raise Exception(
                                "Dimensions of mesh not great enough for "
                                "{}{}".format(xType, dimName)
                            )
                        if xx.size != np.sum(nn):
                            raise Exception(
                                "Vector is not the right size."
                            )
                        start = np.sum(nn[:dim+1])
                        end = np.sum(nn[:dim+2])
                        return outKernal(xx[start:end], nx[dim])

            elif xTypeIsFExyz:
                # This will deal with partial components (x, y or z)
                # lying on edges or faces
                if 'x' in xType:
                    nn = self.vnFx if 'F' in xType else self.vnEx
                elif 'y' in xType:
                    nn = self.vnFy if 'F' in xType else self.vnEy
                elif 'z' in xType:
                    nn = self.vnFz if 'F' in xType else self.vnEz
                if xx.size != np.prod(nn):
                    raise Exception('Vector is not the right size.')
                return outKernal(xx, nn)

        # Check if we are dealing with a vector quantity
        isVectorQuantity = len(x.shape) == 2 and x.shape[1] == self.dim

        if outType in ['F', 'E']:
            if isVectorQuantity:
                raise Exception(
                    'Not sure what to do with a vector vector quantity..'
                )
            outTypeCopy = outType
            out = ()
            for ii, dirName in enumerate(['x', 'y', 'z'][:self.dim]):
                outType = outTypeCopy + dirName
                out += (switchKernal(x),)
            return out
        elif isVectorQuantity:
            out = ()
            for ii in range(x.shape[1]):
                out += (switchKernal(x[:, ii]),)
            return out
        else:
            return switchKernal(x)
