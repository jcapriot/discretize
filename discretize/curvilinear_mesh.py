from __future__ import print_function
import numpy as np
from .pydantic.typing import Array
from typing import List
from pydantic import validator

from discretize import utils
from .base import BaseRectangularMesh, InnerProducts
from discretize.DiffOperators import DiffOperators
from discretize.View import CurviView


# Some helper functions.
def length2D(x):
    return (x[:, 0]**2 + x[:, 1]**2)**0.5


def length3D(x):
    return (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5


def normalize2D(x):
    return x/np.kron(np.ones((1, 2)), utils.mkvc(length2D(x), 2))


def normalize3D(x):
    return x/np.kron(np.ones((1, 3)), utils.mkvc(length3D(x), 2))


class CurvilinearMesh(
    BaseRectangularMesh, DiffOperators, InnerProducts, CurviView
):
    """CurvilinearMesh is a mesh class that deals with curvilinear meshes.

    Example of a curvilinear mesh:

    .. plot::
        :include-source:

        import discretize
        X, Y = discretize.utils.exampleLrmGrid([3,3],'rotate')
        mesh = discretize.CurvilinearMesh([X, Y])
        mesh.plotGrid(showIt=True)
    """

    _meshType = 'Curv'

    nodes = List[Array[float, None]]

    def __init__(self, nodes=None, **kwargs):

        nodes = self._easy_validate(nodes)
        shape = nodes[0].shape
        x0 = [node[0, 0, 0] for node in nodes]
        kwargs.pop('shape', None)
        kwargs.pop('x0', None)
        super().__init__(nodes=nodes, shape=shape, x0=x0, **kwargs)

        # Save nodes to private variable _gridN as vectors
        self._gridN = np.ones((self.nodes[0].size, self.dim))
        for i, node_i in enumerate(self.nodes):
            self._gridN[:, i] = utils.mkvc(node_i.astype(float))

    @validator('nodes')
    def _check_nodes(cls, v, values, **kwargs):
        # check the length of the nodes list is 2 or 3
        dim = len(v)
        if dim not in [2, 3]:
            raise ValueError(f"len(nodes) must be 2 or 3, not {len(v)}")

        # check all vi have shapes of 2 or 3 dimensions, depending on len(v)
        for i, vi in enumerate(v):
            if len(vi.shape) != dim:
                raise ValueError(f"len(nodes[{i}].shape) must be {dim}, not {len(vi.shape)}")

        # check all vi have the same shape
        v0_shape = v[0].shape
        if v0_shape != v[1].shape or (len(v) == 3 and v0_shape != v[2].shape):
            raise ValueError('Nodes must all have the same shape')

        # if shape has been set, ensure that the values are still a valid shape
        if 'shape' in values:
            shape = values['shape']
            for i, vi in enumerate(v):
                if np.any(shape != vi.shape):
                    raise ValueError(f"nodes[{i}] must have shape {shape}, not {vi.shape}")
        return v

    @property
    def gridCC(self):
        """
        Cell-centered grid
        """
        if getattr(self, '_gridCC', None) is None:
            self._gridCC = np.concatenate(
                [self.aveN2CC*self.gridN[:, i] for i in range(self.dim)]
            ).reshape((-1, self.dim), order='F')
        return self._gridCC

    @property
    def gridN(self):
        """
        Nodal grid.
        """
        if getattr(self, '_gridN', None) is None:
            raise Exception("Someone deleted this. I blame you.")
        return self._gridN

    @property
    def gridFx(self):
        """
        Face staggered grid in the x direction.
        """

        if getattr(self, '_gridFx', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [utils.mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                self._gridFx = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [utils.mkvc(0.25 * (n[:, :-1, :-1] + n[:, :-1, 1:] +
                       n[:, 1:, :-1] + n[:, 1:, 1:])) for n in N]
                self._gridFx = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFx

    @property
    def gridFy(self):
        """
        Face staggered grid in the y direction.
        """

        if getattr(self, '_gridFy', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [utils.mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                self._gridFy = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [utils.mkvc(0.25 * (n[:-1, :, :-1] + n[:-1, :, 1:] +
                       n[1:, :, :-1] + n[1:, :, 1:])) for n in N]
                self._gridFy = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFy

    @property
    def gridFz(self):
        """
        Face staggered grid in the y direction.
        """

        if getattr(self, '_gridFz', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            XYZ = [utils.mkvc(0.25 * (n[:-1, :-1, :] + n[:-1, 1:, :] +
                   n[1:, :-1, :] + n[1:, 1:, :])) for n in N]
            self._gridFz = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFz

    @property
    def gridEx(self):
        """
        Edge staggered grid in the x direction.
        """
        if getattr(self, '_gridEx', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [utils.mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                self._gridEx = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [utils.mkvc(0.5 * (n[:-1, :, :] + n[1:, :, :])) for n in N]
                self._gridEx = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEx

    @property
    def gridEy(self):
        """
        Edge staggered grid in the y direction.
        """
        if getattr(self, '_gridEy', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [utils.mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                self._gridEy = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [utils.mkvc(0.5 * (n[:, :-1, :] + n[:, 1:, :])) for n in N]
                self._gridEy = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEy

    @property
    def gridEz(self):
        """
        Edge staggered grid in the z direction.
        """
        if getattr(self, '_gridEz', None) is None and self.dim == 3:
            N = self.r(self.gridN, 'N', 'N', 'M')
            XYZ = [utils.mkvc(0.5 * (n[:, :, :-1] + n[:, :, 1:])) for n in N]
            self._gridEz = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEz

    # --------------- Geometries ---------------------
    #
    #
    # ------------------- 2D -------------------------
    #
    #         node(i,j)          node(i,j+1)
    #              A -------------- B
    #              |                |
    #              |    cell(i,j)   |
    #              |        I       |
    #              |                |
    #             D -------------- C
    #         node(i+1,j)        node(i+1,j+1)
    #
    # ------------------- 3D -------------------------
    #
    #
    #             node(i,j,k+1)       node(i,j+1,k+1)
    #                 E --------------- F
    #                /|               / |
    #               / |              /  |
    #              /  |             /   |
    #       node(i,j,k)         node(i,j+1,k)
    #            A -------------- B     |
    #            |    H ----------|---- G
    #            |   /cell(i,j)   |   /
    #            |  /     I       |  /
    #            | /              | /
    #            D -------------- C
    #       node(i+1,j,k)      node(i+1,j+1,k)

    @property
    def vol(self):
        """
        Construct cell volumes of the 3D model as 1d array
        """

        if getattr(self, '_vol', None) is None:
            if self.dim == 2:
                A, B, C, D = utils.indexCube('ABCD', self.vnC+1)
                normal, area = utils.faceInfo(np.c_[self.gridN, np.zeros(
                                              (self.nN, 1))], A, B, C, D)
                self._vol = area
            elif self.dim == 3:
                # Each polyhedron can be decomposed into 5 tetrahedrons
                # However, this presents a choice so we may as well divide in
                # two ways and average.
                A, B, C, D, E, F, G, H = utils.indexCube('ABCDEFGH', self.vnC +
                                                         1)

                vol1 = (utils.volTetra(self.gridN, A, B, D, E) +  # cutted edge top
                        utils.volTetra(self.gridN, B, E, F, G) +  # cutted edge top
                        utils.volTetra(self.gridN, B, D, E, G) +  # middle
                        utils.volTetra(self.gridN, B, C, D, G) +  # cutted edge bottom
                        utils.volTetra(self.gridN, D, E, G, H))   # cutted edge bottom

                vol2 = (utils.volTetra(self.gridN, A, F, B, C) +  # cutted edge top
                        utils.volTetra(self.gridN, A, E, F, H) +  # cutted edge top
                        utils.volTetra(self.gridN, A, H, F, C) +  # middle
                        utils.volTetra(self.gridN, C, H, D, A) +  # cutted edge bottom
                        utils.volTetra(self.gridN, C, G, H, F))   # cutted edge bottom

                self._vol = (vol1 + vol2)/2
        return self._vol

    @property
    def area(self):
        """
        Area of the faces
        """
        if (getattr(self, '_area', None) is None or
            getattr(self, '_normals', None) is None):
            # Compute areas of cell faces
            if(self.dim == 2):
                xy = self.gridN
                A, B = utils.indexCube('AB', self.vnC+1, np.array([self.nNx,
                                       self.nCy]))
                edge1 = xy[B, :] - xy[A, :]
                normal1 = np.c_[edge1[:, 1], -edge1[:, 0]]
                area1 = length2D(edge1)
                A, D = utils.indexCube('AD', self.vnC+1, np.array([self.nCx,
                                       self.nNy]))
                # Note that we are doing A-D to make sure the normal points the
                # right way.
                # Think about it. Look at the picture. Normal points towards C
                # iff you do this.
                edge2 = xy[A, :] - xy[D, :]
                normal2 = np.c_[edge2[:, 1], -edge2[:, 0]]
                area2 = length2D(edge2)
                self._area = np.r_[utils.mkvc(area1), utils.mkvc(area2)]
                self._normals = [normalize2D(normal1), normalize2D(normal2)]

            elif(self.dim == 3):

                A, E, F, B = utils.indexCube('AEFB', self.vnC+1, np.array(
                                             [self.nNx, self.nCy, self.nCz]))
                normal1, area1 = utils.faceInfo(self.gridN, A, E, F, B,
                                                average=False,
                                                normalizeNormals=False)

                A, D, H, E = utils.indexCube('ADHE', self.vnC+1, np.array(
                                             [self.nCx, self.nNy, self.nCz]))
                normal2, area2 = utils.faceInfo(self.gridN, A, D, H, E,
                                                average=False,
                                                normalizeNormals=False)

                A, B, C, D = utils.indexCube('ABCD', self.vnC+1, np.array(
                                             [self.nCx, self.nCy, self.nNz]))
                normal3, area3 = utils.faceInfo(self.gridN, A, B, C, D,
                                                average=False,
                                                normalizeNormals=False)

                self._area = np.r_[utils.mkvc(area1), utils.mkvc(area2),
                                   utils.mkvc(area3)]
                self._normals = [normal1, normal2, normal3]
        return self._area

    @property
    def normals(self):
        """
        Face normals: calling this will average
        the computed normals so that there is one
        per face. This is especially relevant in
        3D, as there are up to 4 different normals
        for each face that will be different.

        To reshape the normals into a matrix and get the y component::

            NyX, NyY, NyZ = M.r(M.normals, 'F', 'Fy', 'M')
        """

        if getattr(self, '_normals', None) is None:
            self.area  # calling .area will create the face normals
        if self.dim == 2:
            return normalize2D(np.r_[self._normals[0], self._normals[1]])
        elif self.dim == 3:
            normal1 = (
                self._normals[0][0] + self._normals[0][1] +
                self._normals[0][2] + self._normals[0][3]
            )/4
            normal2 = (
                self._normals[1][0] + self._normals[1][1] +
                self._normals[1][2] + self._normals[1][3]
            )/4
            normal3 = (
                self._normals[2][0] + self._normals[2][1] +
                self._normals[2][2] + self._normals[2][3]
            )/4
            return normalize3D(np.r_[normal1, normal2, normal3])

    @property
    def edge(self):
        """Edge lengths"""
        if getattr(self, '_edge', None) is None:
            if(self.dim == 2):
                xy = self.gridN
                A, D = utils.indexCube('AD', self.vnC+1, np.array([self.nCx,
                                                                  self.nNy]))
                edge1 = xy[D, :] - xy[A, :]
                A, B = utils.indexCube('AB', self.vnC+1, np.array([self.nNx,
                                                                   self.nCy]))
                edge2 = xy[B, :] - xy[A, :]
                self._edge = np.r_[utils.mkvc(length2D(edge1)),
                                   utils.mkvc(length2D(edge2))]
                self._tangents = np.r_[edge1, edge2]/np.c_[self._edge,
                                                           self._edge]
            elif(self.dim == 3):
                xyz = self.gridN
                A, D = utils.indexCube('AD', self.vnC+1, np.array([self.nCx,
                                                                   self.nNy,
                                                                   self.nNz]))
                edge1 = xyz[D, :] - xyz[A, :]
                A, B = utils.indexCube('AB', self.vnC+1, np.array([self.nNx,
                                                                   self.nCy,
                                                                   self.nNz]))
                edge2 = xyz[B, :] - xyz[A, :]
                A, E = utils.indexCube('AE', self.vnC+1, np.array([self.nNx,
                                                                   self.nNy,
                                                                   self.nCz]))
                edge3 = xyz[E, :] - xyz[A, :]
                self._edge = np.r_[utils.mkvc(length3D(edge1)),
                                   utils.mkvc(length3D(edge2)),
                                   utils.mkvc(length3D(edge3))]
                self._tangents = (np.r_[edge1, edge2, edge3] /
                                  np.c_[self._edge, self._edge, self._edge])
            return self._edge
        return self._edge

    @property
    def tangents(self):
        """Edge tangents"""
        if getattr(self, '_tangents', None) is None:
            self.edge  # calling .edge will create the tangents
        return self._tangents
