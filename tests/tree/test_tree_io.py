from __future__ import print_function
import numpy as np
import unittest
import os
import discretize
import pickle

try:
    import vtk
except ImportError:
    has_vtk = False
else:
    has_vtk = True


class TestOcTreeMeshIO(unittest.TestCase):

    def setUp(self):
        h = np.ones(16)
        mesh = discretize.TreeMesh([h, 2*h, 3*h])
        cell_points = np.array([[0.5, 0.5, 0.5],
                                [0.5, 2.5, 0.5]])
        cell_levels = np.array([4, 4])
        mesh.insert_cells(cell_points, cell_levels)
        self.mesh = mesh

    def test_UBCfiles(self):

        mesh = self.mesh
        # Make a vector
        vec = np.arange(mesh.nC)
        # Write and read
        mesh.writeUBC('temp.msh', {'arange.txt': vec})
        meshUBC = discretize.TreeMesh.readUBC('temp.msh')
        vecUBC = meshUBC.readModelUBC('arange.txt')

        self.assertEqual(mesh.nC, meshUBC.nC)
        self.assertEqual(mesh.__str__(), meshUBC.__str__())
        self.assertTrue(np.allclose(mesh.gridCC, meshUBC.gridCC))
        self.assertTrue(np.allclose(vec, vecUBC))
        self.assertTrue(np.allclose(np.array(mesh.h), np.array(meshUBC.h)))

        # Write it again with another IO function
        mesh.writeModelUBC(['arange.txt'], [vec])
        vecUBC2 = mesh.readModelUBC('arange.txt')
        self.assertTrue(np.allclose(vec, vecUBC2))
        
        print('IO of UBC octree files is working')
        os.remove('temp.msh')
        os.remove('arange.txt')

    if has_vtk:
        def test_VTUfiles(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            mesh.writeVTK('temp.vtu', {'arange': vec})
            print('Writing of VTU files is working')
            os.remove('temp.vtu')

class TestPickle(unittest.TestCase):

    def test_pickle2D(self):
        mesh0 = discretize.TreeMesh([8, 8])

        def refine(cell):
            xyz = cell.center
            dist = ((xyz - 0.25)**2).sum()**0.5
            if dist < 0.25:
                return 3
            return 2

        mesh0.refine(refine)

        byte_string = pickle.dumps(mesh0)
        mesh1 = pickle.loads(byte_string)

        self.assertEqual(mesh0.nC, mesh1.nC)
        self.assertEqual(mesh0.__str__(), mesh1.__str__())
        self.assertTrue(np.allclose(mesh0.gridCC, mesh1.gridCC))
        self.assertTrue(np.allclose(np.array(mesh0.h), np.array(mesh1.h)))
        print('Pickling of 2D TreeMesh is working')

    def test_pickle3D(self):
        mesh0 = discretize.TreeMesh([8, 8, 8])

        def refine(cell):
            xyz = cell.center
            dist = ((xyz - 0.25)**2).sum()**0.5
            if dist < 0.25:
                return 3
            return 2

        mesh0.refine(refine)

        byte_string = pickle.dumps(mesh0)
        mesh1 = pickle.loads(byte_string)

        self.assertEqual(mesh0.nC, mesh1.nC)
        self.assertEqual(mesh0.__str__(), mesh1.__str__())
        self.assertTrue(np.allclose(mesh0.gridCC, mesh1.gridCC))
        self.assertTrue(np.allclose(np.array(mesh0.h), np.array(mesh1.h)))
        print('Pickling of 3D TreeMesh is working')

if __name__ == '__main__':
    unittest.main()
