import unittest
from spitfire.chemistry.tabulation import Library, Dimension
import numpy as np
from os import remove
from os.path import isfile

machine_epsilon = np.finfo(float).eps

class SaveAndLoad(unittest.TestCase):
    def test_save_and_load_1d(self):
        file_name = 'l1test.pkl'
        if isfile(file_name):
            remove(file_name)

        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        l1['f'] = l1.x_grid
        l1['g'] = np.exp(l1.x_grid)

        l1.save_to_file(file_name)
        l2 = Library.load_from_file(file_name)
        remove(file_name)

        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['g'] - l2['g']) < 10. * machine_epsilon))

    def test_save_and_load_2d(self):
        file_name = 'l1test.pkl'
        if isfile(file_name):
            remove(file_name)

        l1 = Library(Dimension('x', np.linspace(0, 1, 16)),
                     Dimension('y', np.linspace(1, 2, 8)))
        l1['f'] = l1.x_grid + l1.y_grid
        l1['g'] = np.exp(l1.x_grid) * np.cos(np.pi * 2. * l1.y_grid)

        l1.save_to_file(file_name)
        l2 = Library.load_from_file(file_name)
        remove(file_name)

        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['g'] - l2['g']) < 10. * machine_epsilon))

    def test_save_and_load_3d(self):
        file_name = 'l1test.pkl'
        if isfile(file_name):
            remove(file_name)

        l1 = Library(Dimension('x', np.linspace(0, 1, 16)),
                     Dimension('y', np.linspace(1, 2, 8)),
                     Dimension('z', np.linspace(2, 3, 4)))
        l1['f'] = l1.x_grid + l1.y_grid + l1.z_grid
        l1['g'] = np.exp(l1.x_grid) * np.cos(np.pi * 2. * l1.y_grid) * l1.z_grid

        l1.save_to_file(file_name)
        l2 = Library.load_from_file(file_name)
        remove(file_name)

        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['g'] - l2['g']) < 10. * machine_epsilon))


if __name__ == '__main__':
    unittest.main()
