import unittest
from spitfire import Library, Dimension
import numpy as np
from os import remove
from shutil import rmtree
from os.path import isfile
import pickle

machine_epsilon = np.finfo(float).eps


class SaveAndLoad(unittest.TestCase):
    def test_save_and_load_1d(self):
        file_name = 'l1test.pkl'
        if isfile(file_name):
            remove(file_name)

        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        l1['f'] = l1.x_grid
        l1['g'] = np.exp(l1.x_grid)
        l1.extra_attributes['name'] = 'my_library_name'

        l1.save_to_file(file_name)
        l2 = Library.load_from_file(file_name)
        remove(file_name)

        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['g'] - l2['g']) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_save_and_load_2d(self):
        file_name = 'l1test.pkl'
        if isfile(file_name):
            remove(file_name)

        l1 = Library(Dimension('x', np.linspace(0, 1, 16)),
                     Dimension('y', np.linspace(1, 2, 8)))
        l1['f'] = l1.x_grid + l1.y_grid
        l1['g'] = np.exp(l1.x_grid) * np.cos(np.pi * 2. * l1.y_grid)
        l1.extra_attributes['name'] = 'my_library_name'

        l1.save_to_file(file_name)
        l2 = Library.load_from_file(file_name)
        remove(file_name)

        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['g'] - l2['g']) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_save_and_load_3d(self):
        file_name = 'l1test.pkl'
        if isfile(file_name):
            remove(file_name)

        l1 = Library(Dimension('x', np.linspace(0, 1, 16)),
                     Dimension('y', np.linspace(1, 2, 8)),
                     Dimension('z', np.linspace(2, 3, 4)))
        l1['f'] = l1.x_grid + l1.y_grid + l1.z_grid
        l1['g'] = np.exp(l1.x_grid) * np.cos(np.pi * 2. * l1.y_grid) * l1.z_grid
        l1.extra_attributes['name'] = 'my_library_name'

        l1.save_to_file(file_name)
        l2 = Library.load_from_file(file_name)
        remove(file_name)

        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['g'] - l2['g']) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_raw_pickle_3d(self):
        file_name = 'l1test.pkl'
        if isfile(file_name):
            remove(file_name)

        l1 = Library(Dimension('x', np.linspace(0, 1, 16)),
                     Dimension('y', np.linspace(1, 2, 8)),
                     Dimension('z', np.linspace(2, 3, 4)))
        l1['f'] = l1.x_grid + l1.y_grid + l1.z_grid
        l1['g'] = np.exp(l1.x_grid) * np.cos(np.pi * 2. * l1.y_grid) * l1.z_grid
        l1.extra_attributes['name'] = 'my_library_name'

        with open(file_name, 'wb') as f:
            pickle.dump(l1, f)

        with open(file_name, 'rb') as f:
            l2 = pickle.load(f)

        remove(file_name)

        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['g'] - l2['g']) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_save_to_text(self):
        xvalues = np.linspace(0, 1, 16)
        yvalues = np.linspace(1, 2, 8)
        zvalues = np.linspace(2, 3, 4)
        l1 = Library(Dimension('x', xvalues),
                     Dimension('y', yvalues),
                     Dimension('z', zvalues))

        fvalues = l1.x_grid + l1.y_grid + l1.z_grid
        gvalues = np.exp(l1.x_grid) * np.cos(np.pi * 2. * l1.y_grid) * l1.z_grid

        lib_shape = fvalues.shape

        l1['f'] = fvalues
        l1['g'] = gvalues

        lib_name = 'my_library_name'
        l1.extra_attributes['name'] = lib_name

        dir_name = 'out'

        l1.save_to_text_directory(dir_name, ravel_order='F')

        xread = np.loadtxt(dir_name + f'/bulkdata_x.txt')
        yread = np.loadtxt(dir_name + f'/bulkdata_y.txt')
        zread = np.loadtxt(dir_name + f'/bulkdata_z.txt')
        fread = np.loadtxt(dir_name + f'/bulkdata_f.txt').reshape(lib_shape, order='F')
        gread = np.loadtxt(dir_name + f'/bulkdata_g.txt').reshape(lib_shape, order='F')

        with open(dir_name + '/metadata_user_defined_attributes.txt', 'r') as f:
            ea_read = f.readline()
        with open(dir_name + '/metadata_independent_variables.txt', 'r') as f:
            iv_lines = f.readlines()
        with open(dir_name + '/metadata_dependent_variables.txt', 'r') as f:
            dv_lines = f.readlines()

        rmtree(dir_name)

        self.assertTrue(np.all(np.abs(xvalues - xread) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(yvalues - yread) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(zvalues - zread) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvalues - fread) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(gvalues - gread) < 10. * machine_epsilon))
        self.assertTrue(ea_read, str(l1.extra_attributes))
        self.assertTrue(all([ivf.strip() == ivn for (ivf, ivn) in zip(iv_lines, [d.name for d in l1.dims])]))
        self.assertTrue(all([dvf.strip() == dvn.replace(' ', '_') for (dvf, dvn) in zip(dv_lines, l1.props)]))


if __name__ == '__main__':
    unittest.main()
