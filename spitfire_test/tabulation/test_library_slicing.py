import unittest
from spitfire import Library, Dimension
from spitfire.chemistry.library import LibraryIndexError
import numpy as np
from copy import copy, deepcopy

machine_epsilon = np.finfo(float).eps


class Slice1D(unittest.TestCase):
    def test_full(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        l1['f'] = np.exp(l1.x_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        l2 = l1[:]
        self.assertTrue(np.all(np.abs(l1.x_grid - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid.shape)
        self.assertTrue(l2.size == l1.x_grid.size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_partial(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        l1['f'] = np.exp(l1.x_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        n1 = 2
        n2 = 8
        l2 = l1[n1:n2]
        self.assertTrue(np.all(np.abs(l1.x_grid[n1:n2] - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'][n1:n2] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid[n1:n2].shape)
        self.assertTrue(l2.size == l1.x_grid[n1:n2].size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_single(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        l1['f'] = np.exp(l1.x_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        n1 = 2
        n2 = n1 + 1
        l2 = l1[n1:n2]
        self.assertTrue(np.all(np.abs(l1.x_grid[n1:n2] - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'][n1:n2] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid[n1:n2].shape)
        self.assertTrue(l2.size == l1.x_grid[n1:n2].size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_copy(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        l1['f'] = np.exp(l1.x_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        l2 = copy(l1)
        l3 = deepcopy(l1)
        l4 = Library.deepcopy(l1)
        self.assertTrue(np.all(np.abs(l1.x_grid - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.x_grid - l3.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l3['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.x_grid - l4.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l4['f']) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_invalid_number(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        try:
            l2 = l1[:, :]
            self.assertTrue(False)
        except LibraryIndexError:
            self.assertTrue(True)

    def test_multiple_nonslice_args_1(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        try:
            l2 = l1['f', :]
            self.assertTrue(False)
        except LibraryIndexError:
            self.assertTrue(True)

    def test_multiple_nonslice_args_2(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        try:
            l2 = l1[:, 'g']
            self.assertTrue(False)
        except LibraryIndexError:
            self.assertTrue(True)

    def test_multiple_nonslice_args_3(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 16)))
        try:
            l2 = l1['f', 'g']
            self.assertTrue(False)
        except LibraryIndexError:
            self.assertTrue(True)


class Slice2D(unittest.TestCase):
    def test_full(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 2)), Dimension('y', np.linspace(-1, 1, 3)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        l2 = l1[:, :]
        self.assertTrue(np.all(np.abs(l1.x_grid - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.y_grid - l2.y_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid.shape)
        self.assertTrue(l2.size == l1.x_grid.size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

        l3 = l1[:]
        self.assertTrue(np.all(np.abs(l1.x_grid - l3.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.y_grid - l3.y_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l3['f']) < 10. * machine_epsilon))
        self.assertTrue(l3.shape == l1.x_grid.shape)
        self.assertTrue(l3.size == l1.x_grid.size)
        self.assertTrue(l1.extra_attributes['name'] == l3.extra_attributes['name'])

    def test_partial(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)), Dimension('y', np.linspace(-1, 1, 10)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        n1x = 2
        n2x = 8
        n1y = 1
        n2y = -1
        l2 = l1[n1x:n2x, n1y:n2y]
        self.assertTrue(np.all(np.abs(l1.x_grid[n1x:n2x, n1y:n2y] - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'][n1x:n2x, n1y:n2y] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid[n1x:n2x, n1y:n2y].shape)
        self.assertTrue(l2.size == l1.x_grid[n1x:n2x, n1y:n2y].size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_single(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)), Dimension('y', np.linspace(-1, 1, 10)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        n1x = 2
        n2x = n1x + 1
        n1y = 1
        n2y = -1
        l2 = l1[n1x:n2x, n1y:n2y]
        self.assertTrue(np.all(np.abs(l1.x_grid[n1x:n2x, n1y:n2y] - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'][n1x:n2x, n1y:n2y] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid[n1x:n2x, n1y:n2y].shape)
        self.assertTrue(l2.size == l1.x_grid[n1x:n2x, n1y:n2y].size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_squeeze(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)), Dimension('y', np.linspace(-1, 1, 10)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid)
        l1.extra_attributes['name'] = 'my_library_name'

        iy = 3
        l3 = Library.squeeze(l1[:, iy])

        self.assertTrue(np.all(np.abs(l1['f'][:, iy] - l3['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(np.squeeze(l1.x_grid[:, iy]) - l3.x_grid) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l3.extra_attributes['name'])

    def test_copy(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)), Dimension('y', np.linspace(-1, 1, 10)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        l2 = copy(l1)
        l3 = Library.copy(l1)
        self.assertTrue(np.all(np.abs(l1.x_grid - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.y_grid - l2.y_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.x_grid - l3.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.y_grid - l3.y_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l3['f']) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])
        self.assertTrue(l1.extra_attributes['name'] == l3.extra_attributes['name'])

    def test_invalid_number_3(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 2)), Dimension('y', np.linspace(-1, 1, 3)))
        try:
            l2 = l1[:, :, :]
            self.assertTrue(False)
        except LibraryIndexError:
            self.assertTrue(True)


class Slice3D(unittest.TestCase):
    def test_full(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 2)),
                     Dimension('y', np.linspace(-1, 1, 3)),
                     Dimension('z', np.logspace(-1, 1, 4)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        l2 = l1[:, :, :]
        self.assertTrue(np.all(np.abs(l1.x_grid - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.y_grid - l2.y_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.z_grid - l2.z_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid.shape)
        self.assertTrue(l2.size == l1.x_grid.size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

        l3 = l1[:]
        self.assertTrue(np.all(np.abs(l1.x_grid - l3.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.y_grid - l3.y_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.z_grid - l3.z_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l3['f']) < 10. * machine_epsilon))
        self.assertTrue(l3.shape == l1.x_grid.shape)
        self.assertTrue(l3.size == l1.x_grid.size)
        self.assertTrue(l1.extra_attributes['name'] == l3.extra_attributes['name'])

    def test_partial(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)),
                     Dimension('y', np.linspace(-1, 1, 4)),
                     Dimension('z', np.logspace(-1, 1, 7)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        n1x = 2
        n2x = 8
        n1y = 1
        n2y = -1
        n1z = 3
        n2z = 5
        l2 = l1[n1x:n2x, n1y:n2y, n1z:n2z]
        self.assertTrue(np.all(np.abs(l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z] - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'][n1x:n2x, n1y:n2y, n1z:n2z] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z].shape)
        self.assertTrue(l2.size == l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z].size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_single(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)),
                     Dimension('y', np.linspace(-1, 1, 4)),
                     Dimension('z', np.logspace(-1, 1, 7)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        n1x = 2
        n2x = 8
        n1y = 1
        n2y = -1
        n1z = 3
        n2z = n1z + 1
        l2 = l1[n1x:n2x, n1y:n2y, n1z:n2z]
        self.assertTrue(np.all(np.abs(l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z] - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'][n1x:n2x, n1y:n2y, n1z:n2z] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z].shape)
        self.assertTrue(l2.size == l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z].size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_single_internal(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)),
                     Dimension('y', np.linspace(-1, 1, 4)),
                     Dimension('z', np.logspace(-1, 1, 7)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        n1x = 2
        n2x = 8
        n1y = 1
        n2y = n1y + 1
        n1z = 3
        n2z = 6
        l2 = l1[n1x:n2x, n1y:n2y, n1z:n2z]
        self.assertTrue(np.all(np.abs(l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z] - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'][n1x:n2x, n1y:n2y, n1z:n2z] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(l2.shape == l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z].shape)
        self.assertTrue(l2.size == l1.x_grid[n1x:n2x, n1y:n2y, n1z:n2z].size)
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])

    def test_squeeze(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)),
                     Dimension('y', np.linspace(-1, 1, 4)),
                     Dimension('z', np.logspace(-1, 1, 7)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1.extra_attributes['name'] = 'my_library_name'

        iy = 2
        l3 = Library.squeeze(l1[:, iy, :])

        self.assertTrue(np.all(np.abs(np.squeeze(l1.x_grid[:, iy, :]) - l3.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(np.squeeze(l1.z_grid[:, iy, :]) - l3.z_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(np.squeeze(l1['f'][:, iy, :]) - l3['f']) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l3.extra_attributes['name'])

    def test_copy(self):
        l1 = Library(Dimension('x', np.linspace(0, 1, 10)),
                     Dimension('y', np.linspace(-1, 1, 4)),
                     Dimension('z', np.logspace(-1, 1, 7)))
        l1['f'] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1.extra_attributes['name'] = 'my_library_name'
        l2 = copy(l1)
        l3 = Library.copy(l1)
        self.assertTrue(np.all(np.abs(l1.x_grid - l2.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.y_grid - l2.y_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.z_grid - l2.z_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l2['f']) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.x_grid - l3.x_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.y_grid - l3.y_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1.z_grid - l3.z_grid) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1['f'] - l3['f']) < 10. * machine_epsilon))
        self.assertTrue(l1.extra_attributes['name'] == l2.extra_attributes['name'])
        self.assertTrue(l1.extra_attributes['name'] == l3.extra_attributes['name'])

    def test_view(self):
        slices = (slice(0, None, None), slice(1, 3, None), slice(1, -3, None))

        l1 = Library(Dimension('x', np.linspace(0, 1, 10)),
                     Dimension('y', np.linspace(-1, 1, 4)),
                     Dimension('z', np.logspace(-1, 1, 7)))
        gold_float = 0.5
        gold_array = np.exp(-2. * l1.x_grid[slices])

        fvals = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)

        # start with float argument
        g = gold_float
        # set slice of original array
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        fvals[slices] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # set slice of library property
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        l1['f'][slices] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # set property of library slice
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        l1[slices]['f'] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # set slice of library view property
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        l2['f'][:, :, :] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # set property of library view
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        l2['f'] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # repeat with numpy array argument
        g = gold_array
        # set slice of original array
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        fvals[slices] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # set slice of library property
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        l1['f'][slices] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # set property of library slice
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        l1[slices]['f'] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # set slice of library view property
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        l2['f'][:, :, :] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))

        # set property of library view
        fvals[:, :, :] = np.exp(l1.x_grid) * np.cos(l1.y_grid) * np.log(l1.z_grid)
        l1['f'] = fvals
        l2 = l1[slices]
        l2['f'] = g
        self.assertTrue(np.all(np.abs(l1['f'][slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l1[slices]['f'] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(fvals[slices] - g) < 10. * machine_epsilon))
        self.assertTrue(np.all(np.abs(l2['f'] - g) < 10. * machine_epsilon))


if __name__ == '__main__':
    unittest.main()
