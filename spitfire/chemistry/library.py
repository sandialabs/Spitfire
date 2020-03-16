"""
This module defines containers for tabulated chemistry libraries and solution trajectories
"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
#
# You should have received a copy of the 3-clause BSD License
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#
# Questions? Contact Mike Hansen (mahanse@sandia.gov)


import numpy as np
import pickle as pickle


class Dimension(object):
    """A class to contain details of a particular independent variable in a structured or unstructured library

    **Constructor**: specify a name, values, and whether or not the dimension is structured

    Parameters
    ----------
    name : str
        the name of the mechanism - hyphens and spaces may not be used here, use underscore separators
    values: np.array
        the values of the independent variable in the grid
    structured: bool
        whether or not the dimension is structured (optional, default: True)
    """

    def __init__(self, name: str, values: np.array, structured=True, _library_owned=False):
        self._name = name
        self._values = np.copy(values)
        self._min = np.min(values)
        self._max = np.max(values)
        self._npts = values.size
        self._structured = structured
        self._grid = None
        self._library_owned = _library_owned

        if not name.isidentifier():
            raise ValueError(f'Error in building Dimension "{name}", the name cannot contain hyphens or spaces '
                             f'(it must be a valid Python variable name, note that you can check this with name.isidentifier())')

        if len(self._values.shape) != 1:
            raise ValueError(f'Error in building Dimension "{name}", the values object must be one-dimensional.'
                             f' Call the ravel() method to flatten your data (without a deep copy).')

        if structured:
            if self._values.size != np.unique(self._values).size:
                raise ValueError(f'Error in building structured dimension "{name}"'
                                 ', duplicate values identified!')

    @property
    def name(self):
        """Obtain the name of the independent variable"""
        return self._name

    @property
    def values(self):
        """Obtain the one-dimensional np.array of the specified values of this independent variable"""
        return self._values

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def npts(self):
        return self._npts

    @property
    def structured(self):
        return self._structured

    @property
    def grid(self):
        """Obtain the np.ndarray "meshgrid" of the values of this variable in a multi-dimensional library
        Upon construction of the Dimension instance, this returns the given one-dimensional values,
        but after incorporation of the Dimension into a library, this is made into a meshgrid object.
        Note that the library object will always copy Dimension data into a brand new instance,
        so the Dimension instances fed to a Library are not modified. This is consistent in that a Dimension instance
        does not have multidimensional grid data unless incorporated into a multidimensional Library. In the case
        of unstructured data, the grid is always equivalent to the values and consistency is implicit."""
        return self._values if self._grid is None else self._grid

    @grid.setter
    def grid(self, grid):
        """Set the meshgrid object - do not call explicitly, this can only be called by the Library class
        that owns the Dimension."""
        if self._library_owned:
            self._grid = grid
        else:
            raise ValueError(f'Explicitly setting the "grid" property on Dimension "{self._name}" is not allowed.'
                             f'Only an owning Library object can set the multidimensional grid.')

    def _get_dict_for_file_save(self):
        return {'name': self._name, 'values': self._values, 'structured': self._structured}


class Library(object):
    """A class for holding tabulated datasets over structured and unstructured grids

    Upon constructing the Library object, the following properties are made available for each Dimension:
      library.[dimension_name]_name
      library.[dimension_name]_values
      library.[dimension_name]_min
      library.[dimension_name]_max
      library.[dimension_name]_npts
      library.[dimension_name]_structured
      library.[dimension_name]_grid

    **Constructor**: specify the argument list of dimensions defining the grid

    Parameters
    ----------
    dimensions : argument list of Dimension instances
        the dimensions that define the grid
    """

    def __init__(self, *dimensions):
        self._dims = dict({d.name: Dimension(d.name, d.values, d.structured, _library_owned=True) for d in dimensions})
        self._props = dict()
        self._dims_ordering = dict()
        for i, d in enumerate(dimensions):
            self._dims_ordering[i] = d.name

        self._structured = all([self._dims[d].structured for d in self._dims])
        unstructured = all([not self._dims[d].structured for d in self._dims])
        if not self._structured and not unstructured:
            raise ValueError(
                'Error in building Library - Dimensions must be either all structured or all unstructured!')

        if self._structured:
            grid = np.meshgrid(*[self._dims[d].values for d in self._dims], indexing='ij')
            self._grid_shape = grid[0].shape
            for i, d in enumerate(self._dims):
                self._dims[d].grid = grid[i]
        else:
            dim0 = self._dims[next(self._dims.keys())]
            self._grid_shape = dim0.grid.shape
            if not all([self._dims[d].grid.shape == self._grid_shape for d in self._dims]):
                raise ValueError('Unstructured dimensions did not have the same grid shape!')

            if not all([len(self._dims[d].grid.shape) == 1 for d in self._dims]):
                raise ValueError('Unstructured dimensions were not given as flat arrays!')

        for d in self._dims:
            self.__dict__[self._dims[d].name] = d
            for a in self._dims[d].__dict__:
                if a is not '_library_owned':
                    self.__dict__[self._dims[d].name + a] = self._dims[d].__dict__[a]

    def save_to_file(self, file_name):
        """Save a library to a specified file using pickle"""
        instance_dict = dict(
            {'dimensions': {d: self._dims[d]._get_dict_for_file_save() for d in self._dims},
             'dim_ordering': self._dims_ordering,
             'properties': self._props})
        with open(file_name, 'wb') as file_output:
            pickle.dump(instance_dict, file_output)

    @classmethod
    def load_from_file(cls, file_name):
        """Load a library from a specified file name with pickle (following save_to_file)"""
        with open(file_name, 'rb') as file_input:
            instance_dict = pickle.load(file_input)
            ordered_dims = list([None] * len(instance_dict['dimensions'].keys()))
            for index in instance_dict['dim_ordering']:
                name = instance_dict['dim_ordering'][index]
                d = instance_dict['dimensions'][name]
                ordered_dims[index] = Dimension(d['name'], d['values'], d['structured'])
            l = Library(*ordered_dims)
            for prop in instance_dict['properties']:
                l[prop] = instance_dict['properties'][prop]
            return l

    def __setitem__(self, quantity: str, values: np.ndarray):
        """Use the bracket operator, as in lib['myprop'] = values, to add a property defined on the grid
           The np.ndarray of values must be shaped correctly"""
        if values.shape != self._grid_shape:
            raise ValueError(f'The shape of the "{quantity}" array does not conform to that of the library. '
                             f'Given shape = {values.shape}, grid shape = {self._grid_shape}')
        self._props[quantity] = np.copy(values)

    def __getitem__(self, quantity: str):
        """Use the bracket operator, as in lib['myprop'], to obtain the value array of a property"""
        return self._props[quantity]

    def __contains__(self, prop):
        return prop in self._props

    @property
    def props(self):
        """Obtain a list of the names of properties set on the library"""
        return list(self._props.keys())

    @property
    def dims(self):
        """Obtain a list of the Dimension objects associated with the library"""
        dims = []
        for d in self._dims_ordering:
            dims.append(self._dims[self._dims_ordering[d]])
        return dims

    def dim(self, name):
        """Obtain a Dimension object by name"""
        return self._dims[name]

    def get_empty_dataset(self):
        """Obtain an empty dataset in the shape of the grid, to enable filling one point, line, plane, etc. at a time,
        before then possibly setting a library property with the data"""
        return np.ndarray(self._grid_shape)

    def remove(self, *quantities):
        """Remove quantities (argument list of strings) from the library"""
        for quantity in quantities:
            self._props.pop(quantity)
