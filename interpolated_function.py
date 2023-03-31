#!/usr/bin/python3
# interpolated_function.py
"""Class for functions with lookup tables, for faster evaluation."""

__author__ = ("Alexander C. Jenkins",)
__contact__ = ("alex.jenkins@ucl.ac.uk",)
__version__ = "0.2"
__date__ = "2023/03"

import numpy as np

from os.path import isfile
from tqdm import tqdm


#-----------------------------------------------------------------------------#


class InterpolatedFunction(object):
    """A function with a lookup table, for faster evaluation.

    Attributes
    ----------
    func : function
        The base function whose values are interpolated. Must accept a
        single argument.
    file : str
        Name of the file containing the lookup table.
    is_interpolated : bool
        True if the lookup table has been created, False otherwise.
    min : float
        Minimum value of the range of arguments over which the function
        is interpolated. Calling the function at a value outside this
        range will result in a call to the base function, rather than
        using the lookup table.
    max : float
        Maximum value of the range of arguments over which the function
        is interpolated.

    Methods
    -------
    interpolate()
        Create a lookup table.
    reset()
        Undo the interpolation.
    """

    def __init__(self,
                 func,
                 arg_array,
                 file=None,
                 ):
        """Initialise the interpolated function.

        Parameters
        ----------
        func : function
            The base function to interpolate. Must accept a single
            argument.
        arg_array : array, shape (N,)
            Array of gridpoints at which to evaluate the function for
            the lookup table.
        file : str, optional
            Name of the file containing the lookup table. By default
            this is set by the name of the base function. If this file
            doesn't yet exist, then the lookup table will be created
            (warning: this can be a very slow process). If the file
            already exists, then the interpolated function will be
            evaluated using values from that file.
        """

        if func != None:
            self.func = func
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__

            if file == None:
                self.file = func.__name__

        else:
            self.file = file
            self.func = lambda x, y: 0.
            self.__name__ = self.file.rsplit("/")[-1].split(".")[0]

        self._arg_array = arg_array
        self._val_array = []
        self.min = arg_array[0]
        self.max = arg_array[-1]
        self._N = len(arg_array)
        self._N_array = np.array(range(self._N))
        self._linspaced = False

        if np.all(arg_array == np.linspace(self.min, self.max, self._N)):
            self._linspaced = True

        if isfile(self.file + "_val.npy"):
            self.is_interpolated = True
            self._arg_array = np.load(self.file + "_arg.npy")
            self._val_array = np.load(self.file + "_val.npy")

        else:
            self.is_interpolated = False
            self._val_array = []

    def interpolate(self,
                    ):
        """Create a lookup table.

        If a filename is specified, the interpolating values will be written
        to that file.
        """

        if self.is_interpolated:
            return None

        self._val_array = []

        for arg in tqdm(self._arg_array,
                        desc="Creating lookup table for `" + self.file + "'"):
            val = self.func(arg)

            self._val_array.append(val)

        self._val_array = np.array(self._val_array)
        np.save(self.file + "_arg.npy",
                self._arg_array)
        np.save(self.file + "_val.npy",
                self._val_array)
        self.is_interpolated = True

        return None

    def _interpolated_func(self, arg):
        """Internal method for extracting values from the lookup table.

        Parameters
        ----------
        arg : float
            Argument at which the interpolated function is evaluated.

        Returns
        -------
        float
            Approximate value of the function, using the lookup table.
        """

        if self.is_interpolated:

            if self._linspaced:
                n = int(np.floor((self._N-1.) * (arg-self.min)
                                 * (self.max-self.min)**-1.))

            else:
                n = self._N_array[self._arg_array <= arg][-1]

            if n >= self._N-1:
                return self._val_array[-1]

            else:
                return (self._val_array[n]
                        + (self._val_array[n+1]-self._val_array[n])
                        * (arg-self._arg_array[n])
                        * (self._arg_array[n+1]-self._arg_array[n])**-1.)

        else:
            return None

    def reset(self):
        """Undo the interpolation.

        After calling this, you can create a new lookup table by
        calling ``interpolate''.
        Warning: this will overwrite the existing file.
        """

        self.is_interpolated = False
        self._val_array = []

        return None

    def __call__(self, arg):

        if self.is_interpolated and arg >= self.min and arg <= self.max:
            return self._interpolated_func(arg)

        else:
            return self.func(arg)


#-----------------------------------------------------------------------------#
