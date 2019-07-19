#cython: language_level=3

""" Small Cython file to demonstrate the use of PyArray_SimpleNewFromData
in Cython to create an array from already allocated memory.
Cython enables mixing C-level calls and Python-level calls in the same
file with a Python-like syntax and easy type cohersion. See 
http://cython.org for more information
"""

# Author: Gael Varoquaux
# License: BSD

# Declare the prototype of the C function we are interested in calling
cdef extern from "libstrtol.c":
    int* int_deserialize(char* string, int *return_row, int *return_col)

from libc.stdlib cimport free, strtol, calloc
from cpython cimport PyObject, Py_INCREF
# from libc.time cimport clock_gettime, timespec

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as np
cimport cython

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# cdef timespec diff(timespec start, timespec end):
#     cdef timespec temp
#     if (end.tv_nsec-start.tv_nsec) < 0:
#         temp.tv_sec = end.tv_sec-start.tv_sec-1;
#         temp.tv_nsec = 1000000000 + end.tv_nsec-start.tv_nsec;
#     else:
#         temp.tv_sec = end.tv_sec-start.tv_sec;
#         temp.tv_nsec = end.tv_nsec-start.tv_nsec;

#     return temp 

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.
cdef class MatrixWrapper:
    cdef void* data_ptr
    cdef int row
    cdef int col

    cdef set_data(self, int row, int col, void* data_ptr):
        """ Set the data of the array
        This cannot be done in the constructor as it must recieve C-level
        arguments.
        Parameters:
        -----------
        size: int
            Length of the array.
        data_ptr: void*
            Pointer to the data            
        """
        self.data_ptr = data_ptr
        self.row = row
        self.col = col

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.row
        shape[1] = <np.npy_intp> self.col
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(2, shape,
                                               np.NPY_INT, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void*>self.data_ptr)


def cython_deserialize(char *string):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """
    cdef int *array
    cdef np.ndarray ndarray
    cdef int row, col
    # Call the C function
    array = int_deserialize(string, &row, &col)

    matrix_wrapper = MatrixWrapper()
    matrix_wrapper.set_data(row, col, <void*> array) 
    ndarray = np.array(matrix_wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarray.base = <PyObject*> matrix_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(matrix_wrapper)

    return ndarray

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_deserialize2(char *string):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """
    cdef int *array
    cdef char* start
    cdef np.ndarray ndarray
    cdef int row, col, total

    start = string
    # Call the C function
    row = <int> strtol(start, &start, 10);
    col = <int> strtol(start, &start, 10);
    total = row * col
    
    array = <int *> calloc(row * col, sizeof(int))

    with nogil:
        for i from 0 <= i < total:
            array[i] = <int> strtol(start, &start, 10);                

    matrix_wrapper = MatrixWrapper()
    matrix_wrapper.set_data(row, col, <void*> array) 
    ndarray = np.array(matrix_wrapper, copy=False)

    # Assign our object to the 'base' of the ndarray object
    ndarray.base = <PyObject*> matrix_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(matrix_wrapper)

    return ndarray

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_deserialize3(char *string):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """
    cdef char* start
    cdef int row, col
    
    start = string
    # Call the C function
    row = <int> strtol(start, &start, 10);
    col = <int> strtol(start, &start, 10);
    result = np.zeros((row, col), dtype=np.int32)
    cdef int[:, :] result_view = result
    
    with nogil:
        for i from 0 <= i < row:
            for j from 0 <= j < col:
                result_view[i, j] = <int> strtol(start, &start, 10);

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_deserialize4(char *string, int num_per_iter):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """
    cdef char* start
    cdef int row, col, curr_iter
    
    start = string
    # Call the C function
    row = <int> strtol(start, &start, 10);
    col = <int> strtol(start, &start, 10);
    result = np.zeros((row, col), dtype=np.int32)
    cdef int[:, :] result_view = result
    
    curr_iter = 0
    for i from 0 <= i < row:
        for j from 0 <= j < col:            
            if curr_iter == num_per_iter:
                return start, result, i, j
            result_view[i, j] = <int> strtol(start, &start, 10);
            curr_iter += 1 


    return b'', result, row, col

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_deserialize5(char *string, int[:, :] result, int curr_row, int curr_col):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """    
    cdef Py_ssize_t row = result.shape[0]
    cdef Py_ssize_t col = result.shape[1]
    if curr_row == row:
        return result

    cdef char* start = string
    cdef int[:, :] result_view = result
    
    with nogil:
        for j from curr_col <= j < col:
            result_view[curr_row, j] = <int> strtol(start, &start, 10);
    
    curr_col = 0
    curr_row += 1

    with nogil:    
        for i from curr_row <= i < row:
            for j from 0 <= j < col:
                result_view[i, j] = <int> strtol(start, &start, 10);

    return result