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

cdef extern from "math.h":
    double sqrt(double m)

from libc.stdlib cimport free, strtol, calloc
from cpython cimport PyObject, Py_INCREF
from posix.time cimport clock_gettime, timespec, CLOCK_MONOTONIC_COARSE

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as np
cimport cython

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef long time_diff(timespec *start, timespec *end):
    cdef timespec temp
    if (end.tv_nsec - start.tv_nsec) < 0:
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec
    else:
        temp.tv_sec = end.tv_sec - start.tv_sec
        temp.tv_nsec = end.tv_nsec - start.tv_nsec
    return (temp.tv_sec * 1000000000) + temp.tv_nsec

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


@cython.wraparound(False)
@cython.boundscheck(False)
def cython_stddev(np.ndarray[np.float64_t, ndim=1] a not None):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef double mean = 0.0
    for i in range(n):
        mean += a[i]
    mean /= n
    cdef double var = 0.0
    for i in range(n):
        var += (a[i] - mean)**2
    return sqrt(var / n)


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
    cdef int row, col, curr_iter, num_elem_in_time_arr, index_time_arr
    cdef double std, curr_time
    cdef int time_arr_size = 10
    
    start = string
    # Call the C function
    row = <int> strtol(start, &start, 10);
    col = <int> strtol(start, &start, 10);
    time_arr = np.zeros(time_arr_size, dtype=np.float64)
    result = np.zeros((row, col), dtype=np.int32)
    cdef int[:, :] result_view = result
    
    curr_iter = 0
    num_elem_in_time_arr = 0
    index_time_arr = 0

    for i from 0 <= i < row:
        for j from 0 <= j < col:            
            if curr_iter == num_per_iter:
                curr_time = 10.0
                if num_elem_in_time_arr < time_arr_size:
                    num_elem_in_time_arr += 1
                elif num_elem_in_time_arr >= time_arr_size:
                    std = cython_stddev(time_arr)
                    if curr_time >= 2 * std:
                        return start, result, i, j
                time_arr[index_time_arr] = curr_time
                index_time_arr += 1
                index_time_arr %= time_arr_size
                curr_iter = 0
            result_view[i, j] = <int> strtol(start, &start, 10);
            curr_iter += 1 

    return b'', result, row, col

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_deserialize5(char *string, np.ndarray[dtype=np.int32_t, ndim=2, mode='c'] result, int curr_row, int curr_col):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """    
    cdef Py_ssize_t row = result.shape[0]
    cdef Py_ssize_t col = result.shape[1]
    if curr_row == row:
        return result

    cdef char* start = string
    cdef np.ndarray ndarray
    
    ndarray = np.array(result, copy=True)
    cdef int[:, :] result_view = ndarray
    
    with nogil:
        for j from curr_col <= j < col:
            result_view[curr_row, j] = <int> strtol(start, &start, 10);
    
    curr_col = 0
    curr_row += 1

    with nogil:    
        for i from curr_row <= i < row:
            for j from 0 <= j < col:
                result_view[i, j] = <int> strtol(start, &start, 10);

    return ndarray

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_deserialize6(char *string, int num_per_iter, long timeout_in_sec):
    """ Python binding of the 'compute' function in 'c_code.c' that does
        not copy the data allocated in C.
    """
    cdef char* start
    cdef int row, col, curr_iter = 0
    cdef long timeout_in_ns = timeout_in_sec * 1000000000l
    cdef timespec start_time, curr_time, temp_time
    
    clock_gettime(CLOCK_MONOTONIC_COARSE, &start_time)
    start = string

    # Call the C function
    row = <int> strtol(start, &start, 10);
    col = <int> strtol(start, &start, 10);
    result = np.zeros((row, col), dtype=np.int32)
    cdef int[:, :] result_view = result
    
    for i from 0 <= i < row:
        for j from 0 <= j < col:            
            if curr_iter == num_per_iter:
                clock_gettime(CLOCK_MONOTONIC_COARSE, &curr_time)
                if time_diff(&start_time, &curr_time) > timeout_in_ns:
                    return start, result, i, j
                curr_iter = 0
            result_view[i, j] = <int> strtol(start, &start, 10);
            curr_iter += 1 

    return b'', result, row, col