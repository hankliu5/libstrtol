#include <Python.h>
#include <numpy/arrayobject.h>

#include "libstrtol.h"

static PyObject *PyInt_Deserialize(PyObject *self, PyObject *args) {
    npy_intp dims[2];
    char *string;
    int row, col;
    int *deserialized_matrix;
    PyObject *return_array;

    if (!PyArg_ParseTuple(args, "s", &string)) {
        return NULL;
    }

    deserialized_matrix = int_deserialize(string, &row, &col);
    dims[0] = row;
    dims[1] = col;
    return_array = PyArray_SimpleNewFromData(2, dims, NPY_INT, (void *) deserialized_matrix);
    PyArray_ENABLEFLAGS((PyArrayObject *)return_array, NPY_OWNDATA);
    return return_array;
}

static PyMethodDef PyDeserialize_methods[] = { 
    {   
        "deserialize", PyInt_Deserialize, METH_VARARGS,
        "deserializes an ascii matrix file to binary int matrix in a C extension."
    },  
    {NULL, NULL, 0, NULL}
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef PyDeserialize_definition = { 
    PyModuleDef_HEAD_INIT,
    "py_deserialize",
    "A Python module that deserializes an ascii matrix file to binary int matrix from C code.",
    -1, 
    PyDeserialize_methods
};


// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_py_deserialize(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&PyDeserialize_definition);
}

