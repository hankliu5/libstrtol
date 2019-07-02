#include "libstrtol.h"
#include <Python.h>
#include <numpy/arrayobject.h>

int* int_deserialize(char* string, int *return_row, int *return_col) {
    int *return_matrix = NULL;
    char *start, *end;
    const char delimiter[2] = "\n";
    int row, col;
    int i, j;

    start = strtok(string, delimiter);
    row = (int) strtol(start, &end, 10);
    start = end;
    if (errno == ERANGE){
        printf("range error, got ");
        errno = 0;
        return return_matrix;
    }

    col = (int) strtol(start, &end, 10);
    if (errno == ERANGE){
        printf("range error, got ");
        errno = 0;
        return return_matrix;
    }
    printf("row: %d, col: %d\n", row, col);

    return_matrix = calloc(row * col, sizeof(int *));

    for (i = 0; i < row; i++) {
        start = strtok(NULL, delimiter);

        for (j = 0; j < col; j++) {
            *(return_matrix + i*col + j) = (int) strtol(start, &start, 10);
        }
    }
    *return_row = row;
    *return_col = col;
    return return_matrix;
}

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
    printf("after deserializing\n");
    dims[0] = row;
    dims[1] = col;
    return_array = PyArray_SimpleNewFromData(2, dims, NPY_INT, (void *) deserialized_matrix);
    PyArray_ENABLEFLAGS(return_array, NPY_OWNDATA);
    return return_array;
}

static PyMethodDef hello_methods[] = { 
    {   
        "deserialize", PyInt_Deserialize, METH_VARARGS,
        "Print 'hello xxx' from a method defined in a C extension."
    },  
    {NULL, NULL, 0, NULL}
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef hello_definition = { 
    PyModuleDef_HEAD_INIT,
    "hello",
    "A Python module that prints 'hello world' from C code.",
    -1, 
    hello_methods
};


// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_hello(void) {
    Py_Initialize();
    // (void) Py_InitModule("hello", hello_methods);
    import_array();
    return PyModule_Create(&hello_definition);
}

