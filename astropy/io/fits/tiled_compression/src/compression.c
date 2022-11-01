#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <pliocomp.h>

/* Define docstrings */
static char module_docstring[] = "Core compression/decompression functions";
static char compress_plio_1_c_docstring[] = "Compress data using PLIO_1";
static char decompress_plio_1_c_docstring[] = "Decompress data using PLIO_1";

/* Declare the C functions here. */
static PyObject *compress_plio_1_c(PyObject *self, PyObject *args);
static PyObject *decompress_plio_1_c(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"compress_plio_1_c", compress_plio_1_c, METH_VARARGS, compress_plio_1_c_docstring},
    {"decompress_plio_1_c", decompress_plio_1_c, METH_VARARGS, decompress_plio_1_c_docstring},
    {NULL, NULL, 0, NULL}
};

/* This is the function that is called on import. */

#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);

MOD_INIT(_compression)
{
    PyObject *m;
    MOD_DEF(m, "_compression", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    return MOD_SUCCESS_VAL(m);
}


static PyObject *compress_plio_1_c(PyObject *self, PyObject *args) {

    const char* str;
    char * buf;
    Py_ssize_t count;
    PyObject * result;
    int *values;
    int start=1;
    short *compressed_values;
    int compressed_length;

    if (!PyArg_ParseTuple(args, "y#", &str, &count))
    {
        return NULL;
    }

    values = (int*)str;
    count /= 4;

    // TODO: not sure what the maximum length of the compressed data should be - for now
    // use count * 16 which means we assume the number of elements will be less than or equal
    // to the original number of elements.
    compressed_values = (short *)malloc(count * 16);

    // Zero the compressed values array
    for(int i=0;i<count*2;i++) {
        compressed_values[i] = 0;
    }

    compressed_length = pl_p2li(values, start, compressed_values, (int)count);

    buf = (char *)compressed_values;

    result = Py_BuildValue("y#", buf, count * 2);
    free(buf);
    return result;

}

static PyObject *decompress_plio_1_c(PyObject *self, PyObject *args) {

    const char* str;
    char * buf;
    Py_ssize_t count;
    PyObject * result;
    short *values;
    int start=1;
    int *decompressed_values;
    int npix;

    if (!PyArg_ParseTuple(args, "y#", &str, &count))
    {
        return NULL;
    }

    values = (short*)str;

    // TODO: determine how big we need to make the buffer here - we should
    // pass down information about the tile size to this function but for now
    // just hard-code a big size.
    decompressed_values = (int *)malloc(100000);

    npix = pl_l2pi(values, start, decompressed_values, (int)count);

    buf = (char *)decompressed_values;

    result = Py_BuildValue("y#", buf, npix * 4);
    free(buf);
    return result;

}
