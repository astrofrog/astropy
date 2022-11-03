#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <pliocomp.h>
#include <ricecomp.h>

/* Define docstrings */
static char module_docstring[] = "Core compression/decompression functions";
static char compress_plio_1_c_docstring[] = "Compress data using PLIO_1";
static char decompress_plio_1_c_docstring[] = "Decompress data using PLIO_1";
static char compress_rice_1_c_docstring[] = "Compress data using RICE_1";
static char decompress_rice_1_c_docstring[] = "Decompress data using RICE_1";

/* Declare the C functions here. */
static PyObject *compress_plio_1_c(PyObject *self, PyObject *args);
static PyObject *decompress_plio_1_c(PyObject *self, PyObject *args);
static PyObject *compress_rice_1_c(PyObject *self, PyObject *args);
static PyObject *decompress_rice_1_c(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"compress_plio_1_c", compress_plio_1_c, METH_VARARGS, compress_plio_1_c_docstring},
    {"decompress_plio_1_c", decompress_plio_1_c, METH_VARARGS, decompress_plio_1_c_docstring},
    {"compress_rice_1_c", compress_rice_1_c, METH_VARARGS, compress_rice_1_c_docstring},
    {"decompress_rice_1_c", decompress_rice_1_c, METH_VARARGS, decompress_rice_1_c_docstring},
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


static PyObject *compress_rice_1_c(PyObject *self, PyObject *args) {

    const char* str;
    char * buf;
    Py_ssize_t count;
    PyObject * result;
    int *values;
    int start=1;
    unsigned char *compressed_values;
    int compressed_length;
    unsigned short blocksize, bytepix;
    unsigned char *decompressed_values_byte;
    unsigned short *decompressed_values_short;
    unsigned int *decompressed_values_int;

    if (!PyArg_ParseTuple(args, "y#HH", &str, &count, &blocksize, &bytepix))
    {
        return NULL;
    }

    // TODO: not sure what the maximum length of the compressed data should be - for now
    // use count * 16 which means we assume the number of elements will be less than or equal
    // to the original number of elements.
    compressed_values = (short *)malloc(count * 16);

    if (bytepix == 1) {
        decompressed_values_byte = (unsigned char *)str;
        compressed_length = fits_rcomp_byte(decompressed_values_byte, (int)count, compressed_values, count * 16, blocksize);
    } else if (bytepix == 2){
        decompressed_values_short = (unsigned short *)str;
        compressed_length = fits_rcomp_short(decompressed_values_short, (int)count / 2, compressed_values, count * 16, blocksize);
    } else {
        decompressed_values_int = (unsigned int *)str;
        compressed_length = fits_rcomp(decompressed_values_int, (int)count / 4, compressed_values, count * 16, blocksize);
    }

    result = Py_BuildValue("y#", compressed_values, count * 2);
    free(buf);
    return result;

}

static PyObject *decompress_rice_1_c(PyObject *self, PyObject *args) {

    const char* str;
    char * dbytes;
    Py_ssize_t count;
    PyObject * result;
    short *values;
    int start=1;
    unsigned char *decompressed_values_byte;
    unsigned short *decompressed_values_short;
    unsigned int *decompressed_values_int;
    int npix;
    int blocksize, bytepix;
    int i;

    if (!PyArg_ParseTuple(args, "y#iii", &str, &count, &blocksize, &bytepix, &npix))
    {
        return NULL;
    }


    if (bytepix == 1) {
        decompressed_values_byte = (unsigned char *)malloc(npix * 8);
        fits_rdecomp_byte(str, (int)count, decompressed_values_byte, npix, blocksize);
        dbytes = (char *)decompressed_values_byte;
    } else if (bytepix == 2){
        decompressed_values_short = (unsigned short *)malloc(npix * 2 * 8);
        fits_rdecomp_short(str, (int)count, decompressed_values_short, npix, blocksize);
        dbytes = (char *)decompressed_values_short;
    } else {
        decompressed_values_int = (unsigned int *)malloc(npix * 4 * 8);
        fits_rdecomp(str, (int)count, decompressed_values_int, npix, blocksize);
        dbytes = (char *)decompressed_values_int;
    }

    result = Py_BuildValue("y#", dbytes, npix * bytepix);
    free(dbytes);
    return result;

}
