#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <pliocomp.h>
#include <ricecomp.h>
#include <fits_hcompress.h>
#include <fits_hdecompress.h>

void ffpmsg(const char *err_message) {
}

typedef long long LONGLONG;

/* Define docstrings */
static char module_docstring[] = "Core compression/decompression functions";
static char compress_plio_1_c_docstring[] = "Compress data using PLIO_1";
static char decompress_plio_1_c_docstring[] = "Decompress data using PLIO_1";
static char compress_rice_1_c_docstring[] = "Compress data using RICE_1";
static char decompress_rice_1_c_docstring[] = "Decompress data using RICE_1";
static char compress_hcompress_1_c_docstring[] = "Compress data using HCOMPRESS_1";
static char decompress_hcompress_1_c_docstring[] = "Decompress data using HCOMPRESS_1";

/* Declare the C functions here. */
static PyObject *compress_plio_1_c(PyObject *self, PyObject *args);
static PyObject *decompress_plio_1_c(PyObject *self, PyObject *args);
static PyObject *compress_rice_1_c(PyObject *self, PyObject *args);
static PyObject *decompress_rice_1_c(PyObject *self, PyObject *args);
static PyObject *compress_hcompress_1_c(PyObject *self, PyObject *args);
static PyObject *decompress_hcompress_1_c(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"compress_plio_1_c", compress_plio_1_c, METH_VARARGS, compress_plio_1_c_docstring},
    {"decompress_plio_1_c", decompress_plio_1_c, METH_VARARGS, decompress_plio_1_c_docstring},
    {"compress_rice_1_c", compress_rice_1_c, METH_VARARGS, compress_rice_1_c_docstring},
    {"decompress_rice_1_c", decompress_rice_1_c, METH_VARARGS, decompress_rice_1_c_docstring},
    {"compress_hcompress_1_c", compress_hcompress_1_c, METH_VARARGS, compress_hcompress_1_c_docstring},
    {"decompress_hcompress_1_c", decompress_hcompress_1_c, METH_VARARGS, decompress_hcompress_1_c_docstring},
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

/* PLIO/IRAF compression */

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

    result = Py_BuildValue("y#", buf, compressed_length * 2);
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

/* RICE compression */

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

    result = Py_BuildValue("y#", compressed_values, compressed_length);
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

/* HCompress compression */

static PyObject *compress_hcompress_1_c(PyObject *self, PyObject *args) {

    const char* str;
    char * buf;
    Py_ssize_t count;
    PyObject * result;
    int *values;
    int start=1;
    unsigned char *compressed_values;
    int compressed_length;
    unsigned short blocksize, bytepix;
    int *decompressed_values_int;
    LONGLONG *decompressed_values_longlong;
    int nx, ny, scale, status;

    if (!PyArg_ParseTuple(args, "y#iiii", &str, &count, &nx, &ny, &scale, &bytepix))
    {
        return NULL;
    }

    // FIXME: for some reason nx is 0 instead of the value we pass in
    nx = ny;

    // TODO: not sure what the maximum length of the compressed data should be - for now
    // use count * 64 which means we assume the number of elements will be less than or equal
    // to the original number of elements.
    compressed_values = (char *)malloc(count * 64);

    if (bytepix == 4) {
        decompressed_values_int = (int *)str;
        compressed_length = fits_hcompress(decompressed_values_int,
                                           ny, nx, scale,
                                           compressed_values,
                                           &count, &status);
    } else {
        decompressed_values_longlong = (LONGLONG *)str;
        compressed_length = fits_hcompress64(decompressed_values_longlong,
                                             ny, nx, scale,
                                             compressed_values,
                                             &count, &status);
    }

    result = Py_BuildValue("y#", compressed_values, count);
    free(buf);
    return result;

}

static PyObject *decompress_hcompress_1_c(PyObject *self, PyObject *args) {

    const unsigned char* str;
    char * dbytes;
    Py_ssize_t count;
    PyObject * result;
    short *values;
    int start=1;
    int blocksize, bytepix;
    int i;
    unsigned int *decompressed_values_int;
    LONGLONG *decompressed_values_longlong;
    int nx, ny, scale, smooth, status;

    if (!PyArg_ParseTuple(args, "y#iiiii", &str, &count, &nx, &ny, &scale, &smooth, &bytepix))
    {
        return NULL;
    }

    // TODO: raise an error if bytepix is not 4 or 8

    if (bytepix == 4) {
        decompressed_values_int = (int *)malloc(nx * ny * 32);
        fits_hdecompress(str, smooth, decompressed_values_int, &ny, &nx, &scale, &status);
        dbytes = (char *)decompressed_values_int;
    } else {
        decompressed_values_longlong = (LONGLONG *)malloc(nx * ny * 64);
        fits_hdecompress64(str, smooth, decompressed_values_longlong, &ny, &nx, &scale, &status);
        dbytes = (char *)decompressed_values_longlong;
    }

    result = Py_BuildValue("y#", dbytes, nx * ny * bytepix);
    free(dbytes);
    return result;

}
