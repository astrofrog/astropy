#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "convolve.h"

/* Define docstrings */
static char module_docstring[] = "Convolution with a padded boundary";
static char function_docstring[] = "Convolution with a padded boundary";

/* Declare the C functions here. */
static PyObject *convolve_boundary_padded(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"convolve_boundary_padded", convolve_boundary_padded, METH_VARARGS, function_docstring},
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

MOD_INIT(_convolve_boundary_padded)
{
    PyObject *m;
    MOD_DEF(m, "_convolve_boundary_padded", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    return MOD_SUCCESS_VAL(m);
}

/*-------------------------PERFORMANCE NOTES--------------------------------
 * Only 2 scenarios exist for boundary='fill':
 * (image dims: nx, ny, nz; kernel dims: nkx, nky, nkz).
 * (ii, jj, kk, refer to image locations but have iteration length of kernel dims).
 * 1) kernel over-steps image.
 *   I.e. (ii < 0 || ii >= nx) || (jj < 0 || jj >= ny) || (kk < 0 || kk >= nz) == true
 * 2) kernel completely within image.
 *  I.e. (ii < 0 || ii >= nx) || (jj < 0 || jj >= ny) || (kk < 0 || kk >= nz) == false
 * Given that the kernel is normally much smaller than the image, (2)
 * occupies the majority of the iteration space.
 * For (2) the check is exhaustive (no short-circuit), all conditions MUST be checked.
 * It is not possible to completely optimize for both (1) & (2). Better to make
 * majority case more efficient.
 * For (2) that's, at most, 6*nkx*nky*nkz checks. 3*nkx*nky (2D case).
 * 1st tier optimization is to split these out, to un-nest them, i.e. hoist
 * ii condition to ii, kk to kk etc. Worse case -> c*(nkx + nky + nkz), c = 2.
 * For (2) `val` (c.f. code) can only be assigned a value within the inner
 * most loop as all indices are needed. Therefore, with the condition hoisted
 * the result must now be propagated to the next nest level. Use a bool flag
 * for this.
 * I.e. (2D case)
 * ```
 * ...
 *  for (int ii = i_minus_wkx; ii < (int)i_plus_wkx_plus_1; ++ii)
    {
        ...
        bool fill_value_used = false;
        if (ii < 0 || ii >= (int)nx)
        {
            val = fill_value;
            fill_value_used = true;
        }
        ...
        for (int jj = j_minus_wky; jj < (int)j_plus_wky_plus_1; ++jj)
        {
            ...
            if (!fill_value_used)
            {
                if (jj < 0 || jj >= (int)ny)
                    val = fill_value;
                else
                    val = f[(unsigned)ii*ny + jj];
            }
            ...
        }
    ...
    }
    ...
````
 * This yields a perf boost for (1) when a true condition in an outer loop
 * short-circuiting the next loop checks, bar the introduced gateway condition
 * on `fill_value_used`. For (2) it makes the worse case 2*nkx + 3*(nky + nkz)
 * which is still significantly better than 6*nkx*nky*nkz.
 *
 * Further opt:
 * 1) The default fill_value = 0. => sub-nested loops can be
 * skipped entirely. val = fill_value = 0 => val*ker = 0, top += val*ker = top.
 * This is NOT possible for nan interpolation.
 * 2) Scrap all of the above and just pad the image array by, e.g. wkx = nkx/2,
 * with fill_value and remove all conditions. (mem issues for large kernels).
 *--------------------------------------------------------------------------
 */

// 1D
FORCE_INLINE void convolve1d_boundary_padded(DTYPE * const result,
        const DTYPE * const f, const size_t _nx,
        const DTYPE * const g, const size_t _nkx,
        const bool _nan_interpolate,
        const unsigned n_threads)
{
    /*------------------------------WARNING!--------------------------------
     * The size_t variables _nx & _nkx etc, are the lengths of the original
     * image array and NOT the padded version that is actually passed in.
     *------------------------------WARNING!--------------------------------
     */

#ifdef NDEBUG
    if (!result || !f || !g)
        return;
#else
    assert(result);
    assert(f);
    assert(g);
#endif

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel
    { // Code within this block is threaded
#endif

    // Copy these to thread locals to allow compiler to optimize (hoist/loads licm)
    // when threaded. Without these, compile time constant conditionals may
    // not be optimized away.
    const size_t nx = _nx;
    const size_t nkx = _nkx;
    const bool nan_interpolate = _nan_interpolate;

    // Thread locals
    const size_t wkx = nkx / 2;
    const size_t nkx_minus_1 = nkx-1;
    int wkx_minus_i;
    size_t ker_i;
    size_t i_minus_wkx;
    const size_t wkx_plus_1 = wkx + 1;
    omp_iter_var i_plus_wkx_plus_1;
    int nkx_minus_1_minus_wkx_plus_i;
    size_t i_unpadded;

    DTYPE top, bot=0., ker, val;

    {omp_iter_var i;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (i = wkx; i < nx + wkx; ++i)
    {
        wkx_minus_i = wkx - i; // wkx - 1
        i_minus_wkx = i - wkx; //i - wkx
        i_unpadded = i_minus_wkx;
        i_plus_wkx_plus_1 = i + wkx_plus_1; // i + wkx + 1
        nkx_minus_1_minus_wkx_plus_i = nkx_minus_1 - wkx_minus_i; // nkx - 1 - (wkx - i)

        top = 0.;
        if (nan_interpolate) // compile time constant
            bot = 0.;

        {omp_iter_var ii;
        for (ii = i_minus_wkx; ii < i_plus_wkx_plus_1; ++ii)
        {
            val = f[ii];
            ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
            ker = g[ker_i];
            if (nan_interpolate) // compile time constant
            {
                if (!isnan(val))
                {
                    top += val * ker;
                    bot += ker;
                }
            }
            else
                top += val * ker;
        }}

        if (nan_interpolate) // compile time constant
        {
            if (bot == 0) // This should prob be np.isclose(kernel_sum, 0, atol=normalization_zero_tol)
                result[i_unpadded]  = f[i];
            else
                result[i_unpadded]  = top / bot;
        }
        else
            result[i_unpadded] = top;
    }}
#ifdef _OPENMP
    }//end parallel scope
#endif
}

//2D
FORCE_INLINE void convolve2d_boundary_padded(DTYPE * const result,
        const DTYPE * const f, const size_t _nx, const size_t _ny,
        const DTYPE * const g, const size_t _nkx, const size_t _nky,
        const bool _nan_interpolate,
        const unsigned n_threads)
{
    /*------------------------------WARNING!--------------------------------
     * The size_t variables _nx & _nkx etc, are the lengths of the original
     * image array and NOT the padded version that is actually passed in.
     *------------------------------WARNING!--------------------------------
     */

#ifdef NDEBUG
    if (!result || !f || !g)
        return;
#else
    assert(result);
    assert(f);
    assert(g);
#endif

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel
    { // Code within this block is threaded
#endif

    // Copy these to thread locals to allow compiler to optimize (hoist/loads licm)
    // when threaded. Without these, compile time constant conditionals may
    // not be optimized away.
    const size_t nx = _nx, ny = _ny;
    const size_t nkx = _nkx, nky = _nky;
    const bool nan_interpolate = _nan_interpolate;

    // Thread locals
    const size_t wkx = nkx / 2;
    const size_t wky = nky / 2;
    const size_t ny_padded = ny + 2*wky;
    const size_t nkx_minus_1 = nkx-1, nky_minus_1 = nky-1;
    int wkx_minus_i, wky_minus_j;
    size_t ker_i, ker_j;
    int i_minus_wkx, j_minus_wky;
    const size_t wkx_plus_1 = wkx + 1;
    const size_t wky_plus_1 = wky + 1;
    omp_iter_var i_plus_wkx_plus_1, j_plus_wky_plus_1;
    int nkx_minus_1_minus_wkx_plus_i, nky_minus_1_minus_wky_plus_j;
    size_t i_unpadded, j_unpadded;
    DTYPE top, bot=0., ker, val;

    {omp_iter_var i;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (i = wkx; i < nx + wkx; ++i)
    {
        wkx_minus_i = wkx - i; // wkx - 1
        i_minus_wkx = i - wkx; //i - wkx
        i_unpadded = i_minus_wkx;
        i_plus_wkx_plus_1 = i + wkx_plus_1; // i + wkx + 1
        nkx_minus_1_minus_wkx_plus_i = nkx_minus_1 - wkx_minus_i; // nkx - 1 - (wkx - i)

        {omp_iter_var j;
        for (j = wky; j < ny + wky; ++j)
        {
            wky_minus_j = wky - j; // wky - j
            j_minus_wky = j - wky; // j - wky
            j_unpadded = j_minus_wky;
            j_plus_wky_plus_1 = j + wky_plus_1; // j + wky + 1
            nky_minus_1_minus_wky_plus_j = nky_minus_1 - wky_minus_j; // nky - 1 - (wky - i)

            top = 0.;
            if (nan_interpolate) // compile time constant
                bot = 0.;
            {omp_iter_var ii;
            for (ii = i_minus_wkx; ii < i_plus_wkx_plus_1; ++ii)
            {
                ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
                {omp_iter_var jj;
                for (jj = j_minus_wky; jj < j_plus_wky_plus_1; ++jj)
                {
                    val = f[ii*ny_padded + jj];
                    ker_j = nky_minus_1_minus_wky_plus_j - jj; // nky - 1 - (wky + jj - j)
                    ker = g[ker_i*nky + ker_j]; // [ker_i, ker_j];
                    if (nan_interpolate) // compile time constant
                    {
                        if (!isnan(val))
                        {
                            top += val * ker;
                            bot += ker;
                        }
                    }
                    else
                        top += val * ker;
                }}
            }}
            if (nan_interpolate) // compile time constant
            {
                if (bot == 0) // This should prob be np.isclose(kernel_sum, 0, atol=normalization_zero_tol)
                    result[i_unpadded * ny + j_unpadded] = f[i * ny_padded + j];
                else
                    result[i_unpadded * ny + j_unpadded] = top / bot;
            }
            else
            result[i_unpadded * ny + j_unpadded] = top;
        }}
    }}
#ifdef _OPENMP
    }//end parallel scope
#endif
}

// 3D
FORCE_INLINE void convolve3d_boundary_padded(DTYPE * const result,
        const DTYPE * const f, const size_t _nx, const size_t _ny, const size_t _nz,
        const DTYPE * const g, const size_t _nkx, const size_t _nky, const size_t _nkz,
        const bool _nan_interpolate,
        const unsigned n_threads)
{
    /*------------------------------WARNING!--------------------------------
     * The size_t variables _nx & _nkx etc, are the lengths of the original
     * image array and NOT the padded version that is actually passed in.
     *------------------------------WARNING!--------------------------------
     */
#ifdef NDEBUG
    if (!result || !f || !g)
        return;
#else
    assert(result);
    assert(f);
    assert(g);
#endif

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel
    { // Code within this block is threaded
#endif

    // Copy these to thread locals to allow compiler to optimize (hoist/loads licm)
    // when threaded. Without these, compile time constant conditionals may
    // not be optimized away.
    const size_t nx = _nx, ny = _ny, nz = _nz;
    const size_t nkx = _nkx, nky = _nky, nkz = _nkz;
    const bool nan_interpolate = _nan_interpolate;

    // Thread locals
    const size_t wkx = nkx / 2;
    const size_t wky = nky / 2;
    const size_t wkz = nkz / 2;
    const size_t ny_padded = ny + 2*wky;
    const size_t nz_padded = nz + 2*wkz;
    const size_t nkx_minus_1 = nkx-1, nky_minus_1 = nky-1, nkz_minus_1 = nkz-1;
    int wkx_minus_i, wky_minus_j, wkz_minus_k;
    size_t ker_i, ker_j, ker_k;
    int i_minus_wkx, j_minus_wky, k_minus_wkz;
    const size_t wkx_plus_1 = wkx + 1;
    const size_t wky_plus_1 = wky + 1;
    const size_t wkz_plus_1 = wkz + 1;
    omp_iter_var i_plus_wkx_plus_1, j_plus_wky_plus_1, k_plus_wkz_plus_1;
    int nkx_minus_1_minus_wkx_plus_i, nky_minus_1_minus_wky_plus_j, nkz_minus_1_minus_wkz_plus_k;
    size_t i_unpadded, j_unpadded, k_unpadded;

    DTYPE top, bot=0., ker;
    DTYPE val;

    {omp_iter_var i;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (i = wkx; i < nx + wkx; ++i)
    {
        wkx_minus_i = wkx - i; // wkx - 1
        i_minus_wkx = i - wkx; //i - wkx
        i_unpadded = i_minus_wkx;
        i_plus_wkx_plus_1 = i + wkx_plus_1; // i + wkx + 1
        nkx_minus_1_minus_wkx_plus_i = nkx_minus_1 - wkx_minus_i; // nkx - 1 - (wkx - i)

        {omp_iter_var j;
        for (j = wky; j < ny + wky; ++j)
        {
            wky_minus_j = wky - j; // wky - j
            j_minus_wky = j - wky; // j - wky
            j_unpadded = j_minus_wky;
            j_plus_wky_plus_1 = j + wky_plus_1; // j + wky + 1
            nky_minus_1_minus_wky_plus_j = nky_minus_1 - wky_minus_j; // nky - 1 - (wky - i)

            {omp_iter_var k;
            for (k = wkz; k < nz + wkz; ++k)
            {
                wkz_minus_k = wkz - k; // wkz - k
                k_minus_wkz = k - wkz; // k - wkz
                k_unpadded = k_minus_wkz;
                k_plus_wkz_plus_1 = k + wkz_plus_1; // k + wkz + 1
                nkz_minus_1_minus_wkz_plus_k = nkz_minus_1 - wkz_minus_k; // nkz - 1 - (wkz - i)

                top = 0.;
                if (nan_interpolate) // compile time constant
                    bot = 0.;
                {omp_iter_var ii;
                for (ii = i_minus_wkx; ii < i_plus_wkx_plus_1; ++ii)
                {
                    ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
                    {omp_iter_var jj;
                    for (jj = j_minus_wky; jj < j_plus_wky_plus_1; ++jj)
                    {
                        ker_j = nky_minus_1_minus_wky_plus_j - jj; // nky - 1 - (wky + jj - j)
                        {omp_iter_var kk;
                        for (kk = k_minus_wkz; kk < k_plus_wkz_plus_1; ++kk)
                        {
                            ker_k = nkz_minus_1_minus_wkz_plus_k - kk; // nkz - 1 - (wkz + kk - k)
                            val = f[(ii*ny_padded + jj)*nz_padded + kk]; //[ii, jj, kk];
                            ker = g[(ker_i*nky + ker_j)*nkz + ker_k]; // [ker_i, ker_j, ker_k];
                            if (nan_interpolate) // compile time constant
                            {
                                if (!isnan(val))
                                {
                                    top += val * ker;
                                    bot += ker;
                                }
                            }
                            else
                                top += val * ker;
                        }}
                    }}
                }}
                if (nan_interpolate) // compile time constant
                {
                    if (bot == 0) // This should prob be np.isclose(kernel_sum, 0, atol=normalization_zero_tol)
                        result[(i_unpadded*ny + j_unpadded)*nz + k_unpadded]  = f[(i*ny_padded + j)*nz_padded + k] ;
                    else
                        result[(i_unpadded*ny + j_unpadded)*nz + k_unpadded]  = top / bot;
                }
                else
                    result[(i_unpadded*ny + j_unpadded)*nz + k_unpadded] = top;
            }}
        }}
    }}
#ifdef _OPENMP
    }//end parallel scope
#endif
}

static PyObject *convolve_boundary_padded(PyObject *self, PyObject *args) {

  int ndim;
  PyObject *result_obj, *array_obj, *kernel_obj;
  bool nan_interpolate, n_threads;
  PyArrayObject *result_arr, *array_arr, *kernel_arr;
  size_t nx, ny, nz, nkx, nky, nkz;
  DTYPE *result, *array, *kernel;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOii", &result_obj, &array_obj, &kernel_obj, &nan_interpolate, &n_threads)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  result_arr = (PyArrayObject *)PyArray_FROM_OTF(result_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (result_arr == NULL) {
    PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
    Py_XDECREF(result_arr);
    return NULL;
  }

  array_arr = (PyArrayObject *)PyArray_FROM_OTF(array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (array_arr == NULL) {
    PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
    Py_DECREF(result_arr);
    Py_XDECREF(array_arr);
    return NULL;
  }

  kernel_arr = (PyArrayObject *)PyArray_FROM_OTF(kernel_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (result_arr == NULL) {
    PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
    Py_DECREF(result_arr);
    Py_DECREF(array_arr);
    Py_XDECREF(kernel_arr);
    return NULL;
  }

  result = (DTYPE*)PyArray_DATA(result_arr);
  array = (DTYPE*)PyArray_DATA(array_arr);
  kernel = (DTYPE*)PyArray_DATA(kernel_arr);

  /* How many data points are there? */
  ndim = PyArray_NDIM(result_arr);

  nx = PyArray_DIM(result_arr, 0);
  nkx = PyArray_DIM(kernel_arr, 0);

  if (ndim > 1) {
    ny = PyArray_DIM(result_arr, 1);
    nky = PyArray_DIM(kernel_arr, 1);
  }

  if (ndim > 2) {
    nz = PyArray_DIM(result_arr, 2);
    nkz = PyArray_DIM(kernel_arr, 2);
  }

  /*-------------------------PERFORMANCE NOTES--------------------------------
  * The if statements below are designed to take advantage of the following:
  * The preprocessor will inline convolve<N>d_boundary_none(), effectively
  * expanding the two logical branches, replacing nan_interpolate
  * for their literal equivalents. The corresponding conditionals
  * within these functions will then be optimized away, this
  * being the goal - removing the unnecessary conditionals from
  * the loops without duplicating code.
  *--------------------------------------------------------------------------
  */

  if (ndim == 1) {

    if (nan_interpolate)
        convolve1d_boundary_padded(result, array, nx, kernel, nkx, true, n_threads);
    else
        convolve1d_boundary_padded(result, array, nx, kernel, nkx, false, n_threads);

  } else if(ndim == 2) {

    if (nan_interpolate)
        convolve2d_boundary_padded(result, array, nx, ny, kernel, nkx, nky, true, n_threads);
    else
        convolve2d_boundary_padded(result, array, nx, ny, kernel, nkx, nky, false, n_threads);

  } else if(ndim == 3) {

    if (nan_interpolate)
        convolve3d_boundary_padded(result, array, nx, ny, nz, kernel, nkx, nky, nkz, true, n_threads);
    else
        convolve3d_boundary_padded(result, array, nx, ny, nz, kernel, nkx, nky, nkz, false, n_threads);

  }

  return result_obj;
}
