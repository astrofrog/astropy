/*
 Generic WCSLIB distortion (disprm) binding -- see disprm_wrap.h.
*/

#define NO_IMPORT_ARRAY

#include "astropy_wcs/disprm_wrap.h"

#include <wcserr.h>

static PyObject*
PyDisprm_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyDisprm* self = (PyDisprm*)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->x.flag = -1;
    self->x.err = NULL;
  }
  return (PyObject*)self;
}

static void
PyDisprm_dealloc(PyDisprm* self) {
  disfree(&self->x);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
 * __init__(naxis, dtype, records)
 *   naxis    : int
 *   dtype    : sequence of `naxis` distortion-type strings (e.g. "SIP")
 *   records  : sequence of (field:str, axis:int, value:float) DPja keyrecords,
 *              e.g. ("SIP.FWD.2_0", 1, 1.0e-5).  Stored as floating point;
 *              WCSLIB coerces integer-valued records (NAXES, AXIS) as needed.
 */
static int
PyDisprm_init(PyDisprm* self, PyObject* args, PyObject* kwds) {
  int naxis;
  PyObject* dtype_obj;
  PyObject* records_obj;
  PyObject* dtype_seq = NULL;
  PyObject* records_seq = NULL;
  Py_ssize_t ndtype, nrec, i;
  int status;
  struct dpkey* keyp;

  if (!PyArg_ParseTuple(args, "iOO:Distortion.__init__",
                        &naxis, &dtype_obj, &records_obj)) {
    return -1;
  }

  dtype_seq = PySequence_Fast(dtype_obj, "dtype must be a sequence");
  if (dtype_seq == NULL) return -1;
  records_seq = PySequence_Fast(records_obj, "records must be a sequence");
  if (records_seq == NULL) { Py_DECREF(dtype_seq); return -1; }

  ndtype = PySequence_Fast_GET_SIZE(dtype_seq);
  nrec = PySequence_Fast_GET_SIZE(records_seq);

  if (ndtype != naxis) {
    PyErr_SetString(PyExc_ValueError, "len(dtype) must equal naxis");
    goto error;
  }

  status = disinit(1, naxis, &self->x, (int)nrec);
  if (status) {
    PyErr_SetString(PyExc_MemoryError, "disinit failed");
    goto error;
  }

  for (i = 0; i < naxis; ++i) {
    PyObject* s = PySequence_Fast_GET_ITEM(dtype_seq, i);  /* borrowed */
    const char* dt = PyUnicode_AsUTF8(s);
    if (dt == NULL) goto error;
    strncpy(self->x.dtype[i], dt, 71);
    self->x.dtype[i][71] = '\0';
  }

  keyp = self->x.dp;
  for (i = 0; i < nrec; ++i, ++keyp) {
    PyObject* rec = PySequence_Fast_GET_ITEM(records_seq, i);  /* borrowed */
    const char* field;
    int axis;
    double value;
    PyObject* f_obj;
    PyObject* a_obj;
    PyObject* v_obj;

    if (!PyTuple_Check(rec) || PyTuple_GET_SIZE(rec) != 3) {
      PyErr_SetString(PyExc_TypeError,
                      "each record must be a (field, axis, value) tuple");
      goto error;
    }
    f_obj = PyTuple_GET_ITEM(rec, 0);
    a_obj = PyTuple_GET_ITEM(rec, 1);
    v_obj = PyTuple_GET_ITEM(rec, 2);

    field = PyUnicode_AsUTF8(f_obj);
    if (field == NULL) goto error;
    axis = (int)PyLong_AsLong(a_obj);
    if (axis == -1 && PyErr_Occurred()) goto error;
    value = PyFloat_AsDouble(v_obj);
    if (value == -1.0 && PyErr_Occurred()) goto error;

    /* keyword "DP" + field -> dp->field = "DP<axis>.<field>"; type 1 = float */
    dpfill(keyp, "DP", field, axis, 1, 0, value);
  }
  self->x.ndp = (int)nrec;

  status = disset(&self->x);
  if (status) {
    if (self->x.err && self->x.err->msg[0]) {
      PyErr_SetString(PyExc_ValueError, self->x.err->msg);
    } else {
      PyErr_SetString(PyExc_ValueError, "disset failed");
    }
    goto error;
  }

  Py_DECREF(dtype_seq);
  Py_DECREF(records_seq);
  return 0;

error:
  Py_XDECREF(dtype_seq);
  Py_XDECREF(records_seq);
  return -1;
}

/*
 * transform(coords) -> ndarray
 *   coords : (N, naxis) float64 array of "raw" coordinates.
 *   returns the distorted coordinates (disp2x applied row-wise); i.e. the
 *   full corrected coordinate, leaving any reference-pixel/origin handling to
 *   the caller.
 */
static PyObject*
PyDisprm_transform(PyDisprm* self, PyObject* args) {
  PyObject* coords_obj;
  PyArrayObject* coords = NULL;
  PyArrayObject* out = NULL;
  int naxis = self->x.naxis;
  npy_intp nrow, i;
  int status = 0;

  if (!PyArg_ParseTuple(args, "O:transform", &coords_obj)) {
    return NULL;
  }

  coords = (PyArrayObject*)PyArray_ContiguousFromAny(coords_obj, NPY_DOUBLE, 2, 2);
  if (coords == NULL) return NULL;

  if (PyArray_DIM(coords, 1) != naxis) {
    PyErr_Format(PyExc_ValueError,
                 "coordinate array must be (N, %d)", naxis);
    Py_DECREF(coords);
    return NULL;
  }

  out = (PyArrayObject*)PyArray_SimpleNew(2, PyArray_DIMS(coords), NPY_DOUBLE);
  if (out == NULL) { Py_DECREF(coords); return NULL; }

  nrow = PyArray_DIM(coords, 0);

  Py_BEGIN_ALLOW_THREADS
  {
    const double* in_data = (const double*)PyArray_DATA(coords);
    double* out_data = (double*)PyArray_DATA(out);
    for (i = 0; i < nrow; ++i) {
      status = disp2x(&self->x, in_data + i * naxis, out_data + i * naxis);
      if (status) break;
    }
  }
  Py_END_ALLOW_THREADS

  Py_DECREF(coords);

  if (status) {
    Py_DECREF(out);
    PyErr_SetString(PyExc_ValueError, "disp2x failed");
    return NULL;
  }

  return (PyObject*)out;
}

static PyObject*
PyDisprm_get_naxis(PyDisprm* self, void* closure) {
  return PyLong_FromLong(self->x.naxis);
}

static PyGetSetDef PyDisprm_getset[] = {
  {"naxis", (getter)PyDisprm_get_naxis, NULL, "Number of axes", NULL},
  {NULL}
};

static PyMethodDef PyDisprm_methods[] = {
  {"transform", (PyCFunction)PyDisprm_transform, METH_VARARGS,
   "transform(coords) -> distorted coords (disp2x applied row-wise)"},
  {NULL}
};

static PyType_Spec DisprmType_spec = {
  .name = "astropy.wcs._wcs.Distortion",
  .basicsize = sizeof(PyDisprm),
  .itemsize = 0,
  .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .slots = (PyType_Slot[]){
    {Py_tp_dealloc, (destructor)PyDisprm_dealloc},
    {Py_tp_methods, PyDisprm_methods},
    {Py_tp_getset, PyDisprm_getset},
    {Py_tp_init, (initproc)PyDisprm_init},
    {Py_tp_new, PyDisprm_new},
    {0, NULL},
  },
};

PyTypeObject* DisprmType = NULL;

int
_setup_disprm_type(PyObject* m) {
  DisprmType = (PyTypeObject*)PyType_FromSpec(&DisprmType_spec);
  if (DisprmType == NULL) {
    return -1;
  }
  return PyModule_AddObject(m, "Distortion", (PyObject*)DisprmType);
}
