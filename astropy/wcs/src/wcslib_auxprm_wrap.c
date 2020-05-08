#define NO_IMPORT_ARRAY

#include "astropy_wcs/wcslib_auxprm_wrap.h"

#include <wcs.h>
#include <wcsprintf.h>
#include <tab.h>

/*
 It gets to be really tedious to type long docstrings in ANSI C syntax
 (since multi-line strings literals are not valid).  Therefore, the
 docstrings are written in doc/docstrings.py, which are then converted
 by setup.py into docstrings.h, which we include here.
*/
#include "astropy_wcs/docstrings.h"


/***************************************************************************
 * PyAuxprm methods                                                        *
 ***************************************************************************/

static PyObject*
PyAuxprm_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyAuxprm* self;
  self = (PyAuxprm*)type->tp_alloc(type, 0);
  return (PyObject*)self;
}


static int
PyAuxprm_traverse(PyAuxprm* self, visitproc visit, void *arg) {
  Py_VISIT(self->owner);
  return 0;
}


static int
PyAuxprm_clear(PyAuxprm* self) {
  Py_CLEAR(self->owner);
  return 0;
}


static void PyAuxprm_dealloc(PyAuxprm* self) {
  PyAuxprm_clear(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
}


PyAuxprm* PyAuxprm_cnew(PyObject* wcsprm, struct auxprm* x) {
  PyAuxprm* self;
  printf("ici1\n");
  self = (PyAuxprm*)(&PyAuxprmType)->tp_alloc(&PyAuxprmType, 0);
  printf("ici2\n");
  if (self == NULL) return NULL;
  printf("ici3\n");
  self->x = x;
  printf("ici4\n");
  Py_INCREF(wcsprm);
  printf("ici5\n");
  self->owner = wcsprm;
  return self;
}


static void auxprmprt(const struct auxprm *aux) {

  if (aux == 0x0) return;

  wcsprintf("rsun_ref: %d\n", aux->rsun_ref);
  wcsprintf("dsun_obs: %d\n", aux->dsun_obs);
  wcsprintf("crln_obs: %c\n", aux->crln_obs);
  wcsprintf("hgln_obs: %s\n", aux->hgln_obs);
  wcsprintf("hglt_obs: %d\n", aux->hglt_obs);

  return;
}


static PyObject* PyAuxprm_print_contents(PyAuxprm* self) {
  /* This is not thread-safe, but since we're holding onto the GIL,
     we can assume we won't have thread conflicts */
  wcsprintf_set(NULL);
  auxprmprt(self->x);
  printf("%s", wcsprintf_buf());
  fflush(stdout);
  Py_RETURN_NONE;
}


static PyObject* PyAuxprm___str__(PyAuxprm* self) {
  /* This is not thread-safe, but since we're holding onto the GIL,
     we can assume we won't have thread conflicts */
  wcsprintf_set(NULL);
  auxprmprt(self->x);
  return PyUnicode_FromString(wcsprintf_buf());
}


/***************************************************************************
 * Member getters/setters (properties)
 */

static PyObject* PyAuxprm_get_rsun_ref(PyAuxprm* self, void* closure) {
  return get_double("rsun_ref", self->x->rsun_ref);
}

static PyObject* PyAuxprm_get_dsun_obs(PyAuxprm* self, void* closure) {
  return get_double("dsun_obs", self->x->dsun_obs);
}

static PyObject* PyAuxprm_get_crln_obs(PyAuxprm* self, void* closure) {
  return get_double("crln_obs", self->x->crln_obs);
}

static PyObject* PyAuxprm_get_hgln_obs(PyAuxprm* self, void* closure) {
  return get_double("hgln_obs", self->x->hgln_obs);
}

static PyObject* PyAuxprm_get_hglt_obs(PyAuxprm* self, void* closure) {
  return get_double("hglt_obs", self->x->hglt_obs);
}

/***************************************************************************
 * PyAuxprm definition structures
 */

static PyGetSetDef PyAuxprm_getset[] = {
  {"rsun_ref", (getter)PyAuxprm_get_rsun_ref, NULL, (char *) NULL},
  {"dsun_obs", (getter)PyAuxprm_get_dsun_obs, NULL, (char *) NULL},
  {"crln_obs", (getter)PyAuxprm_get_crln_obs, NULL, (char *) NULL},
  {"hgln_obs", (getter)PyAuxprm_get_hgln_obs, NULL, (char *) NULL},
  {"hglt_obs", (getter)PyAuxprm_get_hglt_obs, NULL, (char *) NULL},
  {NULL}
};


static PyMethodDef PyAuxprm_methods[] = {
  {"print_contents", (PyCFunction)PyAuxprm_print_contents, METH_NOARGS, NULL},
  {NULL}
};

PyTypeObject PyAuxprmType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "astropy.wcs.Auxprm",         /*tp_name*/
  sizeof(PyAuxprm),             /*tp_basicsize*/
  0,                            /*tp_itemsize*/
  (destructor)PyAuxprm_dealloc, /*tp_dealloc*/
  0,                            /*tp_print*/
  0,                            /*tp_getattr*/
  0,                            /*tp_setattr*/
  0,                            /*tp_compare*/
  0,                            /*tp_repr*/
  0,                            /*tp_as_number*/
  0,                            /*tp_as_sequence*/
  0,                            /*tp_as_mapping*/
  0,                            /*tp_hash */
  0,                            /*tp_call*/
  (reprfunc)PyAuxprm___str__,   /*tp_str*/
  0,                            /*tp_getattro*/
  0,                            /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  0,                            /* tp_doc */
  (traverseproc)PyAuxprm_traverse, /* tp_traverse */
  (inquiry)PyAuxprm_clear,         /* tp_clear */
  0,                            /* tp_richcompare */
  0,                            /* tp_weaklistoffset */
  0,                            /* tp_iter */
  0,                            /* tp_iternext */
  PyAuxprm_methods,             /* tp_methods */
  0,                            /* tp_members */
  PyAuxprm_getset,              /* tp_getset */
  0,                            /* tp_base */
  0,                            /* tp_dict */
  0,                            /* tp_descr_get */
  0,                            /* tp_descr_set */
  0,                            /* tp_dictoffset */
  0,                            /* tp_init */
  0,                            /* tp_alloc */
  0,                            /* tp_new */
};


int
_setup_auxprm_type(PyObject* m) {
  if (PyType_Ready(&PyAuxprmType) < 0) {
    return -1;
  }

  Py_INCREF(&PyAuxprmType);

  PyModule_AddObject(m, "Auxprm", (PyObject *)&PyAuxprmType);

  return 0;
}
