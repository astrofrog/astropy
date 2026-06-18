/*
 Generic WCSLIB distortion (disprm) binding.

 Wraps a WCSLIB `struct disprm` and exposes a vectorised forward transform
 (`disp2x`).  It carries no distortion-specific maths or knowledge: the caller
 supplies the per-axis distortion type and the DPja keyrecords, and WCSLIB
 (disset) translates them into a TPD which evaluates the transform.  This is
 the compiled engine that the pure-Python distortion classes (e.g. Sip) build
 on top of.
*/

#ifndef __DISPRM_WRAP_H__
#define __DISPRM_WRAP_H__

#include "pyutil.h"
#include <dis.h>

typedef struct {
  PyObject_HEAD
  struct disprm x;
} PyDisprm;

extern PyTypeObject* DisprmType;

int _setup_disprm_type(PyObject* m);

#endif
