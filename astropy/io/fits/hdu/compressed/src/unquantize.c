// This file provides fits_rand_value and fits_init_randoms which are
// needed by quantize.c for dithered quantization. The unquantize_*
// functions that were previously here have been replaced by a pure
// Python+NumPy implementation in _quantization.py.

# include <stdlib.h>
# include <stdio.h>

#define N_RANDOM 10000  /* DO NOT CHANGE THIS;  used when quantizing real numbers */
#define MEMORY_ALLOCATION 113  /* Could not allocate memory */

float *fits_rand_value = 0;

int fits_init_randoms(void) {

/* initialize an array of random numbers */

    int ii;
    double a = 16807.0;
    double m = 2147483647.0;
    double temp, seed;

    if (fits_rand_value) {
       return(0);  /* array is already initialized */
    }

    /* allocate array for the random number sequence */
    /* THIS MEMORY IS NEVER FREED */
    fits_rand_value = calloc(N_RANDOM, sizeof(float));

    if (!fits_rand_value) {
	return(MEMORY_ALLOCATION);
    }

    /*  We need a portable algorithm that anyone can use to generate this
        exact same sequence of random number.  The C 'rand' function is not
	suitable because it is not available to Fortran or Java programmers.
	Instead, use a well known simple algorithm published here:
	"Random number generators: good ones are hard to find", Communications of the ACM,
        Volume 31 ,  Issue 10  (October 1988) Pages: 1192 - 1201
    */

    /* initialize the random numbers */
    seed = 1;
    for (ii = 0; ii < N_RANDOM; ii++) {
        temp = a * seed;
	seed = temp -m * ((int) (temp / m) );
	fits_rand_value[ii] = (float) (seed / m);
    }

    /*
    IMPORTANT NOTE: the 10000th seed value must have the value 1043618065 if the
       algorithm has been implemented correctly */

    if ( (int) seed != 1043618065) {
        printf("fits_init_randoms generated incorrect random number sequence");
	return(1);
    } else {
        return(0);
    }
}
