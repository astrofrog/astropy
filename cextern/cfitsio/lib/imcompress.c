# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <ctype.h>
# include <time.h>
# include "fitsio2.h"

#define NULL_VALUE -2147483647 /* value used to represent undefined pixels */
#define ZERO_VALUE -2147483646 /* value used to represent zero-valued pixels */

/* nearest integer function */
# define NINT(x)  ((x >= 0.) ? (int) (x + 0.5) : (int) (x - 0.5))

/* special quantize level value indicates that floating point image pixels */
/* should not be quantized and instead losslessly compressed (with GZIP) */
#define NO_QUANTIZE 9999

/* string array for storing the individual column compression stats */
char results[999][30];

float *fits_rand_value = 0;

int fits_init_randoms(void) {

/* initialize an array of random numbers */

    int ii;
    double a = 16807.0;
    double m = 2147483647.0;
    double temp, seed;

    FFLOCK;

    if (fits_rand_value) {
       FFUNLOCK;
       return(0);  /* array is already initialized */
    }

    /* allocate array for the random number sequence */
    /* THIS MEMORY IS NEVER FREED */
    fits_rand_value = calloc(N_RANDOM, sizeof(float));

    if (!fits_rand_value) {
        FFUNLOCK;
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

    FFUNLOCK;

    /*
    IMPORTANT NOTE: the 10000th seed value must have the value 1043618065 if the
       algorithm has been implemented correctly */

    if ( (int) seed != 1043618065) {
        ffpmsg("fits_init_randoms generated incorrect random number sequence");
	return(1);
    } else {
        return(0);
    }
}
