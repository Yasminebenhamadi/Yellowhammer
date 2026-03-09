#ifndef FIR_COEFFS_GUARD
#define FIR_COEFFS_GUARD

#include "arm_math.h"

#define PRE_PROCESSING_MAX

/* samplerate=20480, downsample_factor=128 */


const int8_t postShift = 2;

/* SOS Lowpass filter with order=6 and lowcut=16 */
const q31_t firCoefQ31_low [3*5] = {
    182824,
    365649,
    182824,
    554088064,
    -148019568,
    536870912,
    1073741824,
    536870912,
    613632960,
    -221621168,
    536870912,
    1073741824,
    536870912,
    753973376,
    -395091456
};

/* SOS Bandpass filters with order=2 */

// freq_low=3000, freq_high=9000
const q31_t firCoefQ31_band1 [2*5] = {
    202219520,
    -404439040,
    202219520,
    389452896,
    -169370736,
    536870912,
    1073741824,
    536870912,
    -787390976,
    -322348736
};



#endif