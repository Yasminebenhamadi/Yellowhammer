#ifndef FIR_COEFFS_GUARD
#define FIR_COEFFS_GUARD

#include "arm_math.h"

#define PRE_PROECESSING_MAX

const float32_t firCoefF32_low [5*3] = {
	0.00034,
	0.00068, 
	0.00034,
	1.03207,
    -0.275710, 

	1.,  
	2.,  
	1.,
    1.14298, 
	-0.412800, 

	1.,  
	2.,
    1.,  
	1.40438,
	-0.735920
};

const float32_t firCoefF32_band1 [5*2] = {
	0.3766632, 
	-0.7533264,  
	0.3766632 ,  
	0.72541258, 
	-0.31547758,

    1.        ,  
	2.        ,  
	1.        , 
	-1.46663   , 
	-0.60042128
};

#endif