#ifndef BLOCK_CONV1D_GUARD
#define BLOCK_CONV1D_GUARD

#include "arm_math.h"
#include "arm_nnfunctions.h"

// CONV1D params

/* Calculated for audio segments of 30720 samples */

const int32_t pad_left        = 0;
const int32_t pad_right       = 0;

const int32_t BATCH        = 1;

const int32_t INPUT_LEN    = 1024;
const int32_t INPUT_H      = 1;
const int32_t IN_CH        = 1;

const int32_t start_idx    = 79;


const int32_t OUT_CH       = 4;

const int32_t KERNEL_H     = 1;
const int32_t STRIDE_H     = 1;

const int32_t KERNEL_W     = 22;
const int32_t STRIDE_W     = 64;

const int32_t OUTPUT_H     = 1; 
const int32_t OUTPUT_W     = (INPUT_LEN - KERNEL_W) / STRIDE_W + 1;

const int32_t pool_H     	= 1; 
const int32_t pool_W     	= 4; // stride=length

const int32_t POOLED_H   	= 1;
const int32_t POOLED_W   	= OUTPUT_W / pool_W;

// No bias (TODO add later version for bias params)
const cmsis_nn_bias_data bias_data_cmsis = { NULL, false };
const cmsis_nn_dims bias_dims = {0,0,0,0};

// --- CMSIS-NN dimension structs ---
const cmsis_nn_dims input_dims = {
    /*[N, H, W, C_IN]*/
	.n = BATCH,
	.h = INPUT_H,
	.w = INPUT_LEN,
	.c = IN_CH
};
const cmsis_nn_dims filter_dims = {
    /*[C_OUT, HK, WK, C_IN]*/
	.n = OUT_CH,
	.h = KERNEL_H,
	.w = KERNEL_W,
	.c = IN_CH
};

const cmsis_nn_dims output_dims = {
    /*[N, H, W, C_OUT]*/
	.n = BATCH,
	.h = OUTPUT_H,
	.w = OUTPUT_W,
	.c = OUT_CH
};

const cmsis_nn_conv_params conv_params = {
    .input_offset = 0,
	.output_offset = 0,
    .stride  = {STRIDE_W, STRIDE_H},
	.padding = {0,0},
	.dilation = {1, 1},
	.activation = {.min = -32768, .max = 32767}
};

/* AveragePooling1D params */

const cmsis_nn_pool_params pool_params = {
	.stride = {pool_W, pool_H},
	.padding = {0,0},
	.activation = {.min = -32768, .max = 32767}
};

const cmsis_nn_dims pool_filter_dims = {
	/* [H, W] Argument N and C are not used */
	.n = 0,
	.h = pool_H, 
	.w = pool_W, 
	.c = 0
};

const cmsis_nn_dims pool_input_dims = {
	/* [H, W, C_IN] Argument 'N' is not used */
	.n = 0,
	.h = OUTPUT_H,
	.w = OUTPUT_W,
	.c = OUT_CH
};

const cmsis_nn_dims pool_output_dims = {
	/* [H, W, C_OUT=C_IN] Argument 'N' is not used*/
	.n = 0,
	.h = POOLED_H,
	.w = POOLED_W,
	.c = OUT_CH
};


// TODO fix filter_data and quant_params
const cmsis_nn_per_channel_quant_params quant_params = { NULL, NULL };

const int8_t filter_data[1 * 4 * 22] = {
20,18,-17,-48,-20,54,79,0,-103,-83,49,127,49,-83,-103,0,79,54,-20,-48,-17,18,
-16,-19,25,39,-33,-65,34,95,-27,-118,0,127,0,-118,-27,95,34,-65,-33,39,25,-19,
0,28,0,-41,46,27,-87,27,92,-95,-40,127,-40,-95,92,27,-87,27,46,-41,0,28,
0,15,-36,44,-24,-24,76,-97,64,13,-93,127,-93,13,64,-97,76,-24,-24,44,-36,15
}; 

// --- Context buffer ---
const int32_t conv_buf_size = 0; //TODO
uint8_t conv_buffer[0]; //TODO
cmsis_nn_context conv_ctx = {
	.buf = conv_buffer,
	.size = conv_buf_size
};


// --- Context buffer ---
const int32_t pool_buf_size = 0; //TODO
uint8_t pool_buffer[0]; //TODO
cmsis_nn_context pool_ctx = {
	.buf = pool_buffer,
	.size = pool_buf_size
};

#endif