/* trampoline stuff */
#include "tpl_os.h"

/* ----- TFLM header ----- */
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ----- Model TFLM ----- */
#include "model/Band_int8_model.h"


#include "arm_math.h"

#include "fir_coeffs.h"

/* for watchdog*/
#include "em_wdog.h"

/* Envelopes are in SRAM ext, we only share pointers */
extern P2VAR(float, AUTOMATIC, AUTOMATIC) ping_ptr_envelope1;

extern P2VAR(float, AUTOMATIC, AUTOMATIC) pong_ptr_envelope1;

extern VAR(bool, AUTOMATIC) envelop_ping_rdy;

extern VAR(uint8_t, AUTOMATIC) sram_ping_rdy;

extern P2VAR(float32_t, AUTOMATIC, AUTOMATIC) ptr_SRAM_FILTERED_AUDIO_PING_1;

extern P2VAR(float32_t, AUTOMATIC, AUTOMATIC) ptr_SRAM_FILTERED_AUDIO_PONG_1;


extern VAR(float32_t, AUTOMATIC) max_pool_for_low_pass1 [240];
extern P2VAR(uint32_t, AUTOMATIC, AUTOMATIC) indexMax1;

extern VAR(uint8_t, AUTOMATIC) index_buffer_audio_inference;
extern VAR(uint8_t, AUTOMATIC) index_buffer_audio_inference2;

void setupWatchdog(void);
void startWatchdog(void);
void feedWatchdog(void);

#define LED_GPIOPORT                            gpioPortC
#define GREEN_LED                               5
#define RED_LED                                 4

namespace {
	using YellowHammerOpResolver = tflite::MicroMutableOpResolver<12>;

	TfLiteStatus RegisterOps(YellowHammerOpResolver& op_resolver){
		op_resolver.AddExpandDims();
		op_resolver.AddConv2D();
		op_resolver.AddMul();
		op_resolver.AddAdd();
		op_resolver.AddReshape();
		op_resolver.AddMaxPool2D();
		op_resolver.AddMean();
		op_resolver.AddFullyConnected();
		op_resolver.AddLogistic();
		op_resolver.AddStridedSlice();
		op_resolver.AddShape();
		op_resolver.AddPack();
		return kTfLiteOk;
	}

	tflite::MicroInterpreter *interpreter = nullptr;
	TfLiteTensor *input = nullptr;
	TfLiteTensor *output = nullptr;
	const tflite::Model* model = nullptr;
	TfLiteStatus myStatus;
}	//namespace

TfLiteStatus AllocateTensor(){
	TF_LITE_ENSURE_STATUS(interpreter->AllocateTensors());
}

TfLiteStatus ProcessInference(){
	TF_LITE_ENSURE_STATUS(interpreter->Invoke());
}

#define APP_Task_setup_inference_START_SEC_CODE
#include "tpl_memmap.h"
TASK(setup_inference){
	/* Get Model */
	model = ::tflite::GetModel(Band_int8_model_tflite);

	if(model->version() != TFLITE_SCHEMA_VERSION){
		while(1);
	}
	/* Ops */
	YellowHammerOpResolver op_resolver;
	RegisterOps(op_resolver);
	/* ScratchPad Memory for Layers */
	constexpr int kTensorArenaSize = 4500;
	static uint8_t tensor_arena[kTensorArenaSize];
	static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
	interpreter = &static_interpreter;
	myStatus = AllocateTensor();
	if(myStatus != kTfLiteOk){
		while(1);
	}
	/* Input */
	input = interpreter->input(0);
	TFLITE_CHECK_NE(input, nullptr);
	/* Output */
	output = interpreter->output(0);
	TFLITE_CHECK_NE(output, nullptr);
	setupWatchdog();
	startWatchdog();
	ChainTask(inference);
}
#define APP_Task_setup_inference_STOP_SEC_CODE
#include "tpl_memmap.h"

#define APP_Task_inference_START_SEC_CODE
#include "tpl_memmap.h"

VAR(float, AUTOMATIC) input_inference[240];


static float32_t firStateF32_low1 [2*3] = {0};

void setupWatchdog(void){
	WDOG_Init_TypeDef wdogInit = WDOG_INIT_DEFAULT;
	/* Default is 1KHz oscillator */
	/* 8193 clock periods --> */
	wdogInit.perSel = wdogPeriod_8k;
	WDOG_Init(&wdogInit);
}

void startWatchdog(void){
	WDOG_Enable(true);
}

void feedWatchdog(void){
	WDOG_Feed();
}


int control_rate=0;

TASK(inference){

	float *tmp_ptr_envelope1_inference;

	EventMaskType ev1;
	WaitEvent(ev_NORMALIZE);
	GetEvent(inference, &ev1);
	ClearEvent(ev1);
	
	/* Feed watchdog */
	feedWatchdog();

	GPIO_PinModeSet(LED_GPIOPORT, GREEN_LED, gpioModePushPull, true);

	index_buffer_audio_inference2 = index_buffer_audio_inference;

	arm_biquad_cascade_df2T_instance_f32 instFilter_low1;

	arm_biquad_cascade_df2T_init_f32(&instFilter_low1, 3, &firCoefF32_low[0], &firStateF32_low1[0]);
	
	if(sram_ping_rdy){
		sram_ping_rdy = false;

		arm_biquad_cascade_df2T_f32(&instFilter_low1, max_pool_for_low_pass1, ping_ptr_envelope1, 240);

		envelop_ping_rdy = 1;

	}
	else{
		sram_ping_rdy = true;

		arm_biquad_cascade_df2T_f32(&instFilter_low1, max_pool_for_low_pass1, pong_ptr_envelope1, 240);

		envelop_ping_rdy = 0;
	}
	
	float mean_band1 = 0;
	float std_band1 = 0;

	/* If we are here, it means we have 80 data in envelopes */
	/* We normalize for inference input */
	/* First get min/max */
	if(envelop_ping_rdy){
		tmp_ptr_envelope1_inference = ping_ptr_envelope1;
	}
	else{
		tmp_ptr_envelope1_inference = pong_ptr_envelope1;
	}

	for(uint8_t i = 0; i < 240; i++){
		mean_band1 += *tmp_ptr_envelope1_inference; 
		std_band1 += (*tmp_ptr_envelope1_inference)*(*tmp_ptr_envelope1_inference); 

		tmp_ptr_envelope1_inference++;
	}

	mean_band1 = mean_band1/240;

	std_band1 = sqrt(std_band1/240 - mean_band1*mean_band1);

	/* Then normalize */
	if(envelop_ping_rdy){
		tmp_ptr_envelope1_inference = ping_ptr_envelope1;
	}
	else{
		tmp_ptr_envelope1_inference = pong_ptr_envelope1;
	}
	for(uint8_t i = 0; i < 240; i++){
		input_inference[i] = (*tmp_ptr_envelope1_inference++ - mean_band1) / (std_band1);
	}

	// Get data for input
	uint16_t i;

	// float input_scale = input->params.scale;
	const float input_scale = input->params.scale;
    // int input_zero_point = input->params.zero_point;
	const int input_zero_point = input->params.zero_point;
	const float output_scale = output->params.scale;
	const int output_zero_point = output->params.zero_point;


	int8_t quantized_value[240] = {0};
	for(i=0; i<240; i++){
		int value = static_cast<int>(round((input_inference[i] / input_scale) + input_zero_point));
		value = std::min(std::max(value, -128), 127);
		quantized_value[i] = static_cast<int8_t>(value);
		/* Now put quantized data on input of NN */
		input->data.int8[i] = quantized_value[i];
	}

	/* Process NN */
	myStatus = ProcessInference();
	if(myStatus != kTfLiteOk){
		while(1);
	}

	int8_t final_output = output->data.int8[0];
	float output_score = static_cast<float>(static_cast<int>(final_output) - output_zero_point) * output_scale;
	GPIO_PinModeSet(LED_GPIOPORT, GREEN_LED, gpioModePushPull, false);

	if (control_rate%2 ==0){
		ActivateTask(write_audio);
	}
	control_rate++;
	/*if(output_score > 0.3f){
		ActivateTask(write_audio);
	}*/
	
	ChainTask(inference);
}
#define APP_Task_inference_STOP_SEC_CODE
#include "tpl_memmap.h"