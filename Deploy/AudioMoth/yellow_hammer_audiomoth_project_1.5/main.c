#include "tpl_os.h"
#include "tpl_os_event.h"

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "em_device.h"
#include "em_chip.h"
#include "em_cmu.h"

#include "em_adc.h"
#include "em_prs.h"
#include "em_timer.h"
#include "em_dma.h"
#include "em_gpio.h"
#include "em_ebi.h"
#include "em_burtc.h"
#include "em_rtc.h"
#include "em_opamp.h"
#include "em_emu.h"

/* Filter coefficients */
#include "fir_coeffs.h"

/* ARM CMSISDSP */
#include "arm_math.h"

#define ROUNDED_DIV(a, b)                         (((a) + ((b)/2)) / (b))

// #define PRE_PROCESSING_RMS
#define PRE_PROCESSING_MAX

typedef struct {
	char id[4];
	uint32_t size;
  } chunk_t;

  typedef struct {
	chunk_t icmt;
	char comment[384];
  } icmt_t;
  
  typedef struct {
	chunk_t iart;
	char artist[32];
  } iart_t;

typedef struct {
	uint16_t format;
	uint16_t numberOfChannels;
	uint32_t samplesPerSecond;
	uint32_t bytesPerSecond;
	uint16_t bytesPerCapture;
	uint16_t bitsPerSample;
  } wavFormat_t;

typedef struct {
	chunk_t riff;
	char format[4];
	chunk_t fmt;
	wavFormat_t wavFormat;
	chunk_t list;
	char info[4];
	icmt_t icmt;
	iart_t iart;
	chunk_t data;
} wavHeader_t;

FIL fileaudio, filetime;
static UINT bw;

typedef struct {
	uint32_t unix_timestamp;
} timekeeper_t;


const int duration_in_second = 60*55;
const uint16_t samplerate = 20480;

uint32_t file_duration = 0; 
char* audiofilename;
char*timefilename;
char a_filename_buffer[15]; char t_filename_buffer[15];

/* Header detail of wav file */
const wavHeader_t wavHeader = {
	.riff = {.id = "RIFF", .size = 2 * samplerate * duration_in_second +
		sizeof(wavHeader_t) - sizeof(chunk_t)},
	.format = "WAVE",
	.fmt = {.id = "fmt ", .size = sizeof(wavFormat_t)},
	.wavFormat = {.format = 1,							/* PCM format */
					.numberOfChannels = 1,
					.samplesPerSecond = samplerate,
					.bytesPerSecond = 2*samplerate,
					.bytesPerCapture = 2,
					.bitsPerSample = 16},
	.list = {.id = "LIST",
				.size = 4 + sizeof(icmt_t) + sizeof(iart_t)},
	.info = "INFO",
	.icmt = {.icmt.id = "ICMT", .icmt.size = 384, .comment = ""},
	.iart = {.iart.id = "IART", .iart.size = 32, .artist = ""},
	.data = {.id = "data", 
			 .size = 2 * samplerate * duration_in_second}
};

#define APP_Task_write_audio_START_SEC_VAR_NOINIT_UNSPECIFIED
#include "tpl_memmap.h"
VAR(timekeeper_t, AUTOMATIC) timekeeper;
#define APP_Task_write_audio_STOP_SEC_VAR_NOINIT_UNSPECIFIED
#include "tpl_memmap.h"

#define APP_COMMON_START_SEC_CODE
#include "tpl_memmap.h"

P2VAR(float, AUTOMATIC, AUTOMATIC) ping_ptr_envelope1;
P2VAR(float, AUTOMATIC, AUTOMATIC) ping_ptr_envelope2;
P2VAR(float, AUTOMATIC, AUTOMATIC) ping_ptr_envelope3;

P2VAR(float, AUTOMATIC, AUTOMATIC) pong_ptr_envelope1;
P2VAR(float, AUTOMATIC, AUTOMATIC) pong_ptr_envelope2;
P2VAR(float, AUTOMATIC, AUTOMATIC) pong_ptr_envelope3;

// VAR(FATFS, AUTOMATIC) fatfs;
static FATFS fatfs;

FUNC(int, OS_APPL_CODE) main(void){
	CHIP_Init();
	// Enable Clock to GPIO
	CMU_ClockEnable(cmuClock_GPIO, true);
	/* Enable high frequency HFXO clock */
    CMU_OscillatorEnable(cmuOsc_HFXO, true, true);
    CMU_ClockDivSet(cmuClock_HF, cmuClkDiv_1);
    CMU_ClockSelectSet(cmuClock_HF, cmuSelect_HFXO);
    CMU_OscillatorEnable(cmuOsc_HFRCO, false, false);
	/* Enable clock to Low energy module */
	CMU_ClockEnable(cmuClock_CORELE, true);
	/* Setup NVIC for DMA */
	NVIC_ClearPendingIRQ(DMA_IRQn);
  	NVIC_EnableIRQ(DMA_IRQn);
	/* Setup RTCC */
	CMU_OscillatorEnable(cmuOsc_LFXO, true, true);
    CMU_ClockSelectSet(cmuClock_LFA, cmuSelect_LFXO);
	CMU_ClockEnable(cmuClock_RTC, true);
	RTC_Init_TypeDef rtcInit = RTC_INIT_DEFAULT;
	rtcInit.enable = false;
	// rtcInit.debugRun = true;
	RTC_Init(&rtcInit);
	RTC_IntClear(RTC_IF_COMP0);
	RTC_IntClear(RTC_IF_COMP1);
	RTC_CompareSet(0, 32768);
	/* Set Comp1 to Max value to avoid interrupt with comp1 flag on */
	RTC_CompareSet(1, 0xFFFFFFFF);
	NVIC_ClearPendingIRQ(RTC_IRQn);
	NVIC_EnableIRQ(RTC_IRQn);
	RTC_IntEnable(RTC_IF_COMP0);
    RTC_Enable(true);
	/* GPIO A */
	GPIO_PinModeSet(gpioPortA, 0, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 1, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 2, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 3, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 4, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 5, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 6, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 7, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 8, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 9, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 10, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 11, gpioModePushPull, 0);
	GPIO_PinModeSet(gpioPortA, 12, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 13, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 14, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortA, 15, gpioModeDisabled, 0);
	/* GPIO B */
	GPIO_PinModeSet(gpioPortB, 0, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 1, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 2, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 3, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 4, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 5, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 6, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 9, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 10, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 11, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortB, 12, gpioModeDisabled, 0);
	/* GPIO C */
	GPIO_PinModeSet(gpioPortC, 0, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortC, 1, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortC, 3, gpioModePushPull, 0);
	GPIO_PinModeSet(gpioPortC, 7, gpioModeDisabled, 0);	
	GPIO_PinModeSet(gpioPortC, 8, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortC, 9, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortC, 10, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortC, 11, gpioModeDisabled, 0);
	/* GPIO D */
    GPIO_PinModeSet(gpioPortD, 0, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortD, 1, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortD, 2, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortD, 3, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortD, 4, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortD, 5, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortD, 9, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortD, 10, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortD, 11, gpioModePushPull, 1);
	GPIO_PinModeSet(gpioPortD, 12, gpioModePushPull, 1);
	/* GPIO E */
	GPIO_PinModeSet(gpioPortE, 0, gpioModePushPull, 1);
	GPIO_PinModeSet(gpioPortE, 1, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 2, gpioModePushPull, 1);
	GPIO_PinModeSet(gpioPortE, 3, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 4, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 5, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 6, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 7, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 8, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 9, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 10, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 11, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 12, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 13, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 14, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortE, 15, gpioModeDisabled, 0);
	/* GPIO F */
	GPIO_PinModeSet(gpioPortF, 3, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortF, 4, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortF, 5, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortF, 6, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortF, 7, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortF, 8, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortF, 9, gpioModeDisabled, 0);
	GPIO_PinModeSet(gpioPortF, 12, gpioModeDisabled, 0);
	/* Start Trampoline */
	StartOS(OSDEFAULTAPPMODE);
	return 0;
}

#define LED_GPIOPORT                            gpioPortC
#define GREEN_LED                               5
#define RED_LED                                 4

#define APP_COMMON_STOP_SEC_CODE
#include "tpl_memmap.h"

#define APP_Task_start_audio_START_SEC_CODE
#include "tpl_memmap.h"


TASK(start_audio){

	GPIO_PinModeSet(gpioPortA, 14, gpioModeInput, 0);
    GPIO_IntConfig(gpioPortA, 14, true, true, true);
	bool microphone_ext = GPIO_PinInGet(gpioPortA, 14) == 0;
	// Enable Microphone on PE0 --> Disable GPIO 
	GPIO_PinOutClear(gpioPortE, 0);
	// Enable VREF on PA11 to power Microphone
	GPIO_PinOutSet(gpioPortA, 11);
	/* Setup OpAmp */
	CMU_ClockEnable(cmuClock_DAC0, true);
	/* Define the configuration for OPA1 and OPA2 */
	OPAMP_Init_TypeDef opa1Init = OPA_INIT_INVERTING;
	OPAMP_Init_TypeDef opa2Init = OPA_INIT_INVERTING_OPA2;
	opa2Init.outPen = DAC_OPA2MUX_OUTPEN_OUT1;
	/* Set the gain */
    OPAMP_ResSel_TypeDef opamp1NormalGainRange[] = {opaResSelR2eq4_33R1, opaResSelR2eq7R1, opaResSelR2eq15R1, opaResSelR2eq15R1, opaResSelR2eq15R1};
    OPAMP_ResSel_TypeDef opamp2NormalGainRange[] = {opaResSelR2eqR1, opaResSelR2eqR1, opaResSelR2eqR1, opaResSelR1eq1_67R1, opaResSelR2eq2R1};
    OPAMP_ResSel_TypeDef opamp1LowGainRange[] = {opaResSelR2eq0_33R1, opaResSelR2eq0_33R1, opaResSelR2eqR1, opaResSelR2eqR1, opaResSelR2eqR1};
    OPAMP_ResSel_TypeDef opamp2LowGainRange[] = {opaResSelR2eqR1, opaResSelR1eq1_67R1, opaResSelR2eqR1, opaResSelR1eq1_67R1, opaResSelR2eq2R1};
    OPAMP_ResSel_TypeDef *opamp1Gain = 4 == 0 ? opamp1LowGainRange : opamp1NormalGainRange;
    OPAMP_ResSel_TypeDef *opamp2Gain = 4 == 0 ? opamp2LowGainRange : opamp2NormalGainRange;
    uint32_t index = MAX(0, MIN(1, 4));
    opa1Init.resSel = opamp1Gain[index];
    opa2Init.resSel = opamp2Gain[index];
	/* Enable OPA1 and OPA2 */
    OPAMP_Enable(DAC0, OPA1, &opa1Init);
    OPAMP_Enable(DAC0, OPA2, &opa2Init);
    /* Disable the clock */
    CMU_ClockEnable(cmuClock_DAC0, false);
	/* Now setup ADC */
	// Start the clock
	CMU_ClockEnable(cmuClock_ADC0, true);
	ADC_Reset(ADC0);
	// Setup ADC structure
	ADC_Init_TypeDef adcInit = ADC_INIT_DEFAULT;
	adcInit.prescale = (4 - 1);
	adcInit.warmUpMode = adcWarmupKeepADCWarm;
	adcInit.timebase = ADC_TimebaseCalc(0);
	adcInit.lpfMode = adcLPFilterRC;
	// Initialize ADC
	ADC_Init(ADC0, &adcInit);
	/* SCAN mode voltage reference must match the reference selected for SINGLE mode conversions */
	// Reset and set 2.5V Ref
	ADC0->SCANCTRL = ADC_SCANCTRL_REF_2V5;
	/* Configure ADC single conversion structure */
	ADC_InitSingle_TypeDef adcSingleInit = ADC_INITSINGLE_DEFAULT;
	adcSingleInit.prsSel = adcPRSSELCh0;
    adcSingleInit.reference = adcRef2V5;
	adcSingleInit.resolution = adcRes12Bit;
	adcSingleInit.input = adcSingleInpCh0Ch1;
	adcSingleInit.prsEnable = true;
    adcSingleInit.diff = true;
    adcSingleInit.rep = false;
	adcSingleInit.acqTime = adcAcqTime8;
	ADC_InitSingle(ADC0, &adcSingleInit);
	/* Enable ADC interrupt vector */
	ADC_IntClear(ADC0, ADC_IEN_SINGLE);

	ChainTask(start_dma);
}
#define APP_Task_start_audio_STOP_SEC_CODE
#include "tpl_memmap.h"

#define APP_Task_start_dma_START_SEC_CODE
#include "tpl_memmap.h"

static int16_t primaryBuffer[1024];
static int16_t secondaryBuffer[1024];
const uint16_t numberOfSamplesPerTransfer = 1024;
static bool isPrimaryDMABuffer = true;

VAR(bool,AUTOMATIC) envelop_ping_rdy = true;

static void dma_callback(unsigned int channel, bool isPrimaryBuffer, void *user){
	return;
}

TASK(start_dma){
	/* ADC and Timer is started, now link to DMA */
	CMU_ClockEnable(cmuClock_DMA, true);
	/* Initialise the DMA structure */
    DMA_Init_TypeDef dmaInit;
	dmaInit.hprot = 0;
    dmaInit.controlBlock = dmaControlBlock;
	DMA_Init(&dmaInit);
	/* Configure Callback when DMA is done somehow it is needed while not used */
	DMA_CB_TypeDef cb;
	cb.cbFunc = dma_callback;
	cb.userPtr = NULL;
	/* Setup channel */
	DMA_CfgChannel_TypeDef chnlCfg;
	chnlCfg.highPri = false;
    chnlCfg.enableInt = true;
    chnlCfg.select = DMAREQ_ADC0_SINGLE;
	chnlCfg.cb = &cb;
    DMA_CfgChannel(0, &chnlCfg);
	/* Setting up channel descriptor */
    DMA_CfgDescr_TypeDef descrCfg;
    descrCfg.dstInc = dmaDataInc2;
    descrCfg.srcInc = dmaDataIncNone;
    descrCfg.size = dmaDataSize2;
    descrCfg.arbRate = dmaArbitrate1;
    descrCfg.hprot = 0;
	/* Set up both the primary and the secondary transfers */
    DMA_CfgDescr(0, true, &descrCfg);
    DMA_CfgDescr(0, false, &descrCfg);
	/* Set up the first transfer */
	isPrimaryDMABuffer = true;
    DMA_ActivatePingPong(0,
        false,
        (void*)primaryBuffer,
        (void*)&(ADC0->SINGLEDATA),
        numberOfSamplesPerTransfer - 1,
        (void*)secondaryBuffer,
        (void*)&(ADC0->SINGLEDATA),
        numberOfSamplesPerTransfer - 1);
	/* Enable SRAM EXTERN */
	// Clear PD11 --> CPU_FET_SRAM_EN_N
	GPIO->P[gpioPortE].DOUTCLR = 1 << 11;
	/* Enable the external bus interface */
	/* Enable clocks */
    CMU_ClockEnable(cmuClock_EBI, true);
	/* Enable SRAM EBI D0..07 data pins (PortE 8 -- 15)*/
    GPIO_PinModeSet(gpioPortE, 8, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 9, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 10, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 11, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 12, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 13, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 14, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 15, gpioModePushPull, 0);
    /* Enable SRAM EBI A0..15 address pins (PortA 15, 0 -- 6, PortE 1, PortC 9 - 10, PortE 4 -- 7, PortC 8, PortB 0 - 1)*/
    GPIO_PinModeSet(gpioPortA, 15, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortA, 0, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortA, 1, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortA, 2, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortA, 3, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortA, 4, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortA, 5, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortA, 6, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 1, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortC, 9, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortC, 10, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 4, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 5, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 6, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortE, 7, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortC, 8, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortB, 0, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortB, 1, gpioModePushPull, 0);
    /* Enable SRAM EBI CS0-CS1 (PortD 9 - 10)*/
    GPIO_PinModeSet(gpioPortD, 9, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortD, 10, gpioModePushPull, 0);
    /* Enable SRAM EBI WEN/OEN (PortF 8 - 9)*/
    GPIO_PinModeSet(gpioPortF, 8, gpioModePushPull, 0);
    GPIO_PinModeSet(gpioPortF, 9, gpioModePushPull, 0);
	/* Configure EBI controller, changing default values */
	EBI_Init_TypeDef ebiInit = EBI_INIT_DEFAULT;
    ebiInit.mode = ebiModeD8A8;
    ebiInit.banks = EBI_BANK0;
    ebiInit.csLines = EBI_CS0 | EBI_CS1;
    ebiInit.readHalfRE = true;
    ebiInit.aLow = ebiALowA8;
    ebiInit.aHigh = ebiAHighA18;
	/* Address Setup and hold time */
    ebiInit.addrHoldCycles  = 0;
    ebiInit.addrSetupCycles = 0;
	/* Read cycle times */
    ebiInit.readStrobeCycles = 3;
    ebiInit.readHoldCycles   = 1;
    ebiInit.readSetupCycles  = 2;
	/* Write cycle times */
    ebiInit.writeStrobeCycles = 6;
    ebiInit.writeHoldCycles   = 0;
    ebiInit.writeSetupCycles  = 0;
	ebiInit.location = ebiLocation1;
    /* Configure EBI */
    EBI_Init(&ebiInit);

	/* Setup PRS (Peripheral Reflex System) between Timer and ADC */
	CMU_ClockEnable(cmuClock_PRS, true);
	CMU_ClockEnable(cmuClock_TIMER2, true);
	/* Connect PRS channel 0 to TIMER overflow */
	PRS_SourceSignalSet(0, PRS_CH_CTRL_SOURCESEL_TIMER2, PRS_CH_CTRL_SIGSEL_TIMER2OF, prsEdgeOff);
	/* Enable TIMER with default settings */
	TIMER_Init_TypeDef timerInit = TIMER_INIT_DEFAULT;
	timerInit.enable = false;
	TIMER_Init(TIMER2, &timerInit);
	/* Configure TIMER to trigger on sampling rate */
	TIMER_TopSet(TIMER2,  CMU_ClockFreqGet(cmuClock_TIMER2) / 20480 - 1);
	/* Enable Timer on ADC */
	TIMER_Enable(TIMER2, true);
	/* Start ADC sample */
	ADC_Start(ADC0, adcStartSingle);
	TerminateTask();
}
#define APP_Task_start_dma_STOP_SEC_CODE
#include "tpl_memmap.h"


#define APP_Task_copyDMAtoSRAM_START_SEC_CODE
#include "tpl_memmap.h"

VAR(uint8_t,AUTOMATIC) count_env = 0;

/* EBI region 0 */
/* SRAM EXT is 256 Kb, from 0x80000000 to 0x8003E7FF */

// SRAM_EXT_START_ADDR is used for 1024 float32 --> size is 0x1000
#define SRAM_EXT_START_ADDR 	0x80000000

/* We have 2 buffer of 1024 float32 in SRAM EXT --> size is 0x3000 */

#define SRAM_EXT_START_FILTERED_PING_BAND1_ADDR 	0x80001000

#define SRAM_EXT_START_FILTERED_PONG_BAND1_ADDR 	0x80004000


/* SRAM_EXT_ENVELOPE is used for 80 float32 --> size is 0x3C0 each*/
#define SRAM_EXT_PING_ENVELOPE1						0x80007000

#define SRAM_EXT_PONG_ENVELOPE1						0x800073C0

/* SRAM_EXT_ADDR_SOUND_1 is used for 1024 int16 --> size is 0x800 */

#define SRAM_EXT_ADDR_SOUND_1    0x80007780
#define SRAM_EXT_ADDR_SOUND_2    0x80007F80
#define SRAM_EXT_ADDR_SOUND_3    0x80008780
#define SRAM_EXT_ADDR_SOUND_4    0x80008F80
#define SRAM_EXT_ADDR_SOUND_5    0x80009780
#define SRAM_EXT_ADDR_SOUND_6    0x80009F80
#define SRAM_EXT_ADDR_SOUND_7    0x8000A780
#define SRAM_EXT_ADDR_SOUND_8    0x8000AF80
#define SRAM_EXT_ADDR_SOUND_9    0x8000B780
#define SRAM_EXT_ADDR_SOUND_10   0x8000BF80

#define SRAM_EXT_ADDR_SOUND_11    0x8000C780
#define SRAM_EXT_ADDR_SOUND_12    0x8000CF80
#define SRAM_EXT_ADDR_SOUND_13    0x8000D780
#define SRAM_EXT_ADDR_SOUND_14    0x8000DF80
#define SRAM_EXT_ADDR_SOUND_15    0x8000E780
#define SRAM_EXT_ADDR_SOUND_16    0x8000EF80
#define SRAM_EXT_ADDR_SOUND_17    0x8000F780
#define SRAM_EXT_ADDR_SOUND_18    0x8000FF80
#define SRAM_EXT_ADDR_SOUND_19    0x80010780
#define SRAM_EXT_ADDR_SOUND_20    0x80010F80

#define SRAM_EXT_ADDR_SOUND_21    0x80011780
#define SRAM_EXT_ADDR_SOUND_22    0x80011F80
#define SRAM_EXT_ADDR_SOUND_23    0x80012780
#define SRAM_EXT_ADDR_SOUND_24    0x80012F80
#define SRAM_EXT_ADDR_SOUND_25    0x80013780
#define SRAM_EXT_ADDR_SOUND_26    0x80013F80
#define SRAM_EXT_ADDR_SOUND_27    0x80014780
#define SRAM_EXT_ADDR_SOUND_28    0x80014F80
#define SRAM_EXT_ADDR_SOUND_29    0x80015780
#define SRAM_EXT_ADDR_SOUND_30    0x80015F80

#define SRAM_EXT_ADDR_SOUND_31    0x80016780
#define SRAM_EXT_ADDR_SOUND_32    0x80016F80
#define SRAM_EXT_ADDR_SOUND_33    0x80017780
#define SRAM_EXT_ADDR_SOUND_34    0x80017F80
#define SRAM_EXT_ADDR_SOUND_35    0x80018780
#define SRAM_EXT_ADDR_SOUND_36    0x80018F80
#define SRAM_EXT_ADDR_SOUND_37    0x80019780
#define SRAM_EXT_ADDR_SOUND_38    0x80019F80
#define SRAM_EXT_ADDR_SOUND_39    0x8001A780
#define SRAM_EXT_ADDR_SOUND_40    0x8001AF80

#define SRAM_EXT_ADDR_SOUND_41    0x8001B780
#define SRAM_EXT_ADDR_SOUND_42    0x8001BF80
#define SRAM_EXT_ADDR_SOUND_43    0x8001C780
#define SRAM_EXT_ADDR_SOUND_44    0x8001CF80
#define SRAM_EXT_ADDR_SOUND_45    0x8001D780
#define SRAM_EXT_ADDR_SOUND_46    0x8001DF80
#define SRAM_EXT_ADDR_SOUND_47    0x8001E780
#define SRAM_EXT_ADDR_SOUND_48    0x8001EF80
#define SRAM_EXT_ADDR_SOUND_49    0x8001F780
#define SRAM_EXT_ADDR_SOUND_50    0x8001FF80

#define SRAM_EXT_ADDR_SOUND_51    0x80020780
#define SRAM_EXT_ADDR_SOUND_52    0x80020F80
#define SRAM_EXT_ADDR_SOUND_53    0x80021780
#define SRAM_EXT_ADDR_SOUND_54    0x80021F80
#define SRAM_EXT_ADDR_SOUND_55    0x80022780
#define SRAM_EXT_ADDR_SOUND_56    0x80022F80
#define SRAM_EXT_ADDR_SOUND_57    0x80023780
#define SRAM_EXT_ADDR_SOUND_58    0x80023F80
#define SRAM_EXT_ADDR_SOUND_59    0x80024780
#define SRAM_EXT_ADDR_SOUND_60    0x80024F80

#define SRAM_EXT_ADDR_SOUND_61    0x80025780
#define SRAM_EXT_ADDR_SOUND_62    0x80025F80
#define SRAM_EXT_ADDR_SOUND_63    0x80026780
#define SRAM_EXT_ADDR_SOUND_64    0x80026F80
#define SRAM_EXT_ADDR_SOUND_65    0x80027780
#define SRAM_EXT_ADDR_SOUND_66    0x80027F80
#define SRAM_EXT_ADDR_SOUND_67    0x80028780
#define SRAM_EXT_ADDR_SOUND_68    0x80028F80
#define SRAM_EXT_ADDR_SOUND_69    0x80029780
#define SRAM_EXT_ADDR_SOUND_70    0x80029F80

#define SRAM_EXT_ADDR_SOUND_71    0x8002A780
#define SRAM_EXT_ADDR_SOUND_72    0x8002AF80
#define SRAM_EXT_ADDR_SOUND_73    0x8002B780
#define SRAM_EXT_ADDR_SOUND_74    0x8002BF80
#define SRAM_EXT_ADDR_SOUND_75    0x8002C780
#define SRAM_EXT_ADDR_SOUND_76    0x8002CF80
#define SRAM_EXT_ADDR_SOUND_77    0x8002D780
#define SRAM_EXT_ADDR_SOUND_78    0x8002DF80
#define SRAM_EXT_ADDR_SOUND_79    0x8002E780
#define SRAM_EXT_ADDR_SOUND_80    0x8002EF80

#define SRAM_EXT_ADDR_SOUND_81    0x8002F780
#define SRAM_EXT_ADDR_SOUND_82    0x8002FF80
#define SRAM_EXT_ADDR_SOUND_83    0x80030780
#define SRAM_EXT_ADDR_SOUND_84    0x80030F80
#define SRAM_EXT_ADDR_SOUND_85    0x80031780
#define SRAM_EXT_ADDR_SOUND_86    0x80031F80
#define SRAM_EXT_ADDR_SOUND_87    0x80032780
#define SRAM_EXT_ADDR_SOUND_88    0x80032F80
#define SRAM_EXT_ADDR_SOUND_89    0x80033780
#define SRAM_EXT_ADDR_SOUND_90    0x80033F80

#define SRAM_EXT_ADDR_SOUND_91    0x80034780
#define SRAM_EXT_ADDR_SOUND_92    0x80034F80
#define SRAM_EXT_ADDR_SOUND_93    0x80035780
#define SRAM_EXT_ADDR_SOUND_94    0x80035F80
#define SRAM_EXT_ADDR_SOUND_95    0x80036780
#define SRAM_EXT_ADDR_SOUND_96    0x80036F80
#define SRAM_EXT_ADDR_SOUND_97    0x80037780
#define SRAM_EXT_ADDR_SOUND_98    0x80037F80
#define SRAM_EXT_ADDR_SOUND_99    0x80038780
#define SRAM_EXT_ADDR_SOUND_100   0x80038F80
/*5s*/

float32_t *buffer_float_sram;
float32_t *prev_ping_buffer_filtered_band1;
float32_t *prev_pong_buffer_filtered_band1;

float32_t *prev_ping_buffer_filtered_band2;
float32_t *prev_pong_buffer_filtered_band2;

float32_t *prev_ping_buffer_filtered_band3;
float32_t *prev_pong_buffer_filtered_band3;

/* Number of 2nd order stage in filter, order is 2*numStagesIIR */
const uint32_t numStagesIIR = 2;

static float32_t firStateF32_band1 [2*2] = {0};

P2VAR(int16_t, AUTOMATIC, OS_APPL_DATA) buffer_ptr_sram[100] = {
	(int16_t *) SRAM_EXT_ADDR_SOUND_1,
	(int16_t *) SRAM_EXT_ADDR_SOUND_2,
	(int16_t *) SRAM_EXT_ADDR_SOUND_3,
	(int16_t *) SRAM_EXT_ADDR_SOUND_4,
	(int16_t *) SRAM_EXT_ADDR_SOUND_5,
	(int16_t *) SRAM_EXT_ADDR_SOUND_6,
	(int16_t *) SRAM_EXT_ADDR_SOUND_7,
	(int16_t *) SRAM_EXT_ADDR_SOUND_8,
	(int16_t *) SRAM_EXT_ADDR_SOUND_9,
	(int16_t *) SRAM_EXT_ADDR_SOUND_10,
	(int16_t *) SRAM_EXT_ADDR_SOUND_11,
	(int16_t *) SRAM_EXT_ADDR_SOUND_12,
	(int16_t *) SRAM_EXT_ADDR_SOUND_13,
	(int16_t *) SRAM_EXT_ADDR_SOUND_14,
	(int16_t *) SRAM_EXT_ADDR_SOUND_15,
	(int16_t *) SRAM_EXT_ADDR_SOUND_16,
	(int16_t *) SRAM_EXT_ADDR_SOUND_17,
	(int16_t *) SRAM_EXT_ADDR_SOUND_18,
	(int16_t *) SRAM_EXT_ADDR_SOUND_19,
	(int16_t *) SRAM_EXT_ADDR_SOUND_20,
	(int16_t *) SRAM_EXT_ADDR_SOUND_21,
	(int16_t *) SRAM_EXT_ADDR_SOUND_22,
	(int16_t *) SRAM_EXT_ADDR_SOUND_23,
	(int16_t *) SRAM_EXT_ADDR_SOUND_24,
	(int16_t *) SRAM_EXT_ADDR_SOUND_25,
	(int16_t *) SRAM_EXT_ADDR_SOUND_26,
	(int16_t *) SRAM_EXT_ADDR_SOUND_27,
	(int16_t *) SRAM_EXT_ADDR_SOUND_28,
	(int16_t *) SRAM_EXT_ADDR_SOUND_29,
	(int16_t *) SRAM_EXT_ADDR_SOUND_30,
	(int16_t *) SRAM_EXT_ADDR_SOUND_31,
	(int16_t *) SRAM_EXT_ADDR_SOUND_32,
	(int16_t *) SRAM_EXT_ADDR_SOUND_33,
	(int16_t *) SRAM_EXT_ADDR_SOUND_34,
	(int16_t *) SRAM_EXT_ADDR_SOUND_35,
	(int16_t *) SRAM_EXT_ADDR_SOUND_36,
	(int16_t *) SRAM_EXT_ADDR_SOUND_37,
	(int16_t *) SRAM_EXT_ADDR_SOUND_38,
	(int16_t *) SRAM_EXT_ADDR_SOUND_39,
	(int16_t *) SRAM_EXT_ADDR_SOUND_40,
	(int16_t *) SRAM_EXT_ADDR_SOUND_41,
	(int16_t *) SRAM_EXT_ADDR_SOUND_42,
	(int16_t *) SRAM_EXT_ADDR_SOUND_43,
	(int16_t *) SRAM_EXT_ADDR_SOUND_44,
	(int16_t *) SRAM_EXT_ADDR_SOUND_45,
	(int16_t *) SRAM_EXT_ADDR_SOUND_46,
	(int16_t *) SRAM_EXT_ADDR_SOUND_47,
	(int16_t *) SRAM_EXT_ADDR_SOUND_48,
	(int16_t *) SRAM_EXT_ADDR_SOUND_49,
	(int16_t *) SRAM_EXT_ADDR_SOUND_50,
	(int16_t *) SRAM_EXT_ADDR_SOUND_51,
	(int16_t *) SRAM_EXT_ADDR_SOUND_52,
	(int16_t *) SRAM_EXT_ADDR_SOUND_53,
	(int16_t *) SRAM_EXT_ADDR_SOUND_54,
	(int16_t *) SRAM_EXT_ADDR_SOUND_55,
	(int16_t *) SRAM_EXT_ADDR_SOUND_56,
	(int16_t *) SRAM_EXT_ADDR_SOUND_57,
	(int16_t *) SRAM_EXT_ADDR_SOUND_58,
	(int16_t *) SRAM_EXT_ADDR_SOUND_59,
	(int16_t *) SRAM_EXT_ADDR_SOUND_60,
	(int16_t *) SRAM_EXT_ADDR_SOUND_61,
	(int16_t *) SRAM_EXT_ADDR_SOUND_62,
	(int16_t *) SRAM_EXT_ADDR_SOUND_63,
	(int16_t *) SRAM_EXT_ADDR_SOUND_64,
	(int16_t *) SRAM_EXT_ADDR_SOUND_65,
	(int16_t *) SRAM_EXT_ADDR_SOUND_66,
	(int16_t *) SRAM_EXT_ADDR_SOUND_67,
	(int16_t *) SRAM_EXT_ADDR_SOUND_68,
	(int16_t *) SRAM_EXT_ADDR_SOUND_69,
	(int16_t *) SRAM_EXT_ADDR_SOUND_70,
	(int16_t *) SRAM_EXT_ADDR_SOUND_71,
	(int16_t *) SRAM_EXT_ADDR_SOUND_72,
	(int16_t *) SRAM_EXT_ADDR_SOUND_73,
	(int16_t *) SRAM_EXT_ADDR_SOUND_74,
	(int16_t *) SRAM_EXT_ADDR_SOUND_75,
	(int16_t *) SRAM_EXT_ADDR_SOUND_76,
	(int16_t *) SRAM_EXT_ADDR_SOUND_77,
	(int16_t *) SRAM_EXT_ADDR_SOUND_78,
	(int16_t *) SRAM_EXT_ADDR_SOUND_79,
	(int16_t *) SRAM_EXT_ADDR_SOUND_80,
	(int16_t *) SRAM_EXT_ADDR_SOUND_81,
	(int16_t *) SRAM_EXT_ADDR_SOUND_82,
	(int16_t *) SRAM_EXT_ADDR_SOUND_83,
	(int16_t *) SRAM_EXT_ADDR_SOUND_84,
	(int16_t *) SRAM_EXT_ADDR_SOUND_85,
	(int16_t *) SRAM_EXT_ADDR_SOUND_86,
	(int16_t *) SRAM_EXT_ADDR_SOUND_87,
	(int16_t *) SRAM_EXT_ADDR_SOUND_88,
	(int16_t *) SRAM_EXT_ADDR_SOUND_89,
	(int16_t *) SRAM_EXT_ADDR_SOUND_90,
	(int16_t *) SRAM_EXT_ADDR_SOUND_91,
	(int16_t *) SRAM_EXT_ADDR_SOUND_92,
	(int16_t *) SRAM_EXT_ADDR_SOUND_93,
	(int16_t *) SRAM_EXT_ADDR_SOUND_94,
	(int16_t *) SRAM_EXT_ADDR_SOUND_95,
	(int16_t *) SRAM_EXT_ADDR_SOUND_96,
	(int16_t *) SRAM_EXT_ADDR_SOUND_97,
	(int16_t *) SRAM_EXT_ADDR_SOUND_98,
	(int16_t *) SRAM_EXT_ADDR_SOUND_99,
	(int16_t *) SRAM_EXT_ADDR_SOUND_100
};

VAR(uint8_t, AUTOMATIC) count_ptr_buffer = 0;

VAR(uint8_t, AUTOMATIC) is_writting_audio = 0;

VAR(uint8_t, AUTOMATIC) sram_ping_rdy = 0;

VAR(float32_t, AUTOMATIC) max_pool_for_low_pass1 [240] = {0};
P2VAR(uint32_t, AUTOMATIC, AUTOMATIC) indexMax1;

VAR(uint8_t, AUTOMATIC) index_buffer_audio_to_write = 0;

VAR(uint8_t, AUTOMATIC) index_buffer_audio_inference = 0;
VAR(uint8_t, AUTOMATIC) index_buffer_audio_inference2 = 0;

VAR(float32_t, AUTOMATIC) audio_float [1024] = {0};
VAR(float32_t, AUTOMATIC) audio_filtered [1024] = {0};


TASK(copyDMAtoSRAM){
	/* Wait for event to transfer audio to sram */
	EventMaskType ev;
	WaitEvent(ev_DMAtoSRAM);
	GetEvent(copyDMAtoSRAM, &ev);
	ClearEvent(ev);


	int16_t *buffer_dma;
	if(isPrimaryDMABuffer){
		buffer_dma = secondaryBuffer;
	}
	else{
		buffer_dma = primaryBuffer;
	}

	buffer_float_sram = (float32_t *) SRAM_EXT_START_ADDR;
	int16_t *tmp_data_buffer_ptr_sram = buffer_ptr_sram[count_ptr_buffer];
	for(uint16_t i = 0; i < 1024; i++){
		*tmp_data_buffer_ptr_sram++ = *buffer_dma++;
	}
	
	if(is_writting_audio){
		count_env = 0;
		int16_t *tmp_ptr_test = buffer_ptr_sram[count_ptr_buffer];
		count_ptr_buffer++;
		if(count_ptr_buffer == 100) count_ptr_buffer = 0;
		SetEvent(write_audio, ev_write_audio);
		ChainTask(copyDMAtoSRAM);
	}

	arm_biquad_cascade_df2T_instance_f32 instFilter1;

	arm_biquad_cascade_df2T_init_f32(&instFilter1, numStagesIIR, &firCoefF32_band1[0], &firStateF32_band1[0]);

	prev_ping_buffer_filtered_band1 = (float32_t *) SRAM_EXT_START_FILTERED_PING_BAND1_ADDR;
	prev_pong_buffer_filtered_band1 = (float32_t *) SRAM_EXT_START_FILTERED_PONG_BAND1_ADDR;

	ping_ptr_envelope1 = (float32_t *) SRAM_EXT_PING_ENVELOPE1;
	pong_ptr_envelope1 = (float32_t *) SRAM_EXT_PONG_ENVELOPE1;

	float32_t *maxpool_1;

	arm_q15_to_float(buffer_ptr_sram[count_ptr_buffer], audio_float, 1024);

	arm_biquad_cascade_df2T_f32(&instFilter1, audio_float, audio_filtered, 1024);
		
	for(uint8_t i = 0; i < 8; i++){
		arm_absmax_f32(audio_filtered+(128*i), 128, max_pool_for_low_pass1+count_env, indexMax1);
		count_env++;
	}

	if(count_env >= 240){
		index_buffer_audio_inference = count_ptr_buffer;
		count_env = 0;
		SetEvent(inference, ev_NORMALIZE);
	}
	
	count_ptr_buffer++;
	if(count_ptr_buffer == 100) count_ptr_buffer = 0;
	ChainTask(copyDMAtoSRAM);
}
#define APP_Task_copyDMAtoSRAM_STOP_SEC_CODE
#include "tpl_memmap.h"

#define APP_Task_start_sdcard_START_SEC_CODE
#include "tpl_memmap.h"

static void setTime(uint32_t time, uint32_t milliseconds) {
	/* 1024 tick per seconds */
    uint32_t ticks = ROUNDED_DIV(1024 * milliseconds, 1000);
    uint64_t intendedCounter = 1024 * (uint64_t)time + ticks;
    uint64_t offset = intendedCounter - (uint64_t)BURTC_CounterGet();
    BURTC_RetRegSet(1, (uint32_t)(offset >> 32));
    BURTC_RetRegSet(0, (uint32_t)(offset & 0xFFFFFFFF));
    BURTC_RetRegSet(2, 0x11223344);
	return;
}

static void getTime(uint32_t *time, uint32_t *milliseconds) {

    uint64_t offset =  (uint64_t)BURTC_RetRegGet(1) << 32;
    offset += (uint64_t)BURTC_RetRegGet(0);
    uint64_t currentCounter = offset + BURTC_CounterGet();

    if (time != NULL) {
        *time = currentCounter / 1024;
    }

    if (milliseconds != NULL) {
        uint32_t ticks = currentCounter % 1024;
        *milliseconds = ROUNDED_DIV(1000 * ticks, 1024);
    }
	return;
}

static void handleTimeOverflow(void) {
    uint32_t offsetHigh = BURTC_RetRegGet(1);
    BURTC_RetRegSet(1, offsetHigh + 1);
}

/* Time function for FAT file system */
DWORD get_fattime(void) {

    int8_t timezoneHours = 0;

    int8_t timezoneMinutes = 0;

    uint32_t currentTime;

    getTime(&currentTime, NULL);

    if (BURTC_IntGet() & BURTC_IF_OF) {
        handleTimeOverflow();
        getTime(&currentTime, NULL);
        BURTC_IntClear(BURTC_IF_OF);
    }

    time_t fatTime = currentTime + timezoneHours * 60 * 60 + timezoneMinutes * 60;

    struct tm timePtr;

    gmtime_r(&fatTime, &timePtr);

    return (((unsigned int)timePtr.tm_year - 208) << 25) |
            (((unsigned int)timePtr.tm_mon + 1 ) << 21) |
            ((unsigned int)timePtr.tm_mday << 16) |
            ((unsigned int)timePtr.tm_hour << 11) |
            ((unsigned int)timePtr.tm_min << 5) |
            ((unsigned int)timePtr.tm_sec >> 1);
}


TASK(start_sdcard){
	GPIO_PinModeSet(gpioPortD, 12, gpioModePushPull, 1);
	/* Turn on SD card (PortD 12)*/
	GPIO_PinOutClear(gpioPortD, 12);
	/* Init */
	MICROSD_Init();
	/* Check SD card status */
	DSTATUS resCard = disk_initialize(0);
	if (resCard == STA_NOINIT || resCard == STA_NODISK || resCard == STA_PROTECT) {
        while(1);
    }
    /* Initialise file system */
    if (f_mount(&fatfs, "", 1) != FR_OK) {
        while(1);
    }

	TerminateTask();

}
#define APP_Task_start_sdcard_STOP_SEC_CODE
#include "tpl_memmap.h"

#define APP_Task_write_audio_START_SEC_CODE
#include "tpl_memmap.h"

void update_timekeeper(void){
	timekeeper.unix_timestamp++;
}

uint32_t largest_power_of_two(uint32_t x) {
	if (x == 0) return 0;
	uint32_t p = 1;
	while (p * 2 <= x) {
		p *= 2;
	}
	return p;
}

TASK(write_audio){
	GPIO_PinModeSet(LED_GPIOPORT, RED_LED, gpioModePushPull, true); 
	/* We are here from inference hit */
	/* First set writting_variable */
	index_buffer_audio_to_write = 0;
	is_writting_audio = 1;
	/* Create a file */
	char *filename;
	char filename_buffer[15];
	/* Count number of digit in timestamp for offset extension of file */
	uint8_t count_digit = 0;
	uint32_t tmp_timestamp = timekeeper.unix_timestamp;
    while(tmp_timestamp>0){
        count_digit++;
        tmp_timestamp = tmp_timestamp/10;
    }

	filename = __itoa(timekeeper.unix_timestamp, filename_buffer, 10);
	strcpy(filename+count_digit, "\n");

	if (file_duration == 0){
		audiofilename = __itoa(timekeeper.unix_timestamp, a_filename_buffer, 10); 
		strcpy(audiofilename+count_digit, ".wav");

		timefilename = __itoa(timekeeper.unix_timestamp, t_filename_buffer, 10); 
		strcpy(timefilename+count_digit, ".txt");
	}

	FRESULT check_audio = f_open(&fileaudio, audiofilename, FA_OPEN_APPEND | FA_WRITE);

	if (file_duration == 0){
		f_write(&fileaudio, &wavHeader, sizeof(wavHeader_t), &bw);
	}

	f_open(&filetime, timefilename, FA_OPEN_APPEND | FA_WRITE);
	f_write(&filetime, filename, (count_digit+1)*sizeof(char), &bw);
	
	
	/* Current audio buffer to write is count_ptr_buffer */
	/* First write previous buffers */
	uint8_t tmp_count_ptr_buffer = index_buffer_audio_inference2;
	uint8_t read_count_ptr_buffer;

	const uint8_t next_count=30;
	const uint8_t prev_count=30;

	read_count_ptr_buffer = (tmp_count_ptr_buffer + 100 - prev_count) % 100;
	
	uint8_t count_write_audio_pre_inference = 0;
	const uint8_t nb_w = 8;

	while(count_write_audio_pre_inference != prev_count){
		uint8_t nb_mod_w = nb_w;

		if (count_write_audio_pre_inference + nb_mod_w > next_count) {
			nb_mod_w = next_count - count_write_audio_pre_inference;
			nb_mod_w = largest_power_of_two(nb_mod_w);
		}

		if (read_count_ptr_buffer + nb_mod_w > 100){
			nb_mod_w = 100 - read_count_ptr_buffer;
			nb_mod_w = largest_power_of_two(nb_mod_w);
		}

		f_write(&fileaudio, buffer_ptr_sram[read_count_ptr_buffer], 2*1024*nb_mod_w, &bw);

		read_count_ptr_buffer = (read_count_ptr_buffer + nb_mod_w) % 100;
		count_write_audio_pre_inference+=nb_mod_w;
	}

	/* Now write next 2s of sound */
	uint8_t count_write_audio_post_inference = 0;
	while(count_write_audio_post_inference != next_count){
		EventMaskType ev_wr;
		WaitEvent(ev_write_audio);
		GetEvent(write_audio, &ev_wr); 
		ClearEvent(ev_wr);
		count_write_audio_post_inference++;
	}

	count_write_audio_post_inference = 0;
	while(count_write_audio_post_inference != next_count){
		uint32_t nb_mod_w = nb_w;

		if (count_write_audio_post_inference + nb_mod_w > next_count) {
			nb_mod_w = next_count - count_write_audio_post_inference;
			nb_mod_w = largest_power_of_two(nb_mod_w);
		}

		if (read_count_ptr_buffer + nb_mod_w > 100){
			nb_mod_w = 100 - read_count_ptr_buffer;
			nb_mod_w = largest_power_of_two(nb_mod_w);
		}

		f_write(&fileaudio, buffer_ptr_sram[read_count_ptr_buffer], 2*1024*nb_mod_w, &bw);

		read_count_ptr_buffer = (read_count_ptr_buffer + nb_mod_w) % 100;
		count_write_audio_post_inference+=nb_mod_w;
	}

	file_duration = (file_duration + 3) % (3300); // 55min long recordings
	is_writting_audio = 0;
	f_close(&fileaudio);
	f_close(&filetime);
	GPIO_PinModeSet(LED_GPIOPORT, RED_LED, gpioModePushPull, false); 
	TerminateTask();
}
#define APP_Task_write_audio_STOP_SEC_CODE
#include "tpl_memmap.h"

#define APP_ISR_isr_dma_START_SEC_CODE
#include "tpl_memmap.h"
ISR(isr_dma){
	/* Get interrupt mask */
	GPIO_PinOutToggle(gpioPortB, 10);
	uint32_t interruptMask = DMA_IntGet();

	switch (interruptMask){
		case DMA_IF_CH0DONE:
			/* Clear interrupt */
			DMA_IntClear(DMA_IFC_CH0DONE);
			isPrimaryDMABuffer = !isPrimaryDMABuffer;
			/* Re-activate the DMA */
			DMA_RefreshPingPong(0,
				isPrimaryDMABuffer,						/* bool to change between dma descriptor, primaryBuffer or secondaryBuffer DST*/
				false,
				NULL,									/* dst : NULL = same as in descriptor, thus same as in start dma TASK, (void*)primaryBuffer or (void*)secondaryBuffer */
				NULL,									/* src : NULL = same as in descriptor, thus same as in start dma TASK, (void*)&(ADC0->SINGLEDATA) */
				numberOfSamplesPerTransfer - 1,
				false);
			SetEvent(copyDMAtoSRAM, ev_DMAtoSRAM);
			break;

		default: 
			break;
	}
}
#define APP_ISR_isr_dma_STOP_SEC_CODE
#include "tpl_memmap.h"

#define APP_ISR_isr_rtc_START_SEC_CODE
#include "tpl_memmap.h"
ISR(isr_rtc){
	uint32_t interruptFlag = RTC_IntGet();
	switch (interruptFlag){
		case RTC_IF_COMP0:
		RTC_IntClear(RTC_IF_COMP0);
		RTC_CounterReset();
		update_timekeeper();
		break;
	
	default:
		break;
	}
}
#define APP_ISR_isr_rtc_STOP_SEC_CODE
#include "tpl_memmap.h"