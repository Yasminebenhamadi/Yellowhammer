# Yellow Hammer Audiomoth Project

## Overview
This project implements a neural-network to detect and record yellow hammer sound. It is built on [Audiomoth](https://www.openacousticdevices.info/audiomoth).

The system record audio at 20480Hz and then pre-process the audio. Each 500ms of sound is then fed to the neural-network. If the neural network classifies the 500ms of sound as containing a yellow hammer sound, it saves the previous second of sound to the SD card and the next two seconds of sound.

The yellow hammer application is implemented on top of an RTOS, [trampoline](https://github.com/TrampolineRTOS/trampoline).

## Installation
###### Requirements
- git
- python3

First clone the trampoline repository containing the port for audiomoth:
```sh
git clone https://github.com/TrampolineRTOS/trampoline.git
```
Then switch to the audiomoth branch
```sh
git checkout audiomoth
```
Now get submodules, it will download CMSIS_5, CMSIS-DSP, CMSIS-NN and tlfite-micro
```sh
git submodule init
git submodule update
```

Trampoline uses a description language (OIL - Osek Implementation Language) for components used by the RTOS. We need to build the compiler, named goil, for this language.

In the trampoline repository, go under goil/
```sh
cd goil
```
Now based on your host machine, choose the according folder (ie makefile-macosx for mac, makefile-unix for linux ...)

Assuming a Linux host:
```sh
cd makefile-unix
```

Now execute the build python script
```sh
python3 build.py release
```
Upon completion, you should have a goil binary within the same folder (ie makefile-unix in this case).

Add the path to this directory to your environment variable

Open a new terminal and test the following command to check successful installation
```sh
goil --version
```
It should return something similar to this:
```sh
goil : 3.1.12, build with GALGAS 3.5.0
No warning, no error.
```

## Tflite-micro 
This application requires tflite-micro for the neural-network implementation

Within trampoline, it is located in `machines/cortex-m/tflite-micro`.

Now we will build the hello world example from tflite-micro, it is not require with this application but the script within tflite-micro will download all submodule needed to make tflite-micro usable.

In the tflite-micro folder, execute the following command
```sh
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn test_hello_world_test
```
The command is supposed to fail, it is normal. Now you should have all submodules downloaded under ```tensorflow/lite/micro/tools/make/downloads```
It should contains:
- cmsis
- cmsis_nn
- flatbuffers
- gcc_embedded
- gemmlowp
- kissfft
- pigweed
- ruy

## Compiling the application

This example use the CMake build system with VSCode.

- Open the ```main.oil``` file 
- Modify the ```TRAMPOLINE_BASE_PATH``` variable according to your installation
- Open a terminal from this folder
- Then run goil
```sh
goil --target=cortex-m/armv7em/efm32wg --templates=<path_to_trampoline>/goil/templates/ main.oil
```
goil --target=cortex-m/armv7em/efm32wg --templates=/Users/yasminebenhamadi/PhD/YellowHammer/AudioMoth/q15_source/trampoline/goil/templates/ main.oil
- Finally open vscode from this folder
```sh
code .
```
- Now use a toolkit to compile the application

- Finally flash and debug by pressing F5 within VSCode

## Adding time 
To configure automatically the time when flashing the firmware, add the following command in the `launch.json` file. It will execute the `generate_rtc_gdb_command.sh` script and then configure the time variable when flashing the firmware using a `.gdb` file.

```
"preLaunchTask": "set-rtc-time", // Run the shell script before debugging
"postLaunchCommands": [
         "source set_timekeeper.gdb"
      ]
```
