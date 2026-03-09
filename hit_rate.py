import os
import numpy as np
import pandas as pd
import cmsisdsp as dsp

from Training.goertzel import *

import tensorflow as tf

import data_utils.dataset as dataset
import frontends.features as features
from frontends.filterlayers import *


from sklearn.metrics import precision_recall_curve
import eval_utils.evaluate as evaluate
from eval_utils.figures import *

import warnings
warnings.filterwarnings("ignore")

models_folder="Models/"

def get_tf_results (X_test, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get the input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']

    output_X=[]
    for x in X_test:
        test_input = x.astype(np.float32)  # Ensure float32 before scaling

        if "int16" in model_path:
            test_input = (test_input / input_scale + input_zero_point).astype(np.int16)
        elif "int8" in model_path:
            test_input = (test_input / input_scale + input_zero_point).astype(np.int8)

        test_input = np.expand_dims(test_input, axis=0)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32)-output_zero_point)*output_scale
        output_X.append(output_data)
    return np.array(output_X)


def gabor_results(X_test, nb_try=0):
    folder_models = "Models/Gabor/Gabor_"+str(nb_try)+"/"

    name="gabor_InConv1d"
    int16_tflite_path = folder_models+name+"_int16_model.tflite"
    X_int16 = get_tf_results(X_test, int16_tflite_path)
    X_int16 = np.squeeze(X_int16)

    X_int16_log = dsp.arm_vlog_f32(dsp.arm_abs_f32(X_int16) + 1e-6 ).reshape(X_int16.shape)
    X_int16_log = X_int16_log[np.newaxis,...]
    int8_tflite_path = folder_models+name+"_block_int8_model.tflite"
    _, y_pred_scores = evaluate.get_tf_results(X_int16_log, int8_tflite_path)

    return y_pred_scores


def load_audio(folder, sample_files, sample_rate, duration):
    data = []
    new_files = []
    for file_name in sample_files:
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder, file_name)
            data_sample, sr = librosa.load(file_path, sr=sample_rate)
            if data_sample.shape[0]/sr >= duration:
                data.append(np.expand_dims(data_sample[0:int(duration*sr)], axis=1))
                new_files.append(file_name)
    return np.array(data), new_files


def get_X(folder, clip=1.5):

    band_path="Models/Band/Band_0/Band_int8_model.tflite"
    mel_path="Models/Mel/Mel_0/Mel_int8_model.tflite"
    
    cmsis_bands = features.BandEnvelopeCMSIS(samplerate=20480, band_ranges=[(3000,9000)], q31=False)
    cmsis_mel= features.MelSpecCMSIS(samplerate=20480, window_len=512, window_stride=320, nb_mels=32, fmin=2000, fmax=10000)

    list_files = [f for f in os.listdir(folder) if ".WAV" in f and "._" not in f]
    results = []
    for filename in list_files:
        print(filename)
        full_audio, sr = librosa.load(folder+filename, sr=20480)
        duration = full_audio.shape[0]/sr

        for i in range(int(duration/clip)):
            start=i*int(clip*sr)
            end=start+int(clip*sr)
            audio_clip=full_audio[start:end]

            band_input=cmsis_bands.feature(audio_clip)
            band_input=band_input[np.newaxis, ..., np.newaxis]

            mel_input=cmsis_mel.feature(audio_clip)
            mel_input=mel_input[np.newaxis, ..., np.newaxis]

            gabor_input=(audio_clip-np.mean(audio_clip))/np.std(audio_clip)
            gabor_input = gabor_input[np.newaxis, ..., np.newaxis]


            band_score = evaluate.get_tf_results(band_input, band_path)[1][0]
            mel_score = evaluate.get_tf_results(mel_input, mel_path)[1][0]
            gabor_score = gabor_results(gabor_input)[0]
            results.append([filename, i, band_score, mel_score, gabor_score])

    results = np.array(results)
    return np.array(results)

if __name__ == "__main__":

    week_name="March"
    week="March/"
    vizina_folder="/Volumes/LaCie/Vizina/OG/"+week
    output_folder="eval_utils/results/"

    id_AM=1
    subfolder="AM"+str(id_AM)+"/"

    input_folder=os.path.join(vizina_folder, subfolder)

    if True:
        X_vizina = get_X(input_folder)
        df_hitrate = pd.DataFrame(columns=['filename', 'i', 'band_score', 'mel_score', 'gabor_score'], data=X_vizina)
        df_hitrate.to_csv(output_folder+"hit_rate_"+"AM"+str(id_AM)+"_"+week_name+".csv", index=False)


    results = "PhD/TinyYH/models/goertzel/"
    results_folder=output_folder+week+subfolder
    os.makedirs(results_folder, exist_ok=True)

    list_files = [f for f in os.listdir(input_folder) if ".WAV" in f and "._" not in f]
    stats_list = []
    sample_rate=20480
    for filename in list_files:
        print(filename)
        full_audio, sr = librosa.load(input_folder+filename, sr=20480)
        nb_clips = int(len(full_audio)/(1.5*sample_rate))
        X_full = full_audio[0:int(nb_clips*1.5*sample_rate)].reshape((nb_clips, int(1.5*sample_rate), 1))
        full_scores = goertzel_inference(X_full, sr)
        df_file = pd.DataFrame(data=full_scores, columns=["T1" , "T2"])
        df_file.insert(column="max", value=np.max(full_scores, axis=1), loc=2)
        df_file.to_csv(results_folder+filename[0:-3]+"csv", index=False)