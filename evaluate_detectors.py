import os
import numpy as np
import pandas as pd
import cmsisdsp as dsp

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

def results_goertzel():
    df_goertzel_train  = pd.read_csv(models_folder+"goertzel_scores_train_.csv")
    df_goertzel_test  = pd.read_csv(models_folder+"goertzel_scores_test_.csv")

    df_score = df_goertzel_train[["T1", "T2"]]
    train_scores = np.max(df_score, axis=1)
    precision, recall, thresholds = precision_recall_curve(df_goertzel_train["label"], train_scores)
    fscore = 2*precision*recall/(precision+recall+1e-10)
    indx_max = np.argmax(fscore)
    thresholds = np.insert(thresholds, 0, 0)
    threshold_curve = thresholds[indx_max]

    recall_ref = 0.9
    idx_ref_recall=np.argmin(np.abs(recall-recall_ref))
    threshold_recall = thresholds[idx_ref_recall]
    
    precision_ref = 0.9
    idx_ref_precision=np.argmin(np.abs(precision-precision_ref))
    threshold_precision = thresholds[idx_ref_precision]

    print("From train:", threshold_curve, threshold_recall, threshold_precision)


    df_score = df_goertzel_test[["T1", "T2"]]
    test_scores = np.max(df_score, axis=1)

    precision_test, recall_test, thresholds_test = precision_recall_curve(df_goertzel_test["label"], test_scores)

    idx_ref_recall_test=np.argmin(np.abs(recall_test-recall_ref))

    y_pred_recall = (test_scores > thresholds_test[idx_ref_recall_test]).astype(int)

    print("goertzel", thresholds_test[idx_ref_recall_test])

    return df_goertzel_test["label"], y_pred_recall, test_scores, df_goertzel_test["distances"]


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

def inv_sigmoid(x):
    return np.log(x + 1e-30) - np.log(1 - x + 1e-30)

def evaluate_gabor():
    data_yh=dataset.YellowhammerData("full_ID_Set_split", key="test",
                                     duration=1.5, augment=None,
                                     transform=None)
    X_test, y_test, d_test = data_yh.load_audio(binary=True, with_dist=True)
    nb_try=0
    folder_models = "Models/Gabor/Gabor_"+str(nb_try)+"/"

    name="gabor_InConv1d"
    int16_tflite_path = folder_models+name+"_int16_model.tflite"
    X_int16 = get_tf_results(X_test, int16_tflite_path)
    X_int16 = np.squeeze(X_int16)

    #X_int16_log = np.log(np.abs(X_int16)+ 1e-6)#
    X_int16_log = dsp.arm_vlog_f32(dsp.arm_abs_f32(X_int16) + 1e-6 ).reshape(X_int16.shape)

    int8_tflite_path = folder_models+name+"_block_int8_model.tflite"
    _, y_pred_scores = evaluate.get_tf_results(X_int16_log, int8_tflite_path)
    #y_pred_scores = inv_sigmoid(y_pred_scores)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_scores)

    metric_ref = 0.9
    idx_ref_recall=np.argmin(np.abs(recall-metric_ref))
    y_pred = (y_pred_scores>thresholds[idx_ref_recall]).astype(int).flatten()
    print("gabor", thresholds[idx_ref_recall])
    
    recalls_at, results = evaluate.detection_results(y_test, y_pred, y_pred_scores, d_test)

    return recalls_at, results


if __name__ == "__main__":
    split_name="full_ID_Set_split"
    detection=True

    model_names = ["Mel", "Band"]

    cmsis_bands = features.BandEnvelopeCMSIS(samplerate=20480, band_ranges=[(3000,9000)], q31=False)
    cmsis_mel= features.MelSpecCMSIS(samplerate=20480, window_len=512, window_stride=320, nb_mels=32, fmin=2000, fmax=10000)
    all_feautres = [cmsis_mel, cmsis_bands]

    model_types = ["int8", "int8"]


    compare_results = []
    dist_recalls=[]
    for name, cmsis_feature, model_type in zip(model_names, all_feautres, model_types):

        data_yh=dataset.YellowhammerData(split_name, key="test", duration=1.5, augment=None, transform=cmsis_feature)
        X_test, y_test, d_test = data_yh.load_feature(binary=detection, with_dist=True)
        
        i=0
        name_folder="Models/"+name+"/"+name+"_"+str(i)+"/"
        tf_model_path =name_folder+name+"_"+model_type+"_model.tflite"
        keras_model_path=name_folder+"model.keras"

        _, y_pred_scores = evaluate.get_tf_results(X_test, tf_model_path)
        #y_pred_scores = inv_sigmoid(y_pred_scores)


        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_scores)
        metric_ref = 0.9
        idx_ref_recall=np.argmin(np.abs(recall-metric_ref))

        y_pred = (y_pred_scores>thresholds[idx_ref_recall]).astype(int).flatten() #TODO change thresholds
        print(name, thresholds[idx_ref_recall])
        recalls_at, results = evaluate.detection_results(y_test, y_pred, y_pred_scores, d_test)

        results.insert(0, name)
        compare_results.append(results)
        dist_recalls.append(recalls_at)

    # ADD gabor
    recalls_at, results = evaluate_gabor()
    results.insert(0, "Gabor")
    compare_results.append(results)
    dist_recalls.append(recalls_at)
    model_names.append("Gabor")

    # ADD goertzel
    y_true_Goe, y_pred_Goe, goertzel_Scores, d_Goe = results_goertzel()
    recalls_at, results = evaluate.detection_results( y_true_Goe, y_pred_Goe, goertzel_Scores, d_Goe)
    results.insert(0, "goertzel")
    compare_results.append(results)
    dist_recalls.append(recalls_at)
    model_names.append("Goertzel")
    
    plot_folder="eval_utils/results/"
    plot_curve_at_distance(eval_plot_folder=plot_folder, list_all=dist_recalls, names=model_names, plot_name="Recall", show=False)
    pd.DataFrame(columns=['name', 'precision', 'recall', 'f1score', 'accuracy', 'avg_precision'], data=np.array(compare_results)).to_csv(os.path.join(plot_folder, "results-quantized.csv"), index=False)



def FP_evaluate():

    split_name="full_ID_Set_split"
    detection=True

    model_names = ["Mel", "Band", "Gabor"]

    cmsis_bands = features.BandEnvelopeCMSIS(samplerate=20480, band_ranges=[(3000,9000)], q31=False)
    cmsis_mel= features.MelSpecCMSIS(samplerate=20480, window_len=512, window_stride=320, nb_mels=32, fmin=2000, fmax=10000)
    all_feautres = [cmsis_mel, cmsis_bands, None]

    model_types = ["int8", "int8", "int16"]


    compare_results = []
    dist_recalls=[]
    for name, cmsis_feature, model_type in zip(model_names, all_feautres, model_types):

        data_yh=dataset.YellowhammerData(split_name, key="test", duration=1.5, augment=None, transform=cmsis_feature)
        if cmsis_feature is None:
            X_test, y_test, d_test = data_yh.load_audio(binary=detection, with_dist=True)
        else:
            X_test, y_test, d_test = data_yh.load_feature(binary=detection, with_dist=True)
        
        recall_tries = []
        results_tries = []

        for i in range(1):
            name_folder="Models/"+name+"/"+name+"_"+str(i)+"/"
            tf_model_path =name_folder+name+"_"+model_type+"_model.tflite"
            keras_model_path=name_folder+"model.keras"

            _, y_pred_scores = evaluate.get_NN_results (X_test, keras_model_path)
            #_, y_pred_scores = evaluate.get_tf_results(X_test, tf_model_path)


            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_scores)
            metric_ref = 0.9
            idx_ref_recall=np.argmin(np.abs(recall-metric_ref))

            y_pred = (y_pred_scores>thresholds[idx_ref_recall]).astype(int).flatten() #TODO change thresholds
            #print(thresholds[idx_ref_recall])
            recalls_at, results = evaluate.detection_results(y_test, y_pred, y_pred_scores, d_test)
            results_tries.append(results)
            recall_tries.append(recalls_at)

        #print(name,':', np.mean(results_tries, axis=0))
        dist_recalls.append(np.mean(recall_tries, axis=0))
        mean_results = np.mean(results_tries, axis=0).tolist()
        mean_results.insert(0, name)
        compare_results.append(mean_results)

    y_true_Goe, y_pred_Goe, goertzel_Scores, d_Goe = results_goertzel()
    recalls_at, results = evaluate.detection_results( y_true_Goe, y_pred_Goe, goertzel_Scores, d_Goe)
    results.insert(0, "goertzel")
    compare_results.append(results)
    dist_recalls.append(recalls_at)
    model_names.append("Goertzel")
    
    plot_folder="eval_utils/results/"
    plot_curve_at_distance(eval_plot_folder=plot_folder, list_all=dist_recalls, names=model_names, plot_name="Recall", show=False)
    pd.DataFrame(columns=['name', 'precision', 'recall', 'f1score', 'accuracy', 'avg_precision'], data=np.array(compare_results)).to_csv(os.path.join(plot_folder, "results-FP32.csv"), index=False)
