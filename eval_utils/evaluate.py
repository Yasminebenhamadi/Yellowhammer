import os
import math
import numpy as np 
import pandas as pd
from functools import *
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score


def evaluate_at_ditance(metric, y_test, y_pred, d_test):
    df_results = pd.DataFrame(np.array([y_test, y_pred, d_test]).T, columns=['true','predict', 'distance'])
    recalls = []
    for distance, frame in df_results.groupby(['distance']):
        if distance[0]>0:
            recalls.append(metric(frame['true'], frame['predict']))
    return recalls

def detection_results(y_test, y_pred, y_pred_scores, d_test):
    recall_bi = partial(recall_score, average="binary")
    recalls_at = evaluate_at_ditance(recall_bi, y_test, y_pred, d_test)

    return (recalls_at, 
            [precision_score(y_test, y_pred, average="binary"), 
            recall_score(y_test, y_pred, average="binary"), 
            f1_score(y_test, y_pred, average="binary"), 
            accuracy_score(y_test, y_pred), 
            average_precision_score(y_test, y_pred_scores)])


def evaluate(model, X, y_true, test_distances):
    y_one = pd.get_dummies(y_true).astype(int).to_numpy()

    y_pred_scores = model.predict(X)
    pred_y_main = np.zeros_like(y_pred_scores)
    pred_y_main[np.arange(len(y_pred_scores)), np.argmax(y_pred_scores, axis=1)] = 1
    acc, f1 = accuracy_score(y_one, pred_y_main), f1_score(y_one, pred_y_main, average='macro')

    classes, count_classes = np.unique(y_true, return_counts=True)

    dist, dist_counts = np.unique(test_distances, return_counts=True)

    idx_false_classes = [i for i in range(len(y_true)) if (pred_y_main[i] != y_one[i]).any()]
    _, fc_per_dist = np.unique(test_distances[idx_false_classes], return_counts=True)
    _, fc_per_class = np.unique(y_true[idx_false_classes], return_counts=True)

    return acc, f1, fc_per_dist/dist_counts, fc_per_class/count_classes

def get_pred_details(interpreter, output_details, quantized=False):
    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_scale, output_zero_point = output_details[0]['quantization']

    if quantized:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32)-output_zero_point)*output_scale
    
    if len(output_details)==1:
        label = 1 if output_data>0.5 else 0
        return label, output_data
    elif output_data>0.9:
        return 1, output_data
    elif output_data<0.1:
        return 0, output_data
    else:
        output_data_2 = interpreter.get_tensor(output_details[1]['index'])
        if quantized:
            output_scale_2, output_zero_point_2 = output_details[1]['quantization']
            output_data_2 = (output_data_2.astype(np.float32)-output_zero_point_2)*output_scale_2

        label = 1 if output_data_2>0.5 else 0
        return label, output_data_2

def get_tf_results (X_test, model_path, expand=False):

    if expand:
        X_test = X_test[..., np.newaxis]

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get the input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']

    y_pred=[]
    y_pred_scores=[]
    do_quantize="int8" in model_path or "int16" in model_path
    for x in X_test:
        # Normalize and quantize input to INT8
        test_input = x.astype(np.float32)  # Ensure float32 before scaling

        if "int8" in model_path:
            test_input = (test_input / input_scale + input_zero_point).astype(np.int8)
        elif "int16" in model_path:
            test_input = (test_input / input_scale + input_zero_point).astype(np.int16)

        test_input = np.expand_dims(test_input, axis=0)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()

        label, score = get_pred_details(interpreter, output_details, quantized=do_quantize)
        y_pred.append(label)
        y_pred_scores.append(score)

    return np.array(y_pred), np.array(y_pred_scores).flatten()


def get_NN_results (X_test, model_path):
    model = tf.keras.models.load_model(model_path)
    y_pred_scores = model.predict(X_test).flatten()
    y_pred = (y_pred_scores>0.5).astype(int).flatten()
    return y_pred, y_pred_scores