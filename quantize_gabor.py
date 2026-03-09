
import os
import tensorflow as tf
import data_utils.augment as augment
import data_utils.dataset as dataset

from frontends.filterlayers import *
import Training.utils as train_tools 

from tensorflow.keras import models, layers
import Deploy.Convert as convert

import eval_utils.evaluate as evaluate
from sklearn.metrics import auc, precision_recall_curve
from eval_utils.figures import *

def save_conv_gabor(folder_models, input_shape):
    model_path=folder_models+'model.keras'
    gabor_model = tf.keras.models.load_model(model_path)
    gabor_model.summary()

    conv1d, _ =train_tools.get_conv1D(gabor_model, idx=1)

    gabor_model.trainable=False
    input_layer = layers.Input(shape=input_shape)
    output_main = conv1d(input_layer)

    for i in range(2,len(gabor_model.layers)):
        print(gabor_model.layers[i], output_main.shape)
        output_main = train_tools.get_new_layer(gabor_model.layers[i], output_main.shape)(output_main)

    new_model = models.Model(inputs=input_layer, outputs=output_main)
    new_model.summary()

    new_model.trainable=False

    name="gabor_InConv1d"
    model_path=folder_models+name+'.keras'
    new_model.save(model_path)


    input_int16 = layers.Input(shape=input_shape)
    block_int16 = input_int16 
    for i in range(1,2):
        block_int16 = train_tools.get_new_layer(new_model.layers[i], block_int16.shape)(block_int16)
    
    model_int16 = models.Model(inputs=input_int16, outputs=block_int16)
    model_int16.summary()
    model_int16.save(folder_models+name+'_block_int16.keras')


    input_int8 = layers.Input(shape=(int(block_int16.shape[-2]/4), block_int16.shape[-1]))
    print((int(block_int16.shape[-2]/4), block_int16.shape[-1]))
    block_int8 = input_int8
    for i in range(4,len(new_model.layers)):
        print(new_model.layers[i], block_int8.shape)
        block_int8 = train_tools.get_new_layer(new_model.layers[i], block_int8.shape)(block_int8)
    
    model_int8 = models.Model(inputs=input_int8, outputs=block_int8)
    model_int8.summary()
    model_int8.save(folder_models+name+'_block_deploy_int8.keras')

    input_int8_eval = layers.Input(shape=(block_int16.shape[-2], block_int16.shape[-1]))
    block_int8_eval = input_int8_eval
    for i in range(3,len(new_model.layers)):
        block_int8_eval = train_tools.get_new_layer(new_model.layers[i], block_int8_eval.shape)(block_int8_eval)
    
    model_int8_eval = models.Model(inputs=input_int8_eval, outputs=block_int8_eval)
    model_int8_eval.summary()
    model_int8_eval.save(folder_models+name+'_block_int8.keras')

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


def quantize():
    data_yh=dataset.YellowhammerData("full_ID_Set_split", key="train", 
                                     duration=1.5, augment=augment.augment_distances,
                                     transform=None)
    
    X_train, y_train = data_yh.load_audio(binary=True, norm=True)
    input_shape=X_train[0].shape

    nb_try=0
    folder_models = "Models/Gabor/Gabor_"+str(nb_try)+"/"

    name="gabor_InConv1d"
    model_path=folder_models+name+'.keras'
    
    if not os.path.exists(model_path):
        save_conv_gabor(folder_models, input_shape)

    name="gabor_InConv1d"
    int16_path=folder_models+name+'_block_int16.keras'
    convert.convert_model_int16(folder_models, int16_path, name, X_train)

    int16_tflite_path = folder_models+name+"_int16_model.tflite"
    X_int16 = get_tf_results(X_train, int16_tflite_path)
    X_int16 = np.squeeze(X_int16)

    X_int16_log = np.log(np.abs(X_int16)+ 1e-6)#dsp.arm_vlog_f32(dsp.arm_abs_f32(X_int16)).reshape(X_int16.shape)

    
    name="gabor_InConv1d"
    model_path=folder_models+name+'.keras'
    gabor_model = tf.keras.models.load_model(model_path)
    int8_path=folder_models+name+'_block_int8.keras'
    name_block=name+"_block"
    convert.convert_model_full_int(folder_models, int8_path, name_block, X_int16_log)

    model_path=folder_models+name+'.keras'
    gabor_model = tf.keras.models.load_model(model_path)
    int8_path=folder_models+name+'_block_deploy_int8.keras'
    name_deploy=name+"_block_deploy"
    log_layer = train_tools.get_activation_model(gabor_model, ["average_pooling1d"])
    X_float = log_layer(X_train, training=False)
    convert.convert_model_full_int(folder_models, int8_path, name_deploy, X_float)

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
    plt.plot(X_int16[0])
    plt.title("X_int16")
    plt.show()
    X_int16_log = np.log(np.abs(X_int16)+ 1e-6)#dsp.arm_vlog_f32(dsp.arm_abs_f32(X_int16)).reshape(X_int16.shape)
    plt.plot(X_int16_log[0])
    plt.title("X_int16_log")
    plt.show()

    name="gabor_InConv1d"
    model_path=folder_models+name+'.keras'
    gabor_model = tf.keras.models.load_model(model_path)

    log_layer = train_tools.get_activation_model(gabor_model, ["log_layer"])
    X_float = log_layer(X_test, training=False)
    X_float = X_float.numpy()

    int8_tflite_path = folder_models+name+"_int8_model.tflite"
    print(X_float.shape)
    _, y_pred_scores = evaluate.get_tf_results(X_int16_log, int8_tflite_path)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_scores)

    metric_ref = 0.9
    idx_ref_recall=np.argmin(np.abs(recall-metric_ref))
    y_pred = (y_pred_scores>thresholds[idx_ref_recall]).astype(int).flatten()
    
    recalls_at, results = evaluate.detection_results(y_test, y_pred, y_pred_scores, d_test)

    print(results)
    plot_curve_at_distance(eval_plot_folder=None, list_all=[recalls_at], names=['Gabor'], plot_name="Recall", show=True)

if __name__ == "__main__":
    quantize()