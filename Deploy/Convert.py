import os
import numpy as np
import tensorflow as tf

def convert_model(outfolder, model_path, name):
    # Code taken from students
    model = tf.keras.models.load_model(model_path)

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_1 = converter.convert()

    # Save the converted model to a .tflite file
    with open(os.path.join(outfolder,name+'_trained_model.tflite'), 'wb') as f:
        f.write(tflite_model_1)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dynamic_model = converter.convert()

    # Save the converted model to a .tflite file
    with open(os.path.join(outfolder,name+'_dynamic_model.tflite'), 'wb') as f:
        f.write(dynamic_model)

def convert_model_full_int(outfolder, model_path, name, X_train, keep_data=True):
    # Code taken from students
    model = tf.keras.models.load_model(model_path)
    
    # Load sample training data (adjust shape to match model input)
    def representative_dataset(size=2000, keep=keep_data):
        if not keep:
            X_train_reshaped = X_train[..., np.newaxis]
        else:
            X_train_reshaped = X_train
        print(X_train_reshaped.shape)
        for i in range(size):
            sample = X_train_reshaped[i]  # Take one sample
            sample = np.expand_dims(sample, axis=0)  # Add batch dimension
            yield [sample.astype(np.float32)] # Convert to float32

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Specify full int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert and save
    tflite_model = converter.convert()

    with open(os.path.join(outfolder,name+"_int8_model.tflite"), "wb") as f:
        f.write(tflite_model)


def convert_model_int16(outfolder, model_path, name, X_train):
    
    model = tf.keras.models.load_model(model_path)

    # Load your SavedModel
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset generator (REQUIRED for int16 quant)
    # Load sample training data (adjust shape to match model input)
    def representative_dataset(size=4620, keep=True):
        if not keep:
            X_train_reshaped = X_train[..., np.newaxis]
        else:
            X_train_reshaped = X_train
        #X_train_reshaped = shuffle(X_train_reshaped)
        for i in range(size):
            sample = X_train_reshaped[i]  # Take one sample
            sample = np.expand_dims(sample, axis=0)  # Add batch dimension
            yield [sample.astype(np.float32)] # Convert to float32

    converter.representative_dataset = representative_dataset

    # Set supported ops for int16 activations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]

    # Force integer only model (optional)
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16

    tflite_model = converter.convert()

    with open(os.path.join(outfolder,name+"_int16_model.tflite"), "wb") as f:
        f.write(tflite_model)