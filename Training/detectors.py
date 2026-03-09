import tensorflow as tf
from frontends.filterlayers import *
from tensorflow.keras import models, layers


# Detection 

def create_mel_detect_model(input_shape):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 =layers.Conv2D(filters=5, kernel_size=5, strides=2, activation='relu')(input_layer)
    conv1d_1 = layers.BatchNormalization()(conv1d_1)
    output_b1 =  layers.MaxPooling2D(pool_size=4)(conv1d_1)
    output_b1 = layers.Flatten()(output_b1)
    output_b1 = layers.Dense(units=32, activation='relu')(output_b1)
    output_b1 = layers.Dense(units=1, activation='sigmoid', name='output_b1')(output_b1)

    model = models.Model(inputs=input_layer, outputs=output_b1)
    return model

def create_bands_detect_model(input_shape): # from students code
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv1D(filters=4, kernel_size=7, strides=4, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(4, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  
        # Binary classification
    ])
    return model

def create_envelope_detect_model(input_shape):
    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv1D(filters=5, kernel_size=7, strides=4, activation='relu')(input_layer)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=3, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=4, activation='relu')(x)
    output = layers.Dense(units=1, activation='sigmoid', name='output')(x)

    model = models.Model(inputs=input_layer, outputs=output)
    return model

def create_conv_detect_model(input_shape):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 = layers.Conv1D(filters=4, kernel_size=101, strides=84, activation='relu')(conv1d_1)
    conv1d_1 = layers.BatchNormalization()(conv1d_1)
    branch_1 = layers.MaxPooling1D(pool_size=4)(conv1d_1)

    branch_1 = layers.Conv1D(filters=5, kernel_size=7, strides=4, activation='relu')(branch_1)
    branch_1 = layers.MaxPooling1D(pool_size=4)(branch_1)
    branch_1 = layers.Flatten()(branch_1)
    branch_1 = layers.Dense(units=32, activation='relu')(branch_1)
    output_b1 = layers.Dense(units=1, activation='sigmoid', name='output_b1')(branch_1)
    
    model = models.Model(inputs=input_layer, outputs=output_b1)
    return model

def create_gabor_detect_model(input_shape):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 = GaborConv1D(out_channels=4, kernel_size=101, stride=64, fs=20480)(input_layer)
    conv1d_1 = LogLayer()(conv1d_1)
    conv1d_1 = layers.AveragePooling1D(pool_size=4)(conv1d_1)

    output_main = layers.Conv1D(filters=5, kernel_size=7, strides=4, activation='relu')(conv1d_1)
    output_main = layers.MaxPooling1D(pool_size=4)(output_main)
    output_main = layers.Flatten()(output_main)
    output_main = layers.Dense(units=32, activation='relu')(output_main)
    output_main = layers.Dense(units=1, activation='sigmoid', name='output_b1')(output_main)
    
    model = models.Model(inputs=input_layer, outputs=output_main)

    return model