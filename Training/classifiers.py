import tensorflow as tf
from frontends.filterlayers import *
from tensorflow.keras import models, layers


# Song classifiers


def student_raw(input_shape, nb_classes):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 = GaborConv1D(out_channels=32, kernel_size=101, stride=84, fs=20480)(input_layer)

    conv1d_1 = LogLayer()(conv1d_1)
    conv1d_1 = layers.BatchNormalization()(conv1d_1)
    conv1d_1 = layers.AveragePooling1D(pool_size=2)(conv1d_1)

    conv1d_2 = layers.Conv1D(filters=64, kernel_size=7, strides=2, activation='relu')(conv1d_1)
    output_main = layers.MaxPooling1D(pool_size=4)(conv1d_2)
    output_main = layers.Conv1D(filters=32, kernel_size=7, strides=2, activation='relu')(output_main)
    output_main = layers.MaxPooling1D(pool_size=8)(output_main)
    output_main = layers.Flatten()(output_main)
    output_main = layers.Dense(units=64, activation='relu')(output_main)
    output_main = layers.Dense(units=nb_classes, activation='softmax')(output_main)
        
    model = models.Model(inputs=input_layer, outputs={'conv1d_raw': conv1d_2, 'output_main': output_main})
    return model

def create_song_gabor_model(input_shape, nb_classes):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 = GaborConv1D(out_channels=32, kernel_size=101, stride=64, fs=20480, band_min=200)(input_layer)

    conv1d_1 = LogLayer()(conv1d_1)
    conv1d_1 = layers.BatchNormalization()(conv1d_1)
    conv1d_1 = layers.AveragePooling1D(pool_size=4)(conv1d_1)

    output_main = layers.Conv1D(filters=64, kernel_size=7, strides=2, activation='relu')(conv1d_1)
    output_main = layers.MaxPooling1D(pool_size=2)(output_main)
    output_main = layers.Conv1D(filters=32, kernel_size=7, strides=2, activation='relu')(output_main)
    output_main = layers.MaxPooling1D(pool_size=8)(output_main)
    output_main = layers.Flatten()(output_main)
    output_main = layers.Dense(units=64, activation='relu')(output_main)
    output_main = layers.Dense(units=nb_classes, activation='softmax')(output_main)
    
    model = models.Model(inputs=input_layer, outputs=output_main)
    return model

def create_song_sinc_model(input_shape, nb_classes):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 = SincConv1D(out_channels=32, kernel_size=101, stride=64, fs=20480)(input_layer)

    conv1d_1 = LogLayer()(conv1d_1)
    conv1d_1 = layers.BatchNormalization()(conv1d_1)
    conv1d_1 = layers.AveragePooling1D(pool_size=2)(conv1d_1)

    output_main = layers.Conv1D(filters=64, kernel_size=7, strides=2, activation='relu')(conv1d_1)
    output_main = layers.MaxPooling1D(pool_size=4)(output_main)
    output_main = layers.Conv1D(filters=32, kernel_size=3, strides=2, activation='relu')(output_main)
    output_main = layers.MaxPooling1D(pool_size=8)(output_main)
    output_main = layers.Flatten()(output_main)
    output_main = layers.Dense(units=64, activation='relu')(output_main)
    output_main = layers.Dense(units=nb_classes, activation='softmax')(output_main)
    
    model = models.Model(inputs=input_layer, outputs=output_main)
    return model



def create_song_depth_model(input_shape, nb_classes):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 = layers.DepthwiseConv2D(depth_multiplier=5, kernel_size=3, activation='relu')(input_layer)
    output_main = layers.MaxPooling2D(pool_size=2) (conv1d_1)
    output_main = layers.DepthwiseConv2D(depth_multiplier=8, kernel_size=3, activation='relu') (output_main)
    output_main = layers.MaxPooling2D(pool_size=(4,4)) (output_main)
    output_main = layers.DepthwiseConv2D(depth_multiplier=21, kernel_size=3, activation='relu') (output_main)
    output_main = layers.MaxPooling2D(pool_size=(1,4)) (output_main)
    output_main = layers.GlobalAveragePooling2D()(output_main)
    output_main = layers.Dense(units=32, activation='relu') (output_main)
    output_main = layers.Dense(units=nb_classes, activation='softmax', name='output_main') (output_main)

    model = models.Model(inputs=input_layer, outputs=output_main)
    return model



def student_mel(input_shape, nb_classes):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 = layers.DepthwiseConv2D(depth_multiplier=5, kernel_size=3, activation='relu', name='conv1d_1')(input_layer)
    output_main = layers.MaxPooling2D(pool_size=2) (conv1d_1)
    output_main = layers.DepthwiseConv2D(depth_multiplier=8, kernel_size=3, activation='relu') (output_main)
    output_main = layers.MaxPooling2D(pool_size=(4,4)) (output_main)
    output_depth = layers.DepthwiseConv2D(depth_multiplier=21, kernel_size=3, activation='relu') (output_main)
    output_main = layers.MaxPooling2D(pool_size=(1,4)) (output_depth)
    output_main = layers.GlobalAveragePooling2D()(output_main)
    output_main = layers.Dense(units=32, activation='relu') (output_main)
    output_main = layers.Dense(units=nb_classes, activation='softmax', name='output_main') (output_main)


    model = models.Model(inputs=input_layer, outputs={'conv1d_1': conv1d_1, 'output_main': output_main})
    return model



def test_gabor(input_shape, nb_classes):
    input_layer = layers.Input(shape=input_shape)
    conv1d_1 = GaborConv1D(out_channels=32, kernel_size=101, stride=64, fs=20480)(input_layer)

    conv1d_1 = LogLayer()(conv1d_1)
    conv1d_1 = layers.BatchNormalization()(conv1d_1)
    conv1d_1 = layers.AveragePooling1D(pool_size=4)(conv1d_1)


    output_b1 = layers.Lambda(lambda x: x[:, :, ::8])(conv1d_1)
    output_b1 = layers.Conv1D(filters=10, kernel_size=3, strides=1, activation='relu')(output_b1)
    output_b1 = layers.MaxPooling1D(pool_size=4)(output_b1)
    output_b1 = layers.Flatten()(output_b1)
    output_b1 = layers.Dense(units=32, activation='relu')(output_b1)
    output_b1 = layers.Dense(units=1, activation='sigmoid', name='output_b1')(output_b1)

    output_main = layers.Conv1D(filters=64, kernel_size=7, strides=2, activation='relu')(conv1d_1)
    output_main = layers.MaxPooling1D(pool_size=2)(output_main)
    output_main = layers.Conv1D(filters=32, kernel_size=7, strides=2, activation='relu')(output_main)
    output_main = layers.MaxPooling1D(pool_size=8)(output_main)
    output_main = layers.Flatten()(output_main)
    output_main = layers.Dense(units=64, activation='relu')(output_main)
    output_main = layers.Dense(units=nb_classes, activation='softmax', name="output_main")(output_main)
    
    model = models.Model(inputs=input_layer, outputs={'output_b1': output_b1, 'output_main': output_main})
    return model