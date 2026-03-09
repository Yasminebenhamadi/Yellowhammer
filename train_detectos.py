import os
import numpy as np
import pandas as pd
import tensorflow as tf

import data_utils.dataset as dataset
import data_utils.augment as augment

import frontends.features as features
import Training.detectors as detectors
import Training.goertzel as goertzel


from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, losses

import Deploy.Convert as convert


models_folder="Models/"

def train (split_name, create_model, detection, cmsis_feature, save=True, outname='model', try_nb=0):
    data_yh=dataset.YellowhammerData(split_name, key="train", duration=1.5, augment=augment.augment_distances, transform=cmsis_feature)
    
    if cmsis_feature is None:
        X_train, y_train = data_yh.load_audio(binary=detection)
    else:
        X_train, y_train = data_yh.load_feature(binary=detection)

    input_shape=X_train[0].shape
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    class_weights_dict = {0:0.5, 1:2}
    print(X_train.shape, y_train.shape)

    model_detect =create_model(input_shape)

    model_detect.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss=losses.BinaryCrossentropy(),
            metrics=['accuracy'],
        )

    model_detect.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model_detect.fit(x=X_train,y=y_train, 
                            batch_size=32,epochs=50, validation_split=0.2, shuffle=True, 
                            callbacks=[early_stopping],class_weight=class_weights_dict)
    
    if save:
        model_detect.Trainable=False

        out_folder=os.path.join(models_folder, outname)
        model_out_folder=os.path.join(out_folder, outname+"_"+str(try_nb))
        model_path = os.path.join(model_out_folder, "model.keras")
        os.makedirs(out_folder, exist_ok = True)

        model_detect.export(os.path.join(out_folder, outname+"_"+str(try_nb)))
        model_detect.save(model_path)
        
        convert.convert_model_full_int(model_out_folder, model_path, outname, X_train)
        convert.convert_model_int16(model_out_folder, model_path, outname, X_train)



def set_goertzel():
    split_name="full_ID_Set_split"
    data_yh=dataset.YellowhammerData(split_name, key="train", 
                                     duration=1.5, augment=augment.augment_distances,
                                     transform=None)
    
    X_train, y_train = data_yh.load_audio(binary=True, norm=False)

    y_scores = goertzel.goertzel_inference(X_train, sr=20480)
    df_goertzel  = pd.DataFrame(np.array([y_scores[:,0], y_scores[:,1], y_train]).T, columns=["T1", "T2", "label"])
    df_goertzel.to_csv("Models/goertzel_scores_train_.csv", index=False)


    data_yh=dataset.YellowhammerData(split_name, key="test", 
                                     duration=1.5, augment=None,
                                     transform=None)
    
    X_test, y_test, d_test = data_yh.load_audio(binary=True, with_dist=True, norm=False)

    test_scores = goertzel.goertzel_inference(X_test, sr=20480)
    df_goertzel_test  = pd.DataFrame(np.array([test_scores[:,0], test_scores[:,1], y_test, d_test]).T, columns=["T1", "T2", "label", "distances"])
    df_goertzel_test.to_csv("Models/goertzel_scores_test_.csv", index=False)


if __name__ == "__main__":

    split_name="full_ID_Set_split"
    nb_tries=5
    for try_nb in range(1):
        cmsis_mel = features.MelSpecCMSIS(samplerate=20480, window_len=512, window_stride=320, nb_mels=32, fmin=2000, fmax=10000)
        train(split_name=split_name, detection=True, 
            create_model=detectors.create_mel_detect_model, cmsis_feature=cmsis_mel, outname='Mel',
            save=True, try_nb=try_nb)

    for try_nb in range(1):
        cmsis_bands = features.BandEnvelopeCMSIS(samplerate=20480, band_ranges=[(3000,9000)], q31=False)
        train(split_name=split_name, detection=True, 
            create_model=detectors.create_envelope_detect_model, cmsis_feature=cmsis_bands, outname='Band',
            save=True, try_nb=try_nb)
        
    for try_nb in range(1):
        train(split_name=split_name, detection=True, 
            create_model=detectors.create_gabor_detect_model, cmsis_feature=None, outname='Gabor',
            save=True, try_nb=try_nb)

