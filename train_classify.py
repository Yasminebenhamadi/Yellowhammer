import numpy as np
import pandas as pd
import tensorflow as tf
import data_utils.augment as augment
import data_utils.dataset as dataset
import frontends.features as features
import eval_utils.evaluate as eval
import Training.classifiers as yh_models
import Training.utils as train_utils
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping

import Training.utils as training_utils
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix, average_precision_score

import Deploy.Convert as convert

import matplotlib.pyplot as plt

if __name__ == "__main__":
    split_name="ID_Set_split"
    data_yh=dataset.YellowhammerData(split_name, key="train", duration=1.5, augment=augment.augment_distances, transform=None)
    
    X_train, y_train = data_yh.load_audio(binary=False, norm=True)
    #X_train = data_yh.get_features()
    y_train_one = pd.get_dummies(y_train).astype(int).to_numpy()

    print(X_train.shape, y_train_one.shape)

    input_shape=X_train[0].shape
    nb_classes=np.unique(y_train).shape[0]
    gabor_model = yh_models.create_song_gabor_model(input_shape, nb_classes)
    print(gabor_model.summary())

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

    gabor_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.CategoricalCrossentropy(), #losses.CategoricalFocalCrossentropy(),
        metrics=[metrics.F1Score(average='macro')]
    )


    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = gabor_model.fit(x=X_train,y=y_train_one,batch_size=32, validation_split=0.2, shuffle=True, 
                              epochs=25, callbacks=[early_stopping])
    

    folder_models = "Models/"
    name="gabor_conv1d_Feb-02"
    model_path=folder_models+name+'.keras'
    gabor_model.save(model_path)

    new_model = tf.keras.models.load_model(model_path)

    data_test = dataset.YellowhammerData(split_name, key="test", duration=1.5, augment=None, transform=None)
    X_test, y_test = data_test.load_audio(binary=False, norm=True)
    #X_test = data_test.get_features()
    y_test_one = pd.get_dummies(y_test).astype(int).to_numpy()

    print("test: ", X_test.shape, y_test_one.shape)
    
    y_pred_scores = new_model.predict(X_test)
    pred_y_main = np.zeros_like(y_pred_scores)
    pred_y_main[np.arange(len(y_pred_scores)), np.argmax(y_pred_scores, axis=1)] = 1
    acc, f1 = accuracy_score(y_test_one, pred_y_main), f1_score(y_test_one, pred_y_main, average='macro')
    print(acc, f1)

    idx_pos = [i for i in range(len(y_test)) if y_test[i]!=0]
    y_pred_scores = new_model.predict(X_test)[idx_pos]
    pred_y_main = np.zeros_like(y_pred_scores)
    pred_y_main[np.arange(len(y_pred_scores)), np.argmax(y_pred_scores, axis=1)] = 1
    print(accuracy_score(y_test_one[idx_pos], pred_y_main), f1_score(y_test_one[idx_pos], pred_y_main, average='macro'))
    print("AUC", roc_auc_score(y_test_one[idx_pos], pred_y_main))#), auc(y_test_one[idx_pos], pred_y_main))

    print(gabor_model.get_weights()[1])
    _, kernels =train_utils.get_conv1D(gabor_model, idx=1)
    
    squeezed_kernels = np.abs(np.squeeze(kernels))
    binary_prune = (squeezed_kernels <1e-1).astype(int)
    pd.DataFrame(data=binary_prune).to_csv("binary_prune.csv")

    plt.plot(np.squeeze(kernels)[:,:],"-o")
    plt.grid()
    #plt.xlim(40, 50)
    plt.show()