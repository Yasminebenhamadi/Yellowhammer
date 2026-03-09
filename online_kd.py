import numpy as np
import pandas as pd
import tensorflow as tf
import data_utils.augment as augment
import data_utils.dataset as dataset
import frontends.features as features
import eval_utils.evaluate as eval
import Training.classifiers as yh_models
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import auc, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix, average_precision_score


ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

def logit_kd_loss(student_logits, teacher_logits, T=4.0):
    student_soft = tf.nn.softmax(student_logits / T)
    teacher_soft = tf.nn.softmax(teacher_logits / T)
    return tf.reduce_mean(
        tf.reduce_sum(
            teacher_soft * (tf.math.log(teacher_soft + 1e-9) -
                            tf.math.log(student_soft + 1e-9)), axis=1
        )
    ) * (T * T)

def compute_attention_map(F):
    """
    F: (N, L, C) for 1D conv features
    returns A: (N, L)
    """

    # squared activation across channels
    A = tf.reduce_sum(tf.square(F), axis=-1)   # (N, L)

    # flatten over spatial dimension
    A = tf.reshape(A, [tf.shape(A)[0], -1])    # (N, L)

    # normalize each sample
    A = tf.nn.l2_normalize(A, axis=-1)

    return A

def resize_attention_map(A, target_len):
    """
    A: (N, L)
    """

    A = tf.expand_dims(A, axis=2)  # (N, L, 1)

    A = tf.image.resize(
        A,
        size=(A.shape[0], target_len),
        method="bilinear"
    )

    return tf.squeeze(A, axis=2)   # (N, target_len)

def attention_transfer_loss(f_s, f_t):
    """
    f_s: student feature map  (N, Ls, C_s)
    f_t: teacher feature map  (N, Lt, C_t)
    """

    # compute attention maps
    A_s = compute_attention_map(f_s)   # (N, Ls)
    A_t = compute_attention_map(f_t)   # (N, Lt)

    # match temporal length
    A_t = resize_attention_map(A_t, tf.shape(A_s)[1])

    # L2 difference of normalized attention maps
    return tf.reduce_mean(tf.reduce_sum(tf.square(A_s - A_t), axis=-1))

def distillation_loss(
    y_true,
    logits_s,
    logits_t,
    feats_s,
    feats_t,
    alpha=0.5,
    beta=0
):
    # supervised
    sup = ce_loss_fn(y_true, logits_s)

    # logit KD
    kd_logits = logit_kd_loss(logits_s, logits_t)

    # intermediate feature KD (sum over selected layers)
    kd_feats = attention_transfer_loss(feats_s, feats_t) #feature_kd_loss(feats_s, feats_t)
    return sup + alpha * kd_logits + beta * kd_feats

opt_student = tf.keras.optimizers.Adam(1e-3)
opt_teacher = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(student, teacher, x_s, x_t, y):
    with tf.GradientTape(persistent=True) as tape:
        out_s = student(x_s, training=True)
        out_t = teacher(x_t, training=True)
        logits_s = out_s['output_main']
        logits_t = out_t['output_main']
        f1s = out_s['conv1d_raw']
        f1t = out_t['conv1d_1']
        # two losses — both OK now
        loss_A = distillation_loss(y, logits_s, logits_t, f1s, f1t, beta=0, alpha=0)
        loss_B = distillation_loss(y, logits_t, logits_s, f1t, f1s, beta=0, alpha=0)
    grads_A = tape.gradient(loss_A, student.trainable_variables)
    grads_B = tape.gradient(loss_B, teacher.trainable_variables)
    opt_student.apply_gradients(zip(grads_A, student.trainable_variables))
    opt_teacher.apply_gradients(zip(grads_B, teacher.trainable_variables))
    return loss_A, loss_B

if __name__ == "__main__":
    split_name="ID_Set_split"
    data_yh=dataset.YellowhammerData(split_name, key="train", duration=1.5, augment=augment.augment_distances, transform=features.mel_preprocess)
    
    X_train, y_train = data_yh.get_data()
    X_train_mel = data_yh.get_features()
    y_train_one = pd.get_dummies(y_train).astype(int).to_numpy()

    print(X_train.shape, X_train_mel.shape, y_train_one.shape)

    input_shape=X_train[0].shape
    input_mel_shape=X_train_mel[0].shape
    nb_classes=np.unique(y_train).shape[0]


    student = yh_models.student_raw(input_shape, nb_classes)
    teacher = yh_models.student_mel(input_mel_shape, nb_classes)

    print(student.summary())
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train_mel, y_train_one))
    train_ds = train_ds.shuffle(50000).batch(32).prefetch(tf.data.AUTOTUNE)

    for epoch in range(20):
        for x_s_batch, x_t_batch, y_batch in train_ds:
            loss_A, loss_B = train_step(student, teacher, x_s_batch, x_t_batch, y_batch)
        print(epoch, float(loss_A), float(loss_B))
    
    student.trainable=False
    teacher.trainable=False

    data_test = dataset.YellowhammerData(split_name, key="test", duration=1.5, augment=None, transform=features.mel_preprocess)
    X_test, y_test = data_test.get_data()
    X_test_mel = data_test.get_features()
    y_test_one = pd.get_dummies(y_test).astype(int).to_numpy()

    print(X_test.shape, X_test_mel.shape, y_test_one.shape)

    y_pred_scores = student.predict(X_test).get("output_main")
    pred_y_main = np.zeros_like(y_pred_scores)
    pred_y_main[np.arange(len(y_pred_scores)), np.argmax(y_pred_scores, axis=1)] = 1
    print("student:", accuracy_score(y_test_one, pred_y_main), f1_score(y_test_one, pred_y_main, average='macro'))


    y_pred_scores = teacher.predict(X_test_mel).get("output_main")
    pred_y_main = np.zeros_like(y_pred_scores)
    pred_y_main[np.arange(len(y_pred_scores)), np.argmax(y_pred_scores, axis=1)] = 1
    print("teacher:", accuracy_score(y_test_one, pred_y_main), f1_score(y_test_one, pred_y_main, average='macro'))
