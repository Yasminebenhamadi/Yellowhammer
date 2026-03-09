import numpy as np
import tensorflow as tf
#from keras.saving import saving_lib
import tensorflow.keras as keras


# taken from https://medium.com/the-owl/weighted-binary-cross-entropy-losses-in-keras-e3553e28b8db

def weighted_binary_crossentropy(target, output, weights):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output + epsilon_)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return -bce

class WeightedBinaryCrossentropy:
    def __init__(
        self,
        label_smoothing=0.0,
        weights = [1.0, 1.0],
        axis=-1,
        name="weighted_binary_crossentropy",
        fn = None,
    ):
        """Initializes `WeightedBinaryCrossentropy` instance.

        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` contains probabilities (i.e., values in [0,
            1]).
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When >
            0, we compute the loss between the predicted labels and a smoothed
            version of the true labels, where the smoothing squeezes the labels
            towards 0.5.  Larger values of `label_smoothing` correspond to
            heavier smoothing.
          axis: The axis along which to compute crossentropy (the features
            axis).  Defaults to -1.
          name: Name for the op. Defaults to 'weighted_binary_crossentropy'.
        """
        super().__init__()
        self.weights = weights # tf.convert_to_tensor(weights)
        self.label_smoothing = label_smoothing
        self.name = name
        self.fn = weighted_binary_crossentropy if fn is None else fn

    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        self.label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        def _smooth_labels():
            return y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        y_true = tf.__internal__.smart_cond.smart_cond(self.label_smoothing, _smooth_labels, lambda: y_true)

        return tf.reduce_mean(self.fn(y_true, y_pred, self.weights),axis=-1)
    
    def get_config(self):
        config = {"name": self.name, "weights": self.weights, "fn": self.fn}

        # base_config = super().get_config()
        return dict(list(config.items()))
    

def get_conv1D(model, idx, fs=20480):
    center_hz, bandwidth = model.layers[idx].get_weights()
    kernel_size, stride, padding = model.layers[1].kernel_size, model.layers[1].stride, model.layers[1].padding
    out_channels = len(center_hz)
    
    # time grid
    t = tf.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    t = tf.cast(t, tf.float32) / fs
    t = tf.reshape(t, [kernel_size, 1])  # [T,1]
    
    # broadcast params
    f0 = tf.reshape(center_hz, [1, out_channels])   # center frequency
    bw = tf.reshape(bandwidth, [1, out_channels])   # bandwidth
    
    # Gabor = sinusoid * Gaussian window
    pi = tf.constant(3.141592653589793, tf.float32)
    sinusoid = tf.cos(2 * pi * f0 * t)
    gaussian = tf.exp(- (pi * bw * t) ** 2)
    
    kernels = sinusoid * gaussian  # [T, C]
    
    # Conv1D kernel shape
    kernels = tf.reshape(kernels, [kernel_size, 1, out_channels]).numpy()
    
    kernels[np.abs(kernels)<0.1]=0
    
    conv1d = keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding=padding,
                           use_bias=False,
                           kernel_initializer=tf.constant_initializer(kernels), name="conv1D_gabor")
    return conv1d, kernels


@tf.keras.utils.register_keras_serializable()
class FakeQuantLayer(tf.keras.layers.Layer):
    def __init__(self, num_bits=16, min_val=-1.0, max_val=1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.num_bits = num_bits

    def call(self, x):
        # simulate int16 (16-bit symmetric)
        return tf.quantization.fake_quant_with_min_max_vars(
            x, min=self.min_val, max=self.max_val, num_bits=self.num_bits
        )
    

def get_new_layer(old_layer, shape):
    # Get the config and recreate it
    new_layer = type(old_layer).from_config(old_layer.get_config())

    # Copy the weights
    new_layer.build(shape)  # Build layer with correct input shape
    new_layer.set_weights(old_layer.get_weights())
    return new_layer

def get_activation_model(model, layer_names):
    outputs = [model.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=model.input, outputs=outputs)
