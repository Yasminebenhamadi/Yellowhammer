import librosa
import numpy as np
import tensorflow as tf

class SincConv1D(tf.keras.layers.Layer):
    # Generated using ChatGPT
    def __init__(self, out_channels, kernel_size, fs, stride=1, padding="SAME", **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.fs = float(fs)
        self.stride = int(stride)
        self.padding = padding

    def build(self, input_shape):
        self.low_hz = self.add_weight(
            shape=(self.out_channels,),
            initializer=tf.keras.initializers.RandomUniform(minval=50, maxval=1000),
            trainable=True,
            dtype=tf.float32
        )
        self.high_hz = self.add_weight(
            shape=(self.out_channels,),
            initializer=tf.keras.initializers.RandomUniform(minval=2000, maxval=8000),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, x):
        t = tf.linspace(-(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size)
        t = tf.cast(t, tf.float32) / self.fs
        t = tf.reshape(t, [self.kernel_size, 1])

        f1 = tf.reshape(self.low_hz, [1, self.out_channels])
        f2 = tf.reshape(self.high_hz, [1, self.out_channels])

        pi_t = tf.constant(3.14159265359, dtype=tf.float32) * t
        h_low = 2*f1*tf.sin(2*pi_t*f1)/(2*pi_t*f1 + 1e-8)
        h_high = 2*f2*tf.sin(2*pi_t*f2)/(2*pi_t*f2 + 1e-8)
        kernels = h_high - h_low

        kernels = tf.reshape(kernels, [self.kernel_size, 1, self.out_channels])

        return tf.nn.conv1d(x, kernels, stride=self.stride, padding=self.padding)
    


class MinValue(tf.keras.constraints.Constraint):
    """Constrains weights to be >= given min_value (element-wise)."""
    def __init__(self, min_value=0.0):
        self.min_value = min_value

    def __call__(self, w):
        # Clip only lower bound; no upper bound
        return tf.clip_by_value(w, clip_value_min=self.min_value,
                                   clip_value_max=tf.float32.max)

    def get_config(self):
        return {"min_value": self.min_value}


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)
    
@tf.keras.utils.register_keras_serializable()
class GaborConv1D(tf.keras.layers.Layer):
    # Generated using ChatGPT
    def __init__(self, out_channels, kernel_size, fs, stride=1, padding="SAME", band_min=800.0, band_max=1500.0, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.fs = float(fs)
        self.stride = int(stride)
        self.padding = padding
        self.band_min = band_min
        self.band_max = band_max

    def build(self, input_shape):
        # center frequency and bandwidth params (learnable)
        mel_freqs, bandwidths = self.init_from_mels(fmin=3000, fmax=10000)
        self.center_hz = self.add_weight(
            shape=(self.out_channels,),
            initializer=tf.keras.initializers.Constant(mel_freqs),
            #initializer=tf.keras.initializers.RandomUniform(minval=2000, maxval=self.fs/2),
            trainable=True,
            dtype=tf.float32,
        )
        self.bandwidth = self.add_weight(
            shape=(self.out_channels,),
            #initializer=tf.keras.initializers.Constant(bandwidths),
            initializer=tf.keras.initializers.RandomUniform(minval=50, maxval=1000),
            trainable=True,
            dtype=tf.float32,
            constraint=ClipConstraint(self.band_min, self.band_max)
        )

    def init_from_mels(self, fmin=2000, fmax=10000):
        mel_edges = librosa.mel_frequencies(
            self.out_channels + 2, fmin=fmin, fmax=fmax
        )

        mel_freqs = mel_edges[1:-1]                 # centers (length = out_channels)
        bandwidths = mel_edges[2:] - mel_edges[:-2] # length = out_channels
        return mel_freqs, bandwidths

    def call(self, x):
        # time grid
        t = tf.linspace(-(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size)
        t = tf.cast(t, tf.float32) / self.fs
        t = tf.reshape(t, [self.kernel_size, 1])  # [T,1]

        # broadcast params
        f0 = tf.reshape(self.center_hz, [1, self.out_channels])   # center frequency
        bw = tf.reshape(self.bandwidth, [1, self.out_channels])   # bandwidth

        # Gabor = sinusoid * Gaussian window
        pi = tf.constant(3.141592653589793, tf.float32)
        sinusoid = tf.cos(2 * pi * f0 * t)
        gaussian = tf.exp(- (pi * bw * t) ** 2)

        kernels = sinusoid * gaussian  # [T, C]

        # Conv1D kernel shape
        kernels = tf.reshape(kernels, [self.kernel_size, 1, self.out_channels])

        return tf.nn.conv1d(x, kernels, stride=self.stride, padding=self.padding)

@tf.keras.utils.register_keras_serializable()
class LogLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.log(tf.abs(x) + 1e-6)