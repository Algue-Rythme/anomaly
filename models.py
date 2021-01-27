import deel
import tensorflow as tf
from deel.lip.initializers import BjorckInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D
from deel.lip.model import Sequential
from deel.lip.activations import PReLUlip, GroupSort, FullSort
from tensorflow.keras.layers import Input, Lambda, Flatten, MaxPool2D, BatchNormalization


def get_cnn_baseline(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SpectralConv2D(
            filters=32, kernel_size=(3, 3), padding='same',
            activation=PReLUlip(), data_format='channels_last',
            kernel_initializer=BjorckInitializer(15, 50)),
        SpectralConv2D(
            filters=32, kernel_size=(3, 3), padding='same',
            activation=PReLUlip(), data_format='channels_last',
            kernel_initializer=BjorckInitializer(15, 50)),
        MaxPool2D(pool_size=(2, 2), data_format='channels_last'),
        SpectralConv2D(
            filters=64, kernel_size=(3, 3), padding='same',
            activation=PReLUlip(), data_format='channels_last',
            kernel_initializer=BjorckInitializer(15, 50)),
        SpectralConv2D(
            filters=64, kernel_size=(3, 3), padding='same',
            activation=PReLUlip(), data_format='channels_last',
            kernel_initializer=BjorckInitializer(15, 50)),
        MaxPool2D(pool_size=(2, 2), data_format='channels_last'),
        Flatten(),
        SpectralDense(256, activation="relu", kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(1),
    ], k_coef_lip=1., name='baseline')
    return model


def get_fast_baseline(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SpectralDense(32, activation=GroupSort(n=2), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(16, activation=GroupSort(n=2), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(16, activation=GroupSort(n=2), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(1),
    ], k_coef_lip=1., name='baseline')
    return model


def get_mlp_baseline(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SpectralDense(128, activation=FullSort(), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(64, activation=FullSort(), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(32, activation=FullSort(), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(1),
    ], k_coef_lip=1., name='baseline')
    return model


class ShiftLayer(tf.keras.layers.Layer, deel.lip.layers.LipschitzLayer):
    def __init__(self):
        super(ShiftLayer, self).__init__()

    def _compute_lip_coef(self):
        return 1.

    def build(self, input_shape):
        self.shift = self.add_weight(shape=input_shape[1:], initializer="zeros", trainable=True)
    
    def call(self, x):
        return x + self.shift

class FrozenBatchNorm(tf.keras.layers.Layer, deel.lip.layers.LipschitzLayer):
    def __init__(self):
        super(FrozenBatchNorm, self).__init__()
        self.bn = BatchNormalization(center=True, scale=False, trainable=True)

    def _compute_lip_coef(self):
        return 1.

    def call(self, x):
        return self.bn(x)


def get_mlp_no_bias(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        FrozenBatchNorm(),
        SpectralDense(128, activation=FullSort(), use_bias=True, kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(64, activation=FullSort(), use_bias=True, kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(32, activation=FullSort(), use_bias=True, kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(1),
    ], k_coef_lip=1., name='baseline')
    return model
