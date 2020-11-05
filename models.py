from deel.lip.initializers import BjorckInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D
from deel.lip.model import Sequential
from deel.lip.activations import PReLUlip, GroupSort
from tensorflow.keras.layers import Input, Lambda, Flatten, MaxPool2D


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


def get_mlp_baseline(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SpectralDense(128, activation=GroupSort(n=4), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(64, activation=GroupSort(n=4), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(32, activation=GroupSort(n=4), kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(1),
    ], k_coef_lip=1., name='baseline')
    return model