import glob
import os
from re import I
import shutil
from operator import concat
import math

from numpy.core.fromnumeric import nonzero
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import deel
from deel.lip.layers import SpectralConv2D, SpectralDense, FrobeniusDense, InvertibleDownSampling, Condensable
from deel.lip.activations import MaxMin, GroupSort, GroupSort2, FullSort
from deel.lip.utils import load_model
from deel.lip.losses import HKR_loss, KR_loss, hinge_margin_loss
from deel.lip.callbacks import CondenseCallback, MonitorCallback
from deel.lip.model import Sequential
import tensorflow.keras.layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.python.keras.metrics import FalseNegatives
from tensorflow.python.ops.gen_array_ops import fill
from models import OrthoConv2D, ScaledL2NormPooling2D
from numpy import prod
import argparse

from utils import Histogram, AttrDict, GradNormHist, LogitBinaryAccuracy, HKR_binary_accuracy, WeightHist, GradientBag
from models import get_lipschitz_overfitter, get_unconstrained_overfitter
from datasets import prepare_binary_data, binary_train_generator, prepare_all_data, train_generator
from optimizers import preprocess_grads, ArmijoExpensive, ArmijoCheap, ArmijoMetric, Armijo
from losses import MultiClassHKR, CrossEntropyT, MinMarginHKR, MarginWatcher, TopKMarginHKR


config=AttrDict({
    "dataset_name": 'cifar100',
    "loss_type": 'multiminhkr',
    "net_type": 'lip',
    "epochs": 300,  # more epochs
    "batch_size": 300,  # smaller batchs to improve generalization
    "balanced": False,
    "alpha": 500.,
    "margin": 0.01,
    "max_T": 10.,  # big already
    "k_lip": 1.0,
    "optimizer": 'adam',
    "learning_rate": 5.e-4,
    "scale": 128,
    "stiefel": True,
    "condense": True,
    "multihead": None,  # 1 neck
    "scaler": False,
    "deep": True,
    "very_deep": False,
    "target_accuracy": 0.95,
    "random_labels": False,
    })
parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--no-test', dest='test', action='store_false')
parser.add_argument('--name', dest='name', action='store', default=None)
parser.add_argument('--load', dest='load', action='store', default='')
parser.add_argument('--gridsearch', dest='gridsearch', action='store_true')
parser.add_argument('--no-gridsearch', dest='gridsearch', action='store_false')
parser.set_defaults(test=False, gridsearch=False)
args = parser.parse_args()
use_wandb = not args.test
if use_wandb:
    import wandb
    from wandb.keras import WandbCallback
    run = wandb.init(project="supervised_v2", name=args.name, config=config)
tf.keras.backend.set_image_data_format('channels_last')
input_shape = (28, 28, 1) if config.dataset_name == 'mnist' else (32, 32, 3)
binary = config.loss_type in ['hinge', 'hkr', 'bce'] # ['multice', 'multihkr']
if binary:
    output_shape = 1
    class_a = [3]
    class_b = [5]
else:
    if config.dataset_name == 'cifar10':
        output_shape = 10
    elif config.dataset_name == 'cifar100':
        output_shape = 100
eager = True
monitor_grad_norm = False  # speed up computations
monitore_eigenvalues = False
monitore_weights = True
if monitor_grad_norm or 'armijo' in config.optimizer or config.balanced:
    eager = True
if monitor_grad_norm:
    bag = GradientBag()

if config.dataset_name == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif config.dataset_name == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
elif config.dataset_name == 'cifar100':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
if binary:
    x_train, y_train = prepare_binary_data(config, x_train, y_train, input_shape, class_a, class_b)
    x_test, y_test   = prepare_binary_data(config, x_test, y_test, input_shape, class_a, class_b)
    x_train, y_train = binary_train_generator(config, x_train, y_train)
else:
    x_train, y_train = prepare_all_data(config, x_train, y_train, input_shape, random_labels=config.random_labels)
    x_test, y_test   = prepare_all_data(config, x_test, y_test, input_shape)
if input_shape[-1] > 1:
    means = x_train.mean(axis=(0,1,2))
    std = x_train.std(axis=(0,1,2))
    x_train = (x_train - means) / std
    x_test = (x_test - means) / std
print("LOADING OVER", flush=True)

augmenter = tf.keras.Sequential([
    preprocessing.RandomContrast(0.2),
    preprocessing.RandomRotation(0.05),  # = 12° rotation maximum
    preprocessing.RandomTranslation(0.13, 0.13, fill_mode='reflect'),  # nearest better
    preprocessing.RandomFlip('horizontal')
])
augmenter.build((None,)+input_shape)

class CustomSequential(Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def train_step(self, data):
        x, y = data
        x = augmenter(x, training=True)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if config.alpha == 'adaptive':
                self.eager_loss.histogram_update_margins(y, y_pred)
            loss = self.compiled_loss(y, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        if config.stiefel:
            gradients = preprocess_grads(gradients, trainable_vars, retraction=True)
        
        if monitor_grad_norm:
            bag.record_grad_norm(gradients)
        
        if config.optimizer == 'armijo_cheap':
            self.armijo.update_lr(loss, x, y, gradients)

        if config.optimizer == 'armijo':
            self.optimizer.step(self, loss, x, y, gradients)
        else:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if config.optimizer == 'armijo_expensive':
            self.armijo.update_lr(x, y)

        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics

def get_compiled(loss_type, net, optimizer, min_margin):
    if loss_type == 'hkr':
        loss_fn = HKR_loss(alpha=config.alpha,min_margin=min_margin) # HKR stands for the hinge regularized KR loss
        metrics = [KR_loss(),  # shows the KR term of the loss
            hinge_margin_loss(min_margin=min_margin),  # shows the hinge term of the loss
            HKR_binary_accuracy  # shows the classification accuracy
        ]
    elif loss_type == 'hinge':
        loss_fn = hinge_margin_loss(min_margin=min_margin)
        metrics = [Histogram(1),  # shows the KR term of the loss
            hinge_margin_loss(min_margin=min_margin),  # shows the hinge term of the loss
            HKR_binary_accuracy  # shows the classification accuracy
        ]
    elif loss_type == 'bce':
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = [LogitBinaryAccuracy()]
    elif loss_type == 'multihkr':
        margins = tf.constant([1.] * output_shape) * min_margin
        loss_fn = MultiClassHKR(alpha=config.alpha, margins=margins)
        metrics = ['accuracy']
    elif loss_type == 'multice':
        temperatures = tf.constant([1.] * output_shape) * config.max_T
        loss_fn = CrossEntropyT(temperatures=temperatures)
        metrics = ['accuracy']
    elif loss_type == 'multiminhkr':
        margins = tf.constant([1.] * output_shape) * min_margin
        loss_fn = MinMarginHKR(
            alpha=config.alpha,
            margins=margins,
            num_batchs=math.ceil(x_train.shape[0] / config.batch_size),
            perc=config.target_accuracy*100)
        metrics = ['accuracy']
        if config.alpha == 'adaptive':
            metrics.append(MarginWatcher(loss_fn, use_wandb))
    elif loss_type == 'multitopkhkr':
        margins = tf.constant([1.] * output_shape) * min_margin
        loss_fn = TopKMarginHKR(
            alpha=config.alpha,
            margins=margins,
            k=int(output_shape**0.5))
        metrics = ['accuracy']
    if 'armijo' in config.optimizer:
        metrics.append(ArmijoMetric(net.armijo))
    net.compile(loss=loss_fn, metrics=metrics, optimizer=optimizer, run_eagerly=eager)
    net.eager_loss = loss_fn
    return net, loss_fn

def cosine_scheduler(epoch):
    alpha = 0.1
    step = min(epoch, config.epochs)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / config.epochs))
    decayed = (1 - alpha) * cosine_decay + alpha
    return config.learning_rate * decayed

def linear_scheduler(epoch):
    lr = config.learning_rate
    if epoch > config.epochs:
        return lr / 200
    x = [0,
         1*(config.epochs//10),
         3*(config.epochs//5),
         config.epochs]
    y = [lr/10, lr, lr/10, lr/100]
    cur_lr = np.interp([epoch], x, y)[0]
    return cur_lr

def dummy_forward(net):
    _ = net(x_train[0:config.batch_size], training=True)
def dummy_export(net):
    dummy_forward(net)
    vanilla_net = net.vanilla_export()
    dummy_forward(vanilla_net)
    vanilla_net.summary()

def launch_train(net_type, k_coef_lip, min_margin):
    if net_type == 'lip':
        niter_bjorck = 15
        niter_spectral = 3
        net = get_lipschitz_overfitter(CustomSequential, input_shape=input_shape, output_shape=output_shape,
            k_coef_lip=k_coef_lip, scale=config.scale, niter_bjorck=niter_bjorck, niter_spectral=niter_spectral,
            groupsort=True, conv=True, bjorck_forward=True,
            scaler=config.scaler, multihead=config.multihead, deep=config.deep, very_deep=config.very_deep)
    elif net_type == 'notlip':
        net = get_unconstrained_overfitter(input_shape=input_shape, output_shape=output_shape,
                                           k_coef_lip=k_coef_lip, scale=config.scale)
    net.summary()
    dummy_forward(net)  # build network
    if 'rmsprop' in config.optimizer:
        optimizer = RMSprop()
    elif 'adam' in config.optimizer:
        optimizer = Adam()
    elif 'sgd' in config.optimizer:
        optimizer = SGD()
    if 'armijo' in config.optimizer:
        if config.optimizer == 'armijo':
            batch_per_epoch = math.ceil(x_train.shape[0] / config.batch_size)
            optimizer = Armijo(batch_per_epoch, c=0.2, condense=True,
                               polyak_momentum=0.9)
            net.armijo = optimizer
        else:
            if config.optimizer == 'armijo_expensive':
                armijo = ArmijoExpensive(config.batch_size, len(x_train))
            elif config.optimizer == 'armijo_cheap':  # seems bad because no backprop
                # without independance the LR might grow incontrollably
                armijo = ArmijoCheap(config.batch_size, len(x_train),
                                    force_independance=False,  # avoid step size too big
                                    weight_latency=False)  # use current weight to evaluate gradient
            optimizer = SGD(learning_rate=armijo) # momentum=0.6 to speed up convergence
            armijo.update_model(net)
            net.armijo = armijo
    net, _ = get_compiled(config.loss_type, net, optimizer, min_margin)
    if args.load != '':
        weights_file = wandb.restore('weights.h5', run_path='algue/supervised/'+args.load)
        net.load_weights(weights_file.name)
    spec_names = [layer.name for layer in net.layers if 'spectral_dense' in layer.name]
    name = wandb.run.name if use_wandb else 'supervised'
    callbacks = [tf.keras.callbacks.History(),
                 tf.keras.callbacks.ModelCheckpoint(f'checkpoints/{name}.h5', monitor='loss',
                                                    mode='min', save_freq='epoch')]
    if 'armijo' not in config.optimizer and 'noscheduler' not in config.optimizer:
        # callbacks.append(tf.keras.callbacks.LearningRateScheduler(linear_scheduler))
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(cosine_scheduler))
    if (config.stiefel or config.condense) and config.optimizer != 'armijo':
        callbacks.append(CondenseCallback(on_batch=True))
    if monitor_grad_norm:
        callbacks.append(GradNormHist(bag, use_wandb))
    if monitore_weights:
        callbacks.append(WeightHist(use_wandb))
    if monitore_eigenvalues:
        shutil.rmtree('boards/metrics', ignore_errors=True)
        callbacks.append(MonitorCallback(on_epoch=True, on_batch=False,
                         monitored_layers=spec_names, logdir='boards', what='max'))
    if use_wandb:
        callbacks.append(WandbCallback(save_weights_only=True))
    if config.balanced:
        buffer_size = int(config.batch_size / output_shape**0.5)
        buffer_size = max(buffer_size, 1)
        x = train_generator(x_train, y_train,
                            batch_size=config.batch_size,
                            buffer_size=buffer_size)
        y = None
        batch_size = None
    else:
        x, y, batch_size = x_train, y_train, config.batch_size
    last_acc = 0.
    last_epoch, step_epoch = 0, config.epochs
    while last_acc < config.target_accuracy:
        history = net.fit(
            x=x, y=y,
            validation_data=(x_test, y_test),
            initial_epoch=last_epoch,
            epochs=last_epoch+step_epoch,
            verbose=1,
            callbacks=callbacks,
            batch_size=batch_size
        )
        last_epoch = last_epoch+step_epoch
        step_epoch = 50
        last_acc = float(np.array(history.history['accuracy'][-5:]).mean())
    return history, net

if not args.gridsearch:
    _, net = launch_train(config.net_type, config.k_lip, config.margin)
    import invertible
    invertible.plot_moments(use_wandb, config, net, x_test)  # plot moments
else:
    import plotting
    plotting.plot(config, launch_train, binary, training=True)
