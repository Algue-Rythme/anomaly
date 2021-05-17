import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import wandb
import scipy


def tf_dataset(num_batchs, batch_size, sk_func):
    x, y    = sk_func(num_batchs * batch_size)
    x, y    = tf.constant(x, dtype=tf.float32), tf.constant(y, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    return dataset

def dilated_func(f, coef):
    def decorated(*args,**kwargs):
        x, y = f(*args, **kwargs)
        x = x - np.mean(x, axis=0, keepdims=True) # + np.array([[2., 2.]])  # 
        return x * coef, y
    return decorated

def seed_dispatcher(seed=None):
    if seed is None:
        seed = random.randint(1, 1000 * 1000)
    np.random.seed(seed * 2)
    tf.random.set_seed(seed * 3)
    print('Seed used: %d'%seed)

def exp_avg(x_avg, x, m):
    if x_avg is None:
        return x
    return x_avg*m + x*(1-m)

def projected(f, axis):
    def decorated(*args,**kwargs):
        X, y = f(*args, **kwargs)
        return np.expand_dims(X.dot(np.array(axis)), 1), y
    return decorated

class Histogram(tf.keras.metrics.Metric):
    def __init__(self, start_epoch, **kwargs):
        super(Histogram, self).__init__(name='bestacc', **kwargs)
        self.hist_true = None; self.hist_pred = None
        self.epoch = start_epoch
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.hist_pred is None:
            self.hist_pred = tf.squeeze(y_pred)
            self.hist_true = tf.squeeze(y_true)
        else:
            self.hist_pred = tf.concat([self.hist_pred,tf.squeeze(y_pred)],axis=0)
            self.hist_true = tf.concat([self.hist_true,tf.squeeze(y_true)],axis=0)
    def result(self):
        indexes = tf.argsort(self.hist_pred)
        labels = tf.gather(self.hist_true, indexes)
        cumlabels = tf.cumsum(labels, reverse=True)
        cumrlabels = tf.cumsum(1. - labels)
        ok_pred = cumlabels + cumrlabels
        accs = ok_pred / tf.cast(tf.size(labels), dtype=tf.float32)
        max_acc = tf.reduce_max(accs)
        return max_acc
    def reset_states(self):
        self.hist_true = None
        self.hist_pred = None
        self.epoch += 1
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GradientBag():
    def __init__(self):
        self.grad_norms_hist = []
        self.indexes = []
    def record_grad_norm(self, grads):
        for i, grad in enumerate(grads):
            norm = (tf.reduce_sum(grad * grad) ** 0.5)
            if len(self.grad_norms_hist) < i+1:
                self.grad_norms_hist.append([])
            self.grad_norms_hist[i].append(norm)
    def reset(self):
        self.grad_norms_hist = []
    def reset(self):
        self.grad_norms_hist = []

class GradNormHist(tf.keras.callbacks.Callback):
    def __init__(self, bag, use_wandb, **kwargs):
        super().__init__()
        self.bag = bag
        self.use_wandb = use_wandb
        if not use_wandb:
            files = glob.glob('supervised_hists/*')
            for f in files:
                os.remove(f)
    def plot_with_wandb(self, epoch, names):
        grads = {}
        for i, grad_norm in enumerate(self.bag.grad_norms_hist):
            grad_norm = tf.stack(grad_norm).numpy()
            grads[f"gradients_{names[i]}"] = wandb.Histogram(grad_norm)
        wandb.log(grads, commit=False)  # wandb callback ensures the commit later
    def plot_with_plt(self, epoch, names):
        s = 3
        from math import ceil
        _, axes = plt.subplots(nrows=s, ncols=ceil(len(self.bag.grad_norms_hist) / s), figsize=(24, 16))
        axes = [col for row in axes for col in row]
        for i, grad_norm in enumerate(self.bag.grad_norms_hist):
            grad_norm = tf.stack(grad_norm).numpy()
            axes[i].set_xlim(0., float(grad_norm.max())*1.2)
            axes[i].set_xlabel(names[i])
            sns.histplot(data=grad_norm.flatten(), bins=16, ax=axes[i])
        plt.savefig(f'supervised_hists/grad_{int(epoch+1)}.png')
        plt.close()
    def on_epoch_end(self, epoch, logs=None):
        names = [var.name for var in self.model.trainable_variables]
        if self.use_wandb:
            self.plot_with_wandb(epoch, names)
        else:
            self.plot_with_plt(epoch, names)
        self.bag.reset()


class WeightHist(tf.keras.callbacks.Callback):
    def __init__(self, use_wandb, **kwargs):
        super().__init__()
        self.use_wandb = use_wandb
        if not use_wandb:
            files = glob.glob('supervised_hists/*')
            for f in files:
                os.remove(f)
    def plot_with_wandb(self, epoch, names, weights, sigmas):
        to_plot_weights = {}
        for name, weight in zip(names, weights):
            weight = weight.numpy()
            to_plot_weights[f"weights_{name}"] = wandb.Histogram(weight)
            if 'kernel' in name:
                try:
                    weight = weight.reshape((-1,weight.shape[-1]))  # ensure square matrix with rank 2
                    eigenvalues = scipy.linalg.svdvals(weight)  # biggest singular value
                    if 'spectral' in name:
                        prefix = '/'.join(name.split('/')[:-1])  # whole layer name, drop Variable's name
                        sigma_name = prefix + '/sigma:0'
                        sigma = sigmas[sigma_name]  # retrieve the sigma associated
                        ratios = eigenvalues / sigma.numpy()
                        to_plot_weights[f"singular_ratio_{name}"] = wandb.Histogram(ratios)
                    else:
                        to_plot_weights[f"singular_{name}"] = wandb.Histogram(eigenvalues)
                except np.linalg.LinAlgError:
                    print('WARNING: np.eigvals did not converge')
        wandb.log(to_plot_weights, commit=False)  # wandb callback ensures the commit later
    def plot_with_plt(self, epoch, names, weights):
        s = 3
        from math import ceil
        _, axes = plt.subplots(nrows=s, ncols=ceil(len(weights) / s), figsize=(24, 16))
        axes = [col for row in axes for col in row]
        for i, (name, weight) in enumerate(zip(names, weights)):
            weight = weight.numpy()
            axes[i].set_xlim(0., float(weight.max())*1.2)
            axes[i].set_xlabel(name)
            sns.histplot(data=weight.flatten(), bins=16, ax=axes[i])
        plt.savefig(f'supervised_hists/weight_{int(epoch+1)}.png')
        plt.close()
    def on_epoch_end(self, epoch, logs=None):
        names = [var.name for var in self.model.trainable_variables]
        weights = [weight for weight in self.model.trainable_variables]
        sigmas = {weight.name:weight for weight in self.model.non_trainable_variables if 'sigma' in weight.name}
        if self.use_wandb:
            self.plot_with_wandb(epoch, names, weights, sigmas)
        else:
            self.plot_with_plt(epoch, names, weights)


class LogitBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, **kwarg):
        super().__init__(name='accuracy')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)
 
def HKR_binary_accuracy(y_true, y_pred):
    S_true= tf.dtypes.cast(tf.greater_equal(y_true[:,0], 0),dtype=tf.float32)
    S_pred= tf.dtypes.cast(tf.greater_equal(y_pred[:,0], 0),dtype=tf.float32)
    return tf.keras.metrics.binary_accuracy(S_true,S_pred)


class OverFittingChart(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
    def update_data(self, logs):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.accs.append(logs['acc'])
        self.val_accs.append(logs['val_acc'])
    def on_epoch_end(self, epochs, logs):
        self.update_data(logs)
        data_loss = list(zip(self.losses, self.val_losses))
        data_acc = list(zip(self.accs, self.val_accs))
        table_loss = wandb.Table(data=data_loss, columns=["loss", "val_loss"])
        table_acc = wandb.Table(data=data_acc, columns=["acc", "val_acc"])
        # wandb.log(plot_loss, commit=False, step=None)
        # wandb.log(plot_acc, commit=False, step=None)
