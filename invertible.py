from operator import concat
from numpy.core.fromnumeric import nonzero
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

import numpy as np

import deel
from deel.lip.layers import SpectralConv2D, SpectralDense, FrobeniusDense
from deel.lip.activations import MaxMin, GroupSort, GroupSort2, FullSort
from deel.lip.utils import load_model
from deel.lip.losses import HKR_loss, KR_loss, hinge_margin_loss
from deel.lip.callbacks import CondenseCallback, MonitorCallback
from deel.lip.model import Sequential
import tensorflow.keras.layers
from tensorflow.keras.datasets import mnist, cifar10
from models import OrthoConv2D, ScaledL2NormPooling2D
import scipy
import wandb


class FullSortMoments:
    def __init__(self, normalize_rank=False):
        self.means = None
        self.cov = None
        self.examples = 0
        self.normalize_rank = normalize_rank
    def update(self, model, x):
        batchs = self._batch_examples(model, x)
        if self.means is None:
            self.means = [tf.zeros((1,batch.shape[1])) for batch in batchs]
            self.cov = [tf.zeros((mean.shape[-1],mean.shape[-1]),dtype=tf.float32) for mean in self.means]
        self._update_online_means(batchs)
    def _batch_examples(self, model, x):
        batchs = []
        previous = ''
        for layer in model.layers:
            if 'sort' in layer.name and 'spectral_dense' in previous:
                batchs.append(tf.argsort(x, axis=-1, direction='ASCENDING'))
            previous = layer.name
            x = layer(x)
        return batchs
    def fit(self):
        pass
    def _update_online_means(self, batchs):
        for i, batch in enumerate(batchs):
            batch = tf.cast(batch, dtype=tf.float32)
            if self.normalize_rank:
                batch = batch / (batch.shape[-1]-1)
            self.examples += batch.shape[0]
            dx = batch - self.means[i]
            dy = dx
            sdx = tf.reduce_sum(dx, axis=0, keepdims=True)
            self.means[i] = self.means[i] + sdx / self.examples
            dx = batch - self.means[i]
            dcx = tf.matmul(dx, dy, transpose_a=True)
            dcx = dcx - self.cov[i]
            self.cov[i] = self.cov[i] + dcx / self.examples

class GroupSortMoments:
    def __init__(self):
        self.means = None
        self.cov = None
        self.examples = 0
        self.names = []
    def fit(self):
        for mean_idx in range(len(self.means)):
            mean = self.means[mean_idx].flatten()
            index = np.argsort(mean)
            self.means[mean_idx] = mean[index]
            cov = self.cov[mean_idx]
            if cov.shape[0] > 1:
                self.cov[mean_idx] = cov[index][:,index]
            else:
                self.cov[mean_idx] = mean * (1 - mean)  # bernouilli variance
    def update(self, model, x, y):
        assert int(np.prod(x.shape[1:]))%2 == 0
        batchs = self._batch_examples(model, x)
        if self.means is None:
            self.means = [np.zeros((1, batch.shape[1]),dtype=np.float64) for batch in batchs]
            covs = []
            for mean in self.means:
                size = mean.shape[-1]
                if size <= 256:
                    cov = np.zeros((size,size),dtype=np.float64)
                else:
                    cov = np.zeros((1,size),dtype=np.float64)
                covs.append(cov)
            self.cov = covs
        self._update_online_means(batchs)
    def arggroupsort(self, x):
        batch_size = x.shape[0]
        fv = tf.reshape(x, [-1, 2])
        indexes = tf.argmin(fv, axis=1)
        indexes = tf.reshape(indexes, [batch_size, -1])
        return indexes.numpy()
    def _batch_examples(self, model, x):
        batchs = []
        previous = ''
        for layer in model.layers:
            if 'sort' in layer.name:
                self.names.append(layer.name)
                batchs.append(self.arggroupsort(x))
            previous = layer.name
            x = layer(x)
        return batchs # divided by two already
    def _update_online_means(self, batchs):
        self.examples += batchs[0].shape[0]
        for i, batch in enumerate(batchs):
            batch = batch.astype(np.float64)
            dx = batch - self.means[i]
            sdx = dx.sum(axis=0, keepdims=True)
            self.means[i] = self.means[i] + sdx / self.examples
            if self.cov[i].shape[0] > 1:
                dy = dx
                dx = batch - self.means[i]
                dcx = dx.T @ dy
                dcx = dcx - self.cov[i]
                self.cov[i] = self.cov[i] + dcx / self.examples


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
class FullSortGP:
    def __init__(self, train_size, y_target):
        self.gps = None
        self.xs = None
        self.means = []
        self.cov = []
        self.train_size = train_size
        self.y_target = y_target
    def update(self, model, x, y):
        if self.y_target != y and False:  #TODO: handle batch of examples
            return
        batchs = self._batch_examples(model, x)
        if self.gps is None:
            self.gps = [GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds=(0.1,5)) * RBF(0.05, length_scale_bounds=(0.1,1.))
                #kernel=ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(0.05, length_scale_bounds="fixed")
            ) for _ in range(len(batchs))]
            self.xs = [[] for _ in range(len(batchs))]
        for i, batch in enumerate(batchs):
            self.xs[i].append(batch.numpy())
    def _batch_examples(self, model, x):
        batchs = []
        previous = ''
        for layer in model.layers:
            if 'full_sort' in layer.name and 'spectral_dense' in previous:
                batchs.append(tf.argsort(x, axis=-1, direction='ASCENDING'))
            x = layer(x)
            previous = layer.name
        return batchs
    def fit(self):
        for x, gp in zip(self.xs, self.gps):
            num_neurons = x[0].shape[-1]
            x = np.concatenate([xx.ravel() for xx in x], axis=0)
            seq = np.linspace(0,1.,num_neurons)
            y = np.repeat(seq, len(x)/num_neurons, axis=0)
            indexes = np.random.permutation(len(x))
            x = x[indexes]
            y = y[indexes]
            x = x[:self.train_size].reshape(-1, 1)
            y = y[:self.train_size].reshape(-1, 1)
            gp.fit(y, x)  # assume coordinates are independant (weird ?)
            means, cov = gp.predict(seq[:,np.newaxis], return_cov=True)
            # cov = np.diag(cov)
            self.means.append(means)
            self.cov.append(cov)


def plot_moments(use_wandb, config, model, x_test):
    vanilla = model.vanilla_export()
    moments = GroupSortMoments() #FullSortMoments(normalize_rank=True)
    from tqdm import tqdm
    for x in tqdm(tf.data.Dataset.from_tensor_slices(x_test).batch(200)):
        moments.update(vanilla, x, None)
    moments.fit()
    if use_wandb:
        from math import ceil
        numcols = ceil(len(moments.means) / 2)
        numrows = 2
    else:
        numcols = len(moments.means)
        numrows = 2
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(numrows, numcols, figsize=(21, 7))
    axs = axs.reshape((numrows, numcols))
    if not use_wandb:
        for i, img in enumerate(moments.cov):
            if len(img.shape) > 1:
                axs[1,i].imshow(img)
    entropies = []
    for i, (y, var) in enumerate(zip(moments.means, moments.cov)):
        y = np.squeeze(y)
        entropies.append(y)
        if len(var.shape) > 1:
            var = tf.sqrt(tf.linalg.diag_part(var))
        else:
            var = tf.sqrt(var)
        x = np.linspace(0., len(y), len(y))
        if i >= numcols:
            row, col = 1, i-numcols
        else:
            row, col = 0, i
        axs[row, col].plot(x, y, c='red')
        axs[row, col].fill_between(x, y+var, y-var, color='red', alpha=0.15)
        axs[row, col].set_ylim(0., 1.)
        axs[row, col].set_title(moments.names[i])
        # axs[0,i].set_aspect('equal') bug for some reason
    probs = np.concatenate(entropies, axis=0)
    entropy = -(probs*np.log2(probs) + (1-probs)*np.log2(1-probs))
    entropy = np.nan_to_num(entropy, nan=0.)
    entropy = float(np.sum(entropy))  # number of bits of information required
    print(f"Entropy = {int(entropy/1000)} kBits" )
    if use_wandb:
        wandb.log({"groupsort": [wandb.Image(plt)], "entropy_bits": entropy})
    else:
        plt.savefig(f'images/groupsort_{config.dataset_name}_{config.loss_type}_{config.net_type}.png')
        plt.show()

