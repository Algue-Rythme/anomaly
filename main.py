import random

import matplotlib
# matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as activations
from sklearn.datasets import make_moons, make_circles
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

import models
import naive_adversarial
import plotex
from gradient_tools import get_grad_norm_with_tape, oneclass_hinge
from utils import exp_avg, tf_dataset, dilated_func, seed_dispatcher


class PartialModel:

    def __init__(self, model, layer_index):
        self.model = model
        self.layer_index = layer_index

    def __call__(self, x):
        for layer in self.model.layers[:self.layer_index]:
            x = layer(x)
        return x


# @tf.function
def metric_entropy(x, fixed, margin):
    """ See https://arxiv.org/pdf/1908.11184.pdf for the construction.
    """
    x         = tf.reshape(x, shape=[x.shape[0], 1, -1])
    fixed     = tf.reshape(fixed, shape=[1, fixed.shape[0], -1])
    distances = x - fixed
    distances = tf.reduce_sum(distances ** 2., axis=2)
    distances = distances / (margin**2.)  # bigger margin: farther away
    saturated = 5.  # a bit farther away than margin
    distances = tf.minimum(distances, saturated)  # avoid saturation

    similarities = tf.math.exp(distances * (-1.))
    typicality   = tf.reduce_mean(similarities, axis=1)
    entropy      = -tf.reduce_mean(tf.math.log(typicality))
    return entropy


@tf.function
def renormalize(x):
    norms = tf.reduce_sum(x**2., axis=range(len(x.shape[1:])))**0.5
    return x / norms

def renormalize_grads(grads):
    return [renormalize(grad) for grad in grads]


def generate_adversarial(model, x_0, scale, phase):
    return scale * tf.random.uniform(x_0.shape, [[-2., -1.5]], [[3., 2.]])

    max_iter = 10
    lr       = 2.
    h_x_0    = 2.
    h_x      = 1.

    learning_rate = lr * (scale / max_iter)
    optimizer     = SGD(learning_rate=learning_rate)

    # x_init is perturbed x_0, with atmost 10% of a gradient step (which can be admittely quite high)
    x_init = x_0 + 0.1*learning_rate*renormalize(tf.random.uniform(x_0.shape, -1., 1.))
    x      = tf.Variable(initial_value=x_0, trainable=True)
    for _ in range(max_iter):
        with tf.GradientTape() as tape:
            y = model(x)
            dispersion   = h_x   * metric_entropy(x, x, scale)
            fidelity     = h_x_0 * metric_entropy(x, x_0, scale)
            loss = tf.reduce_mean(y) + dispersion + fidelity
            loss = -loss  # minimize -loss <=> maximize loss (but must be regularized)
        grad_f_x = tape.gradient(loss, [x])
        grad_f_x = renormalize_grads(grad_f_x)
        optimizer.apply_gradients(zip(grad_f_x,[x]))
    return x.value()


def one_class_wasserstein(model, x, optimizer, lbda, alpha, scale, phase):
    adv = generate_adversarial(model, x, scale, phase)

    with tf.GradientTape(persistent=True) as tape:
        # tape.watch(x)

        y     = model(x)
        y_adv = model(adv)

        wasserstein = tf.reduce_mean(y) - tf.reduce_mean(y_adv)
        # hinge       = oneclass_hinge(1., y) + oneclass_hinge(1., -y_adv)
        hinge       = oneclass_hinge(1., y) #  + oneclass_hinge(1., -y_adv)
        loss        = -wasserstein + lbda * hinge

    _, norm_nabla_f_x = get_grad_norm_with_tape(tape, y, x)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, wasserstein, hinge, norm_nabla_f_x


def train_OOD_detector(model, dataset, num_batchs, lbda, alpha, scale, phase):
    optimizer = Adam()
    progress = tqdm(total=num_batchs, ascii=True)
    loss_avg, w_avg, hinge_avg, m = None, None, None, 0.97
    penalties = []
    for _, (x, _) in enumerate(dataset):
        loss, w, hinge, penalty = one_class_wasserstein(model, x, optimizer, lbda, alpha, scale, phase)
        loss_avg = exp_avg(loss_avg, loss, m); w_avg = exp_avg(w_avg, w, m); hinge_avg = exp_avg(hinge_avg, hinge, m)
        penalties.append(penalty)
        desc = f'Loss={float(loss_avg):>2.3f} Wasserstein={float(w_avg):>2.3f} Hinge={float(hinge_avg):>2.3f}'
        progress.set_description(desc=desc)
        progress.update(1)
    print('', flush=True)
    return penalties


def draw_adv(model, sk_func, num_examples_draw, margin, phase):
    batch_size_draw = 100
    dataset = tf_dataset(num_examples_draw // batch_size_draw, batch_size_draw, sk_func)
    inf_x, sup_x, inf_y, sup_y = plotex.get_limits(sk_func(100, noise=0.1)[0], 0.6)
    plt.xlim(inf_x, sup_x)
    plt.ylim(inf_y, sup_y)
    for x, _ in dataset:
        adv = generate_adversarial(model, x, margin, phase)
        plt.scatter(x[:,0], x[:,1], c='red', alpha=0.1, marker='.')
        plt.scatter(adv[:,0], adv[:,1], c='green', alpha=0.2, marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    patch_1 = mpatches.Patch(color='red', label=f'support')
    patch_2 = mpatches.Patch(color='green', label=f'adv')
    plt.legend(handles=[patch_1, patch_2])


def plot_levels_lines(sk_func):
    input_shape = (2)
    seed_dispatcher(None)
    model = models.get_mlp_baseline(input_shape)

    num_batchs = 200
    batch_size = 100  # 100
    lbda       = 1.
    alpha      = 0.
    phase      = 'adversarial'
    scale      = 10.
    sk_func    = dilated_func(sk_func, scale)

    dataset    = tf_dataset(num_batchs, batch_size, sk_func)

    X, Y       = sk_func(1000, noise=0.1)
    num_examples_draw = 1000
    fig        = plt.figure(figsize=(20,14))
    plt.subplot(2, 3, 1)
    plotex.plot_levels(X, Y, model)
    plt.subplot(2, 3, 2)
    draw_adv(model, sk_func, num_examples_draw, scale, phase)

    plt.subplot(2, 3, 3)
    penalties = train_OOD_detector(model, dataset, num_batchs, lbda, alpha, scale, phase)
    iterations = np.arange(len(penalties))
    plt.plot(iterations, np.log10(tf.reduce_mean(penalties, axis=1).numpy()))
    plt.plot(iterations, np.log10(tf.reduce_min(penalties, axis=1).numpy()))
    plt.plot(iterations, np.log10(tf.reduce_max(penalties, axis=1).numpy()))
    plt.title(r'Log Gradient Norm $\|\nabla_x f\|_2$')

    plt.subplot(2, 3, 4)
    plotex.plot_levels(X, Y, model)
    plt.subplot(2, 3, 5)
    draw_adv(model, sk_func, num_examples_draw, scale, phase)

    plt.show()


if __name__ == '__main__':
    plot_levels_lines(make_moons)

