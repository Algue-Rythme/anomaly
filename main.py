import argparse
import random

import gin
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as activations
from sklearn.datasets import make_moons, make_circles, make_blobs
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tqdm import tqdm

import models
import plotex
from adversarial import complement_distribution
from gradient_tools import get_grad_norm_with_tape, oneclass_hinge
from utils import exp_avg, tf_dataset, dilated_func, seed_dispatcher, projected


@tf.function
def binary_crossentropy(y, y_adv):
    logloss0 = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones([int(y.shape[0]),1]), y)
    logloss1 = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros([int(y_adv.shape[0]),1]), y_adv)
    return tf.debugging.check_numerics(-(tf.reduce_mean(logloss0) + tf.reduce_mean(logloss1)), 'binary')


@gin.configurable
def one_class_wasserstein(model, x, optimizer,
                          lbda=gin.REQUIRED,
                          scale=gin.REQUIRED,
                          margin=gin.REQUIRED,
                          logloss=gin.REQUIRED):
    adv = complement_distribution(model, x, scale, margin)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)

        y     = tf.debugging.check_numerics(model(x, training=True), 'y_orig')
        y_adv = model(adv, training=True)

        if logloss:
            wasserstein = binary_crossentropy(y, y_adv)
        else:
            wasserstein = tf.reduce_mean(y) - tf.reduce_mean(y_adv)
        hinge       = oneclass_hinge(margin, y) + oneclass_hinge(margin, -y_adv)
        loss        = -wasserstein + lbda*hinge

    _, norm_nabla_f_x = get_grad_norm_with_tape(tape, y, x)  
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.debugging.check_numerics(g, 'model_grad') for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, wasserstein, hinge, norm_nabla_f_x


def train_OOD_detector(model, dataset, num_batchs):
    optimizer = Adam()
    # optimizer = RMSprop()
    # optimizer = SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)
    progress = tqdm(total=num_batchs, ascii=True)
    loss_avg, w_avg, hinge_avg, m = None, None, None, 0.97
    penalties = []
    for _, (x, _) in enumerate(dataset):
        loss, w, hinge, penalty = one_class_wasserstein(model, x, optimizer)
        loss_avg = exp_avg(loss_avg, loss, m); w_avg = exp_avg(w_avg, w, m); hinge_avg = exp_avg(hinge_avg, hinge, m)
        penalties.append(penalty)
        desc = f'Loss={float(loss_avg):>2.5f} Wasserstein={float(w_avg):>2.5f} Hinge={float(hinge_avg):>2.5f}'
        progress.set_description(desc=desc)
        progress.update(1)
    print('', flush=True)
    return penalties


def draw_adv(model, sk_func, X, num_examples_draw, batch_size_draw, fig, index):
    scale = gin.query_parameter('one_class_wasserstein.scale')
    margin = gin.query_parameter('one_class_wasserstein.margin')
    if X.shape[1] == 2:
        fig.add_subplot(index)
    else:
        plt.twinx()
    dataset = tf_dataset(num_examples_draw // batch_size_draw, batch_size_draw, sk_func)
    inf_x, sup_x, inf_y, sup_y = plotex.get_limits(X)
    xs, advs = [], []
    for x, _ in dataset:
        adv = complement_distribution(model, x, scale, margin)
        xs.append(x)
        advs.append(adv)
    xs = np.concatenate(xs)
    advs = np.concatenate(advs)
    if X.shape[1] == 2:
        plt.xlim(inf_x, sup_x)
        plt.ylim(inf_y, sup_y)
        plt.scatter(xs[:,0], xs[:,1], c='red', alpha=0.1, marker='.')
        plt.scatter(advs[:,0], advs[:,1], c='green', alpha=0.2, marker='x')
        plt.xlabel('X')
        plt.ylabel('Y')
    else:
        plt.hist(xs[:,0], bins=100, fc=(1, 0, 0, 0.5), histtype='stepfilled', density=True)
        plt.hist(advs[:,0], bins=100, fc=(0, 1, 0, 0.5), histtype='stepfilled', density=True)
    patch_1 = mpatches.Patch(color='red', label=f'support')
    patch_2 = mpatches.Patch(color='green', label=f'adv')
    plt.legend(handles=[patch_1, patch_2])


@gin.configurable
def plot_levels_lines(sk_func_name=gin.REQUIRED,
                      num_batchs=gin.REQUIRED,
                      batch_size=gin.REQUIRED,
                      num_examples_draw=gin.REQUIRED,
                      batch_size_draw=gin.REQUIRED,
                      proj1D=gin.REQUIRED,
                      init_landscape=gin.REQUIRED):
    seed_dispatcher(None)

    scale      = gin.query_parameter('one_class_wasserstein.scale')
    if sk_func_name == 'make_moons':
        sk_func    = lambda n: make_moons(n, shuffle=True, noise=0.05)
        if proj1D:
            sk_func = projected(sk_func, [1,0])
    elif sk_func_name == 'make_circles':
        sk_func    = lambda n: make_circles(n, shuffle=True, noise=0.05)
        if proj1D:
            sk_func = projected(sk_func, [1,0])
    elif sk_func_name == 'make_blobs':
        dim     = 1 if proj1D else 2
        seed    = random.randint(1, 1000)
        sk_func = lambda n: make_blobs(n, centers=3, cluster_std=1.*scale, n_features=dim,
                                       shuffle=True, random_state=seed)
    
    sk_func    = dilated_func(sk_func, scale)
    X, _       = sk_func(num_examples_draw)
    dataset    = tf_dataset(num_batchs, batch_size, sk_func)
    input_shape = X.shape[1:]
    model = models.get_mlp_baseline(input_shape)  # models.get_mlp_no_bias(input_shape)

    fig = plt.figure(figsize=(22,15))
    plotex.plot_levels(X, model, fig, 121 if proj1D else 231)
    draw_adv(model, sk_func, X, num_examples_draw, batch_size_draw, fig, 232)

    if not proj1D and init_landscape:
        plotex.plot3d(X, model, fig, 233)

    try:
        # raise tf.python.framework.errors_impl.InvalidArgumentError(None, None,"Oulala")
        penalties = train_OOD_detector(model, dataset, num_batchs)
    except tf.python.framework.errors_impl.InvalidArgumentError as e:
        from deel.lip.normalizers import bjorck_normalization, spectral_normalization
        for layer in model.layers:
            W_bar, _u, sigma = spectral_normalization(
                layer.kernel, layer.u, niter=layer.niter_spectral
            )
            norm = tf.reduce_sum(W_bar ** 2.)
            W_bar = bjorck_normalization(W_bar, niter=layer.niter_bjorck)
            print('############################################')
            print(norm, sigma, _u, layer.bias , W_bar)
            print('\n\n\n')
        raise e

    if not proj1D and not init_landscape:
        fig.add_subplot(233)
        iterations = np.arange(len(penalties))
        plt.plot(iterations, np.log10(tf.reduce_mean(penalties, axis=1).numpy()))
        plt.plot(iterations, np.log10(tf.reduce_min(penalties, axis=1).numpy()))
        plt.plot(iterations, np.log10(tf.reduce_max(penalties, axis=1).numpy()))
        plt.title(r'Log Gradient Norm $\log_{10}{\|\nabla_x f\|_2}$')

    plotex.plot_levels(X, model, fig, 122 if proj1D else 234)
    draw_adv(model, sk_func, X, num_examples_draw, batch_size_draw, fig, 235)
    if not proj1D:
        plotex.plot3d(X, model, fig, 236)
    plt.show()


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)
    # tf.debugging.enable_check_numerics()
    tf.debugging.disable_check_numerics()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='store', dest='config', type=str, help='Gin Configuration file')
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    plot_levels_lines()

