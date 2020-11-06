import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as activations
from sklearn.datasets import make_moons, make_circles
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from models import get_mlp_baseline
import plotex


@tf.function
def oneclass_hinge(y):
    return tf.reduce_mean(activations.relu(1 - y))


@tf.function
def order2_penalty(hessian):
    return tf.reduce_sum(hessian ** 2)


@tf.function
def get_orders_12(model, x):
    with tf.GradientTape(persistent=True) as order2_tape:
        order2_tape.watch(x)
        with tf.GradientTape(persistent=True) as order1_tape:
            order1_tape.watch(x)
            y = model(x)
        grad_x = order1_tape.gradient(y, x)
    hessian_x = order2_tape.batch_jacobian(grad_x, x)
    return y, grad_x, hessian_x


def order2_regularization(model, x, optimizer):
    with tf.GradientTape(persistent=True) as parameter_tape:
        y, _, hessian_x = get_orders_12(model, x)
        hinge_loss = oneclass_hinge(y)
        order2_loss = order2_penalty(hessian_x)
        loss = hinge_loss + order2_loss
    grads = parameter_tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# @tf.function
def generate_adversarial(model, x, phase):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    nabla_f_x = tape.gradient(y, x)
    norm_nabla_f_x = tf.reduce_sum(x ** 2, axis=list(range(1, len(x.shape))), keepdims=True)
    direction = nabla_f_x / norm_nabla_f_x
    if phase == 'random':
        noise = tf.random.uniform(x.shape, -1., 1.)
        x_adv = x + 5 * noise
    elif phase == 'adv':
        noise = tf.random.uniform([x.shape[0]]+[1]*(len(x.shape[1:])), 0., 1.)
        x_adv = x - 2 * direction * y * noise
    elif phase == 'inconsistent':
        noise = tf.random.uniform([x.shape[0]]+[1]*(len(x.shape[1:])), 0., 1.)
        x_adv = x + 2 * direction * y * noise
    return x_adv


# @tf.function
def one_class_wasserstein(model, x, optimizer, lbda, phase1):
    x_adv = generate_adversarial(model, x, phase1)
    with tf.GradientTape() as tape:
        y = model(x)
        y_adv = model(x_adv)
        wasserstein = tf.reduce_mean(y) - tf.reduce_mean(y_adv)
        hinge = oneclass_hinge(y) + oneclass_hinge(-y_adv)
        loss = -wasserstein + lbda * hinge
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, wasserstein, hinge


def exp_avg(x_avg, x, m):
    if x_avg is None:
        return x
    return x_avg*m + x*(1-m)


def train_OOD_detector(model, dataset, num_batchs, lbda, phase):
    optimizer = Adam()
    progress = tqdm(total=num_batchs, ascii=True)
    loss_avg, w_avg, hinge_avg, m = None, None, None, 0.9
    for step, (x, _) in enumerate(dataset):
        loss, w, hinge = one_class_wasserstein(model, x, optimizer, lbda, phase)
        loss_avg = exp_avg(loss_avg, loss, m)
        w_avg = exp_avg(w_avg, w, m)
        hinge_avg = exp_avg(hinge_avg, hinge, m)
        desc = f'Loss={float(loss_avg):>2.3f} Wasserstein={float(w_avg):>2.3f} Hinge={float(hinge_avg):>2.3f}'
        progress.set_description(desc=desc)
        progress.update(1)
    print('')


def tf_dataset(num_batchs, batch_size, sk_func):
    x, y    = sk_func(num_batchs * batch_size, shuffle=True, noise=0.1)
    x, y    = tf.constant(x, dtype=tf.float32), tf.constant(y, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    return dataset


def seed_dispatcher(seed=None):
    if seed is None:
        seed = random.randint(1, 1000 * 1000)
    np.random.seed(seed * 2)
    tf.random.set_seed(seed * 3)
    print('Seed used: %d'%seed)


def draw_adv(model, sk_func, num_examples_draw, adv_color):
    batch_size_draw = 1000
    dataset = tf_dataset(num_examples_draw // batch_size_draw, batch_size_draw, sk_func)
    for x, _ in dataset:
        x_adv = generate_adversarial(model, x, 'random')
        plt.scatter(x[:,0], x[:,1], c='red', alpha=0.1, marker='.')
        plt.scatter(x_adv[:,0], x_adv[:,1], c=adv_color, alpha=0.1, marker='.')


def plot_multiple_experiments(sk_func):
    input_shape = (2)
    fig = plt.figure()
    for i in range(9):
        seed_dispatcher(42 + i)
        model = get_mlp_baseline(input_shape)

        plt.subplot(3, 3, i + 1)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        seed_dispatcher(57 + i)

        num_examples_draw = 1000
        draw_adv(model, sk_func, num_examples_draw, 'blue')
        num_batchs = 500
        batch_size = 10
        lbda       = 1.
        dataset    = tf_dataset(num_batchs, batch_size, sk_func)
        train_OOD_detector(model, dataset, num_batchs, lbda, 'random')
        draw_adv(model, sk_func, num_examples_draw, 'green')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        patch_red = mpatches.Patch(color='red', label=f'support')
        patch_blue = mpatches.Patch(color='blue', label=f'before')
        patch_green = mpatches.Patch(color='green', label=f'after')
        plt.legend(handles=[patch_red, patch_blue, patch_green])
    plt.show()


def plot_levels_lines(sk_func):
    input_shape = (2)
    seed_dispatcher(42)
    model = get_mlp_baseline(input_shape)
    num_batchs = 500
    batch_size = 100
    lbda       = 50.
    dataset    = tf_dataset(num_batchs, batch_size, sk_func)
    X, Y       = sk_func(1000, noise=0.1)
    plotex.plot_levels(X, Y, model)
    train_OOD_detector(model, dataset, num_batchs, lbda, 'inconsistent')
    plotex.plot_levels(X, Y, model)
    plt.show()


if __name__ == '__main__':
    plot_multiple_experiments(make_moons)
    # plot_levels_lines(make_circles)
