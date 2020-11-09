import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import models
import plotex
from utils import tf_dataset, dilated_func, exp_avg, seed_dispatcher
from gradient_tools import get_grad_norm, get_grad_norm_with_tape
from gradient_tools import oneclass_hinge, gradient_penalty


# @tf.function
def generate_adversarial(model, x, margin, phase):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    nabla_f_x, norm_nabla_f_x = get_grad_norm_with_tape(tape, y, x)
    direction = nabla_f_x / (tf.maximum(margin*0.01, norm_nabla_f_x))
    if phase == 'random':
        noise = tf.random.uniform(x.shape, -1., 1.)
        x_adv = x + 5 * noise
    elif phase == 'adv':
        noise = tf.random.uniform([x.shape[0]]+[1]*(len(x.shape[1:])), 0., 1.)
        x_adv = x - 2 * direction * y * noise
    elif phase == 'inconsistent':
        noise = tf.random.uniform([x.shape[0]]+[1]*(len(x.shape[1:])), 0., 1.)
        x_adv = x + direction * tf.maximum(y, margin) * noise
    elif phase == 'symmetric':
        noise = tf.random.uniform([x.shape[0]]+[1]*(len(x.shape[1:])), 0.95, 1.)
        delta = direction * tf.maximum(y, margin) * noise
        w_adv = x - delta
        hinge_adv = tf.concat([x + delta, w_adv], axis=0)
        # hinge_adv = x + delta
        return w_adv, hinge_adv
    return x_adv, x_adv, norm_nabla_f_x


# @tf.function
def naive_one_class_wasserstein(model, x, optimizer, lbda, alpha, margin, phase):
    w_adv, hinge_adv = generate_adversarial(model, x, margin, phase)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = model(x)
        y_w_adv = model(w_adv)
        y_hinge_adv = model(hinge_adv)
        # _, norm_nabla_f_x = get_grad_norm_with_tape(tape, y, x)
        _, norm_nabla_f_x = get_grad_norm(model, x)
        wasserstein = tf.reduce_mean(y) - tf.reduce_mean(y_w_adv)
        hinge = oneclass_hinge(margin, y) + oneclass_hinge(margin, -y_hinge_adv)
        loss = -wasserstein + lbda * hinge + alpha * gradient_penalty(norm_nabla_f_x)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, wasserstein, hinge, norm_nabla_f_x


def train_OOD_detector(model, dataset, num_batchs, lbda, alpha, margin, phase):
    optimizer = Adam()
    progress = tqdm(total=num_batchs, ascii=True)
    loss_avg, w_avg, hinge_avg, m = None, None, None, 0.97
    penalties = []
    for _, (x, _) in enumerate(dataset):
        loss, w, hinge, penalty = naive_one_class_wasserstein(model, x, optimizer, lbda, alpha, margin, phase)
        loss_avg = exp_avg(loss_avg, loss, m); w_avg = exp_avg(w_avg, w, m); hinge_avg = exp_avg(hinge_avg, hinge, m)
        penalties.append(penalty)
        desc = f'Loss={float(loss_avg):>2.3f} Wasserstein={float(w_avg):>2.3f} Hinge={float(hinge_avg):>2.3f}'
        progress.set_description(desc=desc)
        progress.update(1)
    print('', flush=True)
    return penalties


def plot_levels_lines(sk_func):
    input_shape = (2)
    seed_dispatcher(None)
    model = models.get_mlp_baseline(input_shape)

    num_batchs = 500
    batch_size = 100
    lbda       = 1.
    alpha      = 10.
    phase      = 'symmetric'
    margin     = 0.2
    dilatation = 1.
    skfunc     = dilated_func(sk_func, dilatation)

    dataset    = tf_dataset(num_batchs, batch_size, sk_func)

    X, Y       = sk_func(100, noise=0.1)
    num_examples_draw = 1000
    fig        = plt.figure(figsize=(20,14))
    plt.subplot(2, 3, 1)
    plotex.plot_levels(X, Y, model)
    plt.subplot(2, 3, 2)
    draw_adv(model, sk_func, num_examples_draw, margin, phase)

    plt.subplot(2, 3, 3)
    penalties = train_OOD_detector(model, dataset, num_batchs, lbda, alpha, margin, phase)
    iterations = np.arange(len(penalties))
    plt.plot(iterations, np.log10(tf.reduce_mean(penalties, axis=1).numpy()))
    plt.plot(iterations, np.log10(tf.reduce_min(penalties, axis=1).numpy()))
    plt.plot(iterations, np.log10(tf.reduce_max(penalties, axis=1).numpy()))
    plt.title(r'Log Gradient Norm $\|\nabla_x f\|_2$')

    plt.subplot(2, 3, 4)
    plotex.plot_levels(X, Y, model)
    plt.subplot(2, 3, 5)
    draw_adv(model, sk_func, num_examples_draw, margin, phase)

    plt.show()


def draw_adv(model, sk_func, num_examples_draw, margin, phase):
    batch_size_draw = 1000
    dataset = tf_dataset(num_examples_draw // batch_size_draw, batch_size_draw, sk_func)
    inf_x, sup_x, inf_y, sup_y = plotex.get_limits(sk_func(100, noise=0.1)[0], 0.6)
    plt.xlim(inf_x, sup_x)
    plt.ylim(inf_y, sup_y)
    for x, _ in dataset:
        w_adv, hinge_adv = generate_adversarial(model, x, margin, phase)
        plt.scatter(x[:,0], x[:,1], c='red', alpha=0.1, marker='.')
        plt.scatter(hinge_adv[:,0], hinge_adv[:,1], c='green', alpha=0.2, marker='x')
        plt.scatter(w_adv[:,0], w_adv[:,1], c='blue', alpha=0.2, marker='+')
    plt.xlabel('X')
    plt.ylabel('Y')
    patch_1 = mpatches.Patch(color='red', label=f'support')
    patch_2 = mpatches.Patch(color='green', label=f'x + delta; x + delta')
    patch_3 = mpatches.Patch(color='blue', label=f'x - delta')
    plt.legend(handles=[patch_1, patch_2, patch_3])
