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

import models
import naive_adversarial
import plotex
from utils import exp_avg, tf_dataset, dilated_func, seed_dispatcher


def one_class_wasserstein(model, x, optimizer, lbda, alpha, margin, phase):
    return 0., 0., 0., 0.


def train_OOD_detector(model, dataset, num_batchs, lbda, alpha, margin, phase):
    optimizer = Adam()
    progress = tqdm(total=num_batchs, ascii=True)
    loss_avg, w_avg, hinge_avg, m = None, None, None, 0.97
    penalties = []
    for _, (x, _) in enumerate(dataset):
        loss, w, hinge, penalty = one_class_wasserstein(model, x, optimizer, lbda, alpha, margin, phase)
        loss_avg = exp_avg(loss_avg, loss, m); w_avg = exp_avg(w_avg, w, m); hinge_avg = exp_avg(hinge_avg, hinge, m)
        penalties.append(penalty)
        desc = f'Loss={float(loss_avg):>2.3f} Wasserstein={float(w_avg):>2.3f} Hinge={float(hinge_avg):>2.3f}'
        progress.set_description(desc=desc)
        progress.update(1)
    print('', flush=True)
    return penalties


if __name__ == '__main__':
    seed_dispatcher(None)

