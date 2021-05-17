import argparse
import pathlib
import random

import gin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from scipy.signal import filtfilt
from tensorflow.python.ops.gen_array_ops import const
import tensorflow.keras.activations as activations
from sklearn.datasets import make_moons, make_circles, make_blobs
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, PrecisionAtRecall, RecallAtPrecision, Recall, Precision
from deel.lip.callbacks import CondenseCallback
import models


@tf.function
def logloss_support(y):
    ones = tf.ones(y.shape)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(ones, y)
    return tf.reduce_mean(loss)

@tf.function
def logloss_border(y):
    border = tf.ones(y.shape) / 2.
    loss = tf.nn.sigmoid_cross_entropy_with_logits(border, y)
    return tf.reduce_mean(loss)

@tf.function
def logloss_ood(y):
    zeros = tf.zeros(y.shape)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(zeros, y)
    return tf.reduce_mean(loss)

@tf.function
def logloss_3parts(y_tp, y_fp, y_tn, lbda, temperature):
    tp_to_tp = logloss_support(y_tp * temperature)
    fp_to_tn = logloss_border(y_fp * temperature)
    tn_to_tn = logloss_ood(y_tn * temperature)
    to_tn = lbda*fp_to_tn + (1-lbda)*tn_to_tn
    loss = tp_to_tp + to_tn
    return loss, tp_to_tp


@tf.function
def border_hinge(y):  # null-margin, on the other side of the frontiere
    return tf.reduce_mean(tf.nn.relu(y))

@tf.function
def border_manifold(y):  # null-margin, exactly on the frontiere
    return tf.reduce_mean(tf.abs(y))

@tf.function
def hinge(y, margin):
    return tf.reduce_mean(tf.nn.relu(margin - y))

@tf.function
def hinge_3parts(y_tp, y_fp, y_tn, lbda, alpha, margin):
    tp_to_tp = -tf.reduce_mean(y_tp) + alpha*hinge(y_tp, margin)
    to_tn = tf.reduce_mean(y_tn) + alpha*border_hinge(y_fp)
    loss = tp_to_tp + to_tn
    return loss, tf.reduce_mean(y_tp)

@tf.function
def triplet_loss(y_tp, y_fp, y_tn, alpha, margin):
    pa = y_tp - y_fp  # positive
    an = y_fp - y_tn  # negative
    delta = an - pa
    triplet = hinge(delta, margin)
    wass = tf.reduce_mean(y_tn) - tf.reduce_mean(y_tp)
    loss = wass + alpha*triplet
    return loss, tf.reduce_mean(delta)

class ManifoldForward():
    def __init__(self, model, n_stop):
        self.model = model
        self.n_stop = n_stop
    def random_feature_map(self, x):
        half = x.shape[0]//2
        if len(x.shape) == 2:
            features = x.shape[1:]
            noise = tf.random.normal(shape=(half,)+tuple(features))
            noise = tf.concat([noise, tf.zeros((half,)+tuple(features))], axis=0)
            z_act = tf.reduce_sum(x*noise, axis=list(range(1, len(noise.shape))), keepdims=True)
            return z_act
        assert len(x.shape) == 4
        features = x.shape[-1]
        idx = np.randint(features, size=half)
        idx = np.concatenate([idx, np.zeros(half)], axis=0)
        idx = np.stack([np.arange(len(idx)), idx], axis=1)
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # swap last dim
        z_act = tf.gather_nd(x, tf.constant(idx))
        z_act = tf.reduce_sum(z_act, axis=[-1, -2])
        return z_act
    def sample(self, x_0):
        if self.n_stop == 0:
            return self.model(x_0, training=False), None
        x = x_0
        for layer in self.model.layers[:-self.n_stop]:
            x = layer(x, training=False)
        z_act = self.random_feature_map(x)
        for layer in self.model.layers[-self.n_stop:]:
            x = layer(x, training=False)  # yield logits in the end
        return x, z_act
    def representation_goal(self, z_act):
        return tf.reduce_mean(z_act)  # average of activations over batch

def renormalize_grads(grads):  # side-effect on parameters: mitigate vanishing gradient
    return [tf.math.l2_normalize(grad, axis=list(range(1,tf.rank(grad)))) for grad in grads]

def add_noise(grads, step_size):  # noisy gradient step
    return [grad + tf.math.sqrt(2.*step_size)*tf.random.normal(shape=grad.shape) for grad in grads]

@tf.function
def ball_project(delta, x_0, adv_radius):
    delta = tf.clip_by_norm(delta, adv_radius, axes=list(range(1,len(x_0.shape))))
    delta = tf.clip_by_value(x_0 + delta, 0., 1.) - x_0
    return delta

def l1_penalty(x):
    l1 = tf.reduce_sum(tf.abs(x), axis=list(range(1,len(x.shape))), keepdims=True)
    non_batch_dims = int(tf.size(x)) / int(x.shape[0])
    return l1 / non_batch_dims

def goal(training_type, adv_type, y):
    if (training_type, adv_type) == ('wasserstein', 'y_fp'):
        return -tf.reduce_mean(y)
    elif (training_type, adv_type) == ('wasserstein', 'y_tn'):
        return tf.reduce_mean(y)  # border_hinge(y)
    elif (training_type, adv_type) == ('logloss', 'y_fp'):
        return logloss_support(y)
    elif (training_type, adv_type) == ('logloss', 'y_tn'):
        return logloss_ood(y)


@gin.configurable
def generate_adversarial(model, x_0, adv_radius, adv_type, training_type, adv_policy,
    step_size=gin.REQUIRED,
    max_iter=gin.REQUIRED,
    n_stop=gin.REQUIRED,
    l1_regularization=gin.REQUIRED):  # l1 special case of intermdiate activation => merge into ManifoldSampler
    if adv_policy == 'adv':
        noise = tf.random.uniform(shape=x_0.shape, minval=-adv_radius, maxval=adv_radius)
        noise = ball_project(noise, x_0, adv_radius)
        delta = tf.Variable(initial_value=noise, trainable=True)
    elif adv_policy == 'gan':
        noise = tf.random.uniform(shape=x_0.shape, minval=0., maxval=1.)
        x = tf.Variable(initial_value=noise, trainable=True)
    lr = step_size * (adv_radius/max_iter)
    optimizer = SGD(learning_rate=lr)  # sufficient considering loss landscape
    manifold = ManifoldForward(model, n_stop)
    for _ in range(max_iter):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            if adv_policy == 'adv':
                tape.watch(delta)
                x = x_0 + delta
            elif adv_policy == 'gan':
                tape.watch(x)
            y, z_map = manifold.sample(x)
            loss = goal(training_type, adv_type, y)
            if z_map is not None:
                z_loss = manifold.representation_goal(z_map)
                loss = loss + z_loss
            if l1_regularization is not None:
                l1 = l1_penalty(x) if adv_policy == 'gan' else l1_penalty(delta)
                loss = loss + l1_regularization * l1
        diff_target = x if adv_policy == 'gan' else delta
        grad_f = tape.gradient(loss, [diff_target])
        grad_f = renormalize_grads(grad_f)
        optimizer.apply_gradients(zip(grad_f,[diff_target]))
        if adv_policy == 'adv':
            delta.assign(ball_project(delta, x_0, adv_radius))  # project back to ball in manifold 
        elif adv_policy == 'gan':
            x.assign(tf.clip_by_value(x, 0., 1.))  # image set
    if adv_policy == 'adv':
        x = x_0+delta.value()
        x = tf.clip_by_value(x, 0., 1.)
        return x
    elif adv_policy == 'gan':
        return x.value()


def print_tensors(x_tp, x_fp, x_tn, adv_policy, epoch, step, prefix='train', grid_print=False):
    if step != 0:
        return
    if grid_print:
        d_tp_fp = tf.unstack(x_tp - x_fp); d_tp_tn = tf.unstack(x_tp - x_tn)
        x_tps = tf.unstack(x_tp); x_fps = tf.unstack(x_fp); x_tns = tf.unstack(x_tn)
        to_print = [x_tps, x_fps, d_tp_fp, x_tns, d_tp_tn]
        numcols = len(to_print); numrows = len(x_tps); index = 1
        for imgs in to_print:
            for img in imgs:
                plt.subplot(numcols,numrows,index); plt.imshow(img)
                index += 1
    else:
        if adv_policy == 'adv':
            d_tp_fp = tf.abs(x_tp - x_fp)
            d_tp_tn = tf.abs(x_tp - x_tn)
            to_print = [x_tp, x_fp, d_tp_fp, x_tn, d_tp_tn]
        elif adv_policy == 'gan':
            to_print = [x_tp, x_fp, x_tn]
        rows = [tf.concat(tf.unstack(imgs),axis=1) for imgs in to_print]
        img = tf.concat(rows, axis=0)
        plt.imshow(img)
    plt.savefig(f'samples/images/{prefix}_{epoch}.png', bbox_inches='tight')
    plt.clf()


@gin.configurable
def adv_training(model, x_tp, epoch, step,
                 training_type=gin.REQUIRED,
                 lbda=gin.REQUIRED,
                 alpha=gin.REQUIRED,
                 false_positive_radius=gin.REQUIRED,
                 true_negative_radius=gin.REQUIRED,
                 margin=gin.REQUIRED,
                 temperature=gin.REQUIRED,
                 adv_policy='gan'):
    non_batch_dims = int(tf.size(x_tp)) / int(x_tp.shape[0])
    unit_length = non_batch_dims ** 0.5  # proportionnal to the square root of the number of pixels
    false_positive_radius = tf.constant(false_positive_radius * unit_length, dtype=tf.float32)
    true_negative_radius = tf.constant(true_negative_radius * unit_length, dtype=tf.float32)
    margin = tf.constant(margin * unit_length, dtype=tf.float32)
    x_fp = generate_adversarial(model, x_tp, false_positive_radius, 'y_fp', training_type, 'gan')
    x_tn = generate_adversarial(model, x_tp, true_negative_radius, 'y_tn', training_type, 'adv')
    with tf.GradientTape() as tape:
        y_tp = model(x_tp, training=False)
        y_fp = model(x_fp, training=False)
        y_tn = model(x_tn, training=False)
        if training_type == 'wasserstein':
            loss, support_weight = hinge_3parts(y_tp, y_fp, y_tn, lbda, alpha, margin)
            # loss, support_weight = triplet_loss(y_tp, y_fp, y_tn, alpha, margin)
        elif training_type == 'logloss':
            loss, support_weight = logloss_3parts(y_tp, y_fp, y_tn, lbda, temperature)
    grads = tape.gradient(loss, model.trainable_variables)
    print_tensors(x_tp, x_fp, x_tn, adv_policy, epoch, step)
    labels_pred = tf.argmax(tf.concat([-y_tp, y_tp], axis=1), axis=1)
    return loss, support_weight, labels_pred, grads


def exp_avg(data_avg, data):
    m_coef = 0.97
    if data_avg is None:
        return data
    return m_coef*data_avg + (1-m_coef)*data

def from_logits(Metric):
    class FromLogits(Metric):
        def __init__(self, in_labels, **kwargs):
            name = Metric.__name__.lower()
            super(FromLogits, self).__init__(name=name, **kwargs)
            self.in_labels = in_labels
        def update_state(self, y_true, y_pred, sample_weights=None):
            y_true = tf.reduce_any(tf.equal(tf.cast(y_true, dtype=tf.int64), self.in_labels), axis=-1, keepdims=True)
            y_pred = tf.nn.sigmoid(y_pred)
            super(FromLogits, self).update_state(y_true, y_pred, sample_weights)
    return FromLogits

class ExpAvg():
    def __init__(self, m_coef):
        self.m_coef = m_coef
        self.state = None
    def update(self, measure):
        if self.state is None: self.state = measure
        else:
            self.state *= self.m_coef
            self.state += (1.0-self.m_coef)*measure
    def result(self):
        return f"{self.state:>2.5f}"


class Histogram(tf.keras.metrics.Metric):
    def __init__(self, in_labels, start_epoch, **kwargs):
        super(Histogram, self).__init__(name='bestacc', **kwargs)
        self.hist_true = None; self.hist_pred = None
        self.in_labels = in_labels; self.epoch = start_epoch
        import glob
        import os
        files = glob.glob('samples/hists/*')
        for f in files:
            os.remove(f)
        files = glob.glob('samples/images/*')
        for f in files:
            os.remove(f)
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.hist_pred is None:
            self.hist_pred = tf.squeeze(y_pred)
            self.hist_true = tf.squeeze(y_true)
        else:
            self.hist_pred = tf.concat([self.hist_pred,tf.squeeze(y_pred)],axis=0)
            self.hist_true = tf.concat([self.hist_true,tf.squeeze(y_true)],axis=0)
    def result(self):
        indexes = tf.argsort(self.hist_pred)
        hist_true = tf.cast(tf.reshape(self.hist_true, shape=(-1, 1)), dtype=tf.int64)
        hist_true = tf.reduce_any(tf.equal(hist_true, self.in_labels), axis=-1, keepdims=True)
        hist_true = tf.cast(hist_true, dtype=tf.float32)
        labels = tf.gather(hist_true, indexes)
        cumlabels = tf.cumsum(labels, reverse=True)
        cumrlabels = tf.cumsum(1. - labels)
        ok_pred = cumlabels + cumrlabels
        accs = ok_pred / tf.cast(tf.size(labels), dtype=tf.float32)
        max_acc = tf.reduce_max(accs)
        return max_acc
    def plot_hist(self):
        labels, _ = tf.unique(self.hist_true)
        hists = {}
        palette = dict()
        palette_in, palette_out = plt.get_cmap('Set1'), plt.get_cmap('Set3')
        hue_in, hue_out = [], []
        for idx, label in enumerate(tf.sort(labels)):
            is_label = tf.equal(self.hist_true, label)
            y_pred = tf.boolean_mask(self.hist_pred, is_label)
            label_name = f"{int(label)}"
            hists[label_name] = pd.Series(y_pred.numpy())
            if int(label) in self.in_labels.numpy():
                palette[label_name] = palette_in(idx / len(labels))
                hue_in.append(label_name)
            else:
                palette[label_name] = palette_out(idx / len(labels))
                hue_out.append(label_name)
        hue_order = hue_in + hue_out
        sns.histplot(pd.DataFrame(hists), stat='density', bins=100, palette=palette, hue_order=hue_order)
        plt.savefig(f'samples/hists/hist_{self.epoch}.png', bbox_inches='tight')
        plt.clf()
    def reset_states(self):
        self.plot_hist()
        self.hist_true = None
        self.hist_pred = None
        self.epoch += 1


@gin.configurable
def train(model, ds_train, ds_test, start_epoch,
          in_labels, num_epochs=gin.REQUIRED):
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(from_logits=True),
                  metrics=[from_logits(BinaryAccuracy)(in_labels),
                           from_logits(PrecisionAtRecall)(in_labels, recall=0.95),
                           from_logits(RecallAtPrecision)(in_labels, precision=0.95),
                           from_logits(Recall)(in_labels), from_logits(Precision)(in_labels),
                           Histogram(in_labels, start_epoch)], run_eagerly=True)
    num_batchs = sum(1 for _ in ds_train)
    model.evaluate(ds_test)
    for epoch in range(start_epoch, num_epochs):
        loss_avg, support_weight_avg = ExpAvg(0.97), ExpAvg(0.97)
        print(f"Epoch {epoch}:")
        progress = tqdm(total=num_batchs, ascii=True)
        for step, (x, _) in enumerate(ds_train):
            loss, support_weight, labels_pred, grads = adv_training(model, x, epoch, step)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            model.condense()
            loss_avg.update(loss); support_weight_avg.update(support_weight)
            desc = f'Loss={loss_avg.result()} Wsupport={support_weight_avg.result()}'
            progress.set_description(desc=desc)
            progress.update(1)
        progress.close()
        model.evaluate(ds_test)
        model.save_weights(f"checkpoints/model_{epoch}.h5")
        if epoch>=3:
            file_to_rem = pathlib.Path(f"checkpoints/model_{epoch-3}.h5")
            file_to_rem.unlink(missing_ok=True)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def process_dataset(name, batch_size=gin.REQUIRED):
    (ds_train, ds_test), ds_info = tfds.load(
        name,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test, ds_info


@gin.configurable
def one_class_dataset(ds, num_classes, num_examples, split,
                      batch_size=gin.REQUIRED,
                      in_labels=gin.REQUIRED,
                      out_labels=gin.REQUIRED):
    assert all(0<=y<num_classes for y in in_labels) and all(0<=y<num_classes for y in out_labels)
    in_labels = tf.constant(in_labels, dtype=tf.int64)
    in_out_labels = tf.concat([in_labels, tf.constant(out_labels, dtype=tf.int64)], axis=0)
    if split == 'train':
        ds = ds.filter(lambda _, y: tf.reduce_any(tf.equal(y, in_labels)))
        ds = ds.cache()  # cache before shuffle for different epochs
        ds = ds.shuffle(num_examples)
    elif split == 'test':
        ds = ds.shuffle(num_examples)
        ds = ds.filter(lambda _, y: tf.reduce_any(tf.equal(y, in_out_labels)))
        ds = ds.cache()  # cache after shuffle
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)
    return ds, in_labels


@gin.configurable
def model_params(in_put_shape, k_lip=gin.REQUIRED, scale=gin.REQUIRED,
                 niter_spectral=gin.REQUIRED, niter_bjorck=gin.REQUIRED, bjorck_forward=gin.REQUIRED):
    tf.keras.backend.set_image_data_format('channels_last')
    return models.get_cnn_baseline(input_shape, k_lip, scale, niter_spectral, niter_bjorck, bjorck_forward)

def dummy_forward(model, ds_test):
    _ = model.predict(next(ds_test.__iter__())[0])  # garbage
    model.condense()  # compute after forward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='store', dest='config', type=str, help='Gin Configuration file')
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', type=str, help='Dataset for Training')
    parser.add_argument('-r', '--resume', action='store', dest='resume', type=str, required=False, default='', help='Resume training at epoch')
    parser.add_argument('-t', '--val_on_trainset', action='store_true', dest='val_on_trainset',
                                                   required=False, default=False,
                                                   help='Use train set for validation (monitore overfiting)')
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    trainset, testset, infos = process_dataset(args.dataset)
    num_classes = infos.features['label'].num_classes
    ds_train, _ = one_class_dataset(trainset, num_classes, infos.splits['train'].num_examples, split='train')
    if args.val_on_trainset:
        num_examples = infos.splits['train'].num_examples
        ds_test, in_labels = one_class_dataset(trainset, num_classes, num_examples, split='test', batch_size=100)
    else:
        num_examples = infos.splits['test'].num_examples
        ds_test, in_labels = one_class_dataset(testset, num_classes, num_examples, split='test', batch_size=100)
    input_shape = infos.features['image'].shape
    model = model_params(input_shape)
    dummy_forward(model, ds_test)  # init variables
    if args.resume == '':
        epoch = 0
    else:
        epoch = int(args.resume.split('_')[-1].split('.')[0]) + 1
        print(f"Load file {args.resume} and resume at epoch {epoch}")
        model.load_weights(args.resume)
    train(model, ds_train, ds_test, epoch, in_labels)
