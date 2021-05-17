import tensorflow as tf
import numpy as np


def prepare_binary_data(config, x, y, input_shape, class_a, class_b):
    x = x.astype('float32')
    x = x.reshape((-1,)+input_shape)
    y = y.astype('float32')
    y = y.flatten()
    np_a = np.array(class_a)
    np_b = np.array(class_b)
    is_aorb = np.equal(y[:,np.newaxis], np.concatenate([np_a,np_b],axis=0)[np.newaxis,:]).sum(axis=1)
    indexes, = is_aorb.nonzero()
    x = x.take(indexes, axis=0).reshape((-1,)+input_shape)
    y = y.take(indexes, axis=0).flatten()
    x = x/255
    if config.dataset_name != 'mnist':
        x = 2*x - 1.
    mask_a = np.equal(y[:,np.newaxis], np_a[np.newaxis,:]).sum(axis=1).astype('bool')
    y[mask_a] = 1.0
    y[~mask_a] = (0. if config.loss_type == 'bce' else -1.0)
    if config.random_labels:
        np.random.shuffle(y)
    return x, y

def prepare_all_data(config, x, y, input_shape, random_labels=False):
    x = x.astype('float32')
    x = x.reshape((-1,)+input_shape)
    y = y.astype('int64')
    y = y.flatten()
    x = x/255
    if config.dataset_name != 'mnist':
        x = 2*x - 1.
    if random_labels:
        np.random.shuffle(y)
    return x, y

def binary_train_generator(config, x, y):
    print('y_train=', y)
    p_mask = y > (0.5 if config.loss_type == 'bce' else 0.)
    q_mask = ~p_mask
    fp, fq = p_mask.sum(), q_mask.sum()
    ratio = fq / fp
    print(f"Ratio={ratio} for fp={fp} fq={fq}",flush=True)
    p_index, = np.squeeze(p_mask).nonzero()
    q_index, = np.squeeze(q_mask).nonzero()
    if ratio > 1.:
        p_index = np.repeat(p_index, round(ratio))
        np.random.shuffle(p_index)
    elif ratio < 1.:
        q_index = np.repeat(q_index, round(1. / ratio))
        np.random.shuffle(q_index)
    px, py = x.take(p_index, axis=0), y.take(p_index, axis=0)
    qx, qy = x.take(q_index, axis=0), y.take(q_index, axis=0)
    xe = np.concatenate([px, qx], axis=0)
    ye = np.concatenate([py, qy], axis=0)
    print('Final: ', xe.shape, ye.shape)
    return xe, ye

def balanced(ds, nb_classes, buffer_size):
    return tf.data.experimental.choose_from_datasets([
        ds.filter(lambda features, label: label == i).shuffle(buffer_size)
        for i in range(nb_classes)],
        tf.data.Dataset.range(nb_classes).repeat(),
    )

def augmenter(x, y, use_keras=False):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_saturation(x, lower=0.8, upper=1.2)
    if use_keras:
        x = tf.keras.preprocessing.image.random_rotation(x, rg=15,
            row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
        x = tf.keras.preprocessing.image.random_shift(x, wrg=0.14, hrg=0.14,
            row_axis=0, col_axis=1, channel_axis=2)
    return x, y

def train_generator(x_train, y_train, batch_size, buffer_size, augment=False):
    num_classes = len(tf.unique(y_train)[0])
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = balanced(ds_train, num_classes, buffer_size)
    if augment:
        ds_train = ds_train.map(augmenter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.shuffle(8)  # next 8 batchs
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train
