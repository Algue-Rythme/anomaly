import random
import tensorflow as tf
import numpy as np


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
