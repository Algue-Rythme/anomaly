import numpy as np
import tensorflow as tf

# from deel.lip.layers import ScaledL2NormPooling2D
from models import ScaledL2NormPooling2D

with tf.device('/cpu:0'):
    pooler = ScaledL2NormPooling2D(pool_size=(3,3))
    ones = tf.ones(shape=(1,3,3,1))
    other = 3*tf.ones(shape=(1,3,3,1))
    print(ones, other, tf.linalg.norm(other-ones))
    po = pooler(ones)
    pt = pooler(other)
    print(po, pt, tf.linalg.norm(pt-po))
