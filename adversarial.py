import gin
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


class PartialModel:

    def __init__(self, model, layer_index):
        self.model = model
        self.layer_index = layer_index

    def __call__(self, x):
        for layer in self.model.layers[:self.layer_index]:
            x = layer(x)
        return x


@tf.function
def metric_entropy(x, fixed, scale, margin):
    """ See https://arxiv.org/pdf/1908.11184.pdf for the construction.
    """
    x         = tf.reshape(x, shape=[x.shape[0], 1, -1])
    fixed     = tf.reshape(fixed, shape=[1, fixed.shape[0], -1])
    distances = x - fixed
    distances = tf.reduce_sum(distances ** 2., axis=2)
    # distances = tf.minimum(distances, margin*margin)  # ONLY TO CROP PENALTY
    distances = distances / (scale**2.)  # normalize to avoid errors
    distances = tf.minimum(distances, 6.)  # avoid saturation (security)

    similarities = tf.math.exp(distances * (-1.))
    typicality   = tf.reduce_mean(similarities, axis=1)
    entropy      = -tf.reduce_mean(tf.math.log(typicality))
    return entropy


@tf.function
def ball_repulsion(x, fixed, scale, margin):
    """Average distance from each 'x' to 'fixed'.

    """
    x         = tf.reshape(x, shape=[x.shape[0], 1, -1])
    fixed     = tf.reshape(fixed, shape=[1, fixed.shape[0], -1])
    distances = x - fixed
    distances = tf.reduce_sum(distances ** 2., axis=2)
    distances = tf.minimum(distances, margin*margin)  # avoid saturation
    distances = distances / (scale**2.)  # just to normalize gradient
    sum_dists = tf.reduce_sum(distances, axis=1)  # square distance instead of hinge
    # because numerical instabilities otherwise :(
    return tf.reduce_mean(sum_dists)


def renormalize_grads(grads):
    return [tf.math.l2_normalize(grad) for grad in grads]


def uniform_noise(x_0, scale):
    coef = 0.3  # a bit farther away
    min_x0 = tf.reduce_min(x_0, axis=0, keepdims=True)
    max_x0 = tf.reduce_max(x_0, axis=0, keepdims=True)
    delta = max_x0 - min_x0
    min_x0 = min_x0 - coef * delta
    max_x0 = max_x0 + coef * delta
    return tf.random.uniform(x_0.shape, min_x0, max_x0)


@tf.function
def frontiere_distance(y, margin):
    return tf.math.abs(y + margin)  # to be minimized


@gin.configurable
def generate_adversarial(model, x_0, scale, margin, true_negative,
                         max_iter=gin.REQUIRED,
                         w_weight=gin.REQUIRED,
                         border  =gin.REQUIRED,
                         h_x_0   =gin.REQUIRED,
                         h_x     =gin.REQUIRED,
                         mult    =gin.REQUIRED,
                         logloss =gin.REQUIRED,
                         reversedlogloss=gin.REQUIRED):
    learning_rate = (mult * margin) / max_iter
    if true_negative:
        learning_rate = learning_rate * model._get_coef()
    optimizer     = SGD(learning_rate=learning_rate)  # no momentum required due to smooth optimization landscape

    # x_init is perturbed x_0, with atmost 10% of a gradient step (which can be admittely quite high)
    x_init = x_0 + 0.1*learning_rate*tf.math.l2_normalize(tf.random.uniform(x_0.shape, -1., 1.))
    x      = tf.Variable(initial_value=x_init, trainable=True)
    for _ in range(max_iter):
        with tf.GradientTape() as tape:
            try:
                y = model(x)
            except tf.python.framework.errors_impl.InvalidArgumentError as e:
                norm = tf.reduce_sum(x ** 2.)
                print('adversarial', norm, x)
                raise e
            
            if logloss:  # binary cross-entropy
                zeros = tf.zeros([int(y.shape[0]),1]) # seek frontiere
                ones = tf.ones([int(y.shape[0]),1])
                if reversedlogloss:
                    zeros, ones = ones, zeros
                if true_negative:  # where f is already negative, add examples
                    ce = tf.nn.sigmoid_cross_entropy_with_logits(zeros+0.5, y)
                else:  # otherwise f is positive but really shouldn't (most of the time), we remove those parts
                    ce = tf.nn.sigmoid_cross_entropy_with_logits(ones, y)
                adversarial_score = tf.reduce_mean(ce)
                if reversedlogloss:
                    adversarial_score = -adversarial_score
            else:  # wasserstein bro: sign is flipped because minimization of loss instead maximization of cost
                if true_negative:
                    adversarial_score = tf.reduce_mean(y)  # seek x of f(x) negative
                else:
                    adversarial_score = -tf.reduce_mean(y)  # seek x of f(x) positive

            loss = w_weight*adversarial_score
            if (h_x_0 + h_x + border) > 0.:
                fidelity   = -h_x_0 * ball_repulsion(x, x_0, scale, margin)  # avoid true positive, irrelevant
                dispersion = -h_x * metric_entropy(x, x, scale, margin)  # regularization to cover space
                frontiere_score = border * frontiere_distance(y, margin)
                loss = loss + dispersion + fidelity + frontiere_score  # minimize loss

        grad_f_x = tape.gradient(loss, [x])
        grad_f_x = renormalize_grads(grad_f_x)  # keep good learning rate
        optimizer.apply_gradients(zip(grad_f_x,[x]))
    return x.value()

"""
Two strategies with Binary Cross Entropy:

1) Maximize Error on Wrong Labels
    Maximize Error on Label 1 (an attack, catch f negative)
    Maximize Error on Label 0 (support, catch f positive)

2) Minimize Error on True Labels
    Minimize Error on Label 0 (a "soft" attack, catch f negative)
    Minimize Error on Label 1 (support, catch f positive)

For now we implemented Strategy 2) since it is more natural to minimize a loss
But strategy 1) may have some potential (warning NaN)
"""


@gin.configurable
def complement_distribution(model, x_0, scale, margin,
                            uniform  =gin.REQUIRED,
                            symmetric=gin.REQUIRED):
    if uniform:
        return uniform_noise(x_0, scale)
    x_false_positive = generate_adversarial(model, x_0, scale, margin, true_negative=False)
    if not symmetric:
        return x_false_positive
    x_true_negative  = generate_adversarial(model, x_0, scale, margin, true_negative=True)
    return tf.concat(values=[x_false_positive, x_true_negative], axis=0)
