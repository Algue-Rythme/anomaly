import tensorflow as tf
from deel.lip.layers import SpectralDense, SpectralConv2D, LipschitzLayer, Condensable, InvertibleDownSampling, FrobeniusDense
from deel.lip.utils import CUSTOM_OBJECTS
from models import MultiLipschitzHead


def OrthogonalOptimizer(BaseOpt, **kwargs):
    class OrthogonalOptimizerClass(BaseOpt):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def _resource_apply_dense(self, grad, var):
            grad = stiefel_manifold(grad, var)
            return super()._resource_apply_dense(grad, var)
        def _resource_apply_sparse(self, grad, var):
            grad = stiefel_manifold(grad, var)
            return super()._resource_apply_sparse(grad, var)
    return OrthogonalOptimizerClass(**kwargs)

class CondenseDenseCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    def on_train_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            if 'dense' in layer.name or 'multihead' in layer.name:
                layer.condense()
        super(CondenseDenseCallback, self).on_train_batch_end(batch, logs)


def stiefel_manifold(grad, var_orig, beta=0.5):
    is_spectral = ('spectral' in var_orig.name)  # or ('spectral_conv2d' in var.name)
    if not is_spectral or ('kernel' not in var_orig.name):
        return grad
    # print('Stiefel on ', var.name)
    flat_shape = (-1, tf.shape(grad)[-1])
    grad = tf.reshape(grad, flat_shape)
    var = tf.reshape(var_orig, flat_shape)
    assert len(grad.shape) == 2
    ogo = tf.linalg.matmul(var, grad, transpose_b=True)@var
    if var.shape[0] == var.shape[1]:
        return (1-beta)*grad - beta*ogo
    if var.shape[0] > var.shape[1]:
        proj_og = tf.linalg.matmul(var, var, transpose_b=True)
        proj_og = proj_og @ var
    else:
        proj_og = tf.linalg.matmul(var, var, transpose_a=True)
        proj_og = var @ proj_og
    grad = grad-beta*(ogo + proj_og)  # final projection
    grad = tf.reshape(grad, tf.shape(var_orig))
    return grad

def sym(x):
    return 0.5*(tf.transpose(x)+x)

def retraction_operator(grad, var_orig):
    is_spectral = ('spectral' in var_orig.name)  # or ('spectral_conv2d' in var.name)
    if not is_spectral or ('kernel' not in var_orig.name):
        return grad
    flat_shape = (-1, tf.shape(grad)[-1])
    grad = tf.reshape(grad, flat_shape)
    var = tf.reshape(var_orig, flat_shape)
    assert len(grad.shape) == 2
    z = sym(tf.linalg.matmul(var, grad, transpose_a=True))
    z = tf.linalg.matmul(var, z)
    grad = grad - z
    grad = tf.reshape(grad, tf.shape(var_orig))
    return grad

def preprocess_grads(gradients, trainable_vars, retraction=False):
    if retraction:
        return [retraction_operator(grad, var) for grad, var in zip(gradients, trainable_vars)]
    else:
        return [stiefel_manifold(grad, var) for grad, var in zip(gradients, trainable_vars)]


class ArmijoCheap:
    def __init__(self, batch_size, train_size, force_independance, weight_latency):
        self.opt = tf.keras.optimizers.SGD(learning_rate=1.)
        self.c = 0.1
        self.beta = 0.9
        self.gamma = 2 ** (batch_size / train_size)
        self.eta = 1.
        self.max_eta = 10.
        self.min_eta = 1e-6
        self.old_eta = None
        # if force_independance=True we use previous batch as a proxy to evaluate generalization
        # of the current gradient update => no theoritical garantee whatsoever
        self.force_independance = force_independance
        self.weight_latency = weight_latency
        self.prev_x = None
        self.prev_y = None
    def update_model(self, model):
        self.model = model
    def c_grad_norm(self, gradients):
        return self.c * tf.linalg.global_norm(gradients)**2.
    def wolfe_condition(target, source, norm, eta):
        return tf.less_equal(target, source - eta*norm)
    @staticmethod
    def rescale(factor, grads):
        return map(lambda grad: factor*grad, grads)
    def try_step(self, old_weights, gradients, eta, trainable_vars):
        new_step = ArmijoExpensive.rescale(eta, gradients)
        self.model.set_weights(old_weights)  # reset old weights
        self.opt.apply_gradients(zip(new_step, trainable_vars))  # gradient step
        self.model.condense()  # gradient step correction
    def update_lr(self, new_source, new_x, new_labels, gradients):
        if self.prev_x is None:
            self.prev_x = new_x
            self.prev_y = new_labels
        # use previous batch to evaluate quality of gradients of current batch
        if self.force_independance:
            x = self.prev_x
            labels = self.prev_y
            source = self.model.compiled_loss(labels, self.model(x, training=True))
        else:
            x = new_x
            labels = new_labels
            source = new_source
        self.old_eta = self.eta
        norm = self.c_grad_norm(gradients)
        old_weights = self.model.get_weights()
        trainable_vars = self.model.trainable_variables
        eta = self.eta * self.gamma
        self.try_step(old_weights, gradients, eta, trainable_vars)
        target = self.model.compiled_loss(labels, self.model(x, training=True))
        while not ArmijoCheap.wolfe_condition(target, source, norm, eta):
            if eta < self.min_eta:
                break
            eta *= self.beta
            self.try_step(old_weights, gradients, eta, trainable_vars)
            target = self.model.compiled_loss(labels, self.model(x, training=True))
        self.model.set_weights(old_weights)
        self.eta = min(self.max_eta, eta)
        self.prev_x = new_x
        self.prev_y = new_labels
    def __call__(self):
        if self.weight_latency:
            if self.old_eta is None:
                return self.eta
            return self.old_eta
        return self.eta  # current batch for gradient direction and step size


class ArmijoMetric(tf.keras.metrics.Metric):
    def __init__(self, armijo, name='lr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.armijo = armijo
    def update_state(self, y_true, y_pred, sample_weight=None):
        pass
    def result(self):
        return self.armijo()
    def reset_states(self):
        pass


class ArmijoExpensive:
    def __init__(self, batch_size, train_size):
        self.opt = tf.keras.optimizers.SGD(learning_rate=1.)
        self.c = 0.05  # threshold
        self.beta = 0.9  # decrease factor
        self.gamma = 2 ** (batch_size / train_size)  # reset policy
        self.eta = 1.  # a bit high ?
        self.max_eta = 10.  # hardcap
        self.min_eta = 1e-6  # soft cap
    def update_model(self, model):
        self.model = model
    def c_grad_norm(self, x, y_true, trainable_vars):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.model.compiled_loss(y_true, y_pred)
        gradients = tape.gradient(loss, trainable_vars)
        norm = self.c * tf.linalg.global_norm(gradients)**2.
        return loss, norm, gradients
    @staticmethod
    def wolfe_condition(target, source, norm, eta):
        return tf.less_equal(target, source - eta*norm)
    @staticmethod
    def rescale(factor, grads):
        return map(lambda grad: factor*grad, grads)
    def try_step(self, old_weights, gradients, eta, trainable_vars):
        new_step = ArmijoExpensive.rescale(eta, gradients)
        self.model.set_weights(old_weights)  # reset old weights
        self.opt.apply_gradients(zip(new_step, trainable_vars))  # gradient step
        self.model.condense()  # gradient step correction
    def update_lr(self, x, labels):
        trainable_vars = self.model.trainable_variables
        source, norm, gradients = self.c_grad_norm(x, labels, trainable_vars)  # another backward :(
        old_weights = self.model.get_weights()
        eta = self.eta * self.gamma  # increase just in case
        self.try_step(old_weights, gradients, eta, trainable_vars)
        target = self.model.compiled_loss(labels, self.model(x, training=True))
        while not ArmijoExpensive.wolfe_condition(target, source, norm, eta):
            if eta < self.min_eta:  # too small
                break
            eta *= self.beta  # decrease
            self.try_step(old_weights, gradients, eta, trainable_vars)
            target = self.model.compiled_loss(labels, self.model(x, training=True))
        self.model.set_weights(old_weights)
        self.eta = min(self.max_eta, eta)
    def __call__(self):
        return self.eta


class Armijo(tf.keras.optimizers.Optimizer):
    def __init__(self, batchs_per_epoch, c=0.1, condense=True, polyak_momentum=None,
                 **kwargs):
        super().__init__(name='armijo', **kwargs)
        self.opt = tf.keras.optimizers.SGD()
        self.condense = condense
        self.c = c
        self.beta = 0.9
        self.batchs_per_epoch = batchs_per_epoch
        self.gamma = 2 ** (1/batchs_per_epoch)
        self.eta = 1.
        self.max_eta = 10.
        self.min_eta = 1e-6
        self.old_weights = None
        self.polyak_momentum = polyak_momentum
    def wolfe_condition(target, source, norm, eta):
        return tf.less_equal(target, source - eta*norm)
    def c_grad_norm(self, gradients):
        return self.c * tf.linalg.global_norm(gradients)**2.
    def step(self, model, source, x, labels, gradients):
        cur_weights = model.get_weights()
        trainable_vars = model.trainable_variables
        if self.polyak_momentum is not None:
            very_old_weights = self.old_weights
            self.old_weights = [var.value() for var in trainable_vars]
        norm = self.c_grad_norm(gradients)
        closure = lambda: model.compiled_loss(labels, model(x, training=True))
        self.eta *= self.gamma
        target = self.try_sgd_step(model, trainable_vars, cur_weights, gradients, closure)
        while not ArmijoExpensive.wolfe_condition(target, source, norm, self.eta):
            self.eta *= self.beta
            if self.eta < self.min_eta or self.eta > self.max_eta:
                break
            target = self.try_sgd_step(model, trainable_vars, cur_weights, gradients, closure)
        self.eta = min(max(self.min_eta, self.eta), self.max_eta)
        if self.polyak_momentum is not None:
            self.apply_polyak(model, very_old_weights, self.old_weights, trainable_vars)
    def try_sgd_step(self, model, trainable_vars, cur_weights, gradients, closure):
        self.opt.learning_rate.assign(self.eta)
        model.set_weights(cur_weights)  # reset old weights
        self.opt.apply_gradients(zip(gradients, trainable_vars))  # gradient step
        if self.condense and self.polyak_momentum is None:
            model.condense()  # gradient step correction
        return closure()  # forward pass
    def apply_polyak(self, model, old_weights, cur_weights, trainable_vars):
        if old_weights is None:
            return
        gradients = [-(wt1 - wt0) for wt1, wt0 in zip(cur_weights, old_weights)]
        gradients = preprocess_grads(gradients, trainable_vars, retraction=True)
        self.opt.learning_rate.assign(self.polyak_momentum)
        self.opt.apply_gradients(zip(gradients, trainable_vars))
        if self.condense:
            model.condense()  # gradient step correction
    def _resource_apply_dense(self, grad, var):
        pass
    def _resource_apply_sparse(self, grad, var):
        pass
    def __call__(self):
        return self.eta
    def get_config(self):
        config = {
            "batchs_per_epoch":self.batchs_per_epoch,
            "c":self.c,
            "condense":self.condense,
            "polyak_momentum":self.polyak_momentum,
        }
        base_config = super(Armijo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

