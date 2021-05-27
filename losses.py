import tensorflow as tf
import wandb


class CrossEntropyT(tf.keras.losses.Loss):
    def __init__(self, temperatures):
        super().__init__(name='CET')
        self.temperatures = tf.expand_dims(temperatures, axis=0)
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype='int64')
        y_true = tf.squeeze(y_true, axis=1)
        y_pred = y_pred * self.temperatures
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
        return tf.reduce_mean(losses)

class MultiClassHKR(tf.keras.losses.Loss):
    def __init__(self, alpha, margins):
        super().__init__(name='multihkr')
        self.depth   = len(margins)
        self.margins = tf.expand_dims(margins, axis=0)
        self.indices = tf.expand_dims(tf.range(self.depth, dtype=tf.int64), axis=0)
        self.alpha   = alpha
    def call(self, y_true, y_pred):
        y_true  = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int64)
        mask    = tf.one_hot(y_true, depth=self.depth, dtype=tf.float32)
        signs   = 2*mask - 1
        y       = signs * y_pred

        # hinge + wasserstein
        hinge   = tf.nn.relu(self.margins - y)
        wass    = -y

        # combination
        losses  = hinge + (1. / self.alpha)*wass

        # correction on the number of classes
        weights = mask*(self.depth-2) + 1.
        losses  = weights * losses

        # average over the batch
        loss = tf.reduce_mean(losses)
        return loss

class MinMarginHKR(tf.keras.losses.Loss):
    def __init__(self, alpha, margins, num_batchs, perc):
        super().__init__(name='multiminhkr')
        self.depth   = len(margins)
        self.margins = tf.Variable(
            initial_value=tf.expand_dims(margins, axis=0),
            trainable=False,
        )
        self.indices = tf.expand_dims(tf.range(self.depth, dtype=tf.int64), axis=0)
        self.alpha   = alpha
        self.perc    = None if alpha != 'adaptive' else perc
        if self.perc is not None:
            self.lr      = 5**(1/num_batchs) - 1
            self.gaussian_prior = 0.3
            self.min_margin = 0.001  # very small non null
            self.alpha = 100 / (100 - perc)
            self.pop = [t for t in tf.split(margins, num_or_size_splits=self.depth)]
            self.buffer_size = 1000
    
    def preprocess_margins(self, y_true, y_pred):
        y_true       = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int64)
        mask         = tf.one_hot(y_true, depth=self.depth, dtype=tf.float32)
        y_pred_false = y_pred - mask * tf.reduce_max(y_pred, axis=1, keepdims=True)
        max_others   = tf.reduce_max(y_pred_false, axis=1)
        y_pred_true  = tf.reduce_sum(y_pred * mask, axis=1)
        margin_other = y_pred_true - max_others
        return margin_other, y_true
    
    def greedy_update_margins(self, y_true, y_pred):
        # instabilities during learning
        if self.perc is None:
            return
        margin_other, y_true = self.preprocess_margins(y_true, y_pred)
        new_margins  = []
        for cl in range(self.depth):
            selected   = tf.boolean_mask(margin_other, y_true == cl)
            batch_size = tf.cast(tf.shape(selected), dtype=tf.float32)
            idx      = tf.cast(self.perc*batch_size, dtype=tf.int64)
            emp_perc = tf.gather(tf.sort(selected), idx)
            emp_std  = tf.math.reduce_std(selected)
            emp_mean = tf.reduce_mean(selected)
            th_perc  = emp_std*(2**0.5)*tf.math.erfinv(2*self.perc/100-1) + emp_mean
            perc     = emp_perc*(1-self.gaussian_prior) + th_perc*self.gaussian_prior
            new_margins.append(perc)
        grad = tf.stack(new_margins, axis=1)
        update = (1-self.lr)*self.margins + self.lr * grad
        update = tf.maximum(update, self.min_margin)
        self.margins.assign(update)
    
    def histogram_update_margins(self, y_true, y_pred):
        margin_other, y_true = self.preprocess_margins(y_true, y_pred)
        margins = []
        for cl in range(self.depth):
            selected = tf.boolean_mask(margin_other, y_true == cl)
            pop = tf.concat([tf.squeeze(selected),self.pop[cl]])
            pop = tf.sort(pop)
            batch_size = tf.cast(tf.size(pop), dtype=tf.float32)
            idx = tf.cast(self.perc*batch_size, dtype=tf.int64)
            emp_perc = tf.gather(tf.sort(selected), idx)
            margins.append(emp_perc)
            if tf.size(pop) > self.buffer_size:
                indexes = tf.random.uniform(self.buffer_size, minval=0,
                                            maxval=tf.size(pop), dtype=tf.int64)
                pop = tf.gather(pop, indexes)
            self.pop[cl] = pop
        grad = tf.stack(margins, axis=1)
        update = (1-self.lr)*self.margins + self.lr * grad
        update = tf.maximum(update, self.min_margin)
        self.margins.assign(update)
    
    def call(self, y_true, y_pred):
        y_true  = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int64)
        mask    = tf.one_hot(y_true, depth=self.depth, dtype=tf.float32)
        signs   = 2*mask - 1
        y       = signs * y_pred

        # wasserstein
        weights = mask*(self.depth-2) + 1.
        wass    = -y * weights
        wass    = tf.reduce_mean(wass, axis=1)

        # hinge
        y_pred_false = y_pred - mask * tf.reduce_max(y_pred, axis=1, keepdims=True)
        max_others   = tf.reduce_max(y_pred_false, axis=1)
        y_pred_true  = tf.reduce_sum(y_pred * mask, axis=1)
        margin_other = y_pred_true - max_others
        true_margin  = tf.reduce_sum(self.margins * mask, axis=1)
        hinge        = tf.nn.relu(true_margin - margin_other)

        # combination
        losses  = hinge + (1. / self.alpha)*wass

        # average over the batch
        loss = tf.reduce_mean(losses)
        return loss

class MarginWatcher(tf.keras.metrics.Metric):
    def __init__(self, loss_obj, use_wandb, name='margins', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_obj = loss_obj
        self.use_wandb = use_wandb
    def update_state(self, y_true, y_pred, sample_weight=None):
        return
    def result(self):
        return tf.reduce_mean(self.loss_obj.margins)
    def reset_states(self):
        if not self.use_wandb:
            return
        margins = wandb.Histogram(tf.squeeze(self.loss_obj.margins))
        wandb.log({'margins_hist':margins}, commit=False)        


class TopKMarginHKR(tf.keras.losses.Loss):
    def __init__(self, alpha, margins, top_k):
        super().__init__(name='multitopkhkr')
        self.depth   = len(margins)
        self.margins = tf.Variable(
            initial_value=tf.expand_dims(margins, axis=0),
            trainable=False,
        )
        self.indices = tf.expand_dims(tf.range(self.depth, dtype=tf.int64), axis=0)
        self.alpha   = alpha
        self.top_k   = top_k
    def call(self, y_true, y_pred):
        y_true  = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int64)
        mask    = tf.one_hot(y_true, depth=self.depth, dtype=tf.float32)
        signs   = 2*mask - 1
        y       = signs * y_pred

        # wasserstein
        weights = mask*(self.depth-2) + 1.
        wass    = -y * weights
        wass    = tf.reduce_mean(wass, axis=1)

        # hinge
        y_pred_false = y_pred - mask * tf.reduce_max(y_pred, axis=1, keepdims=True)
        max_others, _= tf.math.top_k(y_pred_false, k=self.top_k, sorted=False)
        y_pred_true  = tf.reduce_sum(y_pred * mask, axis=1, keepdims=True)
        margin_other = y_pred_true - max_others
        true_margin  = tf.reduce_sum(self.margins * mask, axis=1, keepdims=True)
        hinge        = tf.nn.relu(true_margin - margin_other)
        hinge        = tf.reduce_mean(hinge, axis=1)

        # combination
        losses  = hinge + (1. / self.alpha)*wass

        # average over the batch
        loss = tf.reduce_mean(losses)
        return loss
