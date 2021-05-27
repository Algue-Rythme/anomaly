from os import name
import deel
import tensorflow as tf
from deel.lip.initializers import BjorckInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D, LipschitzLayer, Condensable, InvertibleDownSampling, FrobeniusDense
from deel.lip.activations import MaxMin, GroupSort, GroupSort2, FullSort
from deel.lip.model import Sequential
from deel.lip.activations import PReLUlip, GroupSort, FullSort
from tensorflow.keras.layers import Input, Lambda, Flatten, AveragePooling2D, BatchNormalization, Conv2D, MaxPool2D, Dense
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.keras.initializers import Orthogonal


class ScaledL2NormPooling2D(AveragePooling2D, LipschitzLayer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        k_coef_lip=1.0,
        eps_grad_sqrt=1e-6,
        **kwargs
    ):
        if not ((strides == pool_size) or (strides is None)):
            raise RuntimeError("stride must be equal to pool_size")
        if padding != "valid":
            raise RuntimeError("NormalizedConv only support padding='valid'")
        if eps_grad_sqrt < 0.0:
            raise RuntimeError("eps_grad_sqrt must be positive")
        super(ScaledL2NormPooling2D, self).__init__(
            pool_size=pool_size,
            strides=pool_size,
            padding=padding,
            data_format=data_format,
            **kwargs
        )
        self.set_klip_factor(k_coef_lip)
        self.eps_grad_sqrt = eps_grad_sqrt
        self._kwargs = kwargs
    def build(self, input_shape):
        super(ScaledL2NormPooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True
    def _compute_lip_coef(self, input_shape=None):
        return np.sqrt(np.prod(np.asarray(self.pool_size)))
    @staticmethod
    def _sqrt(eps_grad_sqrt):
        @tf.custom_gradient
        def sqrt_op(x):
            sqrtx = tf.sqrt(x)
            def grad(dy):
                return dy / (2 * (sqrtx + eps_grad_sqrt))
            return sqrtx, grad
        return sqrt_op
    def call(self, x, training=None):
        return (
            ScaledL2NormPooling2D._sqrt(self.eps_grad_sqrt)(
                super(ScaledL2NormPooling2D, self).call(tf.square(x))
            )
            * self._get_coef()
        )
    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(ScaledL2NormPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OrthoConv2D(Conv2D, LipschitzLayer, Condensable):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        niter=3,  # 10
        beta=0.5,
        **kwargs
    ):
        """
        This class is a Conv2D Layer constrained such that all singular of it's kernel
        are 1. The computation based on BjorckNormalizer algorithm. As this is not
        enough to ensure 1 Lipschitzity a coertive coefficient is applied on the
        output.
        The computation is done in three steps:

        1. reduce the largest singular value to 1, using iterated power method.
        2. increase other singular values to 1, using BjorckNormalizer algorithm.
        3. divide the output by the Lipschitz bound to ensure k Lipschitzity.

        Args:
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            kernel_size: An integer or tuple/list of 2 integers, specifying the
                height and width of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the height and width.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            padding: one of `"valid"` or `"same"` (case-insensitive).
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, height, width, channels)` while `channels_first`
                corresponds to inputs with shape
                `(batch, channels, height, width)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".
            dilation_rate: an integer or tuple/list of 2 integers, specifying
                the dilation rate to use for dilated convolution.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Currently, specifying any `dilation_rate` value != 1 is
                incompatible with specifying any stride value != 1.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            kernel_constraint: Constraint function applied to the kernel matrix.
            bias_constraint: Constraint function applied to the bias vector.
            k_coef_lip: lipschitz constant to ensure
            niter_spectral: number of iteration to find the maximum singular value.
            niter_bjorck: number of iteration with BjorckNormalizer algorithm.

        This documentation reuse the body of the original keras.layers.Conv2D doc.
        """
        if not (
            (dilation_rate == (1, 1))
            or (dilation_rate == [1, 1])
            or (dilation_rate == 1)
        ):
            raise RuntimeError("NormalizedConv does not support dilation rate")
        if padding != "same":
            raise RuntimeError("NormalizedConv only support padding='same'")
        super(OrthoConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        self.niter = niter
        self.beta = beta
    def build(self, input_shape):
        super(OrthoConv2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=tf.keras.initializers.RandomNormal(0, 1),
            name="sn",
            trainable=False,
            dtype=self.dtype,
        )

        self.sig = self.add_weight(
            shape=tuple([1, 1]),  # maximum spectral  value
            name="sigma",
            trainable=False,
            dtype=self.dtype,
        )
        self.sig.assign([[1.0]])
        self.built = True
    def _compute_lip_coef(self, input_shape=None):
        return 1
    # @tf.function
    def Clip_OperatorNorm(self, conv, inp_shape):
        """

        Args:
            conv:
            inp_shape:
            beta: if beta is 1 we project tatally

        Returns:

        """
        # reshape to image size before fft
        conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
        conv_shape = conv.get_shape().as_list()
        padding = tf.constant([[0, 0], [0, 0],
                             [0, inp_shape[0] - conv_shape[0]],
                             [0, inp_shape[1] - conv_shape[1]]])
        crop_mask = tf.pad(tf.ones_like(conv_tr), padding, constant_values=(1-self.beta))
        conv_tr_padded = tf.pad(conv_tr, padding)
        for i in range(self.niter):
            # apply FFT
            transform_coeff = tf.signal.fft2d(conv_tr_padded)
            D, U, V = tf.linalg.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]))
            # norm = tf.reduce_max(D)
            # perform pseudo orthogonalization
            D_clipped = tf.cast(self.beta * tf.ones_like(D) + (1-self.beta) * D, tf.complex64)
            # reconstruct kernel in fft domain
            clipped_coeff = tf.matmul(U, tf.matmul(tf.linalg.diag(D_clipped),
                                                 V, adjoint_b=True))
            # perform inverse fft
            clipped_conv_padded = tf.signal.ifft2d(
              tf.transpose(clipped_coeff, perm=[2, 3, 0, 1]))
            # pseudo crop
            conv_tr_padded = tf.multiply(clipped_conv_padded, crop_mask)

        return tf.slice(tf.transpose(tf.math.real(conv_tr_padded), perm=[2, 3, 0, 1]),
                      [0] * len(conv_shape), conv_shape)#, norm
    def call(self, x, training=None):
        if training:
            W_bar = self.Clip_OperatorNorm(self.kernel, x.shape[1:])
        else:
            W_bar = self.kernel
        outputs = K.conv2d(
            x,
            W_bar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "niter": self.niter,
            "beta": self.beta,
        }
        base_config = super(OrthoConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def condense(self):
        self.kernel.assign(self.Clip_OperatorNorm(self.kernel, self._build_input_shape[1:]))
    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.kernel.assign(self.kernel.numpy() * self._get_coef())
        if self.use_bias:
            layer.bias.assign(self.bias.numpy())
        return layer

class Clipper(tf.keras.layers.Layer, LipschitzLayer):
    def __init__(self, inf, sup, k_coef_lip=1.):
        super(Clipper, self).__init__()
        self.set_klip_factor(k_coef_lip)
        self.inf = inf
        self.sup = sup
    def build(self, input_shape):
        super(Clipper, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True
    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer don't require a corrective factor
    def call(self, x, training=None):
        clipped = tf.clip_by_value(x, self.inf, self.sup)
        return clipped * self._get_coef()
    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "inf": self.inf,
            "sup": self.sup
        }
        base_config = super(Clipper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_cnn_baseline(input_shape, k_coef_lip, scale, niter_spectral, niter_bjorck, bjorck_forward):
    model = deel.lip.model.Sequential(
    [
        Input(shape=input_shape),
        SpectralConv2D(filters=4 * scale, kernel_size=(3, 3), data_format='channels_last'),
        GroupSort(n=2),
        SpectralConv2D(filters=4 * scale, kernel_size=(3, 3), data_format='channels_last'),
        GroupSort(n=2),
        ScaledL2NormPooling2D(pool_size=(2, 2), data_format='channels_last'),

        SpectralConv2D(filters=4 * scale, kernel_size=(3, 3), data_format='channels_last'),
        GroupSort(n=2),
        SpectralConv2D(filters=4 * scale, kernel_size=(3, 3), data_format='channels_last'),
        GroupSort(n=2),
        ScaledL2NormPooling2D(pool_size=(2, 2), data_format='channels_last'),

        Flatten(),
        SpectralDense(8 * scale, niter_spectral=niter_spectral, niter_bjorck=niter_bjorck, bjorck_forward=bjorck_forward),
        FullSort(),
        SpectralDense(8 * scale, niter_spectral=niter_spectral, niter_bjorck=niter_bjorck, bjorck_forward=bjorck_forward),
        FullSort(),
        SpectralDense(1, activation="linear", bjorck_forward=bjorck_forward),
    ],
    k_coef_lip=k_coef_lip,
    name='cnn_baseline')
    return model

def get_unconstrained_overfitter(input_shape, output_shape, k_coef_lip, scale, activation='relu', out_act='sigmoid'):
    model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8 * scale, activation=activation),
        tf.keras.layers.Dense(8 * scale, activation=activation),
        tf.keras.layers.Dense(8 * scale, activation=activation),
        tf.keras.layers.Dense(8 * scale, activation=activation),
        tf.keras.layers.Dense(8 * scale, activation=activation),
        tf.keras.layers.Dense(output_shape, activation=out_act),
        tf.keras.layers.Lambda(lambda x: x * k_coef_lip)
    ])
    return model

class MultiLipschitzHead(tf.keras.layers.Layer, Condensable):
    def __init__(self, num_heads, height, scale, niter_bjorck, niter_spectral, bjorck_forward, **kwargs):
        super().__init__(**kwargs)
        self._kwargs = kwargs
        self.num_heads = num_heads
        self.height = height
        self.scale = scale
        self.niter_bjorck = niter_bjorck
        self.niter_spectral = niter_spectral
        self.bjorck_forward = bjorck_forward
        heads = [[] for _ in range(1+2*self.height)]
        for _ in range(self.num_heads):
            for j in range(self.height):
                heads[2*j].append(SpectralDense(self.scale, niter_bjorck=self.niter_bjorck,
                                                niter_spectral=self.niter_spectral,
                                                bjorck_forward=self.bjorck_forward))
                heads[2*j+1].append(GroupSort2())
            heads[2*self.height].append(FrobeniusDense(1))
        self.heads = heads
    def call(self, x, training=False):
        ys = []
        for i in range(self.num_heads):
            y = x
            for j in range(self.height):
                y = self.heads[2*j][i](x, training=training)
                y = self.heads[2*j+1][i](y)
            y = self.heads[2*self.height][i](y)
            ys.append(y)
        y = tf.concat(ys, axis=1)
        return y
    def condense(self):
        for i in range(self.num_heads):
            for j in range(self.height):
                self.heads[2*j][i].condense()
            self.heads[2*self.height][i].condense()
    def stack_neck(self, layers, input_shape):
        layer = tf.keras.layers.Dense(
            units=layers[0].kernel.shape[-1]*self.num_heads,
            activation='linear',
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(input_shape)
        kernels = [head.kernel*head._get_coef() for head in layers]
        big_kernel = tf.concat(kernels, axis=-1)
        layer.kernel.assign(big_kernel)
        biases = [head.bias for head in layers]
        big_bias = tf.concat(biases, axis=-1)
        layer.bias.assign(big_bias.numpy())
        return layer
    def stack_head(self, layers, input_shape):
        layer = tf.keras.layers.Dense(
            units=layers[0].kernel.shape[-1]*self.num_heads,
            activation='linear',
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(input_shape)
        kernels = [head.kernel*head._get_coef() for head in layers]
        zeros = tf.zeros_like(kernels[0])
        n = len(kernels)
        kernels = [[zeros]*i+[ker]+[zeros]*(n-i-1) for i, ker in enumerate(kernels)]
        kernels = [tf.concat(ker_list, axis=0) for ker_list in kernels]
        big_kernel = tf.concat(kernels, axis=-1)
        print(input_shape, big_kernel.shape, layer.kernel.shape)
        layer.kernel.assign(big_kernel)
        biases = [head.bias for head in layers]
        big_bias = tf.concat(biases, axis=-1)
        layer.bias.assign(big_bias.numpy())
        return layer
    def vanilla_export(self):
        necks = [Input(shape=self.input_shape[1:])]
        prev_input_shape = self.input_shape
        for j in range(self.height):
            neck = self.stack_neck(self.heads[2*j], prev_input_shape)
            necks.append(neck)
            necks.append(GroupSort2())
            prev_input_shape = (None, self.heads[2*j+1][0].output_shape[-1]*self.num_heads)
        head = self.stack_head(self.heads[2*self.height], prev_input_shape)
        model = tf.keras.Sequential(necks+[head])
        model.build(self.input_shape)
        return model
    def get_config(self):
        config = {'num_heads':len(self.heads),
                  'height': self.height,
                  'scale': self.scale,
                  'niter_bjorck': self.niter_bjorck,
                  'niter_spectral': self.niter_spectral,
                  'bjorck_forward': self.bjorck_forward}
        base_config = super(MultiLipschitzHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Scaler(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Scaler, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Scaler, self).build(input_shape)
        self.built = True
        self.temperature = self.add_weight(
            shape=(1,)+input_shape[1:],
            initializer=tf.zeros_initializer,
            name="temperature",
            trainable=True,
            dtype=tf.float32,
        )
    def call(self, x, training=None):
        return x * tf.nn.softplus(self.temperature)
    def get_config(self):
        config = {}
        base_config = super(Scaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GlobalL2NormPooling2D(tf.keras.layers.Layer, LipschitzLayer):
    def __init__(
        self,
        data_format=None,
        k_coef_lip=1.0,
        **kwargs
    ):
        super(GlobalL2NormPooling2D, self).__init__(**kwargs)
        self.set_klip_factor(k_coef_lip)
        self._kwargs = kwargs
    def build(self, input_shape):
        super(GlobalL2NormPooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True
    def _compute_lip_coef(self, input_shape=None):
        return 1.
    def call(self, x, training=None):
        assert len(x.shape) == 4
        feature_map_norms = tf.sqrt(tf.reduce_sum(tf.square(x), axis=[-3, -2]))  # CHANNELS_LAST
        return feature_map_norms
    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(GlobalL2NormPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class UnitaryRowsDense(Dense, LipschitzLayer, Condensable):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=Orthogonal(gain=1.0, seed=None),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        **kwargs
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.set_klip_factor(k_coef_lip)
        self.axis = 0
        self._kwargs = kwargs
    def build(self, input_shape):
        self._init_lip_coef(input_shape)
        return super(UnitaryRowsDense, self).build(input_shape)
    def _compute_lip_coef(self, input_shape=None):
        return 1.0
    def call(self, x):
        norms = tf.norm(self.kernel, axis=self.axis, keepdims=True)
        W_bar = self.kernel / norms * self._get_coef()
        kernel = self.kernel
        self.kernel = W_bar
        outputs = Dense.call(self, x)
        self.kernel = kernel
        return outputs
    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(UnitaryRowsDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def condense(self):
        norms = tf.norm(self.kernel, axis=self.axis, keepdims=True)
        W_bar = self.kernel / norms * self._get_coef()
        self.kernel.assign(W_bar)
    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.kernel.assign(self.kernel.numpy() * self._get_coef())
        if self.use_bias:
            layer.bias.assign(self.bias.numpy())
        return layer


class ResidualBlock(tf.keras.layers.Layer, LipschitzLayer, Condensable):
    def __init__(self, scale, stride, conv11, params, **kwargs):
        super().__init__(**kwargs)
        self.path = [
            GroupSort2(),
            SpectralConv2D(scale * 1, (3, 3), strides=(stride, stride), **params),
            GroupSort2(),
            SpectralConv2D(scale * 1, (3, 3), strides=(1, 1))
        ]
        if conv11:
            self.last_conv = SpectralConv2D(scale * 1, (1, 1))
        else:
            self.last_conv = lambda x: x
        if stride > 1:
            self.pool = ScaledL2NormPooling2D(pool_size=(stride,stride))
        else:
            self.pool = None
    def condense(self):
        self.path[1].condense()
        self.path[3].condense()
        return super().condense()
    def call(self, x, training=None):
        y = x
        for layer in self.path:
            y = layer(y)
        if self.pool is not None:
            x = self.pool(x)
        y = tf.concat([x, y], axis=-1) / tf.math.sqrt(2, dtype=tf.float32)
        y = self.last_conv(y)
        return y


def make_resnet(ModelType, input_shape, output_shape, block_depths, block_widths, k_coef_lip,
                reduce_dims, conv11, niter_bjorck, niter_spectral, bjorck_forward, final_pooling):
    params = {'niter_bjorck':niter_bjorck, 'niter_spectral':niter_spectral, 'bjorck_forward':bjorck_forward}
    layers = [tf.keras.layers.Input(shape=input_shape)]
    def makeblocks(block_depth, block_width, stride):
        layers = [ResidualBlock(block_width, stride=stride, conv11=conv11, params=params)]
        for _ in range(block_depth-1):
            layers.append(ResidualBlock(block_width, stride=1, conv11=conv11, params=params))
        return layers
    for idx, (block_depth, block_width) in enumerate(zip(block_depths, block_widths)):
        stride = 2 if reduce_dims == 'stride' and idx != 0 else 1
        layers += makeblocks(block_depth, block_width, stride)
        if reduce_dims == 'stride' and idx+1 < len(block_depths):
            layers.append(ScaledL2NormPooling2D(pool_size=(2,2)))
    if final_pooling == 'flatten':
        layers.append(Flatten())
    else:
        layers.append(GlobalL2NormPooling2D)
    layers += [UnitaryRowsDense(output_shape)]
    return ModelType(layers, k_coef_lip=k_coef_lip, name='lipschitz_resnet')


class IfThen(tf.keras.layers.Layer):
    def __init__(self, temperature=1., **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    def _hard_if_then_else(self, x):
        x_shape = tf.shape(x)
        x = tf.reshape(x, [-1, 4])
        x_key, x_value = tf.split(x, num_or_size_splits=2, axis=-1)
        min_key = tf.argmin(x_key, axis=-1)
        max_key = tf.argmax(x_key, axis=-1)
        min_value = tf.gather(x_value, indices=min_key, axis=-1)
        max_value = tf.gather(x_value, indices=max_key, axis=-1)
        x_value = tf.stack([min_value, max_value], axis=-1)
        x = tf.concat([x_key, x_value], axis=-1)
        x = tf.reshape(x, x_shape)
        return x
    def _smooth_if_then_else(self, x):
        @tf.custom_gradient
        def custom_grad(dy):
            # Forward pass:
            #   [w, t, a, b] => (c, a, b) => [yw, yt, ya, yb]
            #
            # dy = [dy_yw, dy_yt, dy_ya, dy_yb]
            shape = tf.shape(dy)
            x = tf.reshape(x, [-1, 4])
            w, t, a, b = tf.split(x, num_or_size_splits=4, axis=-1)
            dy = tf.reshape(dy, [-1, 4])
            yw, yt, ya, yb = tf.split(dy, num_or_size_splits=4, axis=-1)

            c = self.temperature*(w - t)
            dya_a = tf.math.sigmoid(c)
            dyb_b = tf.math.sigmoid(-c)

            # symmetric because:
            # a - a² = (1-b) - (1-b)² = 1 - b -1 +2b -b² = b - b²

            dy_c = self.temperature*(a - b)
            dy_c = dy_c * (dyb_b - tf.square(dyb_b))  # symmetric

            dy_w = yw + dy_c*(ya - yb)
            dy_t = yt + dy_c*(yb - ya)
            dy_a = dya_a*ya
            dy_b = dyb_b*yb

            grad = tf.concat([dy_w, dy_t, dy_a, dy_b], axis=-1)
            grad = tf.reshape(grad, shape)
            return grad
        y = self._hard_if_then_else(x)
        return y, custom_grad
    def call(self, x):
        assert len(x) % 4 == 0
        return self._smooth_if_then_else(x)


def get_lipschitz_overfitter(ModelType, input_shape, output_shape, k_coef_lip, scale,
                             niter_bjorck, niter_spectral, groupsort,
                             conv, bjorck_forward, scaler, multihead,
                             deep, very_deep, final_pooling):
    Act = (lambda : GroupSort2()) if groupsort else (lambda : FullSort())
    layers = [tf.keras.layers.Input(shape=input_shape)]
    if conv:
        spectral_conv = True
        conv_maker = lambda *args, **kw: (SpectralConv2D(*args, **kw) if spectral_conv else OrthoConv2D(*args, **kw))
        pooling = 'scaled'
        if pooling == 'scaled':
            pooler = lambda : ScaledL2NormPooling2D(pool_size=(2,2))
        elif pooling == 'invertible':
            pooler = lambda : InvertibleDownSampling(pool_size=(2,2))
        layers += [
            conv_maker(scale * 1, (3, 3)),
            GroupSort2(),
            conv_maker(scale * 1, (3, 3)),
            GroupSort2(),
            pooler(),
            conv_maker(scale * 2, (3, 3)),
            GroupSort2(),
            conv_maker(scale * 2, (3, 3)),
            GroupSort2(),
            pooler(),
            conv_maker(scale * 4, (3, 3)),
            GroupSort2(),
            conv_maker(scale * 4, (3, 3)),
            GroupSort2(),
        ]
        if very_deep:
            layers += [
                pooler(),
                conv_maker(scale * 4, (2, 2)),
                GroupSort2(),
                conv_maker(scale * 4, (2, 2)),
                GroupSort2(),
            ]
        if pooling == 'invertible':
            layers.append(conv_maker(scale * 1, (1, 1)))
    if final_pooling == 'flatten':
        layers += [pooler(), tf.keras.layers.Flatten()]
    elif final_pooling == 'global':
        layers.append(GlobalL2NormPooling2D())
    dense_scale = min(512, 4 * scale)
    layers += [SpectralDense(dense_scale, niter_bjorck=niter_bjorck, niter_spectral=niter_spectral, bjorck_forward=bjorck_forward),
               Act()]
    if deep:
        layers += [SpectralDense(dense_scale, niter_bjorck=niter_bjorck, niter_spectral=niter_spectral, bjorck_forward=bjorck_forward),
                   Act()]
    if multihead is not None:
        heads_width = int(scale / (output_shape**0.5))  # heuristic
        layers.append(MultiLipschitzHead(output_shape, height=multihead, scale=heads_width,
                      niter_bjorck=niter_bjorck, niter_spectral=niter_spectral, bjorck_forward=bjorck_forward))
    else:
        layers += [
            SpectralDense(dense_scale, niter_bjorck=niter_bjorck,
                          niter_spectral=niter_spectral, bjorck_forward=bjorck_forward),
            Act(),
            UnitaryRowsDense(output_shape),
        ]
    if scaler:
        layers.append(Scaler())
    model = ModelType(layers, k_coef_lip=k_coef_lip, name='overfitter')
    return model

