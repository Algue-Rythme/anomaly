import tensorflow as tf
import tensorflow.keras.activations as activations


@tf.function
def oneclass_hinge(margin, y):
    return tf.reduce_mean(activations.relu(margin - y))


@tf.function
def gradient_penalty(norm_nabla_f_x):
    return tf.reduce_mean((norm_nabla_f_x - 1.)**2.)


@tf.function
def get_grad_norm_with_tape(tape, y, x):
    nabla_f_x = tape.gradient(y, x)
    norm_nabla_f_x = tf.reduce_sum(x ** 2, axis=list(range(1, len(x.shape))), keepdims=True)
    return nabla_f_x, norm_nabla_f_x**0.5


@tf.function
def get_grad_norm(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    return get_grad_norm_with_tape(tape, y, x)


@tf.function
def order2_penalty(hessian):
    return tf.reduce_sum(hessian ** 2)


@tf.function
def get_orders_012(model, x):
    with tf.GradientTape(persistent=True) as order2_tape:
        order2_tape.watch(x)
        with tf.GradientTape(persistent=True) as order1_tape:
            order1_tape.watch(x)
            y = model(x)
        grad_x = order1_tape.gradient(y, x)
    hessian_x = order2_tape.batch_jacobian(grad_x, x)
    return y, grad_x, hessian_x

