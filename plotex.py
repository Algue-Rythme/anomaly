import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import seaborn as sns
import tensorflow as tf
import gin


@gin.configurable
def get_limits(X, coef=gin.REQUIRED):
    min_x = X[:,0].min()
    max_x = X[:,0].max()
    inf_x = min_x-coef*(max_x-min_x)
    sup_x = max_x+coef*(max_x-min_x)
    if X.shape[1] == 2:
        min_y = X[:,1].min()
        max_y = X[:,1].max()
        inf_y = min_y-coef*(max_y-min_y)
        sup_y = max_y+coef*(max_y-min_y)
    else:
        inf_y, sup_y = -1, 1
    return inf_x, sup_x, inf_y, sup_y

def get_xx_yy(X, model):
    inf_x, sup_x, inf_y, sup_y = get_limits(X)
    x = np.linspace(inf_x, sup_x, 300)
    y = np.linspace(inf_y, sup_y, 300)
    xx, yy = np.meshgrid(x, y, sparse=False)
    X_pred = np.stack((xx.ravel(), yy.ravel()),axis=1)
    pred = model(tf.constant(X_pred), training=True).numpy()
    pred = pred.reshape(x.shape[0],y.shape[0])
    return xx, yy, pred

def plot_levels(X, model, fig, index):
    fig.add_subplot(index)
    if X.shape[1] == 2:
        plt.xlabel('X')
        plt.ylabel('Y')
        xx, yy, pred = get_xx_yy(X, model)
        sns.scatterplot(x=X[:,0], y=X[:,1], alpha=0.1)
        cset = plt.contour(xx, yy, pred, cmap='twilight', levels=20)
        plt.clabel(cset, inline=1, fontsize=10)
    else:  # 1D case
        inf_x, sup_x, _, _ = get_limits(X, 0.3)
        x = np.linspace(inf_x, sup_x, 500)
        x = np.expand_dims(x, axis=1)
        x = tf.constant(x)
        x_pred = model(x, training=True)
        extreme = max(np.max(x_pred), -np.min(x_pred))
        plt.ylim(-extreme*1.2, extreme*1.2)
        plt.axhline(y=0., linewidth=2., color='red', linestyle='-')
        plt.plot(x, x_pred, marker='')


def plot3d(X, model, fig, index):
    if X.shape[1] != 2:
        return
    xx, yy, pred = get_xx_yy(X, model)
    # fig, ax = plt.subplots(ncols=2, subplot_kw={'projection':'3d'}, figsize=(20, 10))
    # surf = ax[0].plot_surface(xx, yy, pred, cmap=cm.coolwarm,
    #                           linewidth=0, antialiased=True)
    # ax[1].view_init(azim=0, elev=90)
    # surf=ax[1].plot_surface(xx, yy, pred,cmap='coolwarm',linewidth=0,antialiased=True)
    ax = fig.add_subplot(index, projection='3d')
    ax.plot_surface(xx, yy, pred, cmap=cm.coolwarm, linewidth=0, antialiased=True)
