import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import seaborn as sns


def get_xx_yy(X, Y, model):
    x = np.linspace(X[:,0].min()-2, X[:,0].max()+2, 200)
    y = np.linspace(X[:,1].min()-2, X[:,1].max()+2, 200)
    xx, yy = np.meshgrid(x, y, sparse=False)
    X_pred = np.stack((xx.ravel(), yy.ravel()),axis=1)
    pred = model(X_pred).numpy()
    pred = pred.reshape(x.shape[0],y.shape[0])
    return xx, yy, pred


def plot_levels(X, Y, model):
    xx, yy, pred = get_xx_yy(X, Y, model)
    fig  = plt.figure(figsize=(10,7))
    ax1  = fig.add_subplot(111)
    sns.scatterplot(x=X[:,0], y=X[:,1], alpha=0.1, ax=ax1)
    cset = ax1.contour(xx, yy, pred, cmap='twilight', levels=20)
    ax1.clabel(cset, inline=1, fontsize=10)


def plot3d(X, Y, model):
    xx, yy, pred = get_xx_yy(X, Y, model)
    fig, ax = plt.subplots(ncols=2, subplot_kw={'projection':'3d'}, figsize=(20, 10))
    #ax[0] = fig.gca(projection='3d')
    #ax[0].view_init(azim=1,elev=45)
    surf = ax[0].plot_surface(xx, yy, pred, cmap=cm.coolwarm,
                              linewidth=0, antialiased=True)
    ax[1].view_init(azim=0, elev=90)
    surf=ax[1].plot_surface(xx, yy, pred,cmap='coolwarm',linewidth=0,antialiased=True)
