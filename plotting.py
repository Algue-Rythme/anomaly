import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot(config, launch_train, binary, training):
    if training:
        df = pd.DataFrame()
        for k_coef_lip, margin in zip([0.125, 0.25, 0.5, 1., 2., 4.], [0.5, 1., 2., 4., 8., 16.]):
            if config.loss_type == 'bce':
                history, _ = launch_train(config.net_type, k_coef_lip, 1.)
                key = f"{k_coef_lip:.3f}"
            else:
                history, _ = launch_train(config.net_type, 1., margin)
                key = f"{margin:.3f}"
            df['loss_'+key] = np.array(history.history['loss'])
            if config.loss_type == 'hkr' and binary:
                df['error_'+key] = 1. - np.array(history.history['HKR_binary_accuracy'])
            elif config.loss_type == 'hinge' and binary:
                df['error_'+key] = 1. - np.array(history.history['bestacc'])
            else:
                df['error_'+key] = 1. - np.array(history.history['accuracy'])
        df.to_csv(f'images/{config.dataset_name}_{config.loss_type}_{config.net_type}.csv')
    else:
        df = pd.read_csv(f'images/{config.dataset_name}_{config.loss_type}_{config.net_type}.csv')
        df.index = np.arange(1, len(df) + 1)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('"Dogs vs Cats" Train Set: '+config.loss_type.upper(), fontsize=20)
    for prefix, axe in zip(['loss_', 'error_'], axes):
        cols = [col for col in df if col.startswith(prefix)]
        ylabel = prefix[:-1]
        if config.loss_type == 'bce':
            to_print = df[cols].dropna().rename(columns=(lambda s: 'L='+s.split('_')[-1]))
        elif 'hkr' in config.loss_type or 'hinge' in config.loss_type:
            to_print = df[cols].dropna().rename(columns=(lambda s: 'margin='+s.split('_')[-1]))
        else:
            to_print = df[cols].dropna().rename(columns=(lambda s: 'temperature='+s.split('_')[-1]))
        if 'error' in prefix:
            logy = None
            ylim = (0, 1.)
        else:
            logy = True
            ylim = None
        to_print.plot(ax=axe, xlabel='step', ylabel=ylabel, grid=True, ylim=ylim, colormap='viridis', linewidth=3, logy=logy)
        axe.legend(loc='best', prop={'size': 16})
        axe.set_xlabel('epoch', fontsize=22)
        axe.set_ylabel(ylabel, fontsize=22)
        axe.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(f'images/{config.dataset_name}_{config.loss_type}_{config.net_type}.png')
    plt.show()
