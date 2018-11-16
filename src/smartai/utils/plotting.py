"""Utility functions for plotting
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def display_image(data, nrows=None, ncols=None, **kwargs):
    pass


def plot_matrix(m, width=5, title=None, vmin=-1, vmax=1, cmap='Greens', show_ticks=True):
    height = m.shape[0] / m.shape[1] * width
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
    im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap)
    if title:
        ax.set_title(title)
    if not show_ticks:
        plt.xticks([])
        plt.yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def _plot_train_history(epochs, history, has_validation, max_cols, subplot_width, subplot_height):
    """Plot the train train_history

    Args:
        train_history: a `keras.callbacks.History` object
        :param max_cols:
        :param train_history:
    """
    metrics = history.keys()

    if has_validation:
        used_metrics = [metric for metric in metrics if not metric.startswith('val_')]
    else:
        used_metrics = metrics

    n_plot = len(used_metrics)
    if n_plot <= max_cols:
        rows, cols = 1, n_plot
    else:
        rows, cols = np.ceil(n_plot / max_cols), max_cols

    w, h = subplot_width * cols, subplot_height * rows
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(w, h))
    if n_plot == 1:
        axs = [axs]

    for i, metric in enumerate(used_metrics):
        metric_str = ' '.join(metric.split('_')).title()
        train_label = 'train {}'.format(metric_str.lower())
        axs[i].plot(
            epochs, history[metric],
            label=train_label,
            linestyle='-', marker='o',
            markersize=2, color='b'
        )
        if has_validation:
            val_metric = 'val_{}'.format(metric)
            val_label = 'validation {}'.format(metric_str.lower())
            axs[i].plot(
                epochs, history[val_metric],
                label=val_label,
                linestyle='--', marker='o',
                markersize=2, color='g'
            )
        axs[i].set_title('Train History: {}'.format(metric_str))
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric_str)
        axs[i].legend()



def plot_keras_train_history(keras_train_history, max_cols=2, subplot_width=6, subplot_height=5):
    """Plot the train train_history from `keras.models.Model.fit` fucntion

    Args:
        train_history: a `keras.callbacks.History` object
        :param max_cols:
        :param keras_train_history:
    """
    epochs = keras_train_history.epoch
    history = keras_train_history.history
    has_validation = keras_train_history.params['do_validation']
    _plot_train_history(
        epochs, history,
        has_validation,
        max_cols=max_cols,
        subplot_width=subplot_width,
        subplot_height=subplot_height
    )


def plot_pytorch_train_history(pytorch_train_history, max_cols=2, subplot_width=6, subplot_height=5):
    epochs = range(pytorch_train_history.epoches)
    history = pytorch_train_history.history
    has_validation = pytorch_train_history.has_validation
    _plot_train_history(
        epochs, history,
        has_validation,
        max_cols=max_cols,
        subplot_width=subplot_width,
        subplot_height=subplot_height
    )
