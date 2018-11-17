"""Utility functions for plotting
"""
import torch
import numpy as np
from PIL.Image import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def display_image(image, label=None, ax=None, figsize=None, show_axis=True, cmap=None, data_format='channels_first'):
    """Display a single image
    :param image: `PIL.Image.Image`, `torch.Tensor`, or `numpy.ndarray`
    :param label: str, the name of the image, shown as title
    :param ax: matplotlib.Axes
    :param figsize: figure size, default None
    :param show_axis: whether show axis or not, default True
    :param cmap: colormap for 2D array, default None (use 'gray')
    :param data_format: 'channels_first' or 'channels_last'
    :return: `matplotlib.image.AxesImage`
    """
    assert isinstance(image, (Image, torch.Tensor, np.ndarray))
    assert data_format in ('channels_first', 'channels_last')
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if not show_axis: ax.set_axis_off()

    if isinstance(image, Image):
        ax.imshow(image)
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if isinstance(image, np.ndarray):
        err_str = "Image dimension can only be 2 or 3. Use display_images for multiple images."
        assert image.ndim in (2, 3), err_str
        if image.ndim == 3:
            n_channel = image.shape[0] if data_format == 'channels_first' else image.shape[-1]
            assert n_channel in (1, 3, 4)
            if n_channel == 1:
                image = image.squeeze()
            if (n_channel in (3, 4)) and (data_format == 'channels_first'):
                image = np.transpose(image, (1, 2, 0))
        if image.ndim == 2:
            cmap = 'gray' if cmap is None else cmap
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)
    if label is not None:
        if isinstance(label, torch.Tensor):
            label = label.item()
        label = str(label)
        ax.set_title(str(label))


def display_images(images, labels=None, subfig_size=1.5, figsize=None, ncols=None,
                   show_axis=False, cmap=None, data_format='channels_first'):
    assert isinstance(images, (list, tuple, dict, np.ndarray, torch.Tensor))
    if isinstance(images, dict):
        labels, images = images.items()
    if labels is not None:
        assert len(images) == len(labels)
    n = len(images)
    ncols = 8 if ncols is None else ncols
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        assert isinstance(subfig_size, (int, float, list, tuple))
        if isinstance(subfig_size, (int, float)):
            subfig_size = (subfig_size, subfig_size)
        assert len(subfig_size) == 2, "subfig_size should be a number or a tuple of two numbers (width, height)"
        figsize = (subfig_size[0] * ncols, subfig_size[1] * nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx < n:
                ax = axs[row][col]
                if labels is not None:
                    label = labels[idx]
                else:
                    label = None
                display_image(
                    images[idx], label=label, ax=ax,
                    show_axis=show_axis, cmap=cmap,
                    data_format=data_format
                )
            else:
                fig.delaxes(axs[row][col])
    plt.tight_layout()


def plot_one_batch_images(dataloader, show_label=True, use_label_name=True, subfig_size=1.5, figsize=None, ncols=None, **kwargs):
    """Plot one batch images of a DataLoader
    :param dataloader: PyTorch DataLoader of Images
    :param show_label: display image label as title
    :param use_label_name: automatically convert integer labels to class names
    :param subfig_size: a number of a tuple of two numbers, default 1.5
    :param figsize: a tuple of two numbers for figure width and height
    :param ncols: an integer for the number of columns
    :return: None
    """
    dataloader_iter = iter(dataloader)
    images, labels = dataloader_iter.next()
    if show_label and use_label_name and hasattr(dataloader, 'label_names'):
        labels = [dataloader.label_names[idx] for idx in labels]
    if show_label:
        display_images(images, labels, subfig_size=subfig_size, figsize=figsize, ncols=ncols, **kwargs)
    else:
        display_images(images, subfig_size=subfig_size, figsize=figsize, ncols=ncols, **kwargs)


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
        :param history: a dict of the train history
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
        :param keras_train_history:
        :param max_cols:
        :param subplot_height:
        :param subplot_width:
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
