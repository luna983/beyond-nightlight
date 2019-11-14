import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from statsmodels.nonparametric.smoothers_lowess import lowess


def transform_coord(transform, to, xy=None, colrow=None):
    """Transforms x/y coord to/from col/row coord in a vectorized manner.

    Args:
        transform (affine.Affine): affine transformation
            (e.g., rasterio.io.DatasetReader.transform)
        to (str): in ['xy', 'colrow']
        xy (numpy.ndarray [N, 2]): x, y coords
        colrow (numpy.ndarray [N, 2]): col, row coords

    Returns:
        numpy.ndarray [N, 2]: transformed array
    """
    if to == 'colrow':
        t = np.array(~transform).reshape((3, 3))
        stacked = np.hstack((xy, np.ones((xy.shape[0], 1))))
        return t.dot(stacked.T).T[:, 0:2]
    elif to == 'xy':
        t = np.array(transform).reshape((3, 3))
        stacked = np.hstack((colrow, np.ones((colrow.shape[0], 1))))
        return t.dot(stacked.T).T[:, 0:2]
    else:
        raise NotImplementedError


def L(coords, A, h):
    """L function describing the clustering of the houses.
    
    Ignoring edge effects.

    Args:
        coords (numpy.ndarray [N, 2]): input coords
        A (float): area in lon/lat space
        h (float): bandwidth in lon/lat space

    Returns:
        float: L function value
    """
    N = coords.shape[0]
    return np.sqrt(
        ((cdist(coords, coords, 'euclidean') < h).sum() - N) * A / (N ** 2 * np.pi)) - h


def plot_scatter(col_x_key, col_y_key, col_x_label, col_y_label, df, out_dir,
                 transform_x=lambda x: x, transform_y=lambda x: x,
                 xlim=None, ylim=None, xticks=None, yticks=None,
                 xticklabels=None, yticklabels=None,
                 alpha=0.5, line=False, square=False, show=False):
    """Generates scatter plots w/ correlation coef.

    Args:
        col_x_key, col_y_key (str): keys for variables
        col_x_label, col_y_label (str): labels for axes
        df (pandas.DataFrame): stores the data
        out_dir (str): output directory
        transform_x, transform_y (function): how the two axes should be
            transformed
        xlim, ylim (tuple of float): limits of the transformed axes
        xticks, yticks, xticklabels, yticklabels (list)
        alpha (float): transparency
        line (bool): whether a non parametric fit line should be displayed
        square (bool): whether x and y axis should be the same and
            whether a 45 degree line should be plotted
        show (bool): whether to show the figure
    """

    cols = df.loc[:, [col_x_key, col_y_key]].dropna().astype('float').values
    coef = pearsonr(cols[:, 0], cols[:, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3))
    x = transform_x(cols[:, 0])
    y = transform_y(cols[:, 1])
    ax.plot(x, y, marker='o', color='slategrey', linestyle='None', alpha=alpha)
    if line:
        f = lowess(y, x, frac=0.3)
        ax.plot(f[:, 0], f[:, 1], '--', color='gray', linewidth=2)
    if square:
        ax.axis('square')
        ax.plot(xlim, ylim, '--', color='gray', linewidth=2)
    ax.set_title('Correlation (in levels): {:.2f}'.format(coef[0]))
    ax.set_xlabel(col_x_label)
    ax.set_ylabel(col_y_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ax.set_frame_on(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    ax.grid()
    plt.tight_layout()
    fig.savefig(os.path.join(
        out_dir, '{}_vs_{}.pdf'.format(col_x_key, col_y_key)))
    if show:
        plt.show()
    plt.close('all')


def gini(x):
    """Calculate the Gini coefficient of a numpy array.
    
    Args:
        x (1d numpy.ndarray): input array, need to be strictly positive

    Returns:
        float: gini coef
    """
    x_nonan = x[~np.isnan(x)]
    x_nonan = np.sort(x_nonan)
    assert len(x_nonan.shape) == 1
    assert x_nonan.min() > 0
    # number of array elements
    n = x_nonan.shape[0]
    # index per array element
    index = np.arange(n) + 1
    # Gini coefficient
    return np.sum((2 * index - n - 1) * x_nonan) / (n * np.sum(x_nonan))
