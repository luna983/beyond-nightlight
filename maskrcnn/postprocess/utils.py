import numpy as np


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
