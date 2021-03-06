import os
import json
import numpy as np

import geopandas as gpd

import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from affine import Affine

from PIL import Image


def load_annotations(file_name, crs):
    """Loads annotations stored in a geojson file.

    Args:
        file_name (str): name of the file.
        crs (dict or str): Output projection parameters
            as string or in dictionary form.

    Returns:
        geopandas.GeoDataFrame: a data frame containing the annotations
            (in reprojected coord system).
    """
    df = gpd.read_file(file_name)
    is_valid = (~df['condition'].isna()) & df.geometry.is_valid
    df = df.loc[is_valid, ['condition', 'geometry']]
    df = df.to_crs(crs)
    return df


def load_chip(dataset, col_min, row_min, cfg):
    """Loads a part of a chip.

    [lon, lat] or [x, y] corresponds to [width, height] and [col, row]

    Args:
        dataset (rasterio.DatasetReader): the opened raster file.
        col_min, row_min (int): min of col, row position in raster.
        cfg (argparse.Namespace): stores multiple configs.

    Returns:
        PIL.Image: the loaded chip image.
        rasterio.Affine: the updated transform.
    """
    # update transforms
    x_min, y_max = dataset.transform * (col_min, row_min)
    transform = Affine(
        dataset.transform.a * cfg.DOWN_RESOLUTION_FACTOR,
        dataset.transform.b,
        x_min,
        dataset.transform.d,
        dataset.transform.e * cfg.DOWN_RESOLUTION_FACTOR,
        y_max)
    # read array from raster
    raster_array = dataset.read(
        window=Window(col_off=col_min, row_off=row_min,
                      width=cfg.WINDOW_SIZE, height=cfg.WINDOW_SIZE),
        out_shape=(dataset.count, cfg.CHIP_SIZE, cfg.CHIP_SIZE),
        resampling=Resampling.bilinear)
    # raster array is color first, move that to color last
    im = Image.fromarray(np.moveaxis(raster_array, 0, 2), mode='RGB')
    return im, transform


def geocode2pixel(geom, transform):
    """Converts geocodes to pixel coordinates

    Args:
        geom (shapely.geometry.polygon.Polygon): geometry to be transformed.
        transform (rasterio.Affine): the transform to be applied.

    Returns:
        dict: a geometry that has been transformed.
            Exterior is a list of coordinates.
            [(x0, y0), (x1, y1), (x2, y2), ...]
            Interior is a list of lists of coordinates.
            Multipart polygon is not supported.
    """
    ext_coords = [(~transform) * coord for coord in geom.exterior.coords]
    int_coords = [[(~transform) * coord for coord in interior.coords]
                  for interior in geom.interiors]
    return {'exterior': ext_coords, 'interiors': int_coords}


def process_file(file_id, cfg):
    """Process one image corresponding to the file ID.

    Args:
        file_id (str): the id of the image file.
        cfg (argparse.Namespace): stores multiple configs.
    """
    with rasterio.open(os.path.join(cfg.IN_IMAGE_DIR,
                                    file_id + '.tif')) as dataset:
        # load all annotations
        df = load_annotations(
            file_name=os.path.join(cfg.IN_ANN_DIR, file_id + '.geojson'),
            crs=dataset.crs.data)
        # sampling random bounding boxes
        N = int((dataset.width / cfg.WINDOW_SIZE) *
                (dataset.height / cfg.WINDOW_SIZE) *
                cfg.SAMPLE_RATIO)
        col_mins = np.random.randint(dataset.width - cfg.WINDOW_SIZE + 1,
                                     size=N)
        row_mins = np.random.randint(dataset.height - cfg.WINDOW_SIZE + 1,
                                     size=N)
        # loop over sampled boxes
        for i, (col_min, row_min) in enumerate(zip(col_mins, row_mins)):
            # sample the chip
            x_min, y_max = dataset.transform * (col_min, row_min)
            x_max, y_min = dataset.transform * (col_min + cfg.WINDOW_SIZE,
                                                row_min + cfg.WINDOW_SIZE)
            # save annotations on the chip
            sliced = df.cx[x_min:x_max, y_min:y_max]
            im, transform = load_chip(dataset, col_min, row_min, cfg)
            im.save(os.path.join(cfg.OUT_IMAGE_DIR,
                                 '{}_s{:06d}.png'.format(file_id, i)))
            ann = {'width': cfg.CHIP_SIZE, 'height': cfg.CHIP_SIZE}
            ann['instances'] = [
                {'category': row['condition'],
                 'polygon': geocode2pixel(row['geometry'], transform)}
                for _, row in sliced.iterrows()]
            with open(os.path.join(cfg.OUT_ANN_DIR,
                                   ('{}_s{:06d}.json'
                                    .format(file_id, i))), 'w') as f:
                json.dump(ann, f)
