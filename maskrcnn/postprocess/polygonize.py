import os
import tqdm
import json
import numpy as np
import pandas as pd
import cv2 as cv
import shapely
import shapely.geometry
import shapely.ops
import rasterio
import rasterio.features
import pycocotools
import pycocotools.mask
import geopandas as gpd


SIZE_CHIP = (800, 800)
SIMPLIFY_PIXEL = 3
AREA_MIN = 0
SCORE_CUTOFF = 0
XMAX = 480
YMAX = 760

def load_ann(ann_file, img_file, extent, out_dir):
    """Load satellite annotation (predictions from ML models) and saves a geojson.

    Args:
        ann_file, img_file (str): paths to the annotation/img file
        extent (tuple): (lon_min, lat_min, lon_max, lat_max)
        our_dir (str): output directory
    """
    lon_min, lat_min, lon_max, lat_max = extent
    idx = os.path.basename(ann_file).split('.')[0]
    with open(ann_file, 'r') as f:
        ann = json.load(f)
    img = cv.imread(img_file)
    img = cv.resize(img, SIZE_CHIP)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # read the image
    output = []
    for ins in ann:
        binary_mask = pycocotools.mask.decode(ins['segmentation'])
        # extract simplified polygon from annotations
        poly = shapely.ops.unary_union([
            shapely.geometry.shape(geom)
            for geom, val in rasterio.features.shapes(binary_mask,
                                                      mask=binary_mask)
            if val == 1])
        # calculate angle
        coord_xs, coord_ys = poly.minimum_rotated_rectangle.exterior.xy
        coord_xs = np.array(coord_xs)
        coord_ys = np.array(coord_ys)
        angles = np.arctan2(coord_ys[1:] - coord_ys[:4],
                            coord_xs[1:] - coord_xs[:4]) / np.pi * 180
        angle, = angles[(angles >= 0) & (angles < 90)]
        # prepare polygon
        poly = poly.simplify(SIMPLIFY_PIXEL,
                             preserve_topology=False)
        if poly.is_empty:
            continue
        # extract RGB mean from image
        binary_mask = binary_mask.astype(np.bool_)
        RGB_mean = np.mean(img[binary_mask, :], axis=0)
        RGB_median = np.median(img[binary_mask, :], axis=0)
        # parse bbox
        xmin, ymin, width, height = ins['bbox']
        output.append({
            'geometry': poly,
            'angle': angle,
            'xmin': xmin,
            'ymin': ymin,
            'width': width,
            'height': height,
            'R_mean': RGB_mean[0],
            'G_mean': RGB_mean[1],
            'B_mean': RGB_mean[2],
            'R_median': RGB_median[0],
            'G_median': RGB_median[1],
            'B_median': RGB_median[2],
            'luminosity': np.mean([np.max(RGB_median), np.min(RGB_median)]) / 255,
            'saturation': (np.max(RGB_median) - np.min(RGB_median)) / 255 /
                           (1 - np.abs(2 * np.mean([np.max(RGB_median),
                                                    np.min(RGB_median)]) / 255 - 1)),
            'area': ins['area'],
            'score': ins['score'],
            'index': ins['image_id_str'],
            'category_id': ins['category_id'],
        })
    if len(output) == 0:
        return None
    # reproject
    output = gpd.GeoDataFrame(pd.DataFrame(output))
    a = (lon_max - lon_min) / SIZE_CHIP[0]
    e = - (lat_max - lat_min) / SIZE_CHIP[1]
    output['geometry'] = output['geometry'].affine_transform(
        (a, 0, 0, e, lon_min, lat_max))

    # drop low score predictions
    output = output.loc[output['score'] >= SCORE_CUTOFF, :]
    
    # drop small predictions
    output = output.loc[output['area'] >= AREA_MIN, :]

    # drop predictions on logos
    output = output.loc[~((output['xmin'] > XMAX) & (output['ymin'] > YMAX)), :]
    if output.shape[0] == 0:
        return None
    output.to_file(os.path.join(out_dir, idx + '.geojson'), driver='GeoJSON', index=False)
