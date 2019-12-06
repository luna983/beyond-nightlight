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
SCORE_CUTOFF = 0.9
XMAX = 480
YMAX = 760


def load_anns(ann_files, img_files, idx_file):
    """Load satellite annotations (predictions from ML models).

    Args:
        ann_files, img_files (list of str): paths to all annotation/img files
        idx_file (str): path to index (metadata) file

    Returns:
        geopandas.GeoDataFrame: annotations
    """
    # read image index data frame
    df_idx = pd.read_csv(idx_file)
    df_idx.set_index('index', inplace=True)
    df = []
    for ann_file, img_file in tqdm.tqdm(zip(ann_files, img_files)):
        idx = os.path.basename(ann_file).split('.')[0]
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        img = cv.imread(img_file)
        img = cv.resize(img, SIZE_CHIP)
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
            poly = poly.simplify(SIMPLIFY_PIXEL,
                                 preserve_topology=False)
            if poly.is_empty:
                continue
            # calculate angle
            coord_xs, coord_ys = poly.minimum_rotated_rectangle.exterior.xy
            coord_xs = np.array(coord_xs)
            coord_ys = np.array(coord_ys)
            angles = np.arctan2(coord_ys[1:] - coord_ys[:4],
                                coord_xs[1:] - coord_xs[:4]) / np.pi * 180
            angle, = angles[(angles >= 0) & (angles < 90)]
            # extract RGB mean from image
            binary_mask = (binary_mask * 255).astype(np.uint8)
            RGB_mean = cv.mean(img, binary_mask)[0:3]
            # parse bbox
            xmin, ymin, width, height = ins['bbox']
            output.append({
                'geometry': poly,
                'angle': angle,
                'xmin': xmin,
                'ymin': ymin,
                'width': width,
                'height': height,
                'RGB_mean': RGB_mean,
                'area': ins['area'],
                'score': ins['score'],
                'index': ins['image_id_str'],
                'category_id': ins['category_id'],
            })
        if len(output) == 0:
            continue
        # reproject
        output = gpd.GeoDataFrame(pd.DataFrame(output))
        xoff, yoff = df_idx.loc[idx, ['lon_min', 'lat_max']].tolist()
        a = (df_idx.loc[idx, 'lon_max'] -
             df_idx.loc[idx, 'lon_min']) / SIZE_CHIP[0]
        e = - (df_idx.loc[idx, 'lat_max'] -
               df_idx.loc[idx, 'lat_min']) / SIZE_CHIP[1]
        output['geometry'] = output['geometry'].affine_transform(
            (a, 0, 0, e, xoff, yoff))
        df.append(output)
    df = pd.concat(df)

    # drop low score predictions
    df = df.loc[df['score'] > SCORE_CUTOFF, :]

    # drop predictions on logos
    df = df.loc[~((df['xmin'] > XMAX) & (df['ymin'] > YMAX)), :]

    # calculate instance characteristics
    df['luminosity'] = df['RGB_mean'].apply(
        lambda x: np.mean([np.max(x[0:3]), np.min(x[0:3])]) / 255)
    df['saturation'] = df['RGB_mean'].apply(
        lambda x: ((np.max(x[0:3]) - np.min(x[0:3])) / 255 /
                   (1 - np.abs(2 * np.mean([np.max(x[0:3]),
                                            np.min(x[0:3])]) / 255 - 1))))
    return df
