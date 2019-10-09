import os
import json
import numpy as np
import geopandas as gpd
import pycocotools.mask as maskutils

import torch
from glob import glob
from PIL import Image
from shapely.geometry import box

from ..utils.visualize_mask import visualize_from_coco_pred


class Resampler(object):
    """Samples new rasters from existing images and predictions.

    Args:
        img_dir, ann_dir (str): path to images or annotations
        chips (geopandas.GeoDataFrame): geo referencing info for the chips
        reverse_x, reverse_y (bool): whether x/y axes should be reversed
    """

    def __init__(self, img_dir, ann_dir, chips, reverse_x, reverse_y):
        self.img_dir = ({} if img_dir is None else
                        {os.path.basename(f).split('.')[0]: f
                         for f in glob(img_dir + '*')})
        self.ann_dir = ({} if ann_dir is None else
                        {os.path.basename(f).split('.')[0]: f
                         for f in glob(ann_dir + '*')})
        self.chips = chips
        self.reverse_x = reverse_x
        self.reverse_y = reverse_y
        self.output = None

    @classmethod
    def from_bounds(cls, img_dir=None, ann_dir=None,
                    indices=None, bounds=None,
                    reverse_x=False, reverse_y=True):
        """Parses geo coord bounds.

        Args:
            img_dir, ann_dir (str): path to images or annotations
            indices (list of str): list of indices of the existing chips
            bounds (list of list of floats): bounds that correspond to chips
                (minx, miny, maxx, maxy)
            reverse_x, reverse_y (bool): whether x/y axes should be reversed

        Returns:
            Resampler: initialized with indices and bounds
        """
        if indices is None or bounds is None:
            chips = gpd.GeoDataFrame()
        else:
            chips = gpd.GeoDataFrame(
                {'index': indices,
                 'geometry': [box(*bound) for bound in bounds]})
        return cls(img_dir, ann_dir, chips, reverse_x, reverse_y)

    @staticmethod
    def _subset_chips(chips, bounds):
        """Subsets chips intersecting with the bounds.

        Args:
            chips (geopandas.GeoDataFrame): geo referencing info for the chips
            bounds (list of floats): bounds for area of interest
                (minx, miny, maxx, maxy)

        Returns:
            geopandas.GeoDataFrame: geo referencing info for subsetted chips
        """
        return chips.cx[bounds[0]:bounds[2], bounds[2]:bounds[3]]

    def _load_chips(self, index, mode):
        """Loads images or annotations.

        Args:
            index (str): index of the chip
            mode (str): set to 'img' or 'ann'

        Returns:
            PIL.Image ('img' mode) or dict ('ann' mode): loaded content
        """
        if mode == 'img':
            output = Image.open(self.img_dir[index])
        elif mode == 'ann':
            with open(self.ann_dir[index], 'r') as f:
                output = json.load(f)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    def _reproject_bbox(bbox, original_width, original_height,
                        chip_width, chip_height,
                        pixel_x, pixel_y):
        """Reprojects bbox from original chip to new chip.

        Args:
            bbox (list of float): original bounding box
            original_width, original_height (int): original width and height
            chip_width, chip_height (int): dim of resized input image
            pixel_x, pixel_y (int): upper left corner to paste the resized
                input image onto the output image

        Returns:
            (list of float): new bounding box
        """
        return [pixel_x + bbox[0] * chip_width / original_width,
                pixel_y + bbox[1] * chip_height / original_height,
                bbox[2] * chip_width / original_width,
                bbox[3] * chip_height / original_height]

    @staticmethod
    def _reproject_mask(rle,
                        width, height,
                        chip_width, chip_height,
                        pixel_x, pixel_y):
        """Reprojects mask from original chip to new chip.

        Args:
            rle (dict): original RLE of the mask
            width, height (int): dim of output image
            chip_width, chip_height (int): dim of resized input image
            pixel_x, pixel_y (int): upper left corner to paste the resized
                input image onto the output image

        Returns:
            dict: new RLE of the mask
        """
        mask = Image.fromarray(maskutils.decode(rle), mode='L')
        output = Image.new(mode='L', size=(width, height))
        output.paste(mask.resize((chip_width, chip_height)),
                     (pixel_x, pixel_y))
        return maskutils.encode(np.asarray(output, dtype=np.uint8, order='F'))

    @staticmethod
    def _postprocess_ann(anns, score_cutoff=0, xmax=None, ymax=None):
        """Postprocess: drop low score ones, and corner annotations on logos.

        Args:
            anns (list of dict): annotations
            score_cutoff (float): score cutoff for dropping annotations
            xmax, ymax (int): upper left corner of a region
                where if an annotation's upper left corner falls there
                it is dropped

        Returns:
            list of dict: postprocessed annotations
        """
        output = [ann for ann in anns if ann['score'] > score_cutoff]
        if not (xmax is None or ymax is None):
            output = [ann for ann in output if not
                      (ann['bbox'][0] > xmax and
                       ann['bbox'][1] > ymax)]
        return output

    def _find_idx(ann, raster_bounds, chip_bounds, width, height):
        """Finds the (flattened) index for an annotation.

        An annotation is assigned to a grid iff its centroid is in the grid.

        Args:
            ann (list of dict): annotations
            raster_bounds (list of floats): bounds for area of interest
                (minx, miny, maxx, maxy)
            chip_bounds (list of floats): bounds for original chip
            width, height (int): dim of output image

        Returns:
            int: the index of the grid to which the annotation belongs
        """
        # TODO
        pass

    def agg(self, bounds, width, height, f, mode, cfg):
        """Aggregates up to a raster.

        Args:
            bounds (list of floats): bounds for area of interest
                (minx, miny, maxx, maxy)
            width, height (int): dim of output image
            f (function): function to aggregate over an annotation/image
            mode (str): in ['ann']
            cfg (argparse.Namespace): aggregation parameters
        """
        assert mode in ['ann'], 'Unknown mode specified'
        chips = self._subset_chips(self.chips, bounds)
        self.output = torch.zeros((height, width)).view(-1)
        for _, chip in chips.iterrows():
            chip_annotations = self._load_chips(chip['index'], mode='ann')
            chip_annotations = self._postprocess_ann(
                chip_annotations,
                score_cutoff=cfg.visual_score_cutoff,
                xmax=cfg.xmax, ymax=cfg.ymax)
            # TODO
            scatter_val = torch.tensor(
                [f(ann) for ann in chip_annotations])
            scatter_idx = torch.tensor(
                [self._find_idx(ann, bounds, chip['geometry'].bounds)
                 for ann in chip_annotations])

    def plot(self, bounds, width, height, mode, cfg=None):
        """Plots images and/or annotations.

        Args:
            bounds (list of floats): bounds for area of interest
                (minx, miny, maxx, maxy)
            width, height (int): dim of output image
            mode (str): in ['img', 'ann']
            cfg (argparse.Namespace): visualization parameters
        """
        chips = self._subset_chips(self.chips, bounds)
        self.output = Image.new(mode='RGB', size=(width, height))
        if mode == 'ann':
            assert cfg is not None, 'Specify visualization config.'
            annotations = []
        for _, chip in chips.iterrows():
            assert mode in ['img', 'ann'], 'Unknown mode specified'
            output = self._load_chips(chip['index'], mode='img')
            chip_width = int((chip['geometry'].bounds[2] -
                              chip['geometry'].bounds[0]) /
                             (bounds[2] - bounds[0]) * width)
            chip_height = int((chip['geometry'].bounds[3] -
                               chip['geometry'].bounds[1]) /
                              (bounds[3] - bounds[1]) * height)
            pixel_x = (int((- chip['geometry'].bounds[2] + bounds[2]) /
                           (bounds[2] - bounds[0]) * width)
                       if self.reverse_x else
                       int((chip['geometry'].bounds[0] - bounds[0]) /
                           (bounds[2] - bounds[0]) * width))
            pixel_y = (int((- chip['geometry'].bounds[3] + bounds[3]) /
                           (bounds[3] - bounds[1]) * height)
                       if self.reverse_y else
                       int((chip['geometry'].bounds[1] - bounds[1]) /
                           (bounds[3] - bounds[1]) * height))
            self.output.paste(output.resize((chip_width, chip_height)),
                              (pixel_x, pixel_y))
            if mode == 'ann':
                chip_annotations = self._load_chips(chip['index'], mode='ann')
                chip_annotations = self._postprocess_ann(
                    chip_annotations,
                    score_cutoff=cfg.visual_score_cutoff,
                    xmax=cfg.xmax, ymax=cfg.ymax)
                # 'reproject' annotations
                for ann in chip_annotations:
                    ann['bbox'] = self._reproject_bbox(
                        ann['bbox'],
                        *ann['segmentation']['size'],
                        chip_width, chip_height,
                        pixel_x, pixel_y)
                    ann['segmentation'] = self._reproject_mask(
                        ann['segmentation'],
                        width, height,
                        chip_width, chip_height,
                        pixel_x, pixel_y)
                    ann['area'] = maskutils.area(ann['segmentation'])
                annotations += [ann for ann in chip_annotations
                                if ann['area'] > 0]
        if mode == 'ann':
            self.annotations = annotations
            self.output = visualize_from_coco_pred(
                cfg=cfg, annotations=annotations, image=self.output)
