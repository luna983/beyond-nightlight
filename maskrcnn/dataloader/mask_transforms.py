import json
import numpy as np
import copy
import warnings
import pycocotools.mask as maskutils

import torch


class Polygon(object):
    """
    Polygons that represents a single instance of an object mask.

    Args:
        width, height (int): width and height of the image, in pixels.
        category (int): value of the category of this instance.
        polygons (list of numpy.ndarray): The first level
            refers to all the polygons that compose the object, and
            the second level to the polygon coordinates. This is consistent
            with the COCO annotation format. Multipart polygon is supported,
            but polygons with holes are not. The coordinates can be floats,
            and can be outside of the bounding box.
    """

    def __init__(self, width=None, height=None, category=None, polygons=None):
        self.width = width
        self.height = height
        self.category = category
        self.polygons = [] if polygons is None else polygons

    def __len__(self):
        return len(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'width={}, '.format(self.width)
        s += 'height={}, '.format(self.height)
        s += 'category={}, '.format(self.category)
        s += 'polygons={})'.format(self.polygons)
        return s

    def load_single_polygon(self, coordinates):
        """
        Load an instance from the coordinates.

        Args:
            coordinates (list/tuple of list/tuple of float):
                coordinates formatted as [[x0, y0], [x1, y1], ...],
                with more than 2 points
        """
        self.polygons.append(np.array(coordinates).astype(np.float32))

    def to_rle(self):
        """Convert instance polygons to run-length encoding (RLE).

        Returns:
            dict: Run-length encoding (RLE) of the binary mask.
        """
        polygons = [list(poly.flatten()) for poly in self.polygons]
        rles = maskutils.frPyObjects(polygons, self.height, self.width)
        rle = maskutils.merge(rles)
        return rle

    def flip(self, FLIP_LEFT_RIGHT=False, FLIP_TOP_BOTTOM=False):
        """
        Flip the image (and annotations).

        Args:
            FLIP_LEFT_RIGHT (bool): default to False
            FLIP_TOP_BOTTOM (bool): default to False

        Returns:
            Polygon: a new instance with flipped polygons
        """

        flipped_polygons = copy.deepcopy(self.polygons)

        if FLIP_LEFT_RIGHT:
            for p in flipped_polygons:
                p[:, 0] = self.width - p[:, 0]

        if FLIP_TOP_BOTTOM:
            for p in flipped_polygons:
                p[:, 1] = self.height - p[:, 1]

        return Polygon(
            polygons=flipped_polygons,
            width=self.width,
            height=self.height,
            category=self.category)

    def crop(self, i, j, h, w):
        """Crop the given polygons.

        Args:
            i (int): i in (i,j) i.e coordinates of the upper left corner.
                Corresponds to height.
            j (int): j in (i,j) i.e coordinates of the upper left corner.
                Corresponds to width.
            h (int): Height of the cropped image.
            w (int): Width of the cropped image.

        Returns:
            Polygon: a new instance with cropped polygons
        """

        assert i >= 0 and i <= self.height, 'Dimension mismatch.'
        assert j >= 0 and j <= self.width, 'Dimension mismatch.'
        assert i + h >= 0 and i + h <= self.height, 'Dimension mismatch.'
        assert j + w >= 0 and j + w <= self.width, 'Dimension mismatch.'

        cropped_polygons = copy.deepcopy(self.polygons)

        for p in cropped_polygons:
            p[:, 0] = p[:, 0] - j
            p[:, 1] = p[:, 1] - i

        return Polygon(
            polygons=cropped_polygons,
            width=w,
            height=h,
            category=self.category)

    def resize(self, h, w):
        """Resize the input labels to the given size.

        Args:
            h, w (int): Desired output size (height, width).

        Returns:
            Polygon: a new instance with resized polygons
        """
        ratio_w, ratio_h = w / self.width, h / self.height
        resized_polygons = copy.deepcopy(self.polygons)
        for p in resized_polygons:
            p[:, 0] *= ratio_w
            p[:, 1] *= ratio_h

        return Polygon(
            polygons=resized_polygons,
            width=w,
            height=h,
            category=self.category)


class InstanceMask(object):
    """
    Instance masks for all objects in one image.

    Args:
        instances (a list of Polygon): Each instance is a Polygon object.
        label_dict (dict): a dict of category names and category values
            {str: int}.
        width, height (int): (width, height) of the image.
    """

    def __init__(self, instances=None, label_dict=None,
                 width=None, height=None):
        self.instances = [] if instances is None else instances
        self.label_dict = label_dict
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'instances={}, '.format(self.instances)
        s += 'label_dict={}, '.format(self.label_dict)
        s += 'width={}, '.format(self.width)
        s += 'height={})'.format(self.height)
        return s

    def from_file(self, file, ann_format, label_dict, verbose=False):
        """
        Load all instances from an image from a file.

        Args:
            file (str): A string of the file path where instance annotations
                are stored.
            ann_format (str): Format of the annotations.
            label_dict (dict): A dict indicating {category name: int} mappings.
            verbose (bool): if True, generate warnings of incorrect imports.
        """

        if ann_format == 'supervisely':
            with open(file, 'r') as f:
                ann = json.load(f)
            self.width = ann['size']['width']
            self.height = ann['size']['height']
            for ins in ann['objects']:
                if not label_dict[ins['classTitle']] == 0:
                    instance = Polygon(
                        width=self.width, height=self.height,
                        category=label_dict[ins['classTitle']])
                    coordinates = ins['points']
                    # check polygon validity - more than 2 points
                    if len(coordinates['exterior']) > 2:
                        instance.load_single_polygon(coordinates['exterior'])
                        self.instances.append(instance)
                    else:
                        if verbose:
                            warnings.warn('Invalid exterior coords ignored: {}'
                                          .format(coordinates['exterior']))
                    if verbose and (len(coordinates['interior']) > 0):
                        warnings.warn('Interior coordinates ignored: {}'
                                      .format(coordinates['interior']))

        elif ann_format == 'openaitanzania':
            with open(file, 'r') as f:
                ann = json.load(f)
            self.width = ann['width']
            self.height = ann['height']
            for ins in ann['instances']:
                if not label_dict[ins['category']] == 0:
                    instance = Polygon(
                        width=self.width, height=self.height,
                        category=label_dict[ins['category']])
                    coordinates = ins['polygon']
                    # check polygon validity - more than 2 points
                    if len(coordinates['exterior']) > 2:
                        instance.load_single_polygon(coordinates['exterior'])
                        self.instances.append(instance)
                    else:
                        if verbose:
                            warnings.warn('Invalid exterior coords ignored: {}'
                                          .format(coordinates['exterior']))
                    if verbose and (len(coordinates['interior']) > 0):
                        warnings.warn('Interior coordinates ignored: {}'
                                      .format(coordinates['interior']))
        else:
            raise NotImplementedError
        self.label_dict = label_dict

    def to_tensor(self):
        """Convert instances to tensors. Some instances may be dropped.

        Returns:
            dict of torch.Tensors: following Mask-RCNN conventions.
                boxes (Tensor[N, 4]): the ground-truth boxes in
                    [x0, y0, x1, y1] format, with values
                    between 0 and H and 0 and W.
                labels (Tensor[N]): the class label for each ground-truth box.
                masks (Tensor[N, H, W]): the segmentation binary masks
                    for each instance.
        """
        # encode to RLEs
        rles = [ins.to_rle() for ins in self.instances]

        # drop instances with zero area
        areas = maskutils.area(rles)
        rles = [rle for rle, area in zip(rles, areas) if area != 0]

        # return None for empty images
        if len(rles) == 0:
            return None
        # convert to masks
        binary_masks = maskutils.decode(rles)
        binary_masks = torch.from_numpy(
            binary_masks.transpose((2, 0, 1)))
        # convert to bounding boxes
        boxes = maskutils.toBbox(rles)  # (x, y, w, h)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = torch.from_numpy(boxes.astype(np.float32))
        # collate labels
        labels = torch.tensor(
            [ins.category for ins, area in zip(self.instances, areas)
             if area != 0],
            dtype=torch.int64)

        return {'boxes': boxes, 'labels': labels, 'masks': binary_masks}

    def flip(self, FLIP_LEFT_RIGHT=False, FLIP_TOP_BOTTOM=False):
        """Flip the annotations on an image.

        Args:
            FLIP_LEFT_RIGHT (bool): default to False
            FLIP_TOP_BOTTOM (bool): default to False

        Returns:
            InstanceMask: a new flipped instance mask.
        """
        return InstanceMask(
            instances=[ins.flip(
                FLIP_LEFT_RIGHT=FLIP_LEFT_RIGHT,
                FLIP_TOP_BOTTOM=FLIP_TOP_BOTTOM) for ins in self.instances],
            width=self.width, height=self.height, label_dict=self.label_dict)

    def crop(self, i, j, h, w):
        """Crop the given instances.

        Args:
            i (int): i in (i,j) i.e coordinates of the upper left corner.
                Corresponds to height.
            j (int): j in (i,j) i.e coordinates of the upper left corner.
                Corresponds to width.
            h (int): Height of the cropped image.
            w (int): Width of the cropped image.

        Returns:
            InstanceMask: a new cropped instance mask.
        """

        assert i >= 0 and i <= self.height, 'Dimension mismatch.'
        assert j >= 0 and j <= self.width, 'Dimension mismatch.'
        assert i + h >= 0 and i + h <= self.height, 'Dimension mismatch.'
        assert j + w >= 0 and j + w <= self.width, 'Dimension mismatch.'

        return InstanceMask(
            instances=[ins.crop(i=i, j=j, h=h, w=w) for ins in self.instances],
            width=w, height=h, label_dict=self.label_dict)

    def resize(self, h, w):
        """Resize the input labels to the given size.

        Args:
            h, w (int): Desired output size (h, w).

        Returns:
            InstanceMask: a new resized instance mask.
        """
        return InstanceMask(
            instances=[ins.resize(h=h, w=w) for ins in self.instances],
            width=w, height=h, label_dict=self.label_dict)
