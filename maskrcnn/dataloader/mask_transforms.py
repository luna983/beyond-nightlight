import json
import numpy as np
import copy
import warnings
from pycocotools import mask

import torch


class Polygon(object):
    """
    Polygons that represents a single instance of an object mask.

    Args:
        polygons (a list of lists of coordinates): The first level
            refers to all the polygons that compose the object, and
            the second level to the polygon coordinates. This is consistent
            with the COCO annotation format.
        width, height: width and height of the image, in pixels.
        category (int): value of the category of this instance.
    """

    def __init__(self, polygons=[], width, height, category):
        self.polygons = polygons
        self.width = width
        self.height = height
        self.category = category

    def from_supervisely(coordinates):
        """
        Load an instance from the Supervisely json format.

        Args:
            coordinates (dict): A dict that follows the Supervisely json format.
        """

        # check polygon validity - more than 2 points
        if len(coordinates['exterior']) > 2:
            self.polygons.append(np.array(
                [i for i in coord for coord in coordinates['exterior']]))
        if len(coordinates['interior']) > 2:
            self.polygons.append(np.array(
                [i for i in coord for coord in coordinates['interior']]))

        # no point outside of image boundaries
        for p in self.polygons:
            try:
                assert p[::2].min() >= 0 and p[::2].max() <= self.width
                assert p[1::2].min() >= 0 and p[1::2].max() <= self.height
            except:
                warnings.warn("Warning: Polygon outside of image boundaries.")

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
                p[0::2] = self.width - p[0::2]

        if FLIP_TOP_BOTTOM:
            for p in flipped_polygons:
                p[1::2] = self.height - p[1::2]

        return Polygon(
            polygons=flipped_polygons,
            width=self.width,
            height=self.height,
            category=self.category)

    def crop(self, i, j, h, w):
        """Crop the given polygons.

        Args:
            i (int): i in (i,j) i.e coordinates of the upper left corner.
            j (int): j in (i,j) i.e coordinates of the upper left corner.
            h (int): Height of the cropped image.
            w (int): Width of the cropped image.

        Returns:
            Polygon: a new instance with cropped polygons
        """

        assert i >= 0 and i <= self.width
        assert j >= 0 and j <= self.height
        assert i + w >= 0 and i + w <= self.width
        assert j + h >= 0 and j + h <= self.height

        cropped_polygons = copy.deepcopy(self.polygons)

        for p in cropped_polygons:
            p[0::2] = p[0::2] - i
            p[1::2] = p[1::2] - j

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
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

        return Polygon(
            polygons=resized_polygons,
            width=w,
            height=h,
            category=self.category)
            
    def to_tensor(self):
        """Convert instance polygons to PyTorch tensors.

        Returns:
            Tensor[H, W]: binary masks of the polygons
        """
        rles = mask.frPyObjects(self.polygons, self.height, self.width)
        rle = mask.merge(rles)
        binary_mask = mask.decode(rle)
        binary_mask = torch.from_numpy(binary_mask)
        return binary_mask

    def get_bbox(self):
        """Get the ground-truth bounding boxes.
        
        Returns:
            list: bounding box in [x0, y0, x1, y1] format,
                with values between 0 and H and 0 and W.
        """
        xmin, xmax, ymin, ymax = [], [], [], []
        for p in self.polygons:
            xmin.append(p[0::2].min())
            xmax.append(p[0::2].max())
            ymin.append(p[1::2].min())
            ymax.append(p[1::2].max())
        xmin = np.array(xmin).min()
        xmax = np.array(xmax).max()
        ymin = np.array(ymin).min()
        ymax = np.array(ymax).max()
        return [np.max([xmin, 0]), 
                np.max([ymin, 0]),
                np.min([xmax, self.width]),
                np.min([ymax, self.height])]

    def is_valid(self):
        """Check validity of the polygon (i.e., inside the image).

        Returns:
            bool: whether the instance is valid.
        """
        is_valid = []
        for p in self.polygons:
            if (p[0::2].min() > self.width or
                p[0::2].max() < 0 or
                p[1::2].min() > self.height or
                p[1::2].max() < 0):
                is_valid.append(False)
            else:
                is_valid.append(True)
        return any(is_valid)

    def __len__(self):
        return len(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_polygons={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.width)
        s += "image_height={})".format(self.height)
        return s


class InstanceMask(object):
    """
    Instance masks for all objects in one image.

    Args:
        instances (a list of Polygon): Each instance is a Polygon object.
        width, height (int): (width, height) of the image.
        label_dict: a dict of category names and category values {str: int}.
    """

    def __init__(self, label_dict=None, instances=None, width=None, height=None):
        self.instances = instances
        self.label_dict = label_dict
        self.width = width
        self.height = height

    def from_supervisely(file, label_dict):
        """
        Load all instances from an image from the Supervisely json format.

        Args:
            file (string): A string of the file path where instance annotations
                are stored using the Supervisely json format.
            label_dict (dict): A dict indicating {category name: int} mappings.
        """

        self.label_dict = label_dict
        with open(file, 'r') as f:
            mask = json.load(f)
        self.width, self.height = mask['size']['width'], mask['size']['height']
        for poly in mask['objects']:
            instance = Polygon(
                width=self.width, height=self.height,
                category=self.label_dict[poly['classTitle']])
            instance.from_supervisely(poly['points'])
            if len(instance.polygons) > 0:
                self.instances.append(instance)

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
            j (int): j in (i,j) i.e coordinates of the upper left corner.
            h (int): Height of the cropped image.
            w (int): Width of the cropped image.

        Returns:
            InstanceMask: a new cropped instance mask.
        """

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

    def to_tensor(self):
        """Convert instances to tensors.

        Prior transforms only operate on polygon coordinates, excessive instances
        (e.g., polygons that are cropped out) are dropped when converted to tensors.
        
        Returns:
            dict of torch.Tensors: following Mask-RCNN conventions
                boxes (Tensor[N, 4]): the ground-truth boxes in [x0, y0, x1, y1] format,
                with values between 0 and H and 0 and W
                labels (Tensor[N]): the class label for each ground-truth box
                masks (Tensor[N, H, W]): the segmentation binary masks for each instance
        """

        # check validity of polygons
        self.instances = [ins for ins in self.instances if ins.is_valid()]

        if len(self.instances) > 0:
            binary_mask = torch.stack([ins.to_tensor() for ins in self.instances])
        else:
            binary_mask = torch.empty([0, self.height, self.width], dtype=torch.uint8)

        boxes = torch.tensor([ins.get_bbox() for ins in self.instances])
        labels = [ins.category for ins in self.instances]

        return {'boxes': boxes, 'labels': labels, 'masks': binary_mask}

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.instances))
        s += "image_width={}, ".format(self.width)
        s += "image_height={})".format(self.height)
        return s