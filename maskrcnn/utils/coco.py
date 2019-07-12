import os
import json
import numpy as np
import pycocotools.mask as maskutils
from itertools import groupby


class COCOSaver(object):
    """This collects torch.Tensors and serialize annotations into a json file.

    Args:
        gt (bool): True: ground truth, False: predictions
        cfg (Config): stores all configurations
    """
    def __init__(self, gt, cfg):
        self.gt = gt
        self.cfg = cfg
        self.annotations = []
        if gt:
            self.images = []
        else:
            self.image_id = 0

    def add(self, annotation):
        """This updates the annotation list.

        Args:
            annotation (dict): dict of torch.Tensor, following Mask RCNN output format.
        """
        if self.gt:
            image_id = len(self.images)
            labels = annotation['labels'].numpy()
            masks = annotation['masks'].numpy().astype(np.uint8)
            boxes = annotation['boxes'].numpy()
            _, height, width = masks.shape
            # for mask.encode()
            # binary mask(s) must have [hxwxn]
            # type np.ndarray(dtype=uint8) in column-major order
            rles = maskutils.encode(np.asfortranarray(masks.transpose(1, 2, 0)))
            areas = maskutils.area(rles)
            # loop over every instance in the image
            for i, (rle, area, label, mask, box) in enumerate(zip(
                rles, areas, labels, masks, boxes)):
                rle['counts'] = rle['counts'].decode('ascii')
                self.annotations.append({
                    'segmentation': rle,
                    'area': int(area),
                    'iscrowd': 0, # set this to 0, otherwise evaluation does not work
                    'bbox': box.tolist(),
                    'image_id': image_id,
                    'category_id': int(label),
                    # assuming no more than 1000 annotations per image
                    # create instance id
                    'id': image_id * 1000 + i})
            self.images.append({
                'id': image_id,
                'height': height,
                'width': width})
        else:
            labels = annotation['labels'].detach().cpu().numpy()
            masks = annotation['masks'].detach().cpu().numpy().astype(np.uint8)
            boxes = annotation['boxes'].detach().cpu().numpy()
            scores = annotation['scores'].detach().cpu().numpy()
            # for mask.encode()
            # binary mask(s) must have [hxwxn]
            # type np.ndarray(dtype=uint8) in column-major order
            rles = maskutils.encode(np.asfortranarray(masks.transpose(1, 2, 0)))
            areas = maskutils.area(rles)
            for i, (rle, area, label, box, score) in enumerate(zip(
                rles, areas, labels, boxes, scores)):
                if area > 0:
                    rle['counts'] = rle['counts'].decode('ascii')
                    self.annotations.append(
                        {'segmentation': rle,
                         'bbox': box.tolist(),
                         'score': float(score),
                         'image_id': self.image_id,
                         'category_id': int(label)})
            self.image_id += 1

    def save(self):
        """This saves the annotations to the default log directory.
        """
        if self.gt:
            with open(os.path.join(self.cfg.run_dir, "annotations_gt.json"), 'w') as f:
                json.dump({
                    'annotations': self.annotations,
                    'images': self.images,
                    'categories': [{'id': i, 'name': name}
                        for name, i in self.cfg.label_dict.items()]
                    }, f)
        else:
            with open(os.path.join(self.cfg.run_dir, "annotations_pred.json"), 'w') as f:
                json.dump(self.annotations, f)