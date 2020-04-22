import os
import json
import numpy as np
import pycocotools.mask as maskutils
import pycocotools.cocoeval


class Evaluator(pycocotools.cocoeval.COCOeval):
    """Modified to summarize results with alternative parameter setting."""

    def _summarize(self, iouThr, precision=True, areaRng='all', maxDets=100):
        """Internal function to summarize average precision/(max) recall.

        Args:
            iouThr (float or NoneType): IoU threshold(s).
            precision (bool): True: precision; False: max recall.
            areaRng (str): area range for size of objects.
            maxDets (int): max number of detections per image.

        Returns:
            float: average precision.
            numpy.ndarray: AR/AP curve.
            numpy.ndarray: scores (if precision=True).
        """
        # parameter settings
        aind = [i for i, aRng in enumerate(self.params.areaRngLbl)
                if aRng == areaRng]
        mind = [i for i, mDet in enumerate(self.params.maxDets)
                if mDet == maxDets]
        # retrieve precision matrix
        s = self.eval['precision'] if precision else self.eval['recall']
        # dimension of precision: [TxRxKxAxM]
        # area under precision-recall curve, given a certain IoU
        s = s[..., aind, mind]
        scores = self.eval['scores'][..., aind, mind] if precision else None
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == self.params.iouThrs)[0]
            s = s[t, ...]
            if precision:
                scores = scores[t, ...]
        if len(s[s > -1]) == 0:
            mean_s = -1  # no gt object
        else:
            mean_s = np.mean(s[s > -1])
        # print results
        titleStr = 'Average Precision' if precision else 'Average Recall'
        iouStr = ('{:0.2f}:{:0.2f}'
                  .format(self.params.iouThrs[0], self.params.iouThrs[-1])
                  if iouThr is None else '{:0.2f}'.format(iouThr))
        print('{:<18} @[ IoU={:<9} ] = {:0.3f}'
              .format(titleStr, iouStr, mean_s))
        return mean_s, s, scores

    def summarize(self):
        """This generates the summarized statistics."""
        ap50, ap50_curve, scores = self._summarize(iouThr=.5)
        ar50, _, _ = self._summarize(iouThr=.5, precision=False)
        self.stats = {
            'AP50': ap50,
            'AR50': ar50,
            'AP50_curve': ap50_curve.squeeze().tolist(),
            'scores': scores.squeeze().tolist(),
        }

class COCOSaver(object):
    """This collects torch.Tensors and serialize annotations into a json file.

    Args:
        cfg (Config): stores all configurations
        gt (bool): True: ground truth, False: predictions
    """

    def __init__(self, cfg, gt):
        self.cfg = cfg
        self.gt = gt
        self.placeholder_int = (self.cfg.num_classes - 1
                                if self.cfg.fillempty else None)

        self.images = []
        self.annotations = []
        self.coco_image_id = 0
        self.coco_instance_id = 0 if gt else None

    def add(self, annotation, image_id):
        """This updates the annotation list.

        Args:
            annotation (dict of torch.Tensor): follow Mask RCNN output format.
            image_id (str): the id of the image.
        """
        self.coco_image_id += 1
        if self.gt:
            labels = annotation['labels'].numpy()
            masks = annotation['masks'].numpy().astype(np.uint8)
            boxes = annotation['boxes'].numpy()
            _, height, width = masks.shape
            # for mask.encode()
            # binary mask(s) must have [hxwxn]
            # type np.ndarray(dtype=uint8) in column-major order
            rles = maskutils.encode(
                np.asfortranarray(masks.transpose(1, 2, 0)))
            areas = maskutils.area(rles)
            # loop over every instance in the image
            for i, (rle, area, label, mask, box) in enumerate(zip(
                    rles, areas, labels, masks, boxes)):
                if label == self.placeholder_int:
                    continue
                self.coco_instance_id += 1
                rle['counts'] = rle['counts'].decode('ascii')
                self.annotations.append({
                    'segmentation': rle,
                    'area': int(area),
                    'iscrowd': 0,  # set this to 0, otherwise eval fails
                    'bbox': [float(box[0]), float(box[1]),
                             float(box[2] - box[0]), float(box[3] - box[1])],
                    'image_id_str': image_id,  # for postprocessing
                    'image_id': self.coco_image_id,  # internal coco eval
                    'category_id': int(label),
                    # 0 reserved, id count from 1
                    'id': self.coco_instance_id})
            self.images.append({
                'id': self.coco_image_id,
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
            rles = maskutils.encode(
                np.asfortranarray(masks.transpose(1, 2, 0)))
            areas = maskutils.area(rles)
            for i, (rle, area, label, box, score) in enumerate(zip(
                    rles, areas, labels, boxes, scores)):
                if label == self.placeholder_int or area == 0:
                    continue
                rle['counts'] = rle['counts'].decode('ascii')
                self.annotations.append(
                    {'segmentation': rle,
                     'bbox': [float(box[0]), float(box[1]),
                              float(box[2] - box[0]),
                              float(box[3] - box[1])],
                     'score': float(score),
                     'area': float(area),
                     'image_id_str': image_id,  # for postprocessing
                     'image_id': self.coco_image_id,  # internal coco eval
                     'category_id': int(label)})

    def save(self, mode, file_name=None):
        """This saves the annotations to the default log directory.

        Args:
            mode (str): the sample to be evaluated (train/val/infer).
            file_name (str): name of the file.
        """
        if mode == 'infer':
            assert file_name is not None, 'file names not provided'
        else:
            file_name = ('gt' if self.gt else 'pred')
        os.makedirs(
            os.path.join(self.cfg.out_mask_dir, mode),
            exist_ok=True)
        with open(os.path.join(
                self.cfg.out_mask_dir, mode, file_name + '.json'), 'w') as f:
            if self.gt:
                json.dump({
                    'annotations': self.annotations,
                    'images': self.images,
                    'categories': [
                        {'id': i, 'name': name}
                        for i, name in self.cfg.int_dict.items()]}, f)
            else:
                json.dump(self.annotations, f)

        self.annotations = []
        self.images = []
