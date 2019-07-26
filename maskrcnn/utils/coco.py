import os
import json
import numpy as np
import pycocotools.mask as maskutils
import pycocotools.cocoeval


class COCOeval(pycocotools.cocoeval.COCOeval):
    """Modified to summarize results with alternative parameter setting.
    """

    def summarize(self):

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = (" {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] "
                    "= {:0.3f}")
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = ("{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                      if iouThr is None else "{:0.2f}".format(iouThr))
            aind = [i for i, aRng in enumerate(p.areaRngLbl)
                    if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr,
                              areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = {'mAP': _summarize(1),
                     'AP50': _summarize(1, iouThr=.5),
                     'AP75': _summarize(1, iouThr=.75)}
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        else:
            raise NotImplementedError
        self.stats = summarize()


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
            annotation (dict of torch.Tensor): follow Mask RCNN output format.
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
            rles = maskutils.encode(
                np.asfortranarray(masks.transpose(1, 2, 0)))
            areas = maskutils.area(rles)
            # loop over every instance in the image
            for i, (rle, area, label, mask, box) in enumerate(zip(
                rles, areas, labels, masks, boxes)):
                rle['counts'] = rle['counts'].decode('ascii')
                self.annotations.append({
                    'segmentation': rle,
                    'area': int(area),
                    'iscrowd': 0,  # set this to 0, otherwise eval fails
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
            rles = maskutils.encode(
                np.asfortranarray(masks.transpose(1, 2, 0)))
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
            with open(os.path.join(self.cfg.run_dir,
                                   "annotations_gt.json"), 'w') as f:
                json.dump({
                    'annotations': self.annotations,
                    'images': self.images,
                    'categories': [
                        {'id': i, 'name': name}
                        for name, i in self.cfg.label_dict.items()]}, f)
        else:
            if self.cfg.infer:
                with open(os.path.join(self.cfg.run_dir,
                                       "annotations_infer.json"), 'w') as f:
                    json.dump(self.annotations, f)
            else:
                with open(os.path.join(self.cfg.run_dir,
                                       "annotations_pred.json"), 'w') as f:
                    json.dump(self.annotations, f)
