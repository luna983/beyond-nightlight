import time
import numpy as np
from collections import defaultdict
import pycocotools.mask as mask
from pycocotools import COCO, COCOeval, Params

def convert_tensor_to_coco(annotation, image_id):
    """Convert PyTorch tensor output from Mask RCNN to COCO annotation format.

    Args:
        annotation (dict): dict of torch.Tensor, following Mask RCNN output format.
        image_id (int): unique image identifier

    Returns:
        dict: annotations in COCO format
    """
    # convert torch tensors to numpy arrays
    labels = annotation['labels'].detach().cpu().numpy()
    masks = annotation['masks'].detach().cpu().numpy()
    scores = annotation['scores'].detach().cpu().numpy()
    boxes = annotation['boxes'].detach().cpu().numpy()
    # get shape
    _, height, width = masks.shape
    # for mask.encode()
    # binary mask(s) must have [hxwxn]
    # type np.ndarray(dtype=uint8) in column-major order
    masks = masks.transpose(1, 2, 0).astype(np.uint8)
    rles = mask.encode(np.asfortranarray(masks))
    areas = mask.area(rles)
    output = []
    for i, rle, area, label, box, score in enumerate(
        zip(rles, areas, labels, boxes, scores)):
        output.append(
            {'segmentation': rle,
             'area': area,
             'iscrowd': 1,
             'image_id': image_id,
             'bbox': box.tolist(),
             'category_id': label,
             # assuming no more than 1000 annotations per image
             # create instance id
             'id': image_id * 1000 + i})
    return output

class AnnotationLoader(COCO):
    """Modified to avoid loading json files and take in dict directly.

    Args:
        annotations (list of dict): list of dict in COCO format
    """
    def __init__(self, annotations):
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        print("Loading annotations into memory...")
        tic = time.time()
        self.dataset = {'annotations': annotations}
        toc = time.time()
        print("Done (t={:0.2f}s)".format(toc - tic))
        self.createIndex()

class AnnotationEvaluator(COCOeval):
    """Modified to summarize results with alternative parameter setting.
    """
    def summarize(self):

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=200):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap==1 else "(AR)"
            iouStr = ("{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None else "{:0.2f}".format(iouThr))
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
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

class Evaluator(object):
    """This evaluates the difference between predictions and ground truth.

    Args:
        preds (list of dict): predictions in COCO format
        targets (list of dict): targets in COCO format
    """
    def __init__(self, preds, targets):
        # instantiate COCOeval object
        self.cocoeval = AnnotationEvaluator(
            cocoGt=AnnotationLoader(targets),
            cocoDt=AnnotationLoader(preds))
        # change default parameters
        self.cocoeval.params.maxDets = [200]
        self.cocoeval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
        self.cocoeval.params.areaRngLbl = ['all']
        
    def evaluate(self):
        self.cocoeval.evaluate()
        self.cocoeval.accumulate()
        self.cocoeval.summarize()