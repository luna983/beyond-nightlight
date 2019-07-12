# from utils.coco import AnnotationEvaluator, AnnotationLoader
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(cfg):
    """This evaluates the difference between predictions and ground truth.

    Args:
        cfg (Config): stores all configurations.
    """
    GT = COCO(os.path.join(cfg.run_dir, "annotations_gt.json"))
    DT = GT.loadRes(os.path.join(cfg.run_dir, "annotations_pred.json"))
    E = COCOeval(cocoGt=GT, cocoDt=DT)
    # change default parameters
    # self.cocoeval.params.maxDets = [20]
    # self.cocoeval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    # self.cocoeval.params.areaRngLbl = ['all']
    
    E.evaluate()
    # import pdb; pdb.set_trace()
    E.accumulate()
    return E.summarize()