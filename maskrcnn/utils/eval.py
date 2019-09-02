import os
from pycocotools.coco import COCO
from utils.coco import COCOeval


def evaluate(cfg, mode):
    """This evaluates the difference between predictions and ground truth.

    Args:
        cfg (Config): stores all configurations.
        mode (str): the sample to be evaluated (train/val).
    """
    GT = COCO(os.path.join(cfg.out_mask_dir, mode,
                           'annotations_gt.json'))
    DT = GT.loadRes(os.path.join(cfg.out_mask_dir, mode,
                                 'annotations_pred.json'))
    E = COCOeval(cocoGt=GT, cocoDt=DT)
    # set new evaluation params
    E.params.maxDets = [100]
    E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    E.params.areaRngLbl = ['all']
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats
