import os
import json
from pycocotools.coco import COCO
from utils.coco import COCOeval


def evaluate(cfg):
    """This evaluates the difference between predictions and ground truth.

    Args:
        cfg (Config): stores all configurations.
    """
    GT = COCO(os.path.join(cfg.run_dir, "annotations_gt.json"))
    try:
        DT = GT.loadRes(os.path.join(cfg.run_dir, "annotations_pred.json"))
    except IndexError:
        # in case no instances were predicted
        with open(os.path.join(cfg.run_dir, "annotations_pred.json"), 'r') as f:
            DT = json.load(f)
            assert len(DT) == 0
            return None
    E = COCOeval(cocoGt=GT, cocoDt=DT)
    # set new evaluation params
    E.params.maxDets = [100]
    E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    E.params.areaRngLbl = ['all']
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats
