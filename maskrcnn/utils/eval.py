import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats