from pycocotools.coco import COCO
from utils.coco import COCOeval


def evaluate(pred_file, gt_file):
    """This evaluates the difference between predictions and ground truth.

    Args:
        pred_file (str): file storing the prediction results.
        gt_file (str): file storing the ground truth.
    """
    GT = COCO(gt_file)
    DT = GT.loadRes(pred_file)
    E = COCOeval(cocoGt=GT, cocoDt=DT)
    # set new evaluation params
    E.params.maxDets = [100]
    E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    E.params.areaRngLbl = ['all']
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats
