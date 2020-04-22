import os
import sys
import json
from pycocotools.coco import COCO

from .coco import Evaluator


def evaluate(cfg, mode):
    """This evaluates the difference between predictions and ground truth.

    Args:
        cfg (Config): stores all configurations.
        mode (str): the sample to be evaluated (train/val).
    """
    # load ground truth and predictions
    GT = COCO(os.path.join(cfg.out_mask_dir, mode,
                           'gt.json'))
    try:
        DT = GT.loadRes(os.path.join(cfg.out_mask_dir, mode,
                                     'pred.json'))
    except IndexError:
        # in case no instances were predicted
        with open(os.path.join(cfg.out_mask_dir, mode,
                               'pred.json'), 'r') as f:
            DT = json.load(f)
        assert len(DT) == 0
        return None
    except:
        print('Unexpected error: ', sys.exc_info())
        return None

    # evaluation starts
    E = Evaluator(cocoGt=GT, cocoDt=DT)
    # set new evaluation params
    E.params.maxDets = [100]
    E.params.areaRng = [[0 ** 2, 1e4 ** 2]]
    E.params.areaRngLbl = ['all']
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats
