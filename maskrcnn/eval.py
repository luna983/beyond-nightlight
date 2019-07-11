from utils.coco import AnnotationEvaluator, AnnotationLoader

class Evaluator(object):
    """This evaluates the difference between predictions and ground truth.

    Args:
        preds (list of dict): predictions in COCO format
        targets (list of dict): targets in COCO format
        width, height (int): width and height of images fed into the model
        num_image (int): number of images loaded
        label_dict (dict): dict of {name: id} pairs for categories
    """
    def __init__(self, preds, targets, width, height, num_image, label_dict):
        # instantiate COCOeval object
        self.cocoeval = AnnotationEvaluator(
            cocoGt=AnnotationLoader(preds, width, height, num_image, label_dict),
            cocoDt=AnnotationLoader(targets, width, height, num_image, label_dict))
        # change default parameters
        self.cocoeval.params.maxDets = [20]
        self.cocoeval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
        self.cocoeval.params.areaRngLbl = ['all']
        
    def evaluate(self):
        self.cocoeval.evaluate()
        self.cocoeval.accumulate()
        self.cocoeval.summarize()
