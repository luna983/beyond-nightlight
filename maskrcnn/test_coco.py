from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask
import json

# MWE for COCO
gt = COCO("annotations/instances_val2014.json")
imgIds=sorted(gt.getImgIds())
imgId=imgIds[3]
print("file opened")
anns = []
for ann in gt.dataset['annotations']:
    if ann['image_id'] == imgId:
        new_ann = ann
        rle = gt.annToRLE(ann)
        rle['counts'] = rle['counts'].decode('ascii')
        new_ann['segmentation'] = rle
        anns.append(new_ann)
gt.dataset['annotations'] = anns
gt.dataset['images'] = [i for i in gt.dataset['images'] if i['id']==imgId]
with open("gt_new.json", 'w') as f:
    json.dump(gt.dataset, f)
print("file altered")
with open("instances_val2014_fakesegm100_results.json", 'r') as f:
    dt = json.load(f)
dt = [i for i in dt if i['image_id']==imgId]
with open("dt_new.json", 'w') as f:
    json.dump(dt, f)
print("file altered")
gt = COCO("gt_new.json")
dt = gt.loadRes("dt_new.json")
cocoEval = COCOeval(gt,dt)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# MWE for me
print("="*72)
gt = COCO("runs/run_05/annotations_gt.json")
dt = gt.loadRes("runs/run_05/annotations_pred.json")
cocoEval = COCOeval(gt,dt)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
