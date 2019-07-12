from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask
import json
import argparse
from dataloader import make_data_loader
from utils.configure import Config
from utils.coco import COCOSaver
import torch
# MWE for COCO
# gt = COCO("annotations/instances_val2014.json")
# imgIds=sorted(gt.getImgIds())
# imgId=imgIds[3]
# print("file opened")
# anns = []
# for ann in gt.dataset['annotations']:
#     if ann['image_id'] == imgId:
#         new_ann = ann
#         rle = gt.annToRLE(ann)
#         rle['counts'] = rle['counts'].decode('ascii')
#         new_ann['segmentation'] = rle
#         anns.append(new_ann)
# gt.dataset['annotations'] = anns
# gt.dataset['images'] = [i for i in gt.dataset['images'] if i['id']==imgId]
# with open("gt_new.json", 'w') as f:
#     json.dump(gt.dataset, f)
# print("file altered")
# with open("instances_val2014_fakesegm100_results.json", 'r') as f:
#     dt = json.load(f)
# dt = [i for i in dt if i['image_id']==imgId]
# with open("dt_new.json", 'w') as f:
#     json.dump(dt, f)
# print("file altered")
print("="*72)
gt = COCO("gt_new.json")
dt = gt.loadRes("dt_new.json")
cocoEval = COCOeval(gt,dt)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# MWE for me
print("="*72)

parser = argparse.ArgumentParser(description="Visualize instance segmentation masks.")
parser.add_argument('--config', nargs='+', type=str,
                    default=["default_config.yaml"],
                    help="Path to config files.")
parser.add_argument('--no-cuda', action='store_true',
                    help="Do not use CUDA.")
parser.add_argument('--cuda-max-devices', type=int, default=1,
                    help="Maximum number of available GPUs.")
parser.add_argument('--resume-run', type=str, default=None,
                    help="Load existing checkpoint and resume training.")
args = parser.parse_args()

# parse configurations
cfg = Config()
cfg.update(args.config)

cfg.run_dir = "runs"
_, val_loader = make_data_loader(cfg, batch_size=2)

cocosaver = COCOSaver(gt=True, cfg=cfg)
for sample in val_loader:
    _, targets = sample
    for target in targets:
        cocosaver.update(target)
        break
cocosaver.save()
cocosaver = COCOSaver(gt=False, cfg=cfg)
for sample in val_loader:
    _, targets = sample
    for target in targets:
        target['scores'] = torch.ones(target['labels'].shape)
        cocosaver.update(target)
        break
cocosaver.save()
gt = COCO("runs/annotations_gt.json")
dt = gt.loadRes("runs/annotations_pred.json")
cocoEval = COCOeval(gt,dt)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
