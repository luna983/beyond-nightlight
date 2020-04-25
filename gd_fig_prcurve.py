import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from maskrcnn.utils.eval import evaluate


matplotlib.rc('pdf', fonttype=42)

# 400 pixels = 22.7 m2 gives the best result
# but that's a rather high threshold, not super sensible
# thresholds between 0-400 do not change results substantially
# I'm deciding against this
# as it complicates the interpretation
# and does not seem to promise performance gains
area_min = 0

cfg = argparse.Namespace()
cfg.out_mask_dir = 'data/Siaya/Pred/cv'

input_pred_files = [f'data/Siaya/Pred/val/pred_cv{i}.json' for i in range(3)]
input_gt_files = [f'data/Siaya/Pred/val/gt_cv{i}.json' for i in range(3)]

output_pred_file = 'data/Siaya/Pred/cv/val/pred.json'
output_gt_file = 'data/Siaya/Pred/cv/val/gt.json'

output_figure = 'output/fig-prcurve/raw.pdf'

preds = []
gts = {'annotations': [],
       'images': [],
       'categories': [{'id': 1, 'name': 'building'}]}
# renumber all the instance ids to prevent non unique ids
instance_id = 0
for i, (pred_file, gt_file) in enumerate(zip(input_pred_files, input_gt_files)):
    # merge predictions
    with open(pred_file, 'r') as f:
        pred = json.load(f)
    for ann in pred:
        ann['image_id'] = ann['image_id'] + i * 40
    preds += [ann for ann in pred if ann['area'] > area_min]
    # merge ground truth
    with open(gt_file, 'r') as f:
        gt = json.load(f)
    for img in gt['images']:
        img['id'] = img['id'] + i * 40
    for ann in gt['annotations']:
        ann['image_id'] = ann['image_id'] + i * 40
        instance_id = instance_id + 1
        ann['id'] = instance_id
    gts['annotations'] += [ann for ann in gt['annotations'] if ann['area'] > area_min]
    gts['images'] += gt['images']

with open(output_pred_file, 'w') as f:
    json.dump(preds, f)
with open(output_gt_file, 'w') as f:
    json.dump(gts, f)

# evaluate
stats = evaluate(cfg=cfg, mode='val')
max_recall = stats['AR50']
auc = stats['AP50']
precision = np.array(stats['AP50_curve'])
scores = np.array(stats['scores'])

recall = np.linspace(0, 1, 101)
precision = precision[recall < max_recall]
scores = scores[recall < max_recall]
recall = recall[recall < max_recall]

f1 = 2 * (precision * recall) / (precision + recall)
optim_recall = recall[np.argmax(f1)]
optim_precision = precision[np.argmax(f1)]
optim_score = scores[np.argmax(f1)]
optim_f1 = np.max(f1)

# plot
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(
    recall, precision,
    linewidth=3,
    color='dimgray',
    clip_on=False,
)
ax.add_patch(Polygon(
    [(recall[0], 0), *zip(recall, precision), (recall[-1], 0)],
    facecolor='dimgray', edgecolor='none', alpha=0.1))
ax.text(
    0.05, 0.05,
    f'Area Under Curve: {auc:.2f}',
)
ax.plot(
    optim_recall, optim_precision,
    marker='o',
    markeredgecolor="none",
    markersize=8,
    color='dimgray',
)
ax.annotate(
    f'Max F1: {optim_f1:.2f}\n' +
    f'(Recall: {optim_recall:.2f}; Precision: {optim_precision:.2f})\n',
    xy=(optim_recall - 0.02, optim_precision - 0.02), xycoords='data',
    xytext=(0.05, 0.3), textcoords='data',
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
)
ax.spines['top'].set_color('dimgray')
ax.spines['right'].set_color('dimgray')
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_color('dimgray')
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_aspect('equal')
ax.set_title(
    f'Optimal Confidence Score Cutoff: {optim_score:.2f}\n' +
    f'Max Recall: {recall[-1]:.2f}')
fig.tight_layout()
fig.savefig(output_figure)
