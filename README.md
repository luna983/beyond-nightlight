## Mask RCNN

[This medium article on Faster RCNN](https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439) and [this medium article on Mask RCNN](https://medium.com/@fractaldle/mask-r-cnn-unmasked-c029aa2f1296) is really helpful in explaining the mechanics of the model.

## Known Issues

* Empty images cannot be handled by the current model (the error is intentional and thrown when trying to go through matchers, not sure how an empty image should be evaluated and what loss should be returned, our data contain plenty of empty space in a non-empty image anyways, so I'm not going to try and fix that; right now the empty images are dropped when going through the collate function, and when preprocessing empty images are sometimes dropped to improve efficiency. Note that because of this, infer images will have a much high proportion of images that are empty, which may or may not be a problem).
* This code is not compatible with the `torch.nn.DataParallel` utilities. This is because when distribution happens within the `DataParallel` module, only `torch.Tensor` objects are treated as mini-batches and can be distributed (`scatter()`ed). This results in errors when the module attempts to split single image onto multiple GPUs and the color dimension is corrupted. `DistributedDataParallel()` have to be used and some sample code is available in [`torchvision/references/detection/train.py`](https://github.com/pytorch/vision/blob/master/references/detection/train.py).

## TODOs

- [ ] post processing annotations
    - [ ] texts on images were classified as houses (remove manually? train with samples?)
    - [ ] drop overlapping annotations

## Experiments

w/ tanazania data
- 00 base spec: COCO pretraind, original anchor, no data aug
- 02 smaller anchors, no pretraining
- 03 stronger data aug
    - validation against census, count .6ish, others low (.2ish)
- 05 smaller scale of training image (.2m, inference GSM zoom 19 = .3m), w/ aug
    - validation against census, count 0.68, others noticably higher (.4ish)
- 06 smaller scale + smaller anchor

w/ google earth pro
- 04 pretrained on tanzania
    - validation against census, count corr 0.72, others (size versus durable) weaker .2ish

## Logs

- [Nov 13 14:57] So I spent most of today looking at why we were not able to predict either house size or overall asset scores very well. I also thought more about how we should approach demonstrating the inequality aspect of our dataset.
    - For the latter, I felt that Gini or other aggregate measures of inequality will be extremely challenging to predict or ground truth in a meaningful way. They are extremely aggregated, or else it makes very little sense, and it is somewhat unclear how we want to interpret differences in inequality across these highly aggregated areas (e.g. states) or if that could be predicted at all. I think predicting neighborhood disparity (potentially using census blocks or locality pairs as observational unit) will both simplify the task, make it more intuitive and easier to predict, and make it easier to implement and debug the pipelines.
    - For the former, I am spending a lot of time looking at images. It's hard to get ground truth labels (mostly because I don't know how valuable those will be, actually, still haven't started the labelling process yet) I sort them by values for a specific variable, and I just look at the 0th, 20th, ..., 180th images, hoping to find some patterns. `VPH_REFRI` seems to correspond to urbanization patterns a little bit. `VPH_SNBIEN` the nodurable variable seems to be actually a bad proxy as it gets much noisier for small localities (or whatever, I don't know, don't think I could predict anything from these images, they look totally random). `VPH_3YMASC` this looks much worse than what I would have thought - just no signal in house size it seems. Plus the ML model makes a lot of false positive predictions on trees and/or roads and it's just a hot mess. `asset_score1` makes maybe a little bit of sense, but not much.
    - These to me suggest that even though we have the choice of creating and extracting more variables from the mask rcnn predictions (neighborhood regularity, which can be measured by road angle?; roof type and density), I am incredibly skeptical that this would help. The census variables that I'm using are not visible enough from space, is how I make sense of this. We need something more direct, e.g. roof materials or market access (?) and asset/wealth measures that are based on those.
    - I think I found the right metric for describing clustering of households - the average no. of neighbors within a certain bandwidth, visually this corresponds to my intuition a lot better, and it is correlated with public good provision. However, when I control for census population there is still a correlation, but when I control for satellite house count it goes away. Not sure why and if that matters.
    - I scanned all the census variables and thought about all the things that we could possibly observe, one other variable that may be of interest is dirt/concrete floor in `VPH_PISODT`, that does seem to be correlated with roof type and more importantly clustering (how many neighbors you have), which seems interesting. Again, I am truly skeptical of what size distribution captures other than noise.