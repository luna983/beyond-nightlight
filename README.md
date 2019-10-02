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
