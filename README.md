## Known Issues

* Empty images cannot be handled by the current model (so random crop data augmentation is turned off, and the `drop_empty` option in `GoogleEarthProInstSeg.__init__()` is turned on).
* This code is not compatible with the `torch.nn.DataParallel` utilities. This is because when distribution happens within the `DataParallel` module, only `torch.Tensor` objects are treated as mini-batches and can be distributed (`scatter()`ed). This results in errors when the module attempts to split single image onto multiple GPUs and the color dimension is corrupted. `DistributedDataParallel()` have to be used and some sample code is available in [`torchvision/references/detection/train.py`](https://github.com/pytorch/vision/blob/master/references/detection/train.py).

## TODO

* [ ] Training loss is not going down, why? Read more on the architecture of Mask-RCNN
* [x] Log train eval and visualization on TB, I did that, and the pattern looks broadly similar to that on val set, I also tried overfitting on a couple of training images, and the following issue persisted.
* [ ] It is not predicting anything for certain categories, why? The scores are not going up (and the scores are always similar), this may have something to do with class weights? Tin roof is always the category that is being predicted. The other categories are mostly missed.
* [ ] If the random seed is not good, the model does not converge, what happens is that each mask has a distinct grid pattern, and that the generated bbox are usually very large. I'm suspecting that this way, the model is not getting proper gradient so the training process does not improve the model. Re-running should solve this issue.
