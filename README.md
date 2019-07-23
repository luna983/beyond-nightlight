## Known Issues

* Empty images cannot be handled by the current model (so random crop data augmentation is turned off, and the `drop_empty` option in `GoogleEarthProInstSeg.__init__()` is turned on).
* This code is not compatible with the `torch.nn.DataParallel` utilities. This is because when distribution happens within the `DataParallel` module, only `torch.Tensor` objects are treated as mini-batches and can be distributed (`scatter()`ed). This results in errors when the module attempts to split single image onto multiple GPUs and the color dimension is corrupted. `DistributedDataParallel()` have to be used and some sample code is available in [`torchvision/references/detection/train.py`](https://github.com/pytorch/vision/blob/master/references/detection/train.py).

## TODO

* [ ] Training loss not going down, why?
* [ ] Read more on the architecture of Mask-RCNN
* [ ] There is definitely a bug in the evaluation codes, the evaluation output is weird (exactly the same for all the epochs) I suspect that the cocosaver is not overwriting existing files properly
