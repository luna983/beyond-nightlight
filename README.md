## Known Issues

* Empty images cannot be handled by the current model (so random crop data augmentation is turned off, and the `drop_empty` option in `GoogleEarthProInstSeg.__init__()` is turned on).
* This code is not compatible with the `torch.nn.DataParallel` utilities. This is because when distribution happens within the `DataParallel` module, only `torch.Tensor` objects are treated as mini-batches and can be distributed (`scatter()`ed). This results in errors when the module attempts to split single image onto multiple GPUs and the color dimension is corrupted. `DistributedDataParallel()` have to be used and some sample code is available in [`torchvision/references/detection/train.py`](https://github.com/pytorch/vision/blob/master/references/detection/train.py).

## TODO

* [ ] Evaluate training samples and generate meaningful batches (intead of logging on tb).
* [ ] Inference mode is loading last checkpoint instead of best model.
* [ ] Change training data to spacenet.
* [ ] The backbone of this model is currently ResNet50 + FPN and that is not necessarily a good choice for the task at hand (small objects of relatively homogenous sizes). Would try to drop FPN (and try smaller models like mobile net or smaller resnets to increase batch sizes).
* [ ] Add post processing that is similar to NMS in spirit.
