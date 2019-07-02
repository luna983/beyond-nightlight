## Known Issues

* Empty images cannot be handled by the current model (so random crop data augmentation is turned off, and the `drop_empty` option in GoogleEarthProInstSeg.__init__() is turned on).
* This code is not compatible with the `torch.nn.DataParallel` utilities. This is because when distribution happens within the `DataParallel` module, only `torch.Tensor` objects are treated as mini-batches and can be distributed (`scatter()`ed). This results in errors when the module attempts to split single image onto multiple GPUs and the color dimension is corrupted.

## TODO

* fix the warnings on loss = np.sum(iterable)
* fix the logging for `loss_dict`
