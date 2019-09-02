## Known Issues

* Empty images cannot be handled by the current model (so random crop data augmentation is turned off, and the `drop_empty` option in `GoogleEarthProInstSeg.__init__()` is turned on).
* This code is not compatible with the `torch.nn.DataParallel` utilities. This is because when distribution happens within the `DataParallel` module, only `torch.Tensor` objects are treated as mini-batches and can be distributed (`scatter()`ed). This results in errors when the module attempts to split single image onto multiple GPUs and the color dimension is corrupted. `DistributedDataParallel()` have to be used and some sample code is available in [`torchvision/references/detection/train.py`](https://github.com/pytorch/vision/blob/master/references/detection/train.py).

## TODOs IN THE FUTURE

* [ ] Add post processing that is similar to NMS in spirit.

## TODOs NOW

* [ ] `utils/preprocess_openaitanzania.py` almost ready to go - would need to debug the alignment between img and mask
* [ ] `model/__init__.py` update
* [ ] update `config/googleearthpro.yaml`, `config/googlestaticmap.yaml`, `config/openaitanzania.yaml`, `config/aerialhistory.yaml`.
* [ ] `dataloader/mask_transforms.py` connect with various annotation formats
* [ ] `dataloader/__init__.py` merge all dataset loaders, drop empty images
* [ ] `maskrcnn/train.py`, `utils/save_ckpt_log_tb.py`, `utils/coco.py` and `utils/eval.py` clean up and make it up to date with new config files (supporting multi datasets), inference mode is loading last checkpoint instead of best model. Evaluate training samples and generate meaningful batches (intead of logging on tb). Add eval for training sample.
* [ ] go through all the config params in all files
