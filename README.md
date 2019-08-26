## Known Issues

* Empty images cannot be handled by the current model (so random crop data augmentation is turned off, and the `drop_empty` option in `GoogleEarthProInstSeg.__init__()` is turned on).
* This code is not compatible with the `torch.nn.DataParallel` utilities. This is because when distribution happens within the `DataParallel` module, only `torch.Tensor` objects are treated as mini-batches and can be distributed (`scatter()`ed). This results in errors when the module attempts to split single image onto multiple GPUs and the color dimension is corrupted. `DistributedDataParallel()` have to be used and some sample code is available in [`torchvision/references/detection/train.py`](https://github.com/pytorch/vision/blob/master/references/detection/train.py).

## TODO

* FEATURE ADDITION
* [ ] Evaluate training samples and generate meaningful batches (intead of logging on tb). Add evaluation for training samples.
* [ ] Evaluate without 100 detections (lots of question marks?)
* [ ] Inference mode is loading last checkpoint instead of best model.
* [ ] Change training data to OpenAITanzania.
* [ ] Add post processing that is similar to NMS in spirit.
* DEBUGGING
* [ ] Resize images?
* [ ] Currently the model is not training properly. This has happened in the past when learning rate is set too low and the generated predictions (1) are clustered with lots of duplicates and (2) the predicted scores are stuck at a low level and not changing over the course of training. From the losses it seems like RPN was not really working (both objectness and rpn_box_reg). I have suspected that this is due to network architecture and switched to mobilenet_v2. (Since Feature Pyramid Network is designed for multiscale object detection which is not super relevant in this context.) I also adjusted AnchorGenerator to better fit our predominantly small object annotations, and dropped COCO pretraining (backbone was pretrained though).