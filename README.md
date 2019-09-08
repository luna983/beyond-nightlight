## Mask RCNN
[This medium article on Faster RCNN](https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439) and [this medium article on Mask RCNN](https://medium.com/@fractaldle/mask-r-cnn-unmasked-c029aa2f1296) is really helpful in explaining the mechanics of the model.

## Known Issues

* Empty images cannot be handled by the current model (the error is intentional and thrown when trying to go through matchers, not sure how an empty image should be evaluated and what loss should be returned, our data contain plenty of empty space in a non-empty image anyways, so I'm not going to try and fix that; right now the empty images are dropped when going through the collate function, and when preprocessing empty images are sometimes dropped to improve efficiency. Note that because of this, val and infer images will have a much high proportion of images that are empty, which may or may not be a problem).
* This code is not compatible with the `torch.nn.DataParallel` utilities. This is because when distribution happens within the `DataParallel` module, only `torch.Tensor` objects are treated as mini-batches and can be distributed (`scatter()`ed). This results in errors when the module attempts to split single image onto multiple GPUs and the color dimension is corrupted. `DistributedDataParallel()` have to be used and some sample code is available in [`torchvision/references/detection/train.py`](https://github.com/pytorch/vision/blob/master/references/detection/train.py).

## TODOs NOW

* [ ] I'm worried that int ids combined with instance ids will be too large, and pycocotools are using this integer to actually do calculations, it seems...for test datat this should be fine, but for google static maps this is really concerning... - not having an issue with it for the moment with openaitanzania, but we will see! Maybe this does slow down eval uncessaraily though. - Look, just do an instance counter would solve a lot of the problems, we could even have a separate image id and a real id, just to make sure that eval is efficient and we can connect this to postprocessing easily.
* [ ] noticing some dramatic errors on unsupported empty images (forests, etc.)
* [ ] a feature needs to be added (sampled training sample eval) as the training sample is large and evaling the whole sample is unnecessary.
