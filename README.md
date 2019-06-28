## TODO

* empty images cannot be handled by the current model (so random crop data augmentation is turned off, and the `drop_empty` option in GoogleEarthProInstSeg.__init__() is turned on).
* fix the warnings on loss = np.sum(iterable)
* fix the logging for `loss_dict`
* this code currently fails on multi GPU cases
