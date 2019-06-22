import random
import PIL
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F

import mask_transforms

class Compose(torchvision.transforms.Compose):
    """Modified to compose transforms together, with target transforms."""
    
    def __call__(self, image, target):
        """Modified to compose transforms together, with target transforms.
        
        Args:
            image: image to be transformed
            target: target to be transformed

        Returns:
            transformed images and targets
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize(object):
    """Modified to resize the input image-label pair to the given size.

    Args:
        width, height (int): width and height of the desired resized object.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, width, height, interpolation=PIL.Image.BILINEAR):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def __call__(self, image, target):
        """Resize instance segmentation image-label pairs.

        Args:
            image (PIL Image): image to be resized
            target (InstanceMask object): instance masks to be resized

        Returns:
            image (PIL Image): resized image
            target (InstanceMask object): resized instance masks
        """
        return (F.resize(image,(self.height, self.width), self.interpolation),
                target.resize(h=self.height, w=self.width))

    def __repr__(self):
        return self.__class__.__name__ + "(size=({0}, {1}), interpolation={2})".format(
            self.height, self.width, self.interpolation)

class ColorJitter(torchvision.transforms.ColorJitter):
    """Modified to add instance mask in return values."""

    def __call__(self, image, target):
        """
        Args:
            img (PIL Image): Input image.
            target (InstanceMask object): Input instance masks.

        Returns:
            PIL Image: Color jittered image.
            target (InstanceMask object): Input instance masks.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(image), target

class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Image to be flipped.
            target (InstanceMask): Instance masks to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
            target (InstanceMask): Randomly flipped instance masks.
        """
        if random.random() < self.p:
            image = F.hflip(image)
            target = target.flip(FLIP_LEFT_RIGHT=True)
        return image, target

class RandomVerticalFlip(torchvision.transforms.RandomVerticalFlip):
    
    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Image to be flipped.
            target (InstanceMask): Instance masks to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
            target (InstanceMask): Randomly flipped instance masks.
        """
        if random.random() < self.p:
            image = F.vflip(image)
            target = target.flip(FLIP_TOP_BOTTOM=True)
        return image, target

class ToTensor(torchvision.transforms.ToTensor):
    def __call__(self, image, target):
        """
        Args:
            image (PIL Image or numpy.ndarray): Image to be converted to tensor.
            target (InstanceMask): Instance Mask to be converted to tensor.

        Returns:
            Tensor: Converted image.
            dict: a dict of tensors following Mask-RCNN conventions.
        """
        return F.to_tensor(image), target.to_tensor()

class Normalize(torchvision.transforms.Normalize):
    def __call__(self, image, target):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W) to be normalized.
            target (InstanceMask): target instance masks.

        Returns:
            Tensor: Normalized Tensor image.
            Tensor: target instance masks.
        """
        return F.normalize(image, self.mean, self.std, self.inplace), target

