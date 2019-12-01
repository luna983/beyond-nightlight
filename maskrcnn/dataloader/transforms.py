import random
import PIL.Image
import PIL.ImageFilter

import torchvision
from torchvision.transforms import functional as F

from .mask_transforms import Polygon


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
        return (F.resize(image, (self.height, self.width), self.interpolation),
                target.resize(h=self.height, w=self.width))


class Blur(object):
    """Add blurring of the original images.

    Args:
        blur_prob (float): blurring is off if None,
            probability of blurring
        blur_times (int): times of blurring
    """

    def __init__(self, blur_prob, blur_times):
        self.blur_prob = blur_prob
        self.blur_times = blur_times

    def __call__(self, image, target):
        """Blurs the images.

        Args:
            image (PIL Image): image to be transformed
            target (InstanceMask object): instance masks

        Returns:
            image (PIL Image): blurred image
            target (InstanceMask object): original instance masks
        """
        if self.blur_prob is not None:
            for _ in range(self.blur_times):
                if random.random() < self.blur_prob:
                    image = image.filter(PIL.ImageFilter.BLUR)
        return image, target


class ColorJitter(torchvision.transforms.ColorJitter):
    """Modified to add instance mask in return values."""

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Input image.
            target (InstanceMask object): Input instance masks.

        Returns:
            PIL Image: Color jittered image.
            target (InstanceMask object): Input instance masks.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(image), target


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    """Modified to add instance mask in return values."""

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
    """Modified to add instance mask in return values."""

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


class RandomCrop(torchvision.transforms.RandomCrop):
    """Modified to add instance mask in return values."""

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Image to be cropped.
            target (InstanceMask): Target instance masks to be cropped.

        Returns:
            PIL Image: Cropped image.
            target (InstanceMask): Cropped target instance masks.
        """
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0),
                          self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]),
                          self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), target.crop(i=i, j=j, h=h, w=w)


class FillEmpty(object):
    """Creates placeholder annotations to add negative samples in training.

    Args:
        activate (bool): if False, no placeholder annotations are added.
        category_int (int or NoneType): integer denoting placeholder class.
    """

    def __init__(self, activate, category_int=None):
        self.activate = activate
        self.category_int = category_int

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Image to be processed.
            target (InstanceMask): Target instance masks to be processed.

        Returns:
            PIL Image: Processed image.
            target (InstanceMask): Processed target instance masks.
        """
        if self.activate:
            if len(target) == 0:
                top = int(target.height * 0.9)
                left = int(target.width * 0.9)
                bottom = target.height
                right = target.width
                # add image patch
                placeholder_patch = PIL.Image.new(
                    mode='RGB',
                    size=(right - left, bottom - top),
                    color=(128, 128, 128))
                image.paste(placeholder_patch, box=(left, top))
                # add instance
                instance = Polygon(
                    width=target.width, height=target.height,
                    category=self.category_int)
                coords = [[top, left], [top, right], [bottom, right],
                          [bottom, left], [top, left]]
                instance.load_single_polygon(coords)
                target.instances.append(instance)
                return image, target
        return image, target


class ToTensor(torchvision.transforms.ToTensor):
    """Modified to return tensors that follow the Mask R-CNN conventions."""

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image or numpy.ndarray): Image to be converted.
            target (InstanceMask): Instance Mask to be converted to tensor.

        Returns:
            Tensor: Converted image.
            dict: a dict of tensors following Mask-RCNN conventions.
        """
        return F.to_tensor(image), target.to_tensor()
