import numpy as np
from PIL import Image, ImageDraw, ImageFont

from torchvision.transforms import functional as F


class InstSegVisualization(object):
    """Visualize an image with instance segmentation annotations.

    Args:
        cfg (argparse Namespace): visualization configurations.
        image (torch.FloatTensor[C, H, W]): The original image,
            in the 0-1 range.
        boxes (torch.Tensor[N, 4]): the predicted boxes
            in [x0, y0, x1, y1] format,
            with values between 0 and H and 0 and W
        labels (torch.Tensor[N]): the predicted labels for each image
        scores (torch.Tensor[N]): the scores or each prediction
        masks (torch.Tensor[N, H, W]): the predicted masks for each instance,
            in 0-1 range. In order to obtain the final segmentation masks,
            the soft masks can be thresholded,
            generally with a value of 0.5 (mask >= 0.5).
    """

    def __init__(self, cfg, image,
                 boxes=None, labels=None, scores=None, masks=None):

        self.cfg = cfg
        self.image = image.detach().cpu()
        self.boxes = (boxes.detach().cpu().numpy()
                      if boxes is not None else None)
        self.labels = (labels.detach().cpu().numpy()
                       if labels is not None else None)
        self.scores = (scores.detach().cpu().numpy()
                       if scores is not None else None)
        self.masks = (masks.detach().cpu().numpy()
                      if masks is not None else None)
        # drop low score instances
        if self.scores is not None:
            if self.boxes is not None:
                self.boxes = self.boxes[
                    self.scores > cfg.visual_score_cutoff, :]
            if self.masks is not None:
                self.masks = self.masks[
                    self.scores > cfg.visual_score_cutoff, :, :]
            if self.labels is not None:
                self.labels = self.labels[
                    self.scores > cfg.visual_score_cutoff]
            self.scores = self.scores[
                self.scores > cfg.visual_score_cutoff]
        self.width = np.floor(image.shape[2] * cfg.up_scale).astype(np.uint16)
        self.height = np.floor(image.shape[1] * cfg.up_scale).astype(np.uint16)
        self.font = ImageFont.truetype(
            cfg.font, cfg.font_size, encoding="unic")
        self.up_scale = cfg.up_scale
        self.output = None

    def plot_image(self, mode='RGB'):
        """Plots an original image (overwrites existing output attribute).

        Args:
            mode (str): PIL Image mode, default to RGB.
        """
        output = F.to_pil_image(self.image, mode=mode)
        self.output = (output.convert('RGBA')
                             .resize(size=(self.width, self.height)))

    def add_bbox(self):
        """Adds bounding boxes for all instances.
        """
        output_draw = ImageDraw.Draw(self.output)
        # check existence of instances
        if not self.boxes.shape[0] == 0:
            for box in self.boxes:
                output_draw.rectangle(
                    box * self.up_scale,
                    outline=tuple(self.cfg.bbox_outline))

    def add_label(self):
        """Adds labels for all instances.
        """
        output_draw = ImageDraw.Draw(self.output)
        # check existence of instances
        if not self.labels.shape[0] == 0:
            for box, label in zip(self.boxes, self.labels):
                output_draw.text(
                    (box[0] * self.up_scale, box[3] * self.up_scale),
                    self.cfg.int_dict[label],
                    font=self.font,
                    fill=tuple(self.cfg.label_fill))

    def add_label_score(self):
        """Adds labels and predicted scores for all instances.
        """
        output_draw = ImageDraw.Draw(self.output)
        # check existence of instances
        if not self.labels.shape[0] == 0 and self.scores is not None:
            for box, label, score in zip(self.boxes, self.labels, self.scores):
                output_draw.text(
                    (box[0] * self.up_scale, box[3] * self.up_scale),
                    '{}: {:d}%'.format(self.cfg.int_dict[label],
                                       int(score * 100)),
                    font=self.font,
                    fill=tuple(self.cfg.label_fill))

    def add_binary_mask(self, threshold=None):
        """Adds binary masks for all instances.

        Args:
            threshold (float): the threshold value for soft segmentation masks
        """

        # check existence of instances
        if not self.masks.shape[0] == 0:
            # threshold to get a [N, H, W] binary mask
            if threshold is not None:
                binary_mask = self.masks > threshold
            else:
                binary_mask = self.masks
            # colored mask, starting out as black and transparent
            color_mask = np.zeros(
                (binary_mask.shape[1], binary_mask.shape[2], 4),
                dtype=np.uint8)
            for val, color in self.cfg.category_palette.items():
                # for every category, take the union of masks of all instances
                category_mask = np.any(
                    binary_mask[self.labels == val, :, :], axis=0)
                # fill color in for each category
                color_mask += (
                    category_mask[:, :, np.newaxis] *
                    np.array(color, dtype=np.uint8)[np.newaxis, np.newaxis, :])
            # overlay color mask and image
            color_mask = Image.fromarray(
                color_mask, mode="RGBA").resize(size=(self.width, self.height))
            self.output = Image.alpha_composite(self.output, color_mask)

    def show(self):
        """Shows the image."""
        if self.output is None:
            raise TypeError("No initialized images.")
        else:
            self.output.show()

    def save(self, out_dir):
        """Saves the image.

        Args:
            out_dir (str): directory where the output image is saved.
        """

        self.output.save(out_dir)
