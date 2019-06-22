from PIL import Image, ImageDraw

from torchvision.transforms import functional as F

class Visualization(object):
    """Visualize an image with instance segmentation annotations.

    Args:
        cfg (argparse Namespace): visualization configurations.
    """
    def __init__(cfg):
        self.cfg = cfg

    def load_image(image,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   mode="RGB"):
        """Loads an original image (overwrites existing images).

        Args:
            image (torch.Tensor[C, H, W]): The original image, in the 0-1 range,
                normalized to mean and std.
            mean, std (list of floats): The mean and std with which images are normalized,
                default to the ImageNet mean and std (FasterRCNN/MaskRCNN use this too).
            mode (str): PIL Image mode, default to RGB.
        """
        if image.is_cuda:
            self.img = image.detach().cpu()
        else:
            self.img = image.detach()

        self.img = F.to_pil_image(self.img, mode=mode):

    def add_bbox(boxes):
        """Adds bounding boxes for all instances.

        Args:
            boxes (torch.Tensor[N, 4]): the predicted boxes in [x0, y0, x1, y1] format,
                with values between 0 and H and 0 and W
        """

    def add_label(label):
        """Adds labels for all instances.

        Args:
            labels (torch.Tensor[N]): the predicted labels for each image
        """

    def add_label_score(label, score):
        """Adds labels and predicted scores for all instances.

        Args:
            labels (torch.Tensor[N]): the predicted labels for each image
            scores (Tensor[N]): the scores or each prediction
        """

    def add_binary_mask(mask, threshold=0.5):
        """Adds binary masks for all instances.

        Args:
            masks (torch.Tensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
                obtain the final segmentation masks, the soft masks can be thresholded, generally
                with a value of 0.5 (mask >= 0.5).
            threshold (float): the threshold value for soft segmentation masks
        """

    def save(out_dir):
        """Saves the image.

        Args:
            out_dir (str): directory where the output image is saved.
        """
    
    mask = Image.open(os.path.join(args.in_mask_dir, file + ".png")).convert("L")
    mask_array = np.array(mask)
    rgba = np.zeros((mask_array.shape[0], mask_array.shape[1], 4))
    for i, col in enumerate(palette):
        for channel in range(4):
            rgba[:, :, channel] += col[channel] * (mask_array == i)
    mask = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    output = Image.alpha_composite(im.resize(mask.size), mask)
    output.save(os.path.join(args.out_dir, file + ".png"))


    from PIL import Image, ImageDraw
blank_image = Image.new('RGBA', (400, 300), 'white')
img_draw = ImageDraw.Draw(blank_image)
img_draw.rectangle((70, 50, 270, 200), outline='red', fill='blue')
img_draw.text((70, 250), 'Hello World', fill='green')
blank_image.save('drawn_image.jpg')