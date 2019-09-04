from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def make_model(cfg):
    """Initializes the model.

    Args:
        cfg (Config): pass in all configurations
    """

    if cfg.model_name == 'maskrcnn_resnet50_fpn':
        if cfg.coco_pretrained:
            model = maskrcnn_resnet50_fpn(pretrained=True)
        else:
            model = maskrcnn_resnet50_fpn(
                num_classes=cfg.num_classes, pretrained=False)
        pretrained_num_classes = (model.roi_heads
                                       .mask_predictor
                                       .mask_fcn_logits
                                       .out_channels)
        swap_predictors = (
            (cfg.num_classes != pretrained_num_classes) or
            cfg.swap_model_predictors)
        if swap_predictors:
            # replace the pre-trained FasterRCNN head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(
                # in_features
                model.roi_heads.box_predictor.cls_score.in_features,
                # num_classes
                cfg.num_classes)
            # replace the pre-trained MaskRCNN head with a new one
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                # in_features_mask
                model.roi_heads.mask_predictor.conv5_mask.in_channels,
                # hidden_layer
                model.roi_heads.mask_predictor.conv5_mask.out_channels,
                # num_classes
                cfg.num_classes)
    elif cfg.model_name == 'adjust_anchor':
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128),),
            aspect_ratios=((1.0,),))
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        model = MaskRCNN(
            backbone=backbone,
            num_classes=cfg.num_classes,
            rpn_anchor_generator=anchor_generator,
            rpn_nms_thresh=0.5,
            box_score_thresh=0.4)
    else:
        raise NotImplementedError
    return model
