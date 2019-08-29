from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def make_model(cfg):

    if cfg.model_name == 'maskrcnn_resnet50_fpn':
        # make model
        if cfg.coco_pretrained:
            model = maskrcnn_resnet50_fpn(pretrained=True)
            # replace the pre-trained FasterRCNN head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(
                # in_features
                model.roi_heads.box_predictor.cls_score.in_features,
                # num_classes
                len(cfg.label_dict) + 1)
            # replace the pre-trained MaskRCNN head with a new one
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                # in_features_mask
                model.roi_heads.mask_predictor.conv5_mask.in_channels,
                # hidden_layer
                model.roi_heads.mask_predictor.conv5_mask.out_channels,
                # num_classes
                len(cfg.label_dict) + 1)
        else:
            params = {
                'num_classes': len(cfg.label_dict) + 1,  # including background
                'pretrained': False}
            model = maskrcnn_resnet50_fpn(**params)
    else:
        raise NotImplementedError
    return model