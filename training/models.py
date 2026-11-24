"""Architectures only: no pretrained builders, checkpoints or downloads."""
import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def block(in_channels, out_channels, stride=2):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class HelmetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2), nn.Linear(128, 2))

    def forward(self, images):
        return self.head(self.features(images))


def build_detector(num_classes=4, feature_stride=16):
    if feature_stride not in (8,16): raise ValueError('Feature stride must be 8 or 16')
    backbone = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128, stride=1 if feature_stride==8 else 2))
    backbone.out_channels = 128
    anchors = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((.5, 1., 2.),))
    return FasterRCNN(backbone, num_classes=num_classes,  # includes background
                      min_size=384, max_size=640,
                      image_mean=[.5, .5, .5], image_std=[.5, .5, .5],
                      rpn_anchor_generator=anchors,
                      box_roi_pool=MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2),
                      rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=500,
                      rpn_pre_nms_top_n_test=500, rpn_post_nms_top_n_test=200,
                      box_detections_per_img=100)


def forbid_external_weights():
    def denied(*args, **kwargs):
        raise RuntimeError('External pretrained weights are prohibited in this project.')
    torch.hub.load_state_dict_from_url = denied
    torch.hub.load = denied
