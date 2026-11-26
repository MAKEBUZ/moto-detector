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


class SpatialContext(nn.Module):
    """Add scene context to the CNN feature map before proposal and ROI detection."""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((12, 12))
        self.position = nn.Parameter(torch.empty(1, 144, channels))
        nn.init.normal_(self.position, std=.02)
        self.encoder = nn.TransformerEncoderLayer(channels, 4, 256, dropout=0.,
                                                   batch_first=True, norm_first=True)
        self.projection = nn.Conv2d(channels, channels, 1)
        # The new branch starts as an exact identity on the parent's features.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features):
        pooled = self.pool(features)
        tokens = pooled.flatten(2).transpose(1, 2) + self.position
        encoded = self.encoder(tokens).transpose(1, 2).reshape_as(pooled)
        correction = nn.functional.interpolate(self.projection(encoded), size=features.shape[-2:],
                                               mode="bilinear", align_corners=False)
        return features + correction


def build_detector(num_classes=4, feature_stride=16, proposal_multiplier=1, deterministic_roi=False, small_head_anchors=False, image_size=640):
    if proposal_multiplier not in (1,4): raise ValueError('Proposal multiplier must be 1 or 4')
    if feature_stride not in (8,16): raise ValueError('Feature stride must be 8 or 16')
    backbone = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128, stride=1 if feature_stride==8 else 2))
    backbone.out_channels = 128
    sizes = (16, 32, 64, 128, 256) + ((8,) if small_head_anchors else ())
    anchors = AnchorGenerator(sizes=(sizes,), aspect_ratios=((.5, 1., 2.),))
    pool_class = MultiScaleRoIAlign
    if deterministic_roi:
        from deterministic_gpu import DeterministicRoIAlign
        pool_class = DeterministicRoIAlign
    return FasterRCNN(backbone, num_classes=num_classes,  # includes background
                      min_size=round(image_size*.6), max_size=image_size,
                      image_mean=[.5, .5, .5], image_std=[.5, .5, .5],
                      rpn_anchor_generator=anchors,
                      box_roi_pool=pool_class(featmap_names=['0'], output_size=7, sampling_ratio=2),
                      rpn_pre_nms_top_n_train=1000*proposal_multiplier, rpn_post_nms_top_n_train=500*proposal_multiplier,
                      rpn_pre_nms_top_n_test=500*proposal_multiplier, rpn_post_nms_top_n_test=200*proposal_multiplier,
                      box_detections_per_img=100)


def forbid_external_weights():
    def denied(*args, **kwargs):
        raise RuntimeError('External pretrained weights are prohibited in this project.')
    torch.hub.load_state_dict_from_url = denied
    torch.hub.load = denied
