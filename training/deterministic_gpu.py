"""Strict GPU determinism and torchvision's reference single-level RoIAlign."""
import importlib
import os
import torch
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.poolers import _filter_input, _setup_scales
from torchvision.ops._utils import convert_boxes_to_roi_format


def configure_deterministic_gpu():
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('Configure CUBLAS_WORKSPACE_CONFIG=:4096:8 before initializing CUDA')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')


class DeterministicRoIAlign(MultiScaleRoIAlign):
    """Same single-level geometry using installed torchvision reference math.

    Native CUDA backward is nondeterministic. On this Windows build, compiled
    reference fallback is unavailable, so call its undecorated eager function.
    No weights, sampling quotas or gradients are substituted with pretrained data.
    """
    def forward(self,x,boxes,image_shapes):
        features=_filter_input(x,self.featmap_names)
        if len(features)!=1 or self.sampling_ratio<=0:
            raise ValueError('Deterministic pool supports one feature level and fixed positive sampling ratio')
        if self.scales is None or self.map_levels is None:
            self.scales,self.map_levels=_setup_scales(features,image_shapes,self.canonical_scale,self.canonical_level)
        module=importlib.import_module('torchvision.ops.roi_align')
        reference=getattr(module._roi_align,'__wrapped__',None)
        if reference is None:raise RuntimeError('Installed torchvision reference RoIAlign API changed')
        rois=convert_boxes_to_roi_format(boxes)
        return reference(features[0],rois,self.scales[0],self.output_size[0],self.output_size[1],self.sampling_ratio,False)
