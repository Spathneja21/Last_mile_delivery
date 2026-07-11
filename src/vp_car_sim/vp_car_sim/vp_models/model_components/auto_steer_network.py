import sys
import types

import torch
from . import common_layers
from . import auto_steer_backbone
from . import auto_steer_neck
from . import auto_steer_percept_head
from .auto_steer_backbone import AutoSteerBackbone
from .auto_steer_neck import AutoSteerNeck
from .auto_steer_percept_head import AutoSteerPerceptHead
from .common_layers import Conv

image_width = 1024
image_height = 512


def _register_legacy_pickle_aliases():
    """
    The AutoSteer 2.0 .pth checkpoint pickles a full model object (not just a plain
    state_dict) saved from the original autoware_vision_pilot source tree, where these
    classes lived under Models.model_components.auto_steer.*. torch.load's unpickler
    needs that exact dotted module path importable to reconstruct the object - register
    sys.modules aliases pointing at this copy so it resolves without needing the
    original autoware_vision_pilot directory present on this machine.
    """
    aliases = {
        'Models': types.ModuleType('Models'),
        'Models.model_components': types.ModuleType('Models.model_components'),
        'Models.model_components.auto_steer': types.ModuleType('Models.model_components.auto_steer'),
        'Models.model_components.common_layers': common_layers,
        'Models.model_components.auto_steer.auto_steer_backbone': auto_steer_backbone,
        'Models.model_components.auto_steer.auto_steer_neck': auto_steer_neck,
        'Models.model_components.auto_steer.auto_steer_percept_head': auto_steer_percept_head,
        'Models.model_components.auto_steer.auto_steer_network': sys.modules[__name__],
    }
    for name, module in aliases.items():
        sys.modules.setdefault(name, module)


_register_legacy_pickle_aliases()

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.net = AutoSteerBackbone(width, depth, csp)
        self.fpn = AutoSteerNeck(width, depth, csp)
        self.head = AutoSteerPerceptHead(width[4])

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        output = self.head(x)
        return output

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class AutoSteerNetwork:
    def __init__(self):
        self.dynamic_weighting = {
            'n': {'csp': [False, True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 16, 32, 64, 128, 256]},
            's': {'csp': [False, True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 32, 64, 128, 256, 512]},
            'm': {'csp': [True, True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 64, 128, 256, 512, 512]},
            'l': {'csp': [True, True], 'depth': [2, 2, 2, 2, 2, 2], 'width': [3, 64, 128, 256, 512, 512]},
            'x': {'csp': [True, True], 'depth': [2, 2, 2, 2, 2, 2], 'width': [3, 96, 192, 384, 768, 768]},
        }

    def build_model(self, version):
        csp = self.dynamic_weighting[version]['csp']
        depth = self.dynamic_weighting[version]['depth']
        width = self.dynamic_weighting[version]['width']
        return YOLO(width, depth, csp)

    def load_model(self, version, num_classes, checkpoint_path, map_location='cpu'):
        csp = self.dynamic_weighting[version]['csp']
        depth = self.dynamic_weighting[version]['depth']
        width = self.dynamic_weighting[version]['width']
        model = YOLO(width, depth, csp)

        # map_location is required here: the checkpoint was saved from a CUDA device,
        # and torch.load defaults to the saving device - this fails outright on a
        # CPU-only host without an explicit map_location (unlike every other model's
        # inference wrapper in this workspace, which already passes one through).
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        loaded_model = ckpt['model']
        state_dict = loaded_model.state_dict()

        model.load_state_dict(state_dict)

        return model
