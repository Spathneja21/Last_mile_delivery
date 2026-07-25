# Full 3D/depth scene model: reuses a frozen backbone taken from a pretrained
# SceneSegNetwork and chains it with a depth-specific context module, neck and
# head to regress a per-pixel depth (3D scene) prediction from a camera image.
from .pre_trained_backbone import PreTrainedBackbone
from .depth_context import DepthContext
from .scene_3d_neck import Scene3DNeck
from .scene_3d_head import Scene3DHead

import torch.nn as nn

class Scene3DNetwork(nn.Module):
    def __init__(self, pretrained):
        super(Scene3DNetwork, self).__init__()

        # Upstream blocks
        self.PreTrainedBackbone = PreTrainedBackbone(pretrained)  # 'pretrained' is a SceneSegNetwork instance whose Backbone is reused (frozen)

        # Depth Context
        self.DepthContext = DepthContext()

        # Neck
        self.DepthNeck = Scene3DNeck()

        # Depth Head
        self.SuperDepthHead = Scene3DHead()


    def forward(self, image):
        features = self.PreTrainedBackbone(image)
        deep_features = features[4]  # deepest/lowest-resolution backbone stage, fed into the depth context module
        context = self.DepthContext(deep_features)
        neck = self.DepthNeck(context, features)
        prediction = self.SuperDepthHead(neck, features)
        return prediction
