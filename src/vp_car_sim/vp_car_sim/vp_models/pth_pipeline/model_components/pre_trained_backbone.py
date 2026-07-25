#! /usr/bin/env python3
# Wraps the backbone of an already-trained SceneSegNetwork so the 3D scene
# pipeline (Scene3DNetwork) can reuse its learned image features as a frozen
# encoder, avoiding retraining the encoder from scratch.
import torch.nn as nn

class PreTrainedBackbone(nn.Module):
    def __init__(self, pretrainedModel):
        super(PreTrainedBackbone, self).__init__()

        self.pretrainedBackBone = pretrainedModel.Backbone  # encoder lifted from an already-trained SceneSegNetwork
        for param in self.pretrainedBackBone.parameters():
            param.requires_grad = False  # freeze weights so this backbone is not updated during 3D-head training

    def forward(self, image):
        features = self.pretrainedBackBone(image)
        return features
