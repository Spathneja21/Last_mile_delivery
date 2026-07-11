#! /usr/bin/env python3
"""
Inference wrapper for the AutoDrive end-to-end driving model.

AutoDrive consumes two consecutive RGB frames and predicts driving signals:
    distance (m), path curvature (1/m), and a binary flag.
See vp_models/model_components/autodrive_network.py.
"""
import torch
from torchvision import transforms

from ..model_components.autodrive_network import AutoDrive, IMAGE_WIDTH, IMAGE_HEIGHT


class AutoDriveInfer():
    def __init__(self, checkpoint_path=''):
        if not checkpoint_path:
            raise ValueError('No path to checkpoint file provided in class initialization')

        # Resize to the model's fixed input size + ImageNet normalisation.
        self.image_loader = transforms.Compose(
            [
                transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for AutoDrive inference')

        self.model = AutoDrive()

        # autodrive.pth is a training checkpoint: {'epoch', 'model' (state_dict), ...}
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def _to_tensor(self, image):
        return self.image_loader(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def inference(self, image_prev, image_curr):
        """
        image_prev, image_curr : PIL.Image (RGB), any size (resized internally).

        Returns (distance_m, curvature_1pm, flag_prob):
            distance_m    : metres, 0..150  (150*(1 - d_norm))
            curvature_1pm : path curvature in 1/m, range (-1, 1)
            flag_prob     : probability in (0, 1)  (sigmoid of the flag logit)
        """
        prev = self._to_tensor(image_prev)
        curr = self._to_tensor(image_curr)

        d_norm, curvature, flag_logit = self.model(prev, curr)

        distance_m = 150.0 * (1.0 - float(d_norm.item()))
        curvature_1pm = float(curvature.item())
        flag_prob = float(torch.sigmoid(flag_logit).item())
        return distance_m, curvature_1pm, flag_prob
