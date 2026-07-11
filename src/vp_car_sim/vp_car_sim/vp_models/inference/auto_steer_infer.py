#! /usr/bin/env python3
import torch
from torchvision import transforms

from ..model_components.auto_steer_network import AutoSteerNetwork

# Model input size required by AutoSteer 2.0
MODEL_WIDTH = 1024
MODEL_HEIGHT = 512


class AutoSteerNetworkInfer():
    def __init__(self, checkpoint_path=''):

        # Image loader. AutoSteer's contract (confirmed against the production C++
        # header, autoware_vision_pilot/VisionPilot/modules/models/include/models/auto_steer.hpp):
        # BGR->RGB, scale to [0,1] (ToTensor), NO ImageNet normalization - unlike every
        # other model in this workspace. `image` must be a BGR numpy array (H,W,3), not PIL.
        self.image_loader = transforms.Compose(
            [
                transforms.Lambda(lambda img: img[:, :, ::-1].copy()),
                transforms.ToTensor(),
            ]
        )

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        if (len(checkpoint_path) > 0):
            self.model = AutoSteerNetwork().load_model(
                version='n', num_classes=4, checkpoint_path=checkpoint_path,
                map_location=self.device)
                print(num_classes)
        else:
            raise ValueError('No path to checkpoint file provided in class initialization')

        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    @torch.no_grad()
    def inference(self, image):
        H, W = image.shape[:2]
        if (W != MODEL_WIDTH or H != MODEL_HEIGHT):
            raise ValueError(
                f'Incorrect input size - input image must have height of {MODEL_HEIGHT}px '
                f'and width of {MODEL_WIDTH}px')

        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)

        # Run model
        xp, h_vector = self.model(image_tensor)
        xp = xp.detach().cpu().numpy().squeeze()        # (64,) normalized [0,1] x per row
        h_vector = h_vector.detach().cpu().numpy().squeeze()  # (64,) confidence per row

        return xp, h_vector
