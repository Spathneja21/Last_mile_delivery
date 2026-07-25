#! /usr/bin/env python3
# Inference wrapper for the trained Scene3DNetwork: loads model weights from
# a checkpoint, preprocesses an input camera image, and runs a forward pass
# to produce a dense depth/3D-scene prediction as a numpy array.
import torch
from torchvision import transforms

from ..model_components.scene_seg_network import SceneSegNetwork
from ..model_components.scene_3d_network import Scene3DNetwork


class Scene3DNetworkInfer():
    def __init__(self, checkpoint_path = ''):

        # Image loader
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet channel mean/std used to normalize input images
            ]
        )

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        # Instantiate model, load to device and set to evaluation mode
        sceneSegNetwork = SceneSegNetwork()  # only used to supply the backbone structure; its weights are overwritten by the checkpoint below
        self.model = Scene3DNetwork(sceneSegNetwork)

        if(len(checkpoint_path) > 0):
            self.model.load_state_dict(torch.load \
                (checkpoint_path, weights_only=True, map_location=self.device))
        else:
            raise ValueError('No path to checkpiont file provided in class initialization')

        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def inference(self, image):

        width, height = image.size
        if(width != 640 or height != 320):  # fixed input resolution the network was trained/architected for
            raise ValueError('Incorrect input size - input image must have height of 320px and width of 640px')

        image_tensor = self.image_loader(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Run model
        prediction = self.model(image_tensor)

        # Get output, find max class probability and convert to numpy array
        prediction = prediction.squeeze(0).cpu().detach()
        prediction = prediction.permute(1, 2, 0)
        output = prediction.numpy()

        return output
