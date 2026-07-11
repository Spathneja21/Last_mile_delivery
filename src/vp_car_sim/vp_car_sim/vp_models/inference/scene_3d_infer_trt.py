#! /usr/bin/env python3
import torch
import tensorrt as trt
from torchvision import transforms

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Scene3DNetworkInferTRT():
    def __init__(self, engine_path=''):

        if len(engine_path) == 0:
            raise ValueError('No path to TensorRT engine file provided in class initialization')

        # Image loader
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.device = torch.device('cuda')
        print(f'Using {self.device} for inference (TensorRT engine)')

        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.output = torch.empty(out_shape, dtype=torch.float32, device=self.device)

        self.stream = torch.cuda.Stream(device=self.device)
        self._live_tensor = None  # keeps input tensor alive until stream sync

    def launch_async(self, image):
        """Preprocess and launch TRT kernel on self.stream without blocking."""
        width, height = image.size
        if width != 640 or height != 320:
            raise ValueError('Incorrect input size - input image must have height of 320px and width of 640px')

        with torch.inference_mode(), torch.cuda.stream(self.stream):
            image_tensor = self.image_loader(image).unsqueeze(0).to(self.device).contiguous()
            self.context.set_input_shape(self.input_name, tuple(image_tensor.shape))
            self.context.set_tensor_address(self.input_name, image_tensor.data_ptr())
            self.context.set_tensor_address(self.output_name, self.output.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
            self._live_tensor = image_tensor  # prevent GC before stream finishes

    def fetch_result(self):
        """Sync stream and return (H, W, C) float depth array."""
        self.stream.synchronize()
        with torch.inference_mode():
            prediction = self.output.squeeze(0).cpu()
            prediction = prediction.permute(1, 2, 0)
            return prediction.numpy()

    def inference(self, image):
        """Synchronous convenience wrapper around launch_async + fetch_result."""
        self.launch_async(image)
        return self.fetch_result()
