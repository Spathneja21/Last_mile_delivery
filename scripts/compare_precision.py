#!/usr/bin/env python3
"""
compare_precision.py
Sanity-check FP16 TensorRT engines against the original FP32 .pth models
(SceneSeg + Scene3D) on a single real frame.

Usage:
  PYTHONPATH="/usr/lib/python3.8/dist-packages:$PYTHONPATH" \
  /data/archit0030/miniforge3/envs/vp_gpu/bin/python3 scripts/compare_precision.py \
      --image "image copy.png"
"""
import argparse
import os
import sys

import numpy as np
import torch
import tensorrt as trt
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(WS, 'src', 'vp_car_sim'))

from vp_car_sim.vp_models.pth_pipeline.inference.scene_seg_infer import SceneSegNetworkInfer
from vp_car_sim.vp_models.pth_pipeline.inference.scene_3d_infer import Scene3DNetworkInfer

MODEL_W, MODEL_H = 640, 320
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

IMAGE_LOADER = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TRTModel:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.output = torch.empty(out_shape, dtype=torch.float32, device='cuda')

    def infer(self, x_gpu):
        x_gpu = x_gpu.contiguous()
        self.context.set_input_shape(self.input_name, tuple(x_gpu.shape))
        self.context.set_tensor_address(self.input_name, x_gpu.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return self.output.clone()


def colorize(class_map):
    colors = {0: (90, 90, 90), 1: (220, 40, 40), 2: (40, 180, 40)}
    h, w = class_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in colors.items():
        out[class_map == cid] = col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=os.path.join(WS, 'image copy.png'))
    ap.add_argument('--seg-ckpt', default=os.path.join(WS, 'models', 'SceneSeg.pth'))
    ap.add_argument('--seg-engine', default=os.path.join(WS, 'models', 'SceneSeg_FP16.engine'))
    ap.add_argument('--depth-ckpt', default=os.path.join(WS, 'models', 'scene3D.pth'))
    ap.add_argument('--depth-engine', default=os.path.join(WS, 'models', 'Scene3D_FP16.engine'))
    ap.add_argument('--outdir', default=HERE)
    args = ap.parse_args()

    img = Image.open(args.image).convert('RGB').resize((MODEL_W, MODEL_H))
    x = IMAGE_LOADER(img).unsqueeze(0).cuda()

    print('=== SceneSeg ===')
    seg_pth = SceneSegNetworkInfer(checkpoint_path=args.seg_ckpt)
    seg_class_map_fp32 = np.asarray(seg_pth.inference(img))

    seg_trt = TRTModel(args.seg_engine)
    with torch.inference_mode():
        seg_logits_fp16 = seg_trt.infer(x)
    seg_class_map_fp16 = torch.max(seg_logits_fp16.squeeze(0).permute(1, 2, 0), dim=2)[1].cpu().numpy()

    agree = 100.0 * np.mean(seg_class_map_fp32 == seg_class_map_fp16)
    print(f'Pixel agreement (FP32 .pth vs FP16 .engine): {agree:.3f} %')
    for cid, name in {0: 'background', 1: 'foreground/object', 2: 'drivable road'}.items():
        inter = np.count_nonzero((seg_class_map_fp32 == cid) & (seg_class_map_fp16 == cid))
        union = np.count_nonzero((seg_class_map_fp32 == cid) | (seg_class_map_fp16 == cid))
        iou = 100.0 * inter / union if union else float('nan')
        print(f'  IoU {name:18s}: {iou:6.2f} %')

    print('\n=== Scene3D ===')
    depth_pth = Scene3DNetworkInfer(checkpoint_path=args.depth_ckpt)
    depth_fp32 = np.asarray(depth_pth.inference(img)).squeeze(-1)

    depth_trt = TRTModel(args.depth_engine)
    with torch.inference_mode():
        depth_fp16_raw = depth_trt.infer(x)
    depth_fp16 = depth_fp16_raw.squeeze(0).squeeze(0).cpu().numpy()

    abs_diff = np.abs(depth_fp32 - depth_fp16)
    print(f'Depth MAE       : {abs_diff.mean():.6f}')
    print(f'Depth max |diff|: {abs_diff.max():.6f}')
    print(f'Depth corr      : {np.corrcoef(depth_fp32.ravel(), depth_fp16.ravel())[0, 1]:.6f}')
    print(f'FP32 depth range: [{depth_fp32.min():.4f}, {depth_fp32.max():.4f}]')
    print(f'FP16 depth range: [{depth_fp16.min():.4f}, {depth_fp16.max():.4f}]')

    os.makedirs(args.outdir, exist_ok=True)
    base = np.asarray(img)

    seg_side = np.concatenate([colorize(seg_class_map_fp32), colorize(seg_class_map_fp16)], axis=1)
    Image.fromarray(seg_side).save(os.path.join(args.outdir, 'compare_sceneseg_fp32_vs_fp16.png'))

    def depth_to_img(d):
        d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
        return (d * 255).astype(np.uint8)

    depth_side = np.concatenate([depth_to_img(depth_fp32), depth_to_img(depth_fp16)], axis=1)
    Image.fromarray(depth_side).save(os.path.join(args.outdir, 'compare_scene3d_fp32_vs_fp16.png'))

    print(f'\nSaved comparison images to {args.outdir}')


if __name__ == '__main__':
    main()
