#!/usr/bin/env python3
"""
Evaluate the trained SceneSeg CNN (models/SceneSeg.pth) on a single image.

This does NOT train a new network — it loads the existing SceneSeg model and runs
inference, reporting how it segments the image into:
    0 = background, 1 = foreground object, 2 = drivable road

Outputs (written next to this script):
    seg_mask.png     colour-coded class map  (bg=grey, foreground=red, road=green)
    overlay.png      mask alpha-blended over the input image
    sidebyside.png   input | overlay

If a ground-truth mask is given (--gt, same HxW, values 0/1/2) it also prints
pixel accuracy and per-class IoU. Without it, it prints the class distribution and a
foreground-detection summary (true "accuracy" needs labelled ground truth).

Usage:
    python eval_sceneseg.py --image ../image.png --ckpt ../models/SceneSeg.pth
    python eval_sceneseg.py --image ../image.png --ckpt ../models/SceneSeg.pth --gt mask.png
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

# SceneSeg fixed input size + class ids
MODEL_W, MODEL_H = 640, 320
BG, FG, ROAD = 0, 1, 2
COLORS = {BG: (90, 90, 90), FG: (220, 40, 40), ROAD: (40, 180, 40)}
NAMES = {BG: 'background', FG: 'foreground/object', ROAD: 'drivable road'}

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
# Reuse the SceneSeg model + wrapper that already live in the workspace package.
sys.path.insert(0, os.path.join(WS, 'src', 'vp_car_sim'))
from vp_car_sim.vp_models.inference.scene_seg_infer import SceneSegNetworkInfer


def colorize(class_map):
    h, w = class_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in COLORS.items():
        out[class_map == cid] = col
    return out


def segment(model, rgb_frame):
    """Run SceneSeg on an RGB numpy frame -> (class_map, overlay_rgb), all at MODEL size."""
    rgb_small = cv2.resize(rgb_frame, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    class_map = np.asarray(model.inference(Image.fromarray(rgb_small)))
    overlay = (0.5 * colorize(class_map) + 0.5 * rgb_small.astype(np.float32)).astype(np.uint8)
    return class_map, overlay


def run_camera(model, source):
    """Live view over an OpenCV capture source (webcam index or video/stream).

    The display loop (capture + blend + show) runs at full camera rate, while SceneSeg
    inference runs on a background thread against the most recent frame. The video stays
    smooth (~camera FPS); the segmentation overlay refreshes whenever inference finishes
    (~1 Hz on CPU). No frame is queued/blocked waiting for the model.
    """
    import threading
    import time

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print(f'ERROR: could not open camera source {source!r}')
        return

    # Reduce capture/decode cost and latency: MJPG, modest resolution, 1-frame buffer.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    state = {'latest_rgb': None, 'class_map': None, 'seg_fps': 0.0, 'run': True}
    lock = threading.Lock()

    def worker():
        t_prev = time.time()
        while state['run']:
            with lock:
                rgb = state['latest_rgb']
            if rgb is None:
                time.sleep(0.005)
                continue
            class_map = segment(model, rgb)[0]
            now = time.time()
            with lock:
                state['class_map'] = class_map
                state['seg_fps'] = 0.8 * state['seg_fps'] + 0.2 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    print('Camera mode: press "q" or ESC to quit.')
    disp_fps, t_prev = 0.0, time.time()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print('Camera read failed / stream ended.')
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with lock:
                state['latest_rgb'] = rgb
                class_map = state['class_map']
                seg_fps = state['seg_fps']

            # Blend the most recent segmentation (upscaled) over the live frame.
            if class_map is not None:
                mask_bgr = cv2.cvtColor(colorize(class_map), cv2.COLOR_RGB2BGR)
                mask_bgr = cv2.resize(mask_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                disp = cv2.addWeighted(mask_bgr, 0.5, frame_bgr, 0.5, 0.0)
                fg_pct = 100.0 * np.count_nonzero(class_map == FG) / class_map.size
            else:
                disp, fg_pct = frame_bgr, 0.0

            now = time.time()
            disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            cv2.putText(disp, f'video {disp_fps:4.1f} FPS  seg {seg_fps:4.1f} FPS  fg:{fg_pct:4.1f}%',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('SceneSeg overlay', disp)

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        state['run'] = False
        th.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=os.path.join(WS, 'image_copy.png'))
    ap.add_argument('--ckpt', default=os.path.join(WS, 'models', 'SceneSeg.pth'))
    ap.add_argument('--gt', default='', help='optional ground-truth mask (0/1/2)')
    ap.add_argument('--outdir', default=HERE)
    ap.add_argument('--camera', nargs='?', const='0', default=None,
                    help='live mode: webcam index (default 0) or a video file / stream URL')
    args = ap.parse_args()

    if args.camera is not None:
        model = SceneSegNetworkInfer(checkpoint_path=args.ckpt)
        print(f'Checkpoint: {args.ckpt}')
        run_camera(model, args.camera)
        return

    print(f'Image     : {args.image}')
    print(f'Checkpoint: {args.ckpt}')

    img = Image.open(args.image).convert('RGB')
    print(f'Input size: {img.size} (resized to {MODEL_W}x{MODEL_H} for the model)')
    img_resized = img.resize((MODEL_W, MODEL_H))

    model = SceneSegNetworkInfer(checkpoint_path=args.ckpt)
    class_map = model.inference(img_resized)          # (320, 640) int
    class_map = np.asarray(class_map)

    # ---- class distribution -------------------------------------------------
    total = class_map.size
    print('\n=== SceneSeg class distribution ===')
    for cid in (BG, FG, ROAD):
        pct = 100.0 * np.count_nonzero(class_map == cid) / total
        print(f'  {NAMES[cid]:18s}: {pct:6.2f} %')

    fg_pct = 100.0 * np.count_nonzero(class_map == FG) / total
    # crude "is the main object detected" proxy: foreground coverage in the central box
    h, w = class_map.shape
    cy0, cy1 = int(0.25 * h), int(0.95 * h)
    cx0, cx1 = int(0.20 * w), int(0.80 * w)
    center = class_map[cy0:cy1, cx0:cx1]
    center_fg = 100.0 * np.count_nonzero(center == FG) / center.size
    print('\n=== Foreground detection ===')
    print(f'  foreground over whole image : {fg_pct:6.2f} %')
    print(f'  foreground in central region: {center_fg:6.2f} %  '
          f'(rough proxy for "object in front" detection)')

    # ---- visual outputs -----------------------------------------------------
    mask_rgb = colorize(class_map)
    base = np.asarray(img_resized).astype(np.float32)
    overlay = (0.5 * mask_rgb + 0.5 * base).astype(np.uint8)

    os.makedirs(args.outdir, exist_ok=True)
    Image.fromarray(mask_rgb).save(os.path.join(args.outdir, 'seg_mask.png'))
    Image.fromarray(overlay).save(os.path.join(args.outdir, 'overlay.png'))
    side = np.concatenate([np.asarray(img_resized), overlay], axis=1)
    Image.fromarray(side).save(os.path.join(args.outdir, 'sidebyside.png'))
    print(f'\nSaved: seg_mask.png, overlay.png, sidebyside.png  ->  {args.outdir}')

    # ---- optional accuracy vs ground truth ----------------------------------
    if args.gt:
        gt = np.asarray(Image.open(args.gt).resize((MODEL_W, MODEL_H), Image.NEAREST))
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        acc = 100.0 * np.mean(gt == class_map)
        print('\n=== Accuracy vs ground truth ===')
        print(f'  pixel accuracy: {acc:6.2f} %')
        for cid in (BG, FG, ROAD):
            inter = np.count_nonzero((class_map == cid) & (gt == cid))
            union = np.count_nonzero((class_map == cid) | (gt == cid))
            iou = 100.0 * inter / union if union else float('nan')
            print(f'  IoU {NAMES[cid]:18s}: {iou:6.2f} %')


if __name__ == '__main__':
    main()
