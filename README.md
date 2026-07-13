# Last_mile_delivery


## 1. Environment Setup

This pipeline uses one conda environment (`vp_gpu`) plus ROS Noetic on the same Jetson machine. Set these up before running anything (Section 1.5 has the quick-start commands).

### 1.1 Confirmed hardware / software versions

| Item | Value |
|---|---|
| GPU model | NVIDIA Jetson AGX Orin Developer Kit (64 GB) |
| L4T / JetPack version | L4T 35.4.1 / JetPack 5.1.2 |
| CUDA version | 11.4.315 (toolkit/`nvcc`) |
| cuDNN version | 8.6.0.166 |
| TensorRT version | **8.5.2.2** |
| PyTorch version (`vp_gpu` env) | **2.1.0a0+41361538.nv23.06** — NVIDIA's Jetson-optimized build |
| torchvision version (`vp_gpu` env) | **0.16.1** |
| Python version (`vp_gpu` env) | 3.8.20 |
| ROS distro | ROS 1 Noetic |
| `trtexec` build command used for the committed `.engine` files | see Section 8.2 |
| ONNX FP32 model source | Downloaded pre-trained from [AutowareFoundation/vision_pilot](https://github.com/autowarefoundation/vision_pilot) — not trained or exported locally, see Section 8.1 |

### 1.2 Environments involved

| Environment | Path | Python | Purpose |
|---|---|---|---|
| `vp_gpu` (conda/miniforge) | `/data/archit0030/miniforge3/envs/vp_gpu` | 3.8.20 | Runs **both** `run_webcam_publisher.sh` (OpenCV + rospy) and `webcam_navigator_node.py` (CUDA PyTorch + torchvision + (via `PYTHONPATH`) the system TensorRT bindings). |
| System TensorRT (apt package) | `/usr/lib/python3.8/dist-packages` | ABI-matches `vp_gpu`'s 3.8 | TensorRT Python bindings — **not installed inside `vp_gpu`**; reached only via `PYTHONPATH` layering (see §6). |
| ROS 1 Noetic | `/opt/ros/noetic` | system 3.8 | Message transport (`rospy`, `sensor_msgs`, `geometry_msgs`). |
| catkin workspace | `/data/archit0030/catkin_workspace` | — | Provides `devel/lib/python3/dist-packages` on `PYTHONPATH` for any custom ROS message/package builds. |

### 1.3 Setting up `vp_gpu` (CUDA inference environment)

```bash
# 1. JetPack 5.1.2 (L4T 35.4.1) must already be flashed — this provides CUDA 11.4.315,
#    cuDNN 8.6.0.166, and the system TensorRT 8.5.2.2 apt packages system-wide.
dpkg -l | grep tensorrt   # sanity check: should show 8.5.2.2-1+cuda11.4

# 2. Create the conda/miniforge env
conda create -n vp_gpu python=3.8
conda activate vp_gpu

# 3. Install the Jetson-built PyTorch wheel matching JetPack 5.1 (NVIDIA's aarch64 wheel,
#    not the generic PyPI build) plus a matching torchvision, e.g.:
#      torch       == 2.1.0a0+41361538.nv23.06
#      torchvision == 0.16.1
pip install numpy==1.24.4 pillow==10.4.0 onnx==1.17.0 opencv-python-headless

# 4. Do NOT pip install `tensorrt` inside this env — it's intentionally left out 
```
`vp_gpu` is the **only** conda environment this pipeline uses — both `run_webcam_publisher.sh` (camera capture) and `run_webcam_navigator.sh` (inference) run under it; only the `PYTHONPATH` layering each script exports differs (Section 5,6).

> TODO: the exact wheel filenames/URLs used to install `torch`/`torchvision` into `vp_gpu` are not captured anywhere in this repo — record them here next time this env is rebuilt from scratch.

### 1.4 ROS environment

```bash
source /opt/ros/noetic/setup.bash
source /data/archit0030/catkin_workspace/devel/setup.bash   # if custom messages/packages are used
roscore &
```

### 1.5 Quick start (running the pipeline)

```bash
# terminal 1
roscore

# terminal 2 — camera publisher (vp_gpu env)
./run_webcam_publisher.sh

# terminal 3 — navigator (vp_gpu env, CUDA + TensorRT inference)
./run_webcam_navigator.sh
```
Both launcher scripts activate the `vp_gpu` environment/`PYTHONPATH` internally — no manual `conda activate` needed before running them.

---

## 2. Pipeline Overview

```
[/dev/video0]
     │  cv2.VideoCapture
     ▼
run_webcam_publisher.sh  ── publishes ──►  /webcam/image_raw (sensor_msgs/Image, rgb8, 640x480 @30Hz)
                                            │
                                            ▼
                                run_webcam_navigator.sh
                 (env setup: ROS Noetic + vp_gpu venv + TensorRT bindings)
                                            │  launches
                                            ▼
                        webcam_navigator_node.py (rospy node: webcam_navigator_node)
                                            │  image_cb (per frame)
                                            ▼
                    resize 640x480→640x320 (cv2.resize + PIL)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
          SceneSegNetworkInferTRT                          Scene3DNetworkInferTRT
          (scene_seg_infer_trt.py)                         (scene_3d_infer_trt.py)
                    │                                               │
          TensorRT .engine (FP16)                          TensorRT .engine (FP16)
          on its own CUDA stream                            on its own CUDA stream
                    │                                               │
          seg_pred (H,W) class ids                      depth_pred (H,W) relative depth
                    └───────────────┬───────────────────────────────┘
                                    ▼
                    compute_command()  (scene_decision.py)
    9-bin scan → steering bearing + depth-weighted obstacle score → linear/angular velocity + blocked flag
                    
                                    │
                                    ▼
                        geometry_msgs/Twist ── publishes──►  /cmd_vel
                                    │
                                    ├──► logs/pipeline_log.csv   (per-frame command log)
                                    ├──► logs/timing_log.csv     (per-frame stage latency)
                                    └──► /vision_pilot/webcam_navigator/overlay (debug image, if subscribed)
```

---

## 3. File locations

| File | Path | Role |
|---|---|---|
| Publisher launcher | [run_webcam_publisher.sh](run_webcam_publisher.sh) | Starts `webcam_publisher` node — reads `/dev/video0`, publishes `/webcam/image_raw`. |
| Navigator launcher | [run_webcam_navigator.sh](run_webcam_navigator.sh) | Sets up ROS + Python env, launches `webcam_navigator_node.py` with model/topic params. |
| Navigator node | [src/vp_car_sim/vp_car_sim/webcam_navigator_node.py](src/vp_car_sim/vp_car_sim/webcam_navigator_node.py) | Main ROS node: subscribes to camera, runs inference, publishes `/cmd_vel`. |
| SceneSeg TensorRT wrapper | [src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/inference/scene_seg_infer_trt.py](src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/inference/scene_seg_infer_trt.py) | Loads/runs the SceneSeg `.engine`, returns per-pixel class map. |
| Scene3D TensorRT wrapper | [src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/inference/scene_3d_infer_trt.py](src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/inference/scene_3d_infer_trt.py) | Loads/runs the Scene3D `.engine`, returns relative depth map. |
| Decision logic | [src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/decision/scene_decision.py](src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/decision/scene_decision.py) | Pure function: seg + depth → `(linear, angular, info)`; also builds the debug overlay image. |
| Command CSV logger | [src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/decision/csv_logger.py](src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/decision/csv_logger.py) | Appends per-frame bin/velocity/brake-state rows to `logs/pipeline_log.csv`. |
| Precision comparison tool | [scripts/compare_precision.py](scripts/compare_precision.py) | Sanity-checks the FP16 `.engine`s against the FP32 `.pth` models — **requires `.pth` checkpoints this repo doesn't ship** (see §8.3); not runnable out of the box. |
| Timing plot generator | [scripts/plot_timings.py](scripts/plot_timings.py) | Reads `logs/timing_log.csv`, plots per-stage latency. |
| Command plot generator | [scripts/plot_pipeline_log.py](scripts/plot_pipeline_log.py) | Reads `logs/pipeline_log.csv`, plots velocity/brake/bin distributions. |
| Model checkpoints | `models/` (gitignored — download from the given links in Section 8.1) | Only `SceneSeg_FP32.onnx`, `Scene3D_FP32.onnx` actually exist — downloaded pre-trained, not trained/exported locally. |

---

## 4. ROS Topics & Message Types

| Topic | Type | Publisher | Subscriber | Notes |
|---|---|---|---|---|
| `/webcam/image_raw` | `sensor_msgs/Image` | `webcam_publisher` (run_webcam_publisher.sh) | `webcam_navigator_node` | `encoding='rgb8'`, 640×480, ~30 Hz. |
| `/cmd_vel` | `geometry_msgs/Twist` | `webcam_navigator_node` | robot base controller | `linear.x` = forward speed (m/s), `angular.z` = yaw rate (rad/s). |
| `/vision_pilot/webcam_navigator/overlay` | `sensor_msgs/Image` | `webcam_navigator_node` | any debug viewer (e.g. `rqt_image_view`) | `encoding='rgb8'`; only rendered/published when something is actually subscribed (`get_num_connections() > 0`). |

### `sensor_msgs/Image` field usage (both `/webcam/image_raw` and the overlay)
```
header.stamp   — capture/production timestamp (rospy.Time.now() at publish)
height, width  — pixel dimensions
encoding       — 'rgb8' (3 channels, 8-bit, RGB order)
is_bigendian   — 0 (little-endian, standard x86)
step           — width * 3 (bytes per row, no padding)
data           — raw row-major byte buffer (frame.tobytes())
```

### `geometry_msgs/Twist` on `/cmd_vel`
Only two fields are used:
```
linear.x   — forward velocity, m/s, always ≥ 0 (this pipeline never commands reverse)
angular.z  — yaw rate, rad/s, positive = turn left (CCW), per ROS REP-103 convention
```

---

## 5. `run_webcam_publisher.sh` — camera capture node

[run_webcam_publisher.sh](run_webcam_publisher.sh) is a self-contained inline Python script (via `python3 -c "..."`) rather than a separate `.py` file. It:

1. Exports a `PYTHONPATH` prioritizing the `vp_gpu` conda/miniforge environment's site-packages (for `cv2`, `numpy`) ahead of ROS's own `dist-packages`.
2. Runs under `/data/archit0030/miniforge3/envs/vp_gpu/bin/python3` — the same environment as the navigator (§6), even though this node only needs OpenCV + rospy and none of `vp_gpu`'s CUDA/TensorRT/PyTorch packages.
3. `rospy.init_node('webcam_publisher', anonymous=True)` — node name `webcam_publisher`, `anonymous=True` appends a random suffix so multiple instances could coexist without name collision.
4. `cv2.VideoCapture(0)` opens `/dev/video0`; explicitly sets capture resolution to 640×480 via `CAP_PROP_FRAME_WIDTH/HEIGHT`.
5. Loop at `rospy.Rate(30)` (30 Hz target): reads a frame, converts BGR→RGB (`cv2.cvtColor`, since OpenCV captures in BGR but `rgb8` encoding is declared), builds a `sensor_msgs/Image`, and publishes.
6. `queue_size=1` on the publisher — no benefit to buffering multiple unsent frames; a slow subscriber should just get the newest one.

This node does **no** processing beyond a color-space swap — it is a pure camera→ROS bridge.

---

## 6. `run_webcam_navigator.sh` — environment setup + launch

[run_webcam_navigator.sh](run_webcam_navigator.sh) is the critical piece wiring together three separate, normally-incompatible environments so a single Python process can use all of them at once:

```bash
VP_GPU_SITE="/data/archit0030/miniforge3/envs/vp_gpu/lib/python3.8/site-packages"
VP_GPU_PY="/data/archit0030/miniforge3/envs/vp_gpu/bin/python3"
```
The node actually **runs under** the `vp_gpu` conda/miniforge environment's Python 3.8 interpreter — this is the environment with CUDA-enabled PyTorch (`torch 2.1.0`, per the script's own comment) and `torchvision 0.16.1`.

### PYTHONPATH layering (order matters — first match wins for any given module name)
```bash
export PYTHONPATH="\
$VP_GPU_SITE:\                                    # 1. numpy, CUDA torch, torchvision, PIL, cv2
$SRC_DIR:\                                         # 2. vp_car_sim package itself (src/vp_car_sim)
/usr/lib/python3.8/dist-packages:\                 # 3. system TensorRT 8.5.2 Python bindings
/usr/lib/python3/dist-packages:\                   # 4. system dist-packages (rospkg, catkin_pkg)
/opt/ros/noetic/lib/python3/dist-packages:\         # 5. rospy, sensor_msgs, geometry_msgs, genpy
$PYTHONPATH"
```
The key subtlety (documented in the script's own comment): **TensorRT's Python bindings are not installed inside the `vp_gpu` environment at all** — they live in the system's `/usr/lib/python3.8/dist-packages` (installed there because TensorRT typically ships as a system/apt package tied to a specific CUDA+driver install, i.e. `libnvinfer`). The `vp_gpu` env's Python 3.8 is ABI-compatible with that system TensorRT build, so `import tensorrt` resolves to the system path (priority 3) while `import torch` resolves to the `vp_gpu` path (priority 1) — one interpreter, two different provenances of native extension modules, glued together purely via `PYTHONPATH` ordering.

### Model + topic configuration
```bash
SEG_ENGINE_PATH="$MODELS_DIR/SceneSeg_FP16.engine"
DEPTH_ENGINE_PATH="$MODELS_DIR/Scene3D_FP16.engine"
CSV_LOG_PATH="$WS_DIR/logs/pipeline_log.csv"
IMAGE_TOPIC="/webcam/image_raw"
CMD_VEL_TOPIC="/cmd_vel"
OVERLAY_TOPIC="/vision_pilot/webcam_navigator/overlay"
```
These become ROS private params passed on the command line as `_name:=value` (ROS 1's remapping/param-override syntax), read inside the node via `rospy.get_param('~name', default)`.

### Launch
```bash
exec "$VP_GPU_PY" "$NODE" \
    _seg_engine_path:="$SEG_ENGINE_PATH" \
    _depth_engine_path:="$DEPTH_ENGINE_PATH" \
    _image_topic:="$IMAGE_TOPIC" \
    _cmd_vel_topic:="$CMD_VEL_TOPIC" \
    _overlay_topic:="$OVERLAY_TOPIC" \
    _csv_log_path:="$CSV_LOG_PATH"
```
`exec` replaces the shell process with the Python process (no lingering wrapper shell), running `webcam_navigator_node.py` directly (not via `rosrun`), which is why explicit `_param:=value` args are used instead of a `.launch` file.

**Precondition:** `roscore` and `run_webcam_publisher.sh` must both already be running — this script only starts the navigator/inference side.

---

## 7. `webcam_navigator_node.py` — the ROS node

[webcam_navigator_node.py](src/vp_car_sim/vp_car_sim/webcam_navigator_node.py)

### 7.1 Pure-Python `cv_bridge` replacement (lines 40-61)
The standard ROS `cv_bridge` package is a C++ extension pinned to a specific NumPy ABI; rather than fight a NumPy 2.x incompatibility, this file hand-rolls the two conversions it needs:
- `imgmsg_to_rgb8(msg)`: `np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)`, with a channel-reversal if `encoding == 'bgr8'`.
- `rgb8_to_imgmsg(arr, header)`: the inverse — builds a `sensor_msgs/Image` from a NumPy array (used for the overlay publish).

### 7.2 `__init__` — node bring-up
- `rospy.init_node('webcam_navigator_node', anonymous=False)` — fixed name, only one instance expected.
- Reads `seg_engine_path` / `depth_engine_path` params; **fatal-exits** (`rospy.logfatal` + `sys.exit(1)`) if either is empty — the node refuses to start without both TensorRT engines.
- Reads the full decision-logic parameter dict `self.p` (steering gain, ROI fraction, obstacle/road weights, braking thresholds, bin count, camera field of view) — all with defaults, all overridable via ROS params.
- Instantiates `SceneSegNetworkInferTRT(engine_path=seg_engine_path)` and `Scene3DNetworkInferTRT(engine_path=depth_engine_path)` — this is where the `.engine` files are deserialized and GPU memory/CUDA streams are allocated (see §9).
- Sets up two CSV logs: `CommandLogger` (per-frame decision output) and a raw `csv.writer` for per-stage timing.
- Publishers: `cmd_pub` (`/cmd_vel`, `Twist`, `queue_size=10`), `overlay_pub` (overlay topic, `Image`, `queue_size=1`).
- Subscribers: `image_topic` → `image_cb` (`queue_size=1`, `buff_size=2**24` — 16 MB buffer, since a 640×480×3 raw image is ~921 KB and would overflow ROS's default 64 KB transport buffer).

### 7.3 `image_cb` — the per-frame pipeline (the core loop)
Runs once per `/webcam/image_raw` message:

1. **Preprocess** (`t0→t1`): `imgmsg_to_rgb8(msg)` decodes the raw bytes to a `(480, 640, 3)` NumPy array; `cv2.resize(cv_image, (640, 320))` squashes it to `MODEL_WIDTH×MODEL_HEIGHT = 640×320` (the fixed TensorRT input shape); wrapped in a PIL `Image` for `torchvision.transforms` compatibility.
2. **Dual async GPU launch**: `self.seg_model.launch_async(pil)` then `self.depth_model.launch_async(pil)` — both enqueue their preprocessing + TensorRT kernel execution on their own dedicated CUDA stream and return immediately (non-blocking). See §9 for the CUDA mechanics.
3. **Sync + fetch** (`t1→t3`): `seg_pred = self.seg_model.fetch_result()` blocks until the seg stream finishes (`t2`); `depth_pred = self.depth_model.fetch_result().squeeze(-1)` blocks until the depth stream finishes (`t3`). Because both kernels were launched before either sync point, their GPU execution overlaps — see §8/§9 for the measured savings.
4. **Decision** (`t3→t4`): `compute_command(seg_pred, depth_pred, self.p)` → `(v, w, info)` (full breakdown in §10). Logged via `CommandLogger.log(...)`.
5. **Publish** (`t4→t5`): builds and publishes the `Twist` on `/cmd_vel`. Also computes `e2e_ms = (rospy.Time.now() - msg.header.stamp) * 1000` — true glass-to-glass latency from camera capture to command publish, as distinct from `total_ms` (in-callback compute time only).
6. **Logging**: writes one row to `logs/timing_log.csv` (`frame, pre_ms, seg_ms, depth_ms, post_ms, pub_ms, total_ms, e2e_ms`) and one `loginfo` line to the console per frame.
7. **Overlay** (conditional): only rendered/published if `overlay_pub.get_num_connections() > 0` — `overlay_image(seg_pred, depth_pred, info, self.p)` builds a colorized debug frame (road=green, obstacles=red intensity-scaled by proximity, chosen bin highlighted yellow, BRAKE/CLEAR banner), converted back to a `sensor_msgs/Image` and published.

---

## 8. Model checkpoint chain: downloaded ONNX FP32 → `.engine`

**This pipeline does not train or export these models.** No `.pth` checkpoint is trained, held, or converted locally — `SceneSeg_FP32.onnx` and `Scene3D_FP32.onnx` are downloaded directly as pre-trained artifacts, and the only step actually performed on this machine is the FP32 → FP16 TensorRT engine build.

Evidence from `models/` (file listing, `.gitignore`d from this repo due to size):
```
SceneSeg_FP32.onnx     # downloaded pre-trained ONNX FP32 model (SceneSeg)
SceneSeg_FP16.engine   # TensorRT engine built locally from the ONNX graph, FP16 precision  ← used by webcam_navigator_node.py

Scene3D_FP32.onnx      # downloaded pre-trained ONNX FP32 model (Scene3D)
Scene3D_FP16.engine    # ← used by webcam_navigator_node.py
```
No `.pth` files exist in `models/` — `SceneSeg.pth` / `scene3D.pth` are only referenced as unused default CLI args in `scripts/compare_precision.py` (carried over from the upstream reference implementation, §8.3) and are not present.

### 8.1 Model source (downloaded, not trained)
`SceneSeg_FP32.onnx` and `Scene3D_FP32.onnx` are downloaded pre-trained model files, sourced from the [AutowareFoundation/vision_pilot](https://github.com/autowarefoundation/vision_pilot) project:

| Model | Task | Source |
|---|---|---|
| `SceneSeg_FP32.onnx` | Scene segmentation | [Google Drive link](https://drive.google.com/file/d/1l-dniunvYyFKvLD7k16Png3AsVTuMl9f/view?usp=drive_link) |
| `Scene3D_FP32.onnx` | Scene3D relative depth | [Google Drive link](https://drive.google.com/file/d/19gMPt_1z4eujo4jm5XKuH-8eafh-wJC6/view?usp=drive_link) |

Both links are the pre-exported FP32 ONNX weights published by `vision_pilot` — no training, fine-tuning, or `torch.onnx.export(...)` step happens in this repo. Download each file into `models/` under the names above before running §8.2's engine build.

### 8.2 TensorRT engine build (`.engine`) — the only step actually performed locally
`SceneSeg_FP16.engine` / `Scene3D_FP16.engine` are built from the downloaded ONNX files using NVIDIA's `trtexec` CLI tool (ships with the TensorRT install):
```bash
trtexec --onnx=models/SceneSeg_FP32.onnx  --saveEngine=models/SceneSeg_FP16.engine  --fp16
trtexec --onnx=models/Scene3D_FP32.onnx   --saveEngine=models/Scene3D_FP16.engine   --fp16
```
`--fp16` instructs TensorRT to build the engine using half-precision (16-bit float) kernels wherever numerically safe, roughly halving both compute time and memory bandwidth versus FP32, at a small accuracy cost. **The exact `trtexec` invocation used to build the currently-committed `.engine` files (workspace size, optimization profile shapes, etc.) is not captured anywhere in this repo** — treat the above as the standard/expected invocation shape, not a verified transcript. File timestamps show both `.onnx` downloads and both engine builds happened within the same session (`.onnx` ~12:49–12:50, `.engine` ~13:20–13:48).

A TensorRT engine is **hardware- and version-locked**: it's compiled against the specific GPU architecture, TensorRT version, and CUDA version present at build time. An engine built on one machine/GPU generally will not load (or will silently underperform/error) on a different GPU architecture or TensorRT version — this is why `.engine` files are typically rebuilt per-deployment-target rather than shared like ordinary model weights.

> GPU model, CUDA version, and TensorRT version for the machine these `.engine` files run on are recorded in §1 (this machine's local `trtexec` also reports v8.5.2, matching). The exact build flags are still not captured — see the TODO row in §1.1.

### 8.3 Unused reference code inherited from `vision_pilot`
`model_components/` ([scene_seg_network.py](src/vp_car_sim/vp_car_sim/vp_models/model_components/scene_seg_network.py), [backbone.py](src/vp_car_sim/vp_car_sim/vp_models/model_components/backbone.py), etc.), `scene_seg_infer.py`, and `scripts/compare_precision.py`'s `.pth` code paths are carried over from the upstream `vision_pilot` reference implementation (`Backbone` — `efficientnet_b0` — feeding `SceneContext → SceneNeck → SceneSegHead`). They describe the architecture the ONNX/engine files were originally trained with, but since no `.pth` weights are present in this repo, none of this code is actually runnable here — it's documentation-by-reference only, not a live code path.

### Why the navigator uses the `.engine`, not the `.onnx`
`scene_seg_infer_trt.py` / `scene_3d_infer_trt.py` (§9) never import `model_components/` or ONNX Runtime at all — they load the `.engine` directly via `tensorrt.Runtime.deserialize_cuda_engine()`. All architecture + weights + FP16 kernel selection are already baked into that binary; the Python wrapper's only remaining job is preprocessing, binding I/O memory addresses, and launching/syncing the precompiled kernels. This is the fastest of the two representations for repeated real-time inference, which is why it's the one wired into `run_webcam_navigator.sh`.

---

## 9. CUDA processing methodology (`scene_seg_infer_trt.py` / `scene_3d_infer_trt.py`)

Both wrapper classes (identical structure, different engine) follow the same three-phase design: **one-time setup**, **async launch**, **blocking fetch**.

### 9.1 One-time setup (`__init__`)
```python
with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    self.engine = runtime.deserialize_cuda_engine(f.read())
self.context = self.engine.create_execution_context()
```
- Reads the `.engine` binary and deserializes it into a live `ICudaEngine` — the compiled inference graph, already resident for execution once an execution context is created.
- `create_execution_context()`: allocates a stateful object that actually runs inference — holds current bound tensor addresses/shapes. One context is created and reused every frame (no per-frame allocation).

```python
self.input_name = self.engine.get_tensor_name(0)
self.output_name = self.engine.get_tensor_name(1)
out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
self.output = torch.empty(out_shape, dtype=torch.float32, device=self.device)
```
- Queries the engine's I/O tensor names/shapes once.
- **Pre-allocates** the output GPU buffer once (`torch.empty(..., device='cuda')`) and reuses it every frame — avoids repeated CUDA memory allocation overhead at inference-rate frequency.

```python
self.stream = torch.cuda.Stream(device=self.device)
self._live_tensor = None
```
- Each model gets its **own dedicated CUDA stream**. A CUDA stream is an ordered queue of GPU operations; operations on *different* streams can execute concurrently on the GPU (hardware permitting). This is the mechanism that lets SceneSeg and Scene3D run **in parallel** rather than one blocking the other.
- `_live_tensor`: a reference-keeping slot preventing Python's garbage collector from freeing the input tensor's GPU memory while an async kernel might still be reading it via a raw pointer.

### 9.2 Async launch (`launch_async`, per frame)
```python
with torch.inference_mode(), torch.cuda.stream(self.stream):
    image_tensor = self.image_loader(image).unsqueeze(0).to(self.device).contiguous()
    self.context.set_input_shape(self.input_name, tuple(image_tensor.shape))
    self.context.set_tensor_address(self.input_name, image_tensor.data_ptr())
    self.context.set_tensor_address(self.output_name, self.output.data_ptr())
    self.context.execute_async_v3(self.stream.cuda_stream)
    self._live_tensor = image_tensor
```
- Input size is validated (must be exactly 640×320 — the engine's fixed shape) before anything else.
- `torch.inference_mode()`: disables autograd tracking (no gradients needed for pure inference).
- `torch.cuda.stream(self.stream)`: every GPU op inside this block — including the host→device copy from `.to(self.device)` — is enqueued on this model's own stream instead of PyTorch's default stream.
- Preprocessing: `image_loader` applies `ToTensor()` + ImageNet `Normalize(mean, std)` (see the separate normalization discussion — must match training-time preprocessing exactly); `.unsqueeze(0)` adds the batch dim → `(1, 3, 320, 640)`; `.contiguous()` guarantees a packed memory layout, required since TensorRT binds raw pointers directly (no copy).
- `set_tensor_address(...)`: binds the engine's input/output slots directly to GPU memory addresses — TensorRT reads/writes those exact locations with **zero additional copying**.
- `execute_async_v3(stream_handle)`: enqueues the compiled kernel(s) on the given stream and **returns immediately** — this is what makes the call "async"; the CPU thread is not blocked waiting for GPU completion.

### 9.3 Concurrency in practice (`webcam_navigator_node.py`, `image_cb`)
```python
self.seg_model.launch_async(pil)     # enqueue on stream A, returns immediately
self.depth_model.launch_async(pil)   # enqueue on stream B, returns immediately
seg_pred = self.seg_model.fetch_result()      # blocks until stream A done
depth_pred = self.depth_model.fetch_result()  # blocks until stream B done (often already finished)
```
Because both launches happen before either `fetch_result()` blocks, the GPU executes both models' kernels **concurrently** rather than sequentially. Measured from this repo's own `logs/timing_log.csv` (1929 frames): `seg_ms` averaged **65.25 ms**, `depth_ms` (the *residual* wait after seg already finished) averaged **46.29 ms** — implying depth's own kernel takes roughly the full ~111 ms span but is almost entirely hidden behind seg's execution. A naive sequential implementation (`seg.inference()` then `depth.inference()`, fully blocking each) would cost roughly `65 + 111 ≈ 176 ms`; the concurrent version measured **~111.5 ms** (`seg_ms + depth_ms`) — an estimated **~35–40% latency reduction** from stream-level concurrency alone.

### 9.4 Blocking fetch (`fetch_result`, per frame)
```python
def fetch_result(self):
    self.stream.synchronize()
    with torch.inference_mode():
        prediction = self.output.squeeze(0).cpu()
        prediction = prediction.permute(1, 2, 0)
        ...
```
- `self.stream.synchronize()`: blocks the CPU thread until every op queued on this stream (preprocessing copy + kernel execution) has actually completed on the GPU — this is the true sync point; nothing before this call guarantees `self.output` contains valid results.
- `.squeeze(0)`: drops the batch dimension.
- `.cpu()`: copies the result from GPU to host (CPU) memory — required since the rest of the pipeline (`compute_command`, NumPy-based) runs on CPU.
- `.permute(1, 2, 0)`: reorders `(C, H, W)` → `(H, W, C)` (channel-last, standard image-array convention).

**SceneSeg-specific finish** (`scene_seg_infer_trt.py`):
```python
_, output = torch.max(prediction, dim=2)
return output.numpy()
```
`torch.max(..., dim=2)` collapses the per-pixel class-logit channel into a single predicted class ID (argmax over classes) → `(H, W)` int array. This is `seg_pred`, consumed by `compute_command()`.

**Scene3D-specific finish** (`scene_3d_infer_trt.py`):
```python
return prediction.numpy()   # (H, W, 1) — single-channel relative depth
```
Returns the raw `(H, W, 1)` float array unchanged (no argmax needed — depth is a continuous regression output, not a class prediction). `webcam_navigator_node.py` immediately applies `.squeeze(-1)` to drop that trailing channel-of-1 dimension, giving a clean `(H, W)` array matching `seg_pred`'s shape (needed for element-wise/boolean-mask alignment in `compute_command`).

### 9.5 Preprocessing normalization (both models)
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
Standard ImageNet channel statistics — required because the shared `Backbone` (`efficientnet_b0`, ImageNet-pretrained) was trained/fine-tuned expecting inputs normalized this exact way. Any deviation here silently degrades model output without erroring.

---

## 10. Decision logic (`compute_command`, `scene_decision.py`)

Given `seg_pred (H,W)` class-id map and `depth_pred (H,W)` relative-depth map (higher = nearer):

1. **ROI crop**: keeps only the bottom `1 - roi_top_frac` of the frame (default bottom 60%) — ignores sky/horizon, focuses on the drivable area directly ahead.
2. **Per-frame depth normalization**: `norm_depth = (roi_depth - d_min) / (d_max - d_min)` within the ROI — depth is *relative*, not metric, so there's no fixed scale; normalization is recomputed every frame.
3. **Column binning**: ROI split into `num_bins` (default 9) vertical strips.
4. **Per-bin features**: `fg_frac` (obstacle pixel fraction), `road_frac` (road pixel fraction), `fg_proximity` (nearest obstacle's depth in that bin), `bin_bearing` (angular offset from image center, computed via `camera_hfov_deg / 2`).
5. **Scoring**: `score = road_weight·road_frac − obstacle_weight·fg_frac·(0.5 + 0.5·fg_proximity)` — rewards road visibility, penalizes obstacles up to 2× harder the closer they are.
6. **Bin selection**: `best = argmax(score)`; `desired_bearing = bin_bearing[best]`.
7. **Steering**: `angular = clamp(kp_steer · desired_bearing, ±max_angular_speed)`.
8. **Braking**: central third of bins checked for `area_blocked` (obstacle coverage ≥ `blocked_stop_fraction`) or `depth_blocked` (proximity ≥ `brake_depth_threshold`); either triggers `blocked=True`.
9. **Speed**: `block_amount = clamp(max(center_fg/blocked_stop_fraction, center_proximity/brake_depth_threshold), 0, 1)`; `linear = max_linear_speed · (1 − block_amount)` — smooth deceleration as an obstacle is approached, forced to exactly `0` once `blocked` or the required turn exceeds `rotate_in_place_angle` (rotate-in-place first).

Returns `(linear, angular, info)` — `info` carries all per-bin arrays plus `blocked`/`area_blocked`/`depth_blocked` for logging and overlay rendering.

---

## 11. Logging outputs

| File | Written by | Columns / content |
|---|---|---|
| `logs/pipeline_log.csv` | `CommandLogger` ([csv_logger.py](src/vp_car_sim/vp_car_sim/vp_models/trt_pipeline/decision/csv_logger.py)) | `bin_number, linear_velocity, angular_velocity, command` (`command` = `BRAKE`/`CLEAR`) — one row per frame. |
| `logs/timing_log.csv` | `webcam_navigator_node.py` directly | `frame, pre_ms, seg_ms, depth_ms, post_ms, pub_ms, total_ms, e2e_ms` — one row per frame. |
| `logs/pipeline_log_plots.png` | manually run [scripts/plot_pipeline_log.py](scripts/plot_pipeline_log.py) (not auto-invoked) | Velocity/brake/bin-distribution plots from `pipeline_log.csv`. |
| (timing plot) | manually run [scripts/plot_timings.py](scripts/plot_timings.py) (not auto-invoked) | Per-stage latency plots from `timing_log.csv`. |

Neither plotting script is called automatically by the node or its launcher — they're standalone post-hoc analysis tools.

---

## 12. Husky Robot Simulation (Gazebo classic, optional)

A ROS 1 catkin checkout of Clearpath's Husky stack lives at `Husky/` (gitignored — a separate nested git clone, not tracked by this repo). It's used to spawn a Husky model in **classic Gazebo** and observe/drive it before wiring in the real vision pipeline.

- **Source:** [Tinker-Twins/Husky](https://github.com/Tinker-Twins/Husky) (a ROS 1 mirror/port of Clearpath's original `husky` meta-repo).
- **Packages:** `husky_msgs`, `husky_description` (URDF), `husky_control` (twist_mux + `diff_drive_controller` via `ros_control`, EKF localization), `husky_gazebo`/`husky_simulator` (classic Gazebo sim), `husky_navigation` (`move_base` demos), `husky_viz` (RViz configs), `husky_desktop` (teleop tools).
- **ROS 1 Noetic** — directly compatible with the rest of this repo's ROS 1 graph. No bridge needed.

### 12.1 One-time setup (verified working on this machine)

The packages aren't in the catkin workspace by default — symlink them in and build:
```bash
cd /data/archit0030/catkin_workspace/src
for pkg in husky_msgs husky_description husky_control husky_gazebo husky_navigation husky_viz husky_desktop husky_simulator; do
    ln -s /data/archit0030/last_mile_delivery/Last_mile_delivery/Husky/$pkg $pkg
done

cd /data/archit0030/catkin_workspace
catkin_make
```
All required ROS dependencies (`gazebo_ros`, `controller_manager`, `diff_drive_controller`, `joint_state_controller`, `robot_localization`, `twist_mux`, `interactive_marker_twist_server`, `joy`, `teleop_twist_joy`) are **already installed** on this machine — `catkin_make` completes with no missing-dependency errors. No `husky_base` (real-hardware serial/CAN driver) is included in this checkout — this stack is for simulation only; see the earlier discussion on what's needed to drive a physical Husky.

> **Gotcha (verified):** run `catkin_make`/`roslaunch`/`xacro` commands with the `vp_gpu` (or any) conda environment **deactivated** (`conda deactivate`). This user's shell activates a conda base env by default, and ROS Noetic's Python tools (`xacro`, etc.) resolve `/usr/bin/env python3` to that conda Python, which lacks `rospkg` — causing `xacro` to fail with `No module named 'rospkg'` and the whole launch to abort. The system Python3 (`/usr/bin/python3`) has `rospkg` installed correctly; conda's does not.

### 12.2 Launching the empty world

```bash
conda deactivate            # see the gotcha above
source /opt/ros/noetic/setup.bash
source /data/archit0030/catkin_workspace/devel/setup.bash

roslaunch husky_gazebo husky_empty_world.launch
```
Verified (dry-run, `roslaunch --files` / `--nodes`) to resolve the full include chain and produce this node graph without errors:
```
/gazebo, /gazebo_gui                      — classic Gazebo server + GUI, empty world
/spawn_husky_model                        — spawns the Husky URDF at the origin
/robot_state_publisher                    — publishes TF from joint states
/base_controller_spawner                  — loads husky_joint_publisher + husky_velocity_controller
/twist_mux                                — merges joystick / interactive-marker / external cmd_vel inputs
/twist_marker_server                      — RViz interactive-marker teleop
/ekf_localization                         — robot_localization, fuses odom + IMU
/joy_teleop/joy_node, /joy_teleop/teleop_twist_joy   — joystick teleop (idles harmlessly with no joystick attached)
```
This actually starts the Gazebo GUI, so run it interactively (not headless/backgrounded) to watch the robot spawn.

### 12.3 Driving it / observing behavior

To manually drive the spawned Husky and see how it responds (no physical joystick needed):
```bash
rosrun husky_control teleop_keyboard.py
```
This is a keyboard teleop script bundled directly in `husky_control/scripts/` (not the separate `teleop_twist_keyboard` package) — it publishes `Twist` messages that flow through `twist_mux` → `husky_velocity_controller/cmd_vel` → the simulated differential-drive controller, the same path any `/cmd_vel` publisher (including a real joystick, or `rostopic pub /cmd_vel ...`) would take.

`twist_mux`'s config (`husky_control/config/twist_mux.yaml`) already defines an `external` input on topic `cmd_vel` (priority 1, lowest — so teleop/joystick can still override it) — which is exactly the topic `webcam_navigator_node.py` publishes to (§4, §7). Topic names line up with no remapping needed if you point your ROS graph at this simulation instead of a real robot.

**Caveat:** this empty-world launch does not wire the simulated Husky's camera to the vision pipeline — there's no camera sensor/topic remap between this Husky URDF and `webcam_navigator_node.py` in either repo. Use §12.2/§12.3 to observe the robot and drivetrain/control behavior standalone (via teleop); closed-loop "vision drives the simulated Husky" testing would need a camera plugin added to the URDF and its topic wired to `IMAGE_TOPIC` in `run_webcam_navigator.sh` (§6) — not currently set up.
