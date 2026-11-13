# MATCH 
Official repository for the paper

**Feed-forward Gaussian Registration for Head Avatar Creation and Editing**, ***CVPR 2026***.

<a href="https://malteprinzler.github.io/" target="_blank">Malte Prinzler</a><sup>1,2</sup>, <a href="https://www.paulogotardo.com/" target="_blank">Paulo Gotardo</a><sup>2</sup>, <a href="https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html/" target="_blank">Siyu Tang</a><sup>1</sup>, <a href="https://sites.google.com/site/bolkartt/" target="_blank">Timo Bolkart</a><sup>2</sup>

<sup>1</sup>ETH Zürich, <sup>2</sup>  
Work partially done while Malte was an intern at Google.

<a href='http://arxiv.org/abs/2603.15811'><img src='https://img.shields.io/badge/arXiv-2603.15811-red'></a> <a href='https://malteprinzler.github.io/projects/match/'><img src='https://img.shields.io/badge/project page-MATCH-Green'></a> <a href='#citation'><img src='https://img.shields.io/badge/cite-blue'></a>

https://github.com/user-attachments/assets/d6d400f4-7d3d-4a86-ad94-2306bb2842a8


TL;DR:  Given calibrated multi-view images, MATCH infers static Gaussian splat textures in 0.5 seconds. The resulting Gaussians are in dense semantic correspondence across subjects and expressions. This allows for various applications such as editing, expression transfer, and fast avatar optimization. 

## ⚡️ Quick start guide
### 🛠️ 1. Create conda environment and install requirements

```bash
# 1. Clone repo
git clone https://github.com/malteprinzler/match/
cd match

# 2. Create conda environment for MATCH:
conda create -f environment.yaml
conda activate match

# 3. Set correct conda environment variables
conda env config vars set CUDA_HOME=<path-to-your-local-cuda12.6-installation>
conda deactivate ; conda activate match

# 4. Install requirements that need more love
pip install --no-build-isolation 'git+https://github.com/mattloper/chumpy@51d5afd92a8ded3637553be8cef41f328a1c863a'
pip install --no-build-isolation 'git+https://github.com/malteprinzler/gsplat_2dgs_sm90.git'
pip install --no-build-isolation 'git+https://github.com/malteprinzler/dqtorch.git'
pip install --no-build-isolation 'git+https://github.com/rahul-goel/fused-ssim/@031f321343102d542efc64f8161f11406d78e9d0'
GCC=gcc-9 CC=gcc-9 CXX=g++-9 pip install --no-build-isolation 'third_party/GEM/submodules/diff-gaussian-rasterization'
pip install --no-build-isolation 'third_party/GEM/submodules/simple-knn'
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
pip install git+https://github.com/MarilynKeller/mesh
pip install opendr==0.78 --no-build-isolation
cd third_party/TEMPEH/modules/liegroups && python setup.py install && cd -
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu126.html
pip install third_party/pyrender
cd third_party/GEM/styleunet/stylegan2_ops/; python setup.py install ; cd -
pip install 'git+https://github.com/MarilynKeller/mesh@e8b04be3aaa1d262a3b7a05ef9e4085b4836b049'
pip install 'mediapipe>=0.10.30' 'protobuf>=5.28,<6'
pip install numpy==2.2.6


# 5. register match as package
pip install -e match


```

### 📥 2. Download assets
Download the assets from https://drive.google.com/drive/folders/1Lq5gvGe6MHHlfSSafKmrieztskAc3-bB?usp=sharing and unzip them under `./assets/`


### 🎬 3. Download example data from Ava256
run `condor/run.py configs/data/download_ava256_quickstart.gin`
this will download a small subset of the Ava-256 dataset to `data/ava-256`

### 🌐 4. Infer coarse mesh
run `condor/run.py configs/tempeh/save_predictions_quickstart.yml`
this will save the coarse mesh and uv renderings to `data/tempeh_predictions/ava-256/quickstart`

### 🚀 5. Inferring MATCH
run `condor/run.py configs/match/save_camera_sweeps_quickstart.gin`
this will infer MATCH and save the results to `experiments/match/quickstart`


## Running MATCH on your own calibrated capture
The `calibrated` dataset type runs both inference stages (TEMPEH coarse mesh, then MATCH camera sweeps) on a synchronized multi-view capture with an OpenCV calibration. It needs no Ava-256 data, no head tracking and no kinematic or FLAME fits. A one-frame run on a real 12-camera capture is in [Calibrated-capture smoke test](#calibrated-capture-smoke-test).

### Capture folder
One capture folder is one subject and one sequence. Folder names may contain spaces.
```
<capture>/
  images/frame{FFFFFF}_cam{CCC}.png   # e.g. frame000000_cam011.png; 8-bit sRGB, alpha is dropped
  opencv_ba.txt                       # the calibration, the single source of truth
```
- Frames and cameras come from the image names. The camera ID is the integer in `cam{CCC}`, and camera `N` is the `N`-th camera line of `opencv_ba.txt`.
- If a frame is missing a camera's image (or mask), the dataset logs a warning and skips that frame/camera pair. With 12 cameras and the 12 pretrained camera angles, that skips the whole frame.
- `opencv_ba.txt` has `#` comment lines and one line of 19 values per camera:
  `fu u0 v0 ar s  k1 k2 p1 p2 k3  qw qx qy qz  tx ty tz  width height`
  - Intrinsics: `K = [[fu, s, u0], [0, fu*ar, v0], [0, 0, 1]]`, OpenCV pixel convention. Nonzero skew `s` raises an error.
  - Distortion: OpenCV's 5-coefficient model. The dataset undistorts images and masks at load time with `K` kept as is, so both models see pinhole cameras.
  - After undistorting, the dataset cuts each image and mask to Ava-256's aspect ratio (width/height 0.651), centered on the head where the image allows: 1024×1024 becomes 667×1024. Both models then use their Ava image sizes.
  - Extrinsics: the quaternion (scalar first) and `t` give `[R | t]`, which maps **world to camera**.
  - Units: the translation is in mm. `scale_factor` (default `1/1000`) converts to meters.
  - `width height` is the native image size. All cameras of a capture must share it.
- The dataset puts the head at the world origin. By default the head center is the point closest to all optical axes. On `background 1` that is (−17, 4, 966) mm, which projects near the mouth and chin. You can override it with `scene_center` (calibration units). `scene_rotation` (3×3, default identity) sets the world axes to MATCH's convention: x toward the subject's left, y down, z toward the back of the head. Identity is right when the calibration's world frame is a roughly front-facing camera (x right, y down, z forward).
- A folder that holds several capture folders (each with its own `opencv_ba.txt`) also works as the root. Subject and sequence IDs default to the capture folder name; set `subject_id` / `sequence_id` to override them (only when the root holds a single capture; with several, the overrides raise an error).

### Mask folder
Both stages need foreground masks: TEMPEH multiplies its input images by them, and MATCH whitens the background with them. The mask folder is separate from the capture folder and holds one mask per image, **with the same file name** (`frame{FFFFFF}_cam{CCC}.png`), in raw (distorted) image space at the native size. Masks may be binary or soft (0–255), single-channel or RGB (the first channel is used). For a root with several captures, use `<mask folder>/<capture folder name>/<image name>`.

Until real masks exist, make interim masks with Sapiens (everything that is not `Background` is foreground). This needs the `sapiens_lite` environment (`source scripts/sapiens/install_environment.sh`) and the bfloat16 Sapiens 1B Goliath segmentation checkpoint at `assets/sapiens/checkpoints/bfloat16/seg/checkpoints/sapiens_1b/` (or pass `--checkpoint`):
```bash
conda activate sapiens_lite
python scripts/sapiens/capture_masks.py "data/captures/background 1" "data/captures/masks_sapiens/background 1" \
  --contact-sheet-frame 0
```
Check `<mask folder>/contact_sheets/frame000000.png` before running the models: hair against foliage and a white shirt against the sky are the usual failure cases.

**Switching to real masks** changes one path in each config (`mask_dir` for TEMPEH, `CAPTURE_MASK_DIR` for MATCH). Because both stages read the masks, **re-run TEMPEH, then MATCH** after the swap. MATCH reads TEMPEH's coarse mesh and UV maps, so running MATCH alone would mix old TEMPEH outputs with new masks.

### Configs
| stage | config | edit |
|---|---|---|
| TEMPEH | `configs/tempeh/save_predictions_calibrated.yml` | `data_dir` (capture folder), `mask_dir`, `out_dir`, `nframes` (`-1` = all), `frame_stride` |
| MATCH | `configs/match/save_camera_sweeps_calibrated.gin` | `CAPTURE_ROOT`, `CAPTURE_MASK_DIR`, `CAPTURE_TEMPEH_DIR` (= TEMPEH's `out_dir` + `/calibrated`), `OUT_DIR`, `NFRAMES`, `CAPTURE_FRAME_STRIDE` |

Use the same capture, masks, frame count and stride in both configs. Run them in order:
```bash
PYTHONPATH=$(pwd) python condor/run.py configs/tempeh/save_predictions_calibrated.yml
PYOPENGL_PLATFORM=egl PYTHONPATH=$(pwd) python condor/run.py configs/match/save_camera_sweeps_calibrated.gin
```
- TEMPEH writes `<out_dir>/calibrated/<subject>/<sequence>/<frame>/{verts.npy, uv_cam<ID>.png, vis.jpg}`. `verts.npy` is in the calibration's frame and units. The UV maps are rendered in the (Ava-aspect) image, as for Ava.
- MATCH writes `<OUT_DIR>/<frame>_input.jpg` (input grid), `<frame>_pred.jpg` (renders of the input views) and `<frame>_sweep.mp4`. As for Ava and NeRSemble, an input view whose UV map is missing gets zeros; the dataset logs a warning for each such view.
- Keep the gin config's path free of spaces: `condor/run.py` does not quote it. Paths inside the configs may contain spaces.

Optional values, not in the shipped configs. Add them to the TEMPEH `--datasets` dict and to the MATCH `dataset_kwargs`:
- `scene_center: [x, y, z]` in calibration units, if the optical-axes estimate is off.
- `scene_rotation: [[...], [...], [...]]`, if the calibration's world frame is not front-facing.
- `subject_id`, `sequence_id`, `scale_factor`.

Framing uses the Ava quickstart's sizes on the Ava-aspect images: TEMPEH's 256×167 input with no head crop and 786×512 UV maps; MATCH's 786×512 image (`CAPTURE_HEIGHT/WIDTH`) and the pretrained 640×512 head crop. For 1024×1024 images, a focal length of about 2400 px and the head about 1 m from the cameras, this makes the head about as large as in Ava. For a rig where the head is much larger or smaller in the image, change the image sizes.
- `CAPTURE_HEAD_CROP_OFFSET_Y = -128` instead of the pretrained −64: on `background 1` with −64, the hair reaches the top edge of the crop in the cameras that look up from below. The crop clamps to the image instead of padding, so a larger offset changes nothing once it reaches the top edge.
- The resize must keep the aspect ratio of the cut images (the dataset raises an error otherwise).

### Debug CLI
Load one sample on the CPU and save a figure of every view: the resized image with the head-crop box and the projected head center, the crop with its mask, and the mask alone. It also logs the camera-to-slot assignment. Its defaults match the MATCH config's framing.
```bash
PYTHONPATH=$(pwd) python match/data/calibrated_capture_dataset_cli.py \
  --root "data/captures/background 1" --mask-dir "data/captures/masks_sapiens/background 1" \
  --out demos/calibrated_sample.png
```
Other options: `--index`, `--height/--width`, `--crop-height/--crop-width/--crop-offset-y/--crop-offset-x`, `--no-crop`, `--scene-center X Y Z`, `--all-cameras` (every camera instead of the 12 pretrained camera-angle slots).

### Opt-in real-capture test
`tests/test_real_capture.py` loads one sample of a real capture with the MATCH config's framing. It checks that all 12 cameras are present with the right shapes and that the head center projects onto the foreground mask in every view. It is skipped unless both variables are set:
```bash
MATCH_REAL_CAPTURE="data/captures/background 1" MATCH_REAL_CAPTURE_MASKS="data/captures/masks_sapiens/background 1" \
  PYTHONPATH=$(pwd) python -m pytest tests/test_real_capture.py
```


## Driving a pretrained MATCH-based GEM avatar with a monocular video
- download the example data from https://drive.google.com/drive/folders/1lbT4Rph91dKs2St0VpJWlNKSqNOthFDe?usp=sharing and place it under `./example_data`
- run `condor/run.py configs/GEM/cross_eval_APP152.yml`. This will create the reenacted videos under `experiments/GEM/avatars/UNION_FINAL_APP152/FINAL_regressor_pcacomponents150/val_predictions`


## Creating a MATCH-based GEM avatar from multi-view videos
We demonstrate how a set of registered Gaussian splat textures of a person predicted by MATCH can be leveraged to create lightweight animatable head avatars. 

- Download the example data by running `condor/run.py configs/data/download_ava256_GEM.gin`
- Infer the coarse mesh initialization on the data: `condor/run.py configs/tempeh/save_predictions_GEM.yml`
- If you haven't done so yet, create a separate `sapiens_lite` conda environment by running `source scripts/sapiens/install_environment.sh`.
- activate the `sapiens_lite` environment
- create the sapiens segmentation masks by running `scripts/sapiens/seg_ava256.sh data/ava-256 data/sapiens_segmentations/ava-256/framestride_1 1 "APP152 PGO261"`
- MATCH prediction: To predict textures of Gaussian splats for a particular subject that can then be converted into an animatable avatar, please run `condor/run.py configs/match/save_GEM_dataset.gin`. By default, this will store the results under `experiments/match/GEM_datasets`
- Then run `condor/run.py configs/GEM/combine_GEM_datasets.gin`. This creates meta-data to combine the predictions on different sequences into one dataset which is stored under `match/GEM_datasets/<subject_id>/UNION_GEM_<subject_id>`.

- Fit FLAME against registrations: `condor/run.py configs/GEM/fit_flame_APP152.yml`

- Compute PCA over the registered Gaussians: `condor/run.py configs/GEM/pca_gauss_APP152.yml`

- Predict expression features for monocular driving for both target and driving subjects: 
    `condor/run.py configs/GEM/predict_face_features_APP152.yml`
    `condor/run.py configs/GEM/predict_face_features_PGO261.yml`

- Launch coefficient regressor training: `condor/run.py configs/GEM/train_avatar_APP152.yml`



## Evaluating pretrained MATCH on a sequence
`scripts/match/eval_sequence.py` runs the pretrained model on every frame of one Ava-256 segment (one forward pass per frame, the 12 standard input cameras) and renders it from held-out cameras that are never inputs. `configs/match/eval_sequence_SEN_a_good_morrow.gin` evaluates the 40 frames of TCE049's `SEN_a_good_morrow_to_you_my_boy` with 3 held-out cameras. It writes a 24 fps grid video (prediction and real image from the held-out camera closest to frontal, the TEMPEH coarse mesh and an orbit around the head), a preview with each source frame shown once, `metrics.json` (masked PSNR/SSIM/LPIPS per frame and held-out camera, plus raw flicker and flow-aligned error per frame pair), per-frame plots and copies of the configs.

- Run the quickstart steps 1 to 4 first, then download the held-out cameras and infer TEMPEH meshes for every frame of the segment. The commands are in steps 1 and 2 of the config's header.
- Smoke check (3 frames, 1 held-out camera, ~3 min). Output goes to `experiments/match/eval_sequence/smoketest`:
    ```bash
    PYTHONPATH=$(pwd) PYOPENGL_PLATFORM=egl torchrun --standalone --nnodes=1 --nproc-per-node=1 \
      scripts/match/eval_sequence.py --gin_configs configs/match/eval_sequence_smoketest.gin
    ```
- Full run (~27 min). Use the same command with `configs/match/eval_sequence_SEN_a_good_morrow.gin`; output goes to `experiments/match/eval_sequence/SEN_a_good_morrow`. It uses the whole 24GB GPU, so nothing else may hold GPU memory. Leave `PYTORCH_CUDA_ALLOC_CONF` unset.
- Check the outputs: `python -m pytest tests/test_eval_sequence_outputs.py tests/test_eval_sequence_smoketest.py`.
- `background 1` with temporal MATCH (flicker-only, sliding windows). This runs all 30 frames of the `background 1` capture through temporal MATCH at initialization, i.e. the pretrained checkpoint with no-op temporal sublayers, in sliding windows of 4 frames. Every camera is an input, so there are no held-out cameras: the fixed view is input camera 0, and `metrics.json` has flicker and jitter scores but no PSNR/SSIM/LPIPS. Its `inference` section records each frame's window start and slot. First make the masks and the TEMPEH predictions for all 30 frames (steps 1 and 2 in the header of `configs/match/eval_sequence_background1_flicker.gin`). Output goes to `experiments/match/eval_sequence/background1_temporal`:
    ```bash
    PYTHONPATH=$(pwd) PYOPENGL_PLATFORM=egl torchrun --standalone --nnodes=1 --nproc-per-node=1 \
      scripts/match/eval_sequence.py --gin_configs configs/match/eval_sequence_background1_temporal.gin
    ```
    For per-frame MATCH, use `configs/match/eval_sequence_background1_flicker.gin` instead (output in `experiments/match/eval_sequence/background1_flicker`). Check both runs with `python -m pytest tests/test_eval_sequence_outputs.py tests/test_eval_sequence_temporal_outputs.py`.

## Training TEMPEH
- Download the entire ava-256 dataset using `condor/run.py configs/data/download_ava256_full.gin`
- Start tempeh training by running `condor/run.py configs/tempeh/train_tempeh.yaml`

## Training MATCH
Coming soon...

### Single-GPU smoke test
`configs/match/train_match_smoketest.gin` runs 50 training steps (with evals at steps 25 and 50) on the quickstart data to check that the training loop works. It is not a real training run. It was tested on a single RTX 4090 (24GB) under WSL2, peaking at ~17GB of GPU memory.

- Run the quickstart steps 3 and 4 first. The smoke test only needs the single frame with TEMPEH predictions under `data/tempeh_predictions/ava-256/quickstart`; Sapiens segmentations and kinematic tracking are not required.
- Run the smoke test:
    ```bash
    PYOPENGL_PLATFORM=egl PYTHONPATH=$(pwd) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      torchrun --standalone --nnodes=1 --nproc-per-node=1 scripts/match/train_match.py \
      --gin_configs configs/match/train_match_smoketest.gin
    ```
    Results are written to `experiments/clean/match_smoketest`.

To fit in 24GB, the config uses 4 input views and a 32x32 grid of UV patches (`uv_patch_size = 16` at `uv_res = 512`). The image-UV patch correspondence step scales with input views x UV patches x pixels, and the full 12-view setup does not fit.

### Temporal MATCH clip-training smoke test
`configs/match/train_temporal_match_smoketest.gin` runs 4 clip steps of temporal MATCH, then one validation pass and one checkpoint, to check that clip training works. It uses the default temporal training config, `configs/match/train_temporal_match.gin` (the pretrained uv_res_1024 weights, clips of 4 frames, 12 input and 12 render views, gradient checkpointing, spatial weights and Sapiens frozen), with different data. It is not a real training run.

- No Ava-256 training subject is on disk, so the smoke test trains on the held-out subject TCE049: its `SEN_a_good_morrow_to_you_my_boy` segment, the only one with TEMPEH predictions at every frame. Create them first with [Evaluating pretrained MATCH on a sequence](#evaluating-pretrained-match-on-a-sequence), step 2. The default config never trains on held-out subjects.
- Run the smoke test:
    ```bash
    PYOPENGL_PLATFORM=egl PYTHONPATH=$(pwd) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      torchrun --standalone --nnodes=1 --nproc-per-node=1 scripts/match/train_match.py \
      --gin_configs configs/match/train_temporal_match_smoketest.gin
    ```
    Results are written to `experiments/match/temporal_match_smoketest`. Delete that folder before rerunning: a run resumes its own checkpoints.

On a single RTX 4090 (24GB) under WSL2, one step on a clip of 4 frames x 12 views takes ~8.3 s and peaks at 16.9 GiB of allocated GPU memory (`torch.cuda.max_memory_allocated`, logged per step as `Resources/GPU/peak_allocated(GB)`), with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Without that setting the caching allocator reserves ~24.9 GiB, more than the card has, and a step takes ~10.2 s. The first step takes ~20 s. The whole smoke test takes a few minutes.

The step fits because, with gradient checkpointing, the frames of a clip share only the temporal sublayers: each block's spatial part, the rendering and the losses are recomputed for backward one frame at a time, LPIPS one image at a time, and the frozen correspondence search and Sapiens encoder run frame by frame. None of this changes what is computed. Under WSL2, a run that needs more than 24GB does not fail with an out-of-memory error. The driver moves the excess into system memory and the run gets 5-10x slower, so check the logged peak, not just that the run finishes.

### Temporal trajectory loss
The option `temporal_loss_weight` (default 0, off) adds a loss on each texel's trajectory over a clip of 3 or more frames: the mean over texels and frame triples of the squared second difference `|x[t+1] - 2 x[t] + x[t-1]|^2` of the texel's final position in mm (TEMPEH part plus offset), plus `temporal_loss_rgb_weight` times the same for its colour. It is added once per clip. Texels are weighted by `geometry_loss_weights_by_part`. Training logs the unweighted terms as `training/temporal_xyz_loss` and `training/temporal_rgb_loss`.

Two configs train with it, both with `temporal_loss_weight = 1e-3` from `configs/match/temporal_loss.gin`:
- `configs/match/train_temporal_match_temporal_loss.gin`: temporal MATCH, i.e. `configs/match/train_temporal_match.gin` plus the loss.
- `configs/match/train_match_clips_temporal_loss.gin`: per-frame MATCH trained on clips, with no temporal sublayers. Every pretrained weight trains, at a learning rate of 4e-6 (`configs/match/per_frame_match_on_clips.gin`).

Each one has a `_smoketest.gin` version that runs on the data of the temporal clip-training smoke test above. The command is the same as above, with a different `--gin_configs`. Results go to `experiments/match/<NAME>`. On TCE049 the pretrained model's `temporal_xyz_loss` is 10-30 mm² per clip, so the loss adds 0.01-0.03 to a clip loss of ~0.77, and the colour term is under 1% of it. A temporal MATCH step takes ~8.4 s and peaks at 14.7 GiB allocated. A per-frame-on-clips step takes ~8.7 s and peaks at 17.4 GiB.

#### Troubleshooting (WSL2)
- `EGL_NOT_INITIALIZED` / `failed to open /dev/dri/renderD128: Permission denied`: the render device is owned by a group your user is not in. Check with `ls -ln /dev/dri`, then either run `sudo chmod 666 /dev/dri/renderD128` (resets when WSL restarts) or add yourself to the owning group permanently, e.g. for gid 992: `sudo groupadd -g 992 render992 && sudo usermod -aG render992 $USER`, followed by `wsl --shutdown`.
- `pyglet.gl.ContextException: Could not create GL context` during evaluation: `PYOPENGL_PLATFORM=egl` is not set.
- `XStartError: Xvfb program closed ... server already running`: an Xvfb from a crashed run still holds display `:99`. Kill it with `pkill -f "[X]vfb -br.*:99"`.

### Calibrated-capture smoke test
A one-frame inference run on a real capture: interim masks, TEMPEH, then MATCH camera sweeps. It checks that the whole [calibrated-capture pipeline](#running-match-on-your-own-calibrated-capture) works on real data; it says little about output quality. It was run on frame 0 of `background 1` (12 cameras, 1024×1024, outdoor) on a single RTX 4090 under WSL2, with the shipped configs and only the paths changed.

1. Masks (skip if you have real ones). In the `sapiens_lite` env, `python scripts/sapiens/capture_masks.py "<capture>" "<mask folder>" --contact-sheet-frame 0`. For one frame this took 24 s.
2. TEMPEH. Set `data_dir`, `mask_dir`, `out_dir` in `configs/tempeh/save_predictions_calibrated.yml` (`nframes: 1`), then:
    ```bash
    PYTHONPATH=$(pwd) python condor/run.py configs/tempeh/save_predictions_calibrated.yml
    ```
    Expect `verts.npy`, `vis.jpg` and 12 files `uv_cam0.png` … `uv_cam11.png` (786×512, framed like the dataset's 667×1024 images) in `<out_dir>/calibrated/background 1/background 1/000000/`. The run took 20 s. On `background 1`:
    - The coarse mesh, shifted by the head center and scaled to meters, has a bounding box of 0.20 × 0.34 × 0.24 m centered within 12 mm of the origin. The 0.34 m height includes the neck; Ava-256's quickstart mesh is 0.22 × 0.33 × 0.27 m.
    - 98–99% of each UV map's covered pixels fall inside the Sapiens foreground mask (undistorted and cut like the images).
3. MATCH. Set `CAPTURE_ROOT`, `CAPTURE_MASK_DIR`, `CAPTURE_TEMPEH_DIR` (TEMPEH's `out_dir` + `/calibrated`) and `OUT_DIR` in `configs/match/save_camera_sweeps_calibrated.gin`, then:
    ```bash
    PYOPENGL_PLATFORM=egl PYTHONPATH=$(pwd) python condor/run.py configs/match/save_camera_sweeps_calibrated.gin
    ```
    Expect `000000_input.jpg`, `000000_pred.jpg` and `000000_sweep.mp4` in `OUT_DIR`, with a single pass over the dataset (`Dataset [calibrated_val]: 1/1`) and no skipped frames or zero-filled UV maps. The run took 96 s end to end, 41 s of it on the one sample (200-frame sweep render included). On `background 1` the renders show the whole head in all 12 views, with a slightly cool color cast and a dark, noisy shirt and foliage at the bottom of the crop.

When the dataset is built (MATCH run, debug CLI), it logs which camera fills each of the pretrained model's 12 camera-angle slots and how far off it is. For `background 1`:
```
INFO:match.data.calibrated_capture_dataset:/path/to/background 1: camera-angle slot assignment
  camera 0 -> slot 0 [az 0, el 0], angular error 1.1 deg
  camera 9 -> slot 1 [az 0, el -15], angular error 2.9 deg
  camera 1 -> slot 2 [az 20, el 0], angular error 0.6 deg
  camera 7 -> slot 3 [az 20, el 36], angular error 25.2 deg
  camera 4 -> slot 4 [az 20, el -15], angular error 14.3 deg
  camera 2 -> slot 5 [az -20, el 0], angular error 3.1 deg
  camera 8 -> slot 6 [az -20, el 36], angular error 32.3 deg
  camera 6 -> slot 7 [az -20, el -15], angular error 6.2 deg
  camera 3 -> slot 8 [az 40, el -5], angular error 11.9 deg
  camera 10 -> slot 9 [az 40, el 25], angular error 59.8 deg
  camera 5 -> slot 10 [az -40, el -5], angular error 14.9 deg
  camera 11 -> slot 11 [az -40, el 25], angular error 68.5 deg
```
This rig is a horizontal arc (azimuth −26° to +33°) plus a vertical column at azimuth ≈ 0° (elevation +18° to −30°). The training rigs have upper side cameras at +25° to +36° elevation, and this rig has nothing there. So the four upper slots (3, 6, 9, 11) go to cameras 25–69° away. Slots are filled in order, each taking the nearest camera still unused, so the last slots get what is left. Expect this to lower quality, mostly on the top of the head and the forehead. It does not break the run. TEMPEH builds the same dataset but does not print the log, as `save_predictions.py` does not configure logging.

The masks in this run covered frame 0 only. That is fine for TEMPEH and MATCH with `nframes: 1`. The debug CLI loads every frame, so with those masks it logs a "missing mask" warning for each of frames 1–29.
<!-- - Make sure you downloaded the entire ava-256 dataset as described [here](#training-tempeh).
- Download the NeRSemble dataset following the instructions from the [official repository](https://github.com/tobias-kirschstein/nersemble-data). For all further steps, we assume that the nersemble dataset root is `data/nersemble/`
-  -->



<!-- 
### Data preprocessing
#### Sapiens segmentation: 
We require sapiens segmentation masks for training. 
- Please follow the instructions <a href='https://github.com/facebookresearch/sapiens/blob/main/lite/README.md#-installation'>here</a> to set up the `sapiens_lite` conda environment.
- Download the pretrained segmentation weights for the bfloat16 version of the `sapiens-seg-1b` model from <a href='https://huggingface.co/facebook/sapiens'>here</a> and place it under `assets/sapiens/checkpoints/bfloat16/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_bfloat16.pt2`
- edit `configs/sapiens/segment_ava256.gin` according to your needs
- run `condor/run.py configs/sapiens/segment_ava256.gin` which will start the semantic segmentation.  -->

## License
MATCH is distributed under the MIT license. However, it builds on the third party software [GEM](https://github.com/Zielon/GEM/blob/main/LICENSE), [pyrender](https://github.com/mmatl/pyrender/blob/master/LICENSE), [sapiens](https://github.com/facebookresearch/sapiens/blob/main/LICENSE), and [TEMPEH](https://github.com/TimoBolkart/TEMPEH/blob/main/LICENSE) for which different licenses may apply. If you use their code, please make sure to obey the respective licenses. 

## 📖 Citation

```tex
@inproceedings{prinzler2026match,
  title={Feed-forward Gaussian Registration for Head Avatar Creation and Editing},
  author={Prinzler, Malte and Gotardo, Paulo and Tang, Siyu and Bolkart, Timo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```


