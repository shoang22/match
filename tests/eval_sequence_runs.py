"""The sequence-evaluation runs whose output folders the tests check, and how to load their outputs.

SMOKE and FULL render held-out cameras; FLICKER is the flicker-only run (no held-out cameras, an input camera as
the fixed view, no image-quality scores) on SMOKE's frames.
BACKGROUND1_FLICKER and BACKGROUND1_TEMPORAL are flicker-only runs on the 30 frames of the `background 1` capture,
with per-frame MATCH (#29) and with temporal MATCH in sliding windows (#36).
The runs need the GPU, the pretrained model and the data, so their output tests skip until the run has been
done (run commands in the configs' headers). EVAL_SEQUENCE_SMOKE_DIR / EVAL_SEQUENCE_FULL_DIR /
EVAL_SEQUENCE_FLICKER_DIR (and the other EVAL_SEQUENCE_*_DIR below) point the tests at a run written elsewhere, e.g. with
--gin_bindings "OUT_DIR = 'experiments/match/eval_sequence/<name>'".
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import numpy as np
import pytest

import background1
from tce049 import (FIXED_VIEW_CAMERA, FRAME_IDS_PER_SECOND, FRONTAL_INPUT_CAMERA, HELDOUT_CAMERAS, INPUT_CAMERA_IDS,
                    REPO_ROOT, SEGMENT, SEGMENT_FRAMES)

FPS = 24
PREVIEW_FPS = 10
PIX_FMT = 'yuv420p'
PANEL_W, PANEL_H, LABEL_H = 512, 640, 32
GRID_COLS = 2
VIDEO_W, VIDEO_H = GRID_COLS * PANEL_W, 2 * (LABEL_H + PANEL_H)  # 2x2 grid of labelled panels
IMAGE_METRICS_PLOT = 'plot_image_metrics.png'
FLICKER_PLOTS = ['plot_temporal_metrics.png', 'plot_geometric_jitter.png']
FRAME_SPACING_S = 12 / FRAME_IDS_PER_SECOND  # between consecutive 4TB frames
# Top-level sections of metrics.json. An output folder whose metrics.json lacks one was written by older code.
METRICS_SECTIONS = ['model', 'subject', 'segments', 'frames', 'heldout_cameras', 'fixed_view_camera', 'fixed_view_kind',
                    'input_cameras', 'video', 'output_frames', 'preview', 'per_frame', 'summary', 'temporal', 'jitter',
                    'plots']
IMAGE_QUALITY_SECTIONS = ['per_frame', 'summary']  # only in runs with held-out cameras

# How far temporal MATCH at initialization may score from per-frame MATCH. At initialization the temporal sublayers
# are exact no-ops, so the only differences are GPU nondeterminism and fp16 numerics of running the 3 frames as one
# batch instead of one at a time. Each bound is ~3x the largest noise floor measured on the smoke frames (largest
# absolute difference over frames, cameras and sources) in two ways:
#                         run to run (the same computation   batch size (per-frame MATCH, one frame
#                         twice; largest of 4 pairs)         at a time vs one window of 3)
#   psnr (dB)             0.0076                             0.0070
#   ssim                  0.0069                             0.0028
#   lpips                 5.2e-4                             6.3e-4
#   raw flicker           3.5e-5                             4.1e-5
#   flow-aligned error    2.1e-5                             1.4e-5
#   jitter (mm)           1.1e-3                             1.8e-3
# Temporal MATCH at initialization vs per-frame MATCH measured (two pairs of runs) at most 0.0045 dB, 0.0068,
# 4.0e-4, 8.1e-6, 9.8e-6 and 1.6e-3 mm.
TOLERANCE = {
    'psnr': 0.025,  # dB
    'ssim': 0.02,
    'lpips': 2e-3,
    'raw_flicker': 1.5e-4,
    'flow_aligned_error': 1.5e-4,
    'jitter_mm': 0.005,
}


def panel_labels(camera, kind):
    """The grid's four panels, from the fixed view's camera; kind is 'held-out' or 'input'."""
    return [f'MATCH prediction (cam {camera})', f'Real {kind} (cam {camera})', f'TEMPEH coarse mesh (cam {camera})',
            'Orbit']


PANELS = panel_labels(FIXED_VIEW_CAMERA, 'held-out')
PLOTS = [IMAGE_METRICS_PLOT] + FLICKER_PLOTS


@dataclass(frozen=True)
class EvalRun:
    name: str
    config: str  # repo-relative, as passed on the command line
    out_dir_setting: str  # OUT_DIR as the run was configured (relative to the repo root unless absolute)
    frames: list
    heldout_cameras: list  # none in a flicker-only run
    n_output_frames: int  # source frames 2/15 s apart, so len(frames) * 2/15 s * 24 fps, rounded
    fixed_view_camera: str = FIXED_VIEW_CAMERA
    fixed_view_kind: str = 'heldout'  # 'input' in a flicker-only run
    panels: tuple = tuple(PANELS)
    plots: tuple = tuple(PLOTS)
    # The capture: the defaults are TCE049's SEN_a_good_morrow_to_you_my_boy segment.
    segments: tuple = (SEGMENT,)
    input_cameras: tuple = tuple(INPUT_CAMERA_IDS)  # the same on every frame, in camera-angle slot order
    capture_fps: int = FRAME_IDS_PER_SECOND  # CAPTURE_FPS: frame IDs per second (Ava) or frames per second
    frame_spacing_s: float = FRAME_SPACING_S  # between consecutive source frames
    holds: tuple = (3, 4)  # grid-video frames per source frame: frame_spacing_s * 24 fps, rounded either way
    # Mean TEMPEH jitter per triple, roughly 0.3x to 3x the range measured (a unit slip is 10x or 1000x).
    # TCE049: 1.2 to 5.9 mm per triple (mean 2.86 mm, from verts.npy).
    tempeh_jitter_range_mm: tuple = (0.5, 20.0)

    @property
    def metrics_sections(self):
        if self.heldout_cameras:
            return METRICS_SECTIONS
        return [s for s in METRICS_SECTIONS if s not in IMAGE_QUALITY_SECTIONS]

    @property
    def out_dir(self) -> Path:
        return REPO_ROOT / self.out_dir_setting

    @property
    def duration(self):
        return len(self.frames) * self.frame_spacing_s


SMOKE = EvalRun(
    name='smoke', config='configs/match/eval_sequence_smoketest.gin',
    out_dir_setting=os.environ.get('EVAL_SEQUENCE_SMOKE_DIR', 'experiments/match/eval_sequence/smoketest'),
    frames=SEGMENT_FRAMES[:3], heldout_cameras=[FIXED_VIEW_CAMERA], n_output_frames=10)
FULL = EvalRun(
    name='full', config='configs/match/eval_sequence_SEN_a_good_morrow.gin',
    out_dir_setting=os.environ.get('EVAL_SEQUENCE_FULL_DIR', 'experiments/match/eval_sequence/SEN_a_good_morrow'),
    frames=SEGMENT_FRAMES, heldout_cameras=list(HELDOUT_CAMERAS), n_output_frames=128)
# The smoke run with temporal MATCH at initialization in sliding windows (one clip of the 3 frames).
TEMPORAL_SMOKE = EvalRun(
    name='temporal_smoke', config='configs/match/eval_sequence_smoketest_temporal.gin',
    out_dir_setting=os.environ.get('EVAL_SEQUENCE_TEMPORAL_SMOKE_DIR',
                                   'experiments/match/eval_sequence/smoketest_temporal'),
    frames=SEGMENT_FRAMES[:3], heldout_cameras=[FIXED_VIEW_CAMERA], n_output_frames=10)
# The temporal smoke run on 6 frames, so that several windows of 4 run: sliding (3 windows) and block (2 windows).
# Commands in configs/match/eval_sequence_smoketest_temporal.gin.
TEMPORAL_SLIDING6, TEMPORAL_BLOCK6 = (
    EvalRun(name=f'temporal_{mode}6', config='configs/match/eval_sequence_smoketest_temporal.gin',
            out_dir_setting=os.environ.get(f'EVAL_SEQUENCE_TEMPORAL_{mode.upper()}6_DIR',
                                           f'experiments/match/eval_sequence/smoketest_temporal_{mode}6'),
            frames=SEGMENT_FRAMES[:6], heldout_cameras=[FIXED_VIEW_CAMERA], n_output_frames=19)
    for mode in ['sliding', 'block'])
FLICKER = EvalRun(
    name='flicker', config='configs/match/eval_sequence_flicker_smoketest.gin',
    out_dir_setting=os.environ.get('EVAL_SEQUENCE_FLICKER_DIR', 'experiments/match/eval_sequence/flicker_smoketest'),
    frames=SEGMENT_FRAMES[:3], heldout_cameras=[], n_output_frames=10,
    fixed_view_camera=FRONTAL_INPUT_CAMERA, fixed_view_kind='input',
    panels=tuple(panel_labels(FRONTAL_INPUT_CAMERA, 'input')), plots=tuple(FLICKER_PLOTS))


def background1_run(name, config):
    """A flicker-only run on all 30 frames of `background 1`, where every camera is an input; its output folder is
    experiments/match/eval_sequence/<name> unless EVAL_SEQUENCE_<NAME>_DIR is set."""
    return EvalRun(
        name=name, config=config,
        out_dir_setting=os.environ.get(f'EVAL_SEQUENCE_{name.upper()}_DIR', f'experiments/match/eval_sequence/{name}'),
        frames=background1.FRAMES, heldout_cameras=[], n_output_frames=72,  # 30 frames 0.1 s apart at 24 fps
        fixed_view_camera=background1.FRONTAL_INPUT_CAMERA, fixed_view_kind='input',
        panels=tuple(panel_labels(background1.FRONTAL_INPUT_CAMERA, 'input')), plots=tuple(FLICKER_PLOTS),
        segments=(background1.CAPTURE,), input_cameras=tuple(background1.INPUT_CAMERA_IDS),
        capture_fps=background1.CAPTURE_FPS, frame_spacing_s=1 / background1.CAPTURE_FPS, holds=(2, 3),
        # TEMPEH jitter on `background 1` is 0.52 to 1.44 mm per triple (mean 0.91 mm; #29).
        tempeh_jitter_range_mm=(0.15, 5.0))


# Per-frame MATCH (#29) and temporal MATCH at initialization in sliding windows of 4 (#36).
BACKGROUND1_FLICKER = background1_run('background1_flicker', 'configs/match/eval_sequence_background1_flicker.gin')
BACKGROUND1_TEMPORAL = background1_run('background1_temporal', 'configs/match/eval_sequence_background1_temporal.gin')


def load_metrics(run: EvalRun):
    """The run's metrics.json. Skips the test if the run hasn't been done, or was done by older code."""
    path = run.out_dir / 'metrics.json'
    if not path.exists():
        pytest.skip(f'run the {run.name} config first: {run.config} (no {path})')
    metrics = json.loads(path.read_text())
    missing = [s for s in run.metrics_sections if s not in metrics]
    if missing:
        pytest.skip(f'{path} predates the metrics sections {missing}; rerun {run.config}')
    return metrics


def probe_video(path):
    """(video stream info with frames counted, duration in s)."""
    probe = ffmpeg.probe(str(path), count_frames=None)
    (stream,) = [s for s in probe['streams'] if s['codec_type'] == 'video']
    return stream, float(probe['format']['duration'])


def decode_video(path, width=VIDEO_W, height=VIDEO_H):
    """All frames as (N,H,W,3) float32 in 0-255."""
    out, _ = (ffmpeg.input(str(path)).output('pipe:', format='rawvideo', pix_fmt='rgb24')
              .run(capture_stdout=True, quiet=True))
    return np.frombuffer(out, np.uint8).reshape(-1, height, width, 3).astype(np.float32)


def config_bindings(path):
    """The config's lines without comment lines, so header edits after a run don't count as a change."""
    return [line for line in Path(path).read_text().splitlines() if not line.lstrip().startswith('#')]
