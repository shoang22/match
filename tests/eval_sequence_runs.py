"""The sequence-evaluation runs whose output folders the tests check, and how to load their outputs.

SMOKE and FULL render held-out cameras; FLICKER is the flicker-only run (no held-out cameras, an input camera as
the fixed view, no image-quality scores) on SMOKE's frames.
The runs need the GPU, the pretrained model and the data, so their output tests skip until the run has been
done (run commands in the configs' headers). EVAL_SEQUENCE_SMOKE_DIR / EVAL_SEQUENCE_FULL_DIR /
EVAL_SEQUENCE_FLICKER_DIR point the tests at a run written elsewhere, e.g. with
--gin_bindings "OUT_DIR = 'experiments/match/eval_sequence/<name>'".
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import numpy as np
import pytest

from tce049 import (FIXED_VIEW_CAMERA, FRAME_IDS_PER_SECOND, FRONTAL_INPUT_CAMERA, HELDOUT_CAMERAS, REPO_ROOT,
                    SEGMENT_FRAMES)

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
        return len(self.frames) * FRAME_SPACING_S


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
