"""Outputs of the sequence-evaluation smoke run (#8, #9).

Checks the experiment folder written by the smoke config: the grid video's timing, size, panels and
holds, the metrics JSON, and the config copies. Skipped until the smoke run has been done (it needs the
GPU, the pretrained model and the data; run command in the config's header):
  PYTHONPATH=$(pwd) PYOPENGL_PLATFORM=egl conda run -n match --no-capture-output torchrun --standalone \
    --nnodes=1 --nproc-per-node=1 scripts/match/eval_sequence.py --gin_configs configs/match/eval_sequence_smoketest.gin
Then, from the repo root:
  conda run -n match python -m pytest tests/test_eval_sequence_smoketest.py
To check a run written elsewhere (e.g. with --gin_bindings "OUT_DIR = '<dir>'"), set EVAL_SEQUENCE_SMOKE_DIR=<dir>.
"""
import json
import math
import os
from pathlib import Path

import ffmpeg
import numpy as np
import pytest

SMOKE_CONFIG = Path('configs/match/eval_sequence_smoketest.gin')
OUT_DIR = Path(os.environ.get('EVAL_SEQUENCE_SMOKE_DIR', 'experiments/match/eval_sequence/smoketest'))
FRAMES = ['32885', '32897', '32909']  # first 3 frames of SEN_a_good_morrow_to_you_my_boy, 0.4 s apart
HELDOUT_CAMERA = '402040'
INPUT_CAMERA_IDS = ['401875', '401919', '402020', '402012', '401920', '401965',
                    '401630', '401916', '401860', '402038', '401966', '401969']
FPS = 24
# 3 source frames x 0.4 s = 1.2 s = 28.8 output frames at 24 fps, rounded to whole frames.
N_OUTPUT_FRAMES = 29
PANEL_W, PANEL_H, LABEL_H = 512, 640, 32
PANELS = [f'MATCH prediction (cam {HELDOUT_CAMERA})', f'Real held-out (cam {HELDOUT_CAMERA})',
          f'TEMPEH coarse mesh (cam {HELDOUT_CAMERA})']
PREDICTION, REAL, TEMPEH_MESH = range(3)
GRID_COLS = 2  # 3-4 panels in a 2x2 grid, row-major
VIDEO_W, VIDEO_H = GRID_COLS * PANEL_W, 2 * (LABEL_H + PANEL_H)

pytestmark = pytest.mark.skipif(
    not (OUT_DIR / 'metrics.json').exists(),
    reason=f'run the smoke config first: {SMOKE_CONFIG}',
)


@pytest.fixture(scope='module')
def metrics():
    return json.loads((OUT_DIR / 'metrics.json').read_text())


@pytest.fixture(scope='module')
def video_frames():
    out, _ = (ffmpeg.input(str(OUT_DIR / 'grid.mp4'))
              .output('pipe:', format='rawvideo', pix_fmt='rgb24')
              .run(capture_stdout=True, quiet=True))
    return np.frombuffer(out, np.uint8).reshape(-1, VIDEO_H, VIDEO_W, 3).astype(np.float32)


def panel(video_frames, p):
    """(N, LABEL_H + PANEL_H, PANEL_W, 3) slice of panel p, label band included."""
    r, c = divmod(p, GRID_COLS)
    return video_frames[:, r * (LABEL_H + PANEL_H):(r + 1) * (LABEL_H + PANEL_H), c * PANEL_W:(c + 1) * PANEL_W]


def silhouette(image):
    """Foreground of a panel image on a white background."""
    return image.min(axis=-1) < 200


def test_grid_video_is_24_fps_with_the_true_duration_and_size():
    probe = ffmpeg.probe(str(OUT_DIR / 'grid.mp4'), count_frames=None)
    (stream,) = [s for s in probe['streams'] if s['codec_type'] == 'video']
    assert stream['avg_frame_rate'] == '24/1'
    assert (int(stream['width']), int(stream['height'])) == (VIDEO_W, VIDEO_H)
    assert int(stream['nb_read_frames']) == N_OUTPUT_FRAMES
    assert abs(float(probe['format']['duration']) - len(FRAMES) * 0.4) <= 1 / FPS


def test_every_output_frame_records_its_source_frame(metrics):
    output_frames = metrics['output_frames']
    assert len(output_frames) == N_OUTPUT_FRAMES
    assert sorted(set(output_frames), key=int) == FRAMES
    assert output_frames == sorted(output_frames, key=int)
    # each source frame is held for 0.4 s = 9.6 output frames, rounded to 9 or 10
    holds = [output_frames.count(f) for f in FRAMES]
    assert all(h in (9, 10) for h in holds), holds


def test_panels_are_labelled_and_hold_still_between_source_frames(metrics, video_frames):
    assert metrics['video']['panels'] == PANELS
    assert len(video_frames) == N_OUTPUT_FRAMES
    source = metrics['output_frames']
    for p in range(len(PANELS)):
        tile = panel(video_frames, p)
        label_band = tile[:, :LABEL_H]
        assert label_band.max() > 200, f'panel {p} has no label text'
        change = np.abs(np.diff(tile[:, LABEL_H:], axis=0)).mean(axis=(1, 2, 3))  # change into frame k+1
        within_hold = [c for k, c in enumerate(change) if source[k + 1] == source[k]]
        at_new_frame = [c for k, c in enumerate(change) if source[k + 1] != source[k]]
        # Held frames are repeats (codec noise only), never blends; both panels switch at the same output frames.
        assert max(within_hold) < 0.5, (p, max(within_hold))
        assert min(at_new_frame) > 2 * max(within_hold), (p, at_new_frame, max(within_hold))


def test_tempeh_mesh_lines_up_with_the_head_in_the_prediction_and_real_panels(metrics, video_frames):
    source = metrics['output_frames']
    first_of_each_frame = [source.index(f) for f in FRAMES]
    for k in first_of_each_frame:
        mesh = silhouette(panel(video_frames, TEMPEH_MESH)[k, LABEL_H:])
        pred = silhouette(panel(video_frames, PREDICTION)[k, LABEL_H:])
        real = silhouette(panel(video_frames, REAL)[k, LABEL_H:])
        assert 0.05 < mesh.mean() < 0.8, f'output frame {k}: mesh covers {mesh.mean():.2f} of the panel'
        # The coarse mesh has no hair or shoulders, so it sits inside the head rather than matching its outline.
        # Aligned, 99.7% of it lies on the head; shifting it by 20 px drops that to ~97%.
        for name, head in [('prediction', pred), ('real', real)]:
            inside = (mesh & head).sum() / mesh.sum()
            assert inside > 0.98, f'output frame {k}: {inside:.3f} of the mesh lies on the {name} head'


def test_metrics_are_finite_for_every_frame_and_heldout_camera(metrics):
    assert metrics['frames'] == FRAMES
    assert metrics['heldout_cameras'] == [HELDOUT_CAMERA]
    assert metrics['fixed_view_camera'] == HELDOUT_CAMERA
    assert sorted(metrics['per_frame']) == FRAMES
    for frame in FRAMES:
        scores = metrics['per_frame'][frame][HELDOUT_CAMERA]
        assert sorted(scores) == ['lpips', 'psnr', 'ssim']
        assert all(isinstance(v, float) and math.isfinite(v) for v in scores.values()), scores


def test_summary_has_mean_and_worst_frames(metrics):
    summary = metrics['summary'][HELDOUT_CAMERA]
    for name, worst_is_lowest in [('psnr', True), ('ssim', True), ('lpips', False)]:
        values = {f: metrics['per_frame'][f][HELDOUT_CAMERA][name] for f in FRAMES}
        assert summary[name]['mean'] == pytest.approx(sum(values.values()) / len(values))
        worst = summary[name]['worst_frames'][0]
        expected = min(values, key=values.get) if worst_is_lowest else max(values, key=values.get)
        assert worst == {'frame': expected, 'value': values[expected]}


def test_pretrained_model_with_the_12_standard_inputs_on_every_frame(metrics):
    assert metrics['model'] == {'path': 'assets/pretrained_models/match/uv_res_1024', 'checkpoint': '860000'}
    assert metrics['input_cameras'] == {f: INPUT_CAMERA_IDS for f in FRAMES}


def test_gin_configs_are_copied():
    assert (OUT_DIR / 'config_0.gin').read_text() == SMOKE_CONFIG.read_text()
    assert f"OUT_DIR = '{OUT_DIR}'" in (OUT_DIR / 'config_all.gin').read_text()
