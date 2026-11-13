"""Output contract of the single-frame novel-view benchmark, that per-frame checkpoints score there exactly as the
sequence evaluation scores a held-out camera, and that temporal MATCH at initialization scores like per-frame MATCH.

Runs checked (each test skips until its runs have been done; commands in configs/match/eval_novel_view*.gin):
- PER_FRAME / TEMPORAL: the few-sample check (configs/match/eval_novel_view_smoketest.gin, 6 TCE049 test frames),
  per-frame MATCH and temporal MATCH at initialization in sliding windows (NOVEL_VIEW_SMOKE_DIR,
  NOVEL_VIEW_SMOKE_TEMPORAL_DIR),
- FULL_PER_FRAME / FULL_TEMPORAL: the same on all 40 eligible TCE049 frames (configs/match/eval_novel_view.gin;
  NOVEL_VIEW_FULL_DIR, NOVEL_VIEW_FULL_TEMPORAL_DIR),
- the full sequence-evaluation run (eval_sequence_runs.FULL, EVAL_SEQUENCE_FULL_DIR): per-frame MATCH on every
  frame of the segment at every held-out camera, the reference for "scores as before".
"""
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

# TOLERANCE: ~3x the run-to-run and batch-size noise floors measured for the sequence evaluation (#35).
from eval_sequence_runs import FULL, TOLERANCE, load_metrics as load_sequence_metrics
from tce049 import HELDOUT_CAMERAS, INPUT_CAMERA_IDS, REPO_ROOT, SEGMENT, SEGMENT_FRAMES

SECTIONS = ['model', 'protocol', 'inference', 'summary', 'samples']
IMAGE_SCORES = ['psnr', 'ssim', 'lpips']  # the sequence evaluation's image metrics
ERROR_SCORES = ['l1', 'l2']  # masked mean absolute and squared error
SCORES = IMAGE_SCORES + ERROR_SCORES
# Temporal MATCH at initialization vs per-frame MATCH, relative: TOLERANCE['psnr'] = 0.025 dB is a 0.58% change of the
# squared error, and the absolute error moves less than that.
RELATIVE_TOLERANCE = {'l1': 0.006, 'l2': 0.006}
CLIP_LENGTH = 4


@dataclass(frozen=True)
class BenchmarkRun:
    name: str
    out_dir_setting: str
    mode: str
    n_samples: int

    @property
    def out_dir(self):
        return REPO_ROOT / self.out_dir_setting


def run(name, env, default, mode, n_samples):
    return BenchmarkRun(name, os.environ.get(env, default), mode, n_samples)


PER_FRAME = run('per_frame', 'NOVEL_VIEW_SMOKE_DIR', 'experiments/match/eval_novel_view/smoketest', 'per_frame', 6)
TEMPORAL = run('temporal', 'NOVEL_VIEW_SMOKE_TEMPORAL_DIR', 'experiments/match/eval_novel_view/smoketest_temporal',
               'sliding', 6)
FULL_PER_FRAME = run('full_per_frame', 'NOVEL_VIEW_FULL_DIR', 'experiments/match/eval_novel_view/per_frame',
                     'per_frame', 40)
FULL_TEMPORAL = run('full_temporal', 'NOVEL_VIEW_FULL_TEMPORAL_DIR', 'experiments/match/eval_novel_view/temporal',
                    'sliding', 40)
RUNS = [PER_FRAME, TEMPORAL, FULL_PER_FRAME, FULL_TEMPORAL]
PAIRS = [(PER_FRAME, TEMPORAL), (FULL_PER_FRAME, FULL_TEMPORAL)]


def load(run: BenchmarkRun):
    path = run.out_dir / 'metrics.json'
    if not path.exists():
        pytest.skip(f'run the {run.name} benchmark first (no {path})')
    metrics = json.loads(path.read_text())
    missing = [s for s in SECTIONS if s not in metrics]
    if missing:
        pytest.skip(f'{path} predates the sections {missing}')
    missing = [s for s in SCORES if s not in metrics['summary']]
    if missing:
        pytest.skip(f'{path} predates the scores {missing}')
    return metrics


def keys(metrics):
    return [(s['subject'], s['frame'], s['target_camera']) for s in metrics['samples']]


by_name = pytest.mark.parametrize('run', RUNS, ids=lambda r: r.name)
by_pair = pytest.mark.parametrize('per_frame, temporal', PAIRS, ids=lambda r: r.name)


@by_name
def test_the_output_reports_the_inference_mode_alongside_the_scores(run):
    metrics = load(run)
    summary = metrics['summary']

    assert metrics['inference']['mode'] == summary['inference_mode'] == run.mode
    assert summary['n_samples'] == len(metrics['samples']) == run.n_samples
    for name in SCORES:
        assert summary[name] == pytest.approx(sum(s[name] for s in metrics['samples']) / run.n_samples)


@by_name
def test_l2_is_the_squared_error_that_psnr_reports_in_db_and_l1_lies_below_its_root(run):
    for s in load(run)['samples']:
        assert s['psnr'] == pytest.approx(-10 * math.log10(s['l2']), abs=1e-4), s['frame']
        assert 0 < s['l1'] <= math.sqrt(s['l2']), s['frame']  # mean |e| <= sqrt(mean e^2)


@by_name
def test_test_frames_are_tce049_frames_with_the_12_inputs_and_a_target_camera_that_is_no_input(run):
    samples = load(run)['samples']

    assert {(s['subject'], s['sequence']) for s in samples} == {('TCE049', SEGMENT)}
    assert all(s['frame'] in SEGMENT_FRAMES for s in samples)
    assert all(s['input_cameras'] == INPUT_CAMERA_IDS for s in samples)
    assert all(s['target_camera'] in HELDOUT_CAMERAS for s in samples)


@by_name
def test_a_per_frame_run_has_no_windows_and_a_temporal_run_keeps_each_frame_from_its_sliding_window(run):
    metrics = load(run)

    if run.mode == 'per_frame':
        assert metrics['inference'] == {'mode': 'per_frame'}
        assert all('window' not in s for s in metrics['samples'])
        return
    assert metrics['inference'] == {'mode': 'sliding', 'clip_length': CLIP_LENGTH}
    assert metrics['model']['temporal'] is True
    for s in metrics['samples']:
        t = SEGMENT_FRAMES.index(s['frame'])
        start = min(max(t - 1, 0), len(SEGMENT_FRAMES) - CLIP_LENGTH)  # one before, shifted inward at the ends
        assert s['window'] == {'frames': SEGMENT_FRAMES[start:start + CLIP_LENGTH], 'slot': t - start}, s['frame']


@by_pair
def test_per_frame_and_temporal_runs_score_the_same_samples(per_frame, temporal):
    assert keys(load(per_frame)) == keys(load(temporal))


@by_pair
def test_temporal_match_at_initialization_scores_the_same_as_per_frame_match(per_frame, temporal):
    reference, windowed = load(per_frame), load(temporal)

    for a, b in zip(reference['samples'], windowed['samples']):
        for name in IMAGE_SCORES:
            assert b[name] == pytest.approx(a[name], abs=TOLERANCE[name]), (a['frame'], a['target_camera'], name)
        for name in ERROR_SCORES:
            assert b[name] == pytest.approx(a[name], rel=RELATIVE_TOLERANCE[name]), (a['frame'], a['target_camera'], name)


@pytest.mark.parametrize('run', [PER_FRAME, FULL_PER_FRAME], ids=lambda r: r.name)
def test_a_per_frame_checkpoint_scores_as_the_sequence_evaluation_scores_that_frame_and_held_out_camera(run):
    benchmark = load(run)
    sequence = load_sequence_metrics(FULL)['per_frame']

    for s in benchmark['samples']:
        reference = sequence[s['frame']][s['target_camera']]
        for name in IMAGE_SCORES:
            assert s[name] == pytest.approx(reference[name], abs=TOLERANCE[name]), (s['frame'], s['target_camera'], name)
