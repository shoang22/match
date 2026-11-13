"""Single-frame novel-view benchmark under MATCH's protocol: held-out subjects, 12 input views, and a target camera
disjoint from the inputs, scored with masked PSNR, SSIM and LPIPS (the sequence evaluation's image metrics) and masked
L1 and L2. CSIM is not computed (follow-up #45).

Test frames and target cameras are drawn with a fixed seed from the frames with complete inputs
(match/novel_view/samples.py), independent of the checkpoint, so every model is scored on the same samples.
Per-frame MATCH predicts each test frame alone. Temporal MATCH predicts it from its sliding window (one frame
before, the frame, two after, shifted inward at the ends of the run of frames with TEMPEH predictions) and keeps
that frame's output, as the sequence evaluation runs it on sequences.

Writes `out_dir/metrics.json`: model, protocol, inference (mode, and clip_length for windowed runs), summary
(inference mode, number of samples, mean scores) and samples (per test frame: subject, sequence, frame, target
camera, input cameras, scores, and for windowed runs the window's frames and the test frame's slot), plus copies
of the gin configs. Configs and run commands: configs/match/eval_novel_view*.gin.
"""
import os

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # before anything imports pyrender

import argparse
from pathlib import Path

import gin
import torch

from match import data
from match.novel_view.benchmark import run_benchmark
from match.novel_view.data import frame_inputs, load_frame, target_cameras_on_disk
from match.novel_view.samples import plan_samples
from match.options import Options
from match.runner import MatchRunner
from match.sequence_eval import outputs as seq_outputs
from match.sequence_eval import predictors as seq_predictors
from match.sequence_eval.context import render_views
from match.sequence_eval.metrics import IMAGE_METRICS
from match.utils import metric_util


def load_captures(opt):
    """The held-out captures of the first val_loader_params entry, by subject ID."""
    params = opt.val_loader_params[0]
    dataset = data.get_dataset_cls(params['cls'])(**params['kwargs']['dataset_kwargs'])
    captures = {}
    for capture in dataset.single_capture_datasets.values():
        if capture.sid in captures:
            raise ValueError(f'Two captures of subject {capture.sid}: {captures[capture.sid].dir} and {capture.dir}')
        captures[capture.sid] = capture
    return captures


# The scores of a target view: the sequence evaluation's image metrics, plus masked L1 and L2.
SCORES = {**{name: m.fn for name, m in IMAGE_METRICS.items()}, 'l1': metric_util.masked_l1, 'l2': metric_util.masked_l2}


def target_view_scores(runner, batch, gaussians):
    """The SCORES of the Gaussians rendered from the batch's target camera, as the sequence evaluation scores a
    held-out camera."""
    camera = str(batch['extra_camera'][0, 0].item())
    view = render_views(runner, gaussians, batch['extra_C2W'][0], batch['extra_fxfycxcy'][0],
                        batch['extra_image'][0].float(), batch['extra_mask'][0].float(), [camera])[camera]
    return {name: fn(view.pred, view.real, view.mask) for name, fn in SCORES.items()}


@gin.configurable()
def eval_novel_view(opt: Options, out_dir: str = None, target_camera_ids: list = None, n_samples: int = 1000,
                    seed: int = 0, context_length: int = 4, inference_mode: str = None,
                    store_config_files: list = ()):
    """Scores the configured checkpoint on the benchmark's samples. See the module docstring.

    target_camera_ids: cameras a target may be drawn from (those with data for the test frame and not an input of
      any frame of its context window). n_samples: test frames (all eligible frames when fewer, or None).
    context_length: the sliding window whose inputs the target avoids; test frames need a run of at least this
      many frames with TEMPEH predictions. A temporal checkpoint's clip_length must not exceed it.
    inference_mode: 'per_frame' or 'sliding'; None picks 'sliding' for a temporal checkpoint, else 'per_frame'.
    """
    if not out_dir:
        raise ValueError('eval_novel_view.out_dir must be set in the config')
    if not target_camera_ids:
        raise ValueError('eval_novel_view.target_camera_ids must list the cameras targets may be drawn from')
    mode = seq_predictors.resolve_inference_mode(opt.temporal, inference_mode)

    # everything that can fail on the data fails before the model is loaded
    captures = load_captures(opt)
    targets = {sid: target_cameras_on_disk(c, target_camera_ids) for sid, c in captures.items()}
    plan = plan_samples({sid: frame_inputs(c) for sid, c in captures.items()}, targets, n_samples=n_samples,
                        seed=seed, context_length=context_length)
    if not plan.samples:
        raise ValueError(f'No test frames: {plan.n_without_inputs} frames lack complete inputs (TEMPEH predictions) '
                         f'and {plan.n_in_short_runs} lie in runs shorter than {context_length} frames')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_outputs.copy_configs(out_dir, store_config_files)

    runner = MatchRunner(opt)
    runner.logger.info(f'Scoring {len(plan.samples)} novel-view samples ({mode}) into [{out_dir}]')
    runner.val_loaders = []  # this run loads its own frames; accelerate_prepare only places the model
    runner.accelerate_prepare()
    runner.eval()

    with torch.autocast('cuda', dtype=torch.float16), torch.no_grad():
        result = run_benchmark(
            plan,
            load_frame=lambda sid, frame, target: load_frame(captures[sid], frame, target, device=runner.device),
            forward_frame=seq_predictors.frame_forward(runner), forward_clip=seq_predictors.clip_forward(runner),
            clip_length=opt.clip_length, mode=mode,
            score=lambda batch, gaussians: target_view_scores(runner, batch, gaussians))

    metrics = {
        'model': {'path': opt.load_pretrained_model, 'checkpoint': runner.pretrained_model_iter,
                  'temporal': opt.temporal},
        'protocol': {'subjects': list(captures), 'target_cameras': [str(c) for c in target_camera_ids],
                     'seed': seed, 'n_samples': n_samples, 'context_length': context_length,
                     'n_eligible_frames': plan.n_eligible, 'n_frames_without_inputs': plan.n_without_inputs,
                     'n_frames_in_short_runs': plan.n_in_short_runs},
        **result,
    }
    path = seq_outputs.write_metrics_json(out_dir, metrics)
    runner.logger.info(f'Wrote {path}: {metrics["summary"]}')
    runner.graceful_exit(0)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_configs', action='append', default=[])
    parser.add_argument('--gin_bindings', action='append', default=[])
    FLAGS = parser.parse_args(argv)
    gin.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs, bindings=FLAGS.gin_bindings, skip_unknown=True)
    opt = Options()
    eval_novel_view(opt, store_config_files=FLAGS.gin_configs)


if __name__ == '__main__':
    main()
