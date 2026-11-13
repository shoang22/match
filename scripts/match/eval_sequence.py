"""Sequence evaluation of MATCH: an independent forward pass of the pretrained model on every frame of an
Ava-256 segment, rendered from held-out cameras that are never inputs.

Writes to `out_dir`: a 24 fps grid video of labelled panels (each source frame held for its true
duration), a faster preview of the grid (each source frame shown once, at preview_fps), a metrics JSON
keyed by source frame ID and held-out camera, per-frame metric plots, and copies of the gin configs.
Configs and run command: configs/match/eval_sequence_*.gin.

Extension points (one place each):
- how each frame's Gaussian texture is predicted: `build_predictor` (predictors in match/sequence_eval/predictors.py)
- grid panels: `build_panels` (panel classes in match/sequence_eval/panels.py)
- metrics: `build_metric_collectors` (collectors in match/sequence_eval/metrics.py)
- outputs: `write_outputs` (helpers in match/sequence_eval/outputs.py)
"""
import os

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # before anything imports pyrender

import argparse
from pathlib import Path

import gin
import torch
import tqdm

from match import data
from match.options import Options
from match.runner import MatchRunner
from match.sequence_eval import metrics as seq_metrics
from match.sequence_eval import outputs as seq_outputs
from match.sequence_eval import panels as seq_panels
from match.sequence_eval import plots as seq_plots
from match.sequence_eval import predictors as seq_predictors
from match.sequence_eval.context import SequencePlan, build_frame_context
from match.sequence_eval.frames import (FrameLoadError, SequenceFrames, check_heldout_cameras, closest_to_frontal,
                                        frame_ids, single_capture)
from match.utils import data_util


def build_panels(plan, opt) -> list[seq_panels.Panel]:
    """The grid's panels, in layout order (row-major)."""
    return [
        seq_panels.PredictionPanel(plan.fixed_view_camera),
        seq_panels.RealPanel(plan.fixed_view_camera),
        seq_panels.TempehMeshPanel(plan.fixed_view_camera, opt.template_mesh_path),
        seq_panels.OrbitPanel(plan.n_output_frames),  # loops from the config
    ]


def build_metric_collectors() -> list[seq_metrics.MetricCollector]:
    # RAFT for the temporal metrics runs on the CPU (~2.5 s per pair): the model still holds the GPU.
    return [seq_metrics.ImageMetrics(), seq_metrics.TemporalMetrics(device='cpu'), seq_metrics.GeometricJitter()]


def load_sequence(opt, heldout_camera_ids):
    """Builds the dataset from the first val_loader_params entry and checks the held-out cameras on disk.

    Raises HeldOutCameraError before the model is loaded if a held-out camera is incomplete.
    """
    params = opt.val_loader_params[0]
    dataset = data.get_dataset_cls(params['cls'])(**params['kwargs']['dataset_kwargs'])
    capture = single_capture(dataset)
    check_heldout_cameras(capture, heldout_camera_ids, frame_ids(capture))
    return SequenceFrames(capture, heldout_camera_ids)


def build_predictor(runner) -> seq_predictors.FramePredictor:
    """How each frame's Gaussian texture is predicted: one forward pass per frame."""
    model = runner.get_unwrapped_model()
    return seq_predictors.PerFramePredictor(
        forward=lambda batch: model.mask_gaussians(
            runner.model(batch, func_name='forward_gaussians', return_tempeh_xyz=True)))


def load_batches(runner, sequence):
    """Yields the batch (B=1, on the runner's device) of every frame of the sequence, in order."""
    num_workers = runner.opt.val_loader_params[0]['kwargs']['dataloader_kwargs'].get('num_workers', 0)
    loader = torch.utils.data.DataLoader(sequence, batch_size=1, shuffle=False, num_workers=num_workers)
    for i, sample in enumerate(loader):
        batch = data_util.MatchBatch(sample)
        batch.to_device(runner.device)
        frame = batch['frame'][0]
        if frame != sequence.frame_ids[i]:
            raise FrameLoadError(f'Expected frame {sequence.frame_ids[i]} at position {i}, got {frame}')
        yield batch


def predicted_in_order(frames, frame_ids):
    """Passes the predictor's frames through, raising FrameLoadError unless they are exactly the sequence, in order."""
    n = 0
    for frame in frames:
        emitted = frame.batch['frame'][0]
        if n == len(frame_ids) or frame.frame_index != n or emitted != frame_ids[n]:
            expected = f'frame {frame_ids[n]} at position {n}' if n < len(frame_ids) else 'no more frames'
            raise FrameLoadError(f'The predictor emitted frame {emitted} at position {frame.frame_index}, '
                                 f'expected {expected}')
        yield frame
        n += 1
    if n != len(frame_ids):
        raise FrameLoadError(f'The predictor stopped after {n} of {len(frame_ids)} frames')


def run_frames(runner, plan, predictor, panels, collectors, video, preview):
    """Metrics and grid frames for every frame the predictor emits, in frame order. Returns the input camera IDs per frame.

    The predictor reads the frames' batches and may read several before it emits one. The preview gets the
    first grid frame of every source frame, so it shows each source frame exactly once.
    """
    sequence = plan.sequence
    input_cameras = {}
    with torch.autocast('cuda', dtype=torch.float16), torch.no_grad():
        frames = predicted_in_order(predictor.predict(load_batches(runner, sequence)), sequence.frame_ids)
        for predicted in tqdm.tqdm(frames, desc='Frames', total=len(sequence)):
            predicted = predicted.to(runner.device)
            i, batch = predicted.frame_index, predicted.batch
            input_cameras[batch['frame'][0]] = [str(c) for c in batch['camera'][0].tolist()]

            ctx = build_frame_context(runner, batch, predicted.gaussians, plan, frame_index=i)
            for collector in collectors:
                collector.add_frame(ctx)
            for k, grid in seq_panels.render_grid_frames(panels, ctx):
                video.write(k, grid)
                if k == ctx.output_frame_indices[0]:
                    preview.write(i, grid)
            del ctx, predicted, batch
    return input_cameras


def write_outputs(out_dir, video, preview, metrics):
    """Finishes the files of the run. Configs are copied at the start of the run."""
    video.finish()
    preview.finish()
    metrics['plots'] = seq_plots.write_metric_plots(out_dir, metrics)
    seq_outputs.write_metrics_json(out_dir, metrics)


@gin.configurable()
def eval_sequence(opt: Options, out_dir: str = None, heldout_camera_ids: list = None, fps: int = 24,
                  capture_fps: int = 90, preview_fps: int = 10, store_config_files: list = ()):
    """Evaluates the pretrained model on every frame of the configured segment. See the module docstring."""
    if not out_dir or not heldout_camera_ids:
        raise ValueError('eval_sequence.out_dir and eval_sequence.heldout_camera_ids must be set in the config')
    heldout_camera_ids = [str(c) for c in heldout_camera_ids]
    sequence = load_sequence(opt, heldout_camera_ids)  # fails before anything is loaded or written
    capture = sequence.capture
    plan = SequencePlan(sequence, schedule=seq_outputs.output_frame_schedule(sequence.frame_ids, fps, capture_fps),
                        fixed_view_camera=closest_to_frontal(capture, heldout_camera_ids))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_outputs.copy_configs(out_dir, store_config_files)

    runner = MatchRunner(opt)
    runner.logger.info(f'Evaluating {len(sequence)} frames of {capture.dir} into [{out_dir}]')
    runner.val_loaders = []  # this run loads its own frames; accelerate_prepare only places the model
    runner.accelerate_prepare()
    runner.eval()

    panels = build_panels(plan, opt)
    collectors = build_metric_collectors()
    video = seq_outputs.GridVideoWriter(out_dir, n_frames=plan.n_output_frames, fps=fps)
    preview = seq_outputs.GridVideoWriter(out_dir, n_frames=len(sequence), fps=preview_fps,
                                          name=seq_outputs.PREVIEW_VIDEO)
    predictor = build_predictor(runner)
    input_cameras = run_frames(runner, plan, predictor, panels, collectors, video, preview)

    metrics = {
        'model': {'path': opt.load_pretrained_model, 'checkpoint': runner.pretrained_model_iter},
        'subject': capture.sid,
        'segments': sorted(set(capture.framelist.seg_id.tolist())),
        'frames': sequence.frame_ids,
        'heldout_cameras': heldout_camera_ids,
        'fixed_view_camera': plan.fixed_view_camera,
        'input_cameras': input_cameras,
        'video': {'file': seq_outputs.GRID_VIDEO, 'fps': fps, 'capture_fps': capture_fps,
                  'n_frames': plan.n_output_frames, 'panels': [p.label for p in panels]},
        'output_frames': [sequence.frame_ids[s] for s in plan.schedule],
        'preview': {'file': seq_outputs.PREVIEW_VIDEO, 'fps': preview_fps, 'n_frames': len(sequence),
                    'output_frames': list(sequence.frame_ids)},
    }
    for source, sections in [(predictor, predictor.metrics())] + [(c, c.finish()) for c in collectors]:
        clash = set(sections) & set(metrics)
        if clash:
            raise KeyError(f'{type(source).__name__} returned metrics sections that already exist: {clash}')
        metrics.update(sections)
    write_outputs(out_dir, video, preview, metrics)
    runner.logger.info(f'Wrote {out_dir / seq_outputs.GRID_VIDEO}, {out_dir / seq_outputs.PREVIEW_VIDEO}, '
                       f'{out_dir / seq_outputs.METRICS_JSON} and plots {metrics["plots"]}')
    runner.graceful_exit(0)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_configs', action='append', default=[])
    parser.add_argument('--gin_bindings', action='append', default=[])
    FLAGS = parser.parse_args(argv)
    gin.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs, bindings=FLAGS.gin_bindings, skip_unknown=True)
    opt = Options()
    eval_sequence(opt, store_config_files=FLAGS.gin_configs)


if __name__ == '__main__':
    main()
