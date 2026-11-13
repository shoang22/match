"""The single-frame novel-view benchmark loop: predict each test frame, render it from its target camera, score it.

Per-frame inference predicts the test frame alone, exactly as the sequence evaluation's per-frame predictor
does. Sliding inference (temporal MATCH) predicts it from its sliding window, planned over the test frame's
run of frames with complete inputs (match/novel_view/samples.py, match/sequence_eval/windows.py), and keeps the
test frame's slot, as the sequence evaluation's windowed predictor does on a sequence. Only the test frame is
loaded with its target camera, and the target view never reaches the model.
"""
import math
from typing import Callable

from match.novel_view.samples import SamplePlan
from match.sequence_eval.predictors import stack_window
from match.sequence_eval.windows import window_plan

MODES = ('per_frame', 'sliding')


def _without_target(batch):
    return type(batch)({k: v for k, v in batch.items() if not k.startswith('extra_')})


def run_benchmark(plan: SamplePlan, load_frame: Callable, forward_frame: Callable, forward_clip: Callable,
                  clip_length: int, mode: str, score: Callable) -> dict:
    """Scores every sample of the plan. Returns the 'inference', 'samples' and 'summary' sections of the output.

    load_frame(subject, frame, target_camera): the frame's batch (B=1) with its input cameras, plus the target
      camera as the 'extra_*' view unless target_camera is None.
    forward_frame(batch): masked Gaussians (1,1,C,L,1). forward_clip(clip): masked Gaussians (1,T,1,C,L,1) of a
      clip batch (see predictors.stack_window).
    clip_length: window length T of sliding inference (at most plan.context_length).
    score(batch, gaussians): {metric: value} of the Gaussians rendered from the batch's target camera.
    """
    if mode not in MODES:
        raise ValueError(f'The novel-view benchmark runs inference mode {MODES}, got {mode!r}')
    if mode == 'sliding' and clip_length > plan.context_length:
        raise ValueError(f'clip_length {clip_length} is longer than the context window of {plan.context_length} '
                         'frames whose inputs the target cameras avoid; plan the samples with a longer context_length')

    rows = []
    for sample in plan.samples:
        batch = load_frame(sample.subject, sample.frame, sample.target_camera)
        row = {'subject': sample.subject, 'sequence': sample.sequence, 'frame': sample.frame,
               'target_camera': sample.target_camera,
               'input_cameras': [str(c) for c in batch['camera'][0].tolist()]}
        if mode == 'per_frame':
            gaussians = forward_frame(batch)
        else:
            slot = window_plan(len(sample.run), clip_length, 'sliding')[sample.index]
            frames = sample.run[slot.start:slot.start + slot.length]
            window = [_without_target(batch) if f == sample.frame else load_frame(sample.subject, f, None)
                      for f in frames]
            gaussians = {k: v[:, slot.slot] for k, v in forward_clip(stack_window(window)).items()}
            row['window'] = {'frames': list(frames), 'slot': slot.slot}
        rows.append({**row, **score(batch, gaussians)})

    inference = {'mode': mode} if mode == 'per_frame' else {'mode': mode, 'clip_length': clip_length}
    return {'inference': inference, 'samples': rows, 'summary': summarize(rows, mode)}


def summarize(rows, mode):
    """Mean of every score over the samples (non-finite values left out), with the inference mode."""
    names = [k for k, v in rows[0].items() if isinstance(v, float)] if rows else []
    summary = {'inference_mode': mode, 'n_samples': len(rows)}
    for name in names:
        finite = [r[name] for r in rows if math.isfinite(r[name])]
        summary[name] = sum(finite) / len(finite) if finite else math.nan
    return summary
