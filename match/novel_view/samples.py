"""Sample plan of the single-frame novel-view benchmark: the test frames, one target camera each, and the frames
a temporal model's window around each test frame may use.

MATCH's protocol: held-out subjects, 12 input views, and a target camera disjoint from the inputs. Here:
- A frame has complete inputs when its input cameras can be picked and it has a TEMPEH prediction (coarse mesh
  and the UV maps of its input cameras). A run is a maximal stretch of consecutive frames with complete inputs
  in one sequence (Ava-256 segment), so a frame without a TEMPEH prediction splits its sequence.
- Test frames are drawn only from runs of at least `context_length` frames, so the sliding window of every
  test frame lies inside its run and never needs a frame without a TEMPEH prediction. A temporal model's window
  (`clip_length` <= `context_length`) is planned over the run, so at a run's edges it shifts inward exactly as at
  a sequence's edges in the sequence evaluation. Frames of shorter runs are not test frames.
- The target camera is drawn from the cameras available for the test frame that are not an input of any frame of
  its sliding context window, so neither model sees the target view.
- The plan depends only on the data, the seed, `n_samples` and `context_length`, never on the checkpoint or
  inference mode, so per-frame and temporal runs score the same frames from the same cameras.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from match.sequence_eval.windows import window_plan


class NoTargetCameraError(RuntimeError):
    """A test frame has no available camera outside the inputs of its context window."""


@dataclass(frozen=True)
class FrameInputs:
    """One frame of a capture's frame list: its input cameras, or None when its inputs are incomplete."""
    frame: str
    sequence: str
    inputs: tuple | None


@dataclass(frozen=True)
class BenchmarkSample:
    """A test frame and its target camera.

    run: the frames of its run (consecutive frames with complete inputs, in order); index: its position in the run.
    """
    subject: str
    sequence: str
    frame: str
    target_camera: str
    run: tuple
    index: int


@dataclass(frozen=True)
class SamplePlan:
    samples: list
    n_eligible: int  # frames that could be test frames
    n_without_inputs: int  # frames without complete inputs (e.g. no TEMPEH prediction)
    n_in_short_runs: int  # frames with complete inputs in runs shorter than the context window


def complete_runs(frames: list[FrameInputs]) -> list[list[FrameInputs]]:
    """Maximal stretches of consecutive frames with complete inputs within one sequence, in frame-list order."""
    runs, current = [], []
    for f in frames:
        if f.inputs is None or (current and current[-1].sequence != f.sequence):
            if current:
                runs.append(current)
            current = []
        if f.inputs is not None:
            current.append(f)
    if current:
        runs.append(current)
    return runs


def plan_samples(captures: dict[str, list[FrameInputs]], target_cameras: Callable[[str, str], list],
                 n_samples: int | None, seed: int, context_length: int) -> SamplePlan:
    """Draws up to n_samples test frames (all eligible frames if None or fewer) and a target camera for each.

    captures: subject -> its frame list. target_cameras(subject, frame): the cameras with data for that frame
    that may be targets. Samples keep the frames' order (subjects in the given order).
    """
    eligible, n_without, n_short = [], 0, 0
    for subject, frame_list in captures.items():
        n_without += sum(f.inputs is None for f in frame_list)
        for run in complete_runs(frame_list):
            if len(run) < context_length:
                n_short += len(run)
                continue
            plan = window_plan(len(run), context_length, 'sliding')
            for index, slot in enumerate(plan):
                context = run[slot.start:slot.start + slot.length]
                eligible.append((subject, run, index, context))

    rng = np.random.default_rng(seed)
    if n_samples is None or n_samples >= len(eligible):
        chosen = range(len(eligible))
    else:
        chosen = sorted(rng.choice(len(eligible), size=n_samples, replace=False).tolist())

    samples = []
    for i in chosen:
        subject, run, index, context = eligible[i]
        frame = run[index]
        seen = {c for f in context for c in f.inputs}
        candidates = sorted(str(c) for c in target_cameras(subject, frame.frame) if str(c) not in seen)
        if not candidates:
            raise NoTargetCameraError(
                f'{subject} frame {frame.frame}: no target camera outside the inputs {sorted(seen)} of its window '
                f'{[f.frame for f in context]}')
        samples.append(BenchmarkSample(
            subject=subject, sequence=frame.sequence, frame=frame.frame,
            target_camera=candidates[int(rng.integers(len(candidates)))],
            run=tuple(f.frame for f in run), index=index))
    return SamplePlan(samples, n_eligible=len(eligible), n_without_inputs=n_without, n_in_short_runs=n_short)
