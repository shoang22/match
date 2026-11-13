"""Window plan of windowed sequence inference: which forward pass of a temporal model produces each output frame.

A temporal checkpoint predicts a clip of up to T frames at once. To get one output per frame of an
n-frame sequence, the model runs on windows of min(T, n) consecutive frames, and each output frame is kept
from exactly one window, at one slot:
- 'sliding': frame t comes from the window starting at t - 1, clamped to [0, n - T], at slot t - start
  (one frame before it, the frame itself and T - 2 after it). Windows shift inward at the sequence edges.
- 'block': non-overlapping windows starting at multiples of T; the last one is shifted inward when n is
  not a multiple of T, and keeps only the frames the previous block did not cover. A stress test for
  flicker at window boundaries.
"""
from dataclasses import dataclass

MODES = ('sliding', 'block')


@dataclass(frozen=True)
class WindowSlot:
    """The window (frames start .. start + length - 1) and slot in it that produce one output frame."""
    start: int
    slot: int
    length: int


def window_plan(n_frames: int, clip_length: int, mode: str) -> list[WindowSlot]:
    """One WindowSlot per frame of an n_frames sequence, in frame order. See the module docstring."""
    if mode not in MODES:
        raise ValueError(f'Unknown windowed inference mode {mode!r}; expected one of {MODES}')
    if n_frames < 1 or clip_length < 1:
        raise ValueError(f'Need at least one frame and a window of at least one frame, got {n_frames} and {clip_length}')
    length = min(clip_length, n_frames)
    plan = []
    for t in range(n_frames):
        start = t - 1 if mode == 'sliding' else t // length * length
        start = max(0, min(start, n_frames - length))
        plan.append(WindowSlot(start=start, slot=t - start, length=length))
    return plan
