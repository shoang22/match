"""Frame predictors: how the sequence evaluation gets each frame's Gaussian texture.

A predictor reads the sequence's per-frame batches in order and emits one PredictedFrame per frame, also
in order. It may read several batches before it emits a frame (a texture filter needs the whole
sequence, a temporal model needs a window of frames around the one it emits); the evaluation scores and
renders each frame as soon as it is emitted. The default is PerFramePredictor: one forward pass per frame.
"""
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Protocol

import torch

from match.sequence_eval import texture_filter
from match.sequence_eval.windows import window_slot

INFERENCE_MODES = ('per_frame', 'sliding', 'block')


def resolve_inference_mode(temporal: bool, inference_mode: str | None) -> str:
    """The inference mode to run: the one asked for, or by default 'sliding' for a temporal checkpoint and
    'per_frame' otherwise."""
    mode = inference_mode or ('sliding' if temporal else 'per_frame')
    if mode not in INFERENCE_MODES:
        raise ValueError(f'Unknown inference mode {mode!r}; expected one of {INFERENCE_MODES}')
    return mode


def frame_forward(runner) -> Callable[[dict], dict]:
    """A frame's batch (B=1, on the device) -> its masked Gaussians (1,1,C,L,1), TEMPEH part included: per-frame MATCH."""
    model = runner.get_unwrapped_model()
    return lambda batch: model.mask_gaussians(
        runner.model(batch, func_name='forward_gaussians', return_tempeh_xyz=True))


def clip_forward(runner) -> Callable[[dict], dict]:
    """A clip batch (B=1, T frames; see `stack_window`) -> its masked Gaussians (1,T,1,C,L,1), TEMPEH part included."""
    from match.utils import data_util  # pulls in the model utilities; keep this module light to import

    def forward(clip):
        clip = data_util.MatchBatch(clip).to_device(runner.device)
        _, masked = runner.model(clip, func_name='forward_gaussians', return_masked_gaussians=True,
                                 return_tempeh_xyz=True)
        return masked
    return forward


@dataclass
class PredictedFrame:
    """One frame of the sequence and its predicted Gaussian texture.

    frame_index: position of the frame in the sequence (0-based).
    batch: the frame's MatchBatch (B=1) as loaded: inputs, 'coarse_verts', and the 'extra_*' held-out views.
    gaussians: masked Gaussians in dataset coordinates (not centred), dict of (1,1,C,L,1), as from
      `model.mask_gaussians(model(batch, func_name='forward_gaussians'))`.

    A predictor that holds frames back may keep them on the CPU; the evaluation moves each emitted frame
    back to its device.
    """
    frame_index: int
    batch: dict
    gaussians: dict

    def to(self, device) -> 'PredictedFrame':
        batch = type(self.batch)({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.batch.items()})
        gaussians = {k: v.to(device) for k, v in self.gaussians.items()}
        return PredictedFrame(self.frame_index, batch, gaussians)


class FramePredictor(Protocol):
    def predict(self, batches: Iterable[dict]) -> Iterator[PredictedFrame]:
        """Reads the sequence's batches (B=1, in frame order) and yields every frame's prediction in frame order.

        Runs under the evaluation's autocast and no_grad.
        """

    def metrics(self) -> dict:
        """Top-level metrics.json sections describing how frames were predicted ({} for none).

        Called once, after `predict` has yielded every frame.
        """


class PerFramePredictor:
    """One independent forward pass per frame, emitted before the next batch is read."""

    def __init__(self, forward: Callable[[dict], dict]):
        """forward: a frame's batch -> its masked Gaussians (see PredictedFrame.gaussians)."""
        self.forward = forward

    def predict(self, batches):
        for i, batch in enumerate(batches):
            yield PredictedFrame(i, batch, self.forward(batch))

    def metrics(self):
        return {}


class TextureFilterPredictor:
    """Another predictor's frames smoothed over time by the texture filter: the temporal-filter baseline.

    Frame t is emitted as soon as the inner predictor has emitted the last frame its kernel reaches
    (t + radius, or the end of the sequence), so at most 2 * radius + 1 frames are held, on the CPU.
    """

    def __init__(self, inner: FramePredictor, sigma: float):
        """inner: the per-frame predictor whose textures are filtered. sigma: kernel width in frames (> 0)."""
        if not sigma > 0:
            raise ValueError(f'The texture filter needs sigma > 0, got {sigma}')
        self.inner, self.sigma = inner, float(sigma)
        self.radius = texture_filter.kernel_radius(self.sigma)

    def predict(self, batches):
        held = {}  # frame index -> PredictedFrame, on the CPU
        t = n = 0  # next frame to emit, frames read so far
        for frame in self.inner.predict(batches):
            held[frame.frame_index] = frame.to('cpu')
            n = frame.frame_index + 1
            while t + self.radius < n:  # frame t's kernel ends at t + radius: the same window whatever n becomes
                yield self._filtered(held, t, n)
                t += 1
        while t < n:
            yield self._filtered(held, t, n)
            t += 1

    def _filtered(self, held, t, n):
        """Frame t filtered over the held frames (n read so far); drops the frames no later frame needs."""
        start, weights = texture_filter.kernel(t, n, self.sigma)
        textures = [held[s].gaussians for s in range(start, start + len(weights))]
        frame = PredictedFrame(t, held[t].batch, texture_filter.blend(textures, weights, centre=t - start))
        held.pop(t - self.radius, None)  # frame t + 1's kernel starts at t + 1 - radius
        return frame

    def metrics(self):
        sections = {'texture_filter': {'sigma': self.sigma, 'radius_frames': self.radius,
                                       'filtered': list(texture_filter.FILTERED_KEYS)}}
        inner = self.inner.metrics()
        clash = set(inner) & set(sections)
        if clash:
            raise KeyError(f'The inner predictor returned metrics sections the texture filter writes: {clash}')
        return {**inner, **sections}


def _on_cpu(batch: dict) -> dict:
    return type(batch)({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()})


def stack_window(batches: list[dict]) -> dict:
    """A clip batch (B=1) from consecutive frames' batches (B=1), as `default_collate` makes it from one clip
    sample: tensors (1, T, ...), `frame` as T lists of one ID, other lists (subject, sequence) from the first frame."""
    first = batches[0]
    clip = type(first)()
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            clip[k] = torch.stack([b[k] for b in batches], dim=1)
        elif k == 'frame':
            clip[k] = [list(b[k]) for b in batches]
        else:
            clip[k] = v
    return clip


class WindowedPredictor:
    """A temporal model run on windows of consecutive frames, keeping each frame from one window and slot
    (the window plan in match/sequence_eval/windows.py: 'sliding' or 'block').

    Frame t is emitted as soon as its window has been read and run. Each window runs once; the frames of
    the current window and its outputs are held on the CPU.
    """

    def __init__(self, forward_clip: Callable[[dict], dict], clip_length: int, mode: str):
        """forward_clip: a clip batch (B=1, T frames; see `stack_window`) -> its masked Gaussians with a time
          axis, dict of (1, T, 1, C, L, 1).
        clip_length: window length T (a shorter sequence runs as one shorter window).
        mode: 'sliding' or 'block'."""
        window_slot(0, None, clip_length, mode)  # validates the mode and length
        self.forward_clip, self.clip_length, self.mode = forward_clip, clip_length, mode
        self.per_frame, self.window_length, self.n_windows = {}, None, 0

    def predict(self, batches):
        held, window = {}, None  # frame index -> batch on the CPU; (WindowSlot, its Gaussians on the CPU)
        t = n = 0  # next frame to emit, frames read so far
        for batch in batches:
            held[n] = _on_cpu(batch)
            n += 1
            while t < n:
                slot = window_slot(t, None, self.clip_length, self.mode)
                if slot.start + slot.length > n:
                    break  # its window is not read yet
                frame, window = self._emit(held, window, t, slot)
                yield frame
                t += 1
        while t < n:  # the sequence has ended, so its last windows shift inward
            frame, window = self._emit(held, window, t, window_slot(t, n, self.clip_length, self.mode))
            yield frame
            t += 1

    def _emit(self, held, window, t, slot):
        """Frame t from its window (run unless it is the current one); drops frames no later window needs."""
        if window is None or (window[0].start, window[0].length) != (slot.start, slot.length):
            clip = stack_window([held[i] for i in range(slot.start, slot.start + slot.length)])
            window = (slot, {k: v.cpu() for k, v in self.forward_clip(clip).items()})
            self.n_windows += 1
        self.window_length = slot.length
        self.per_frame[held[t]['frame'][0]] = {
            'window_start': slot.start, 'window_start_frame': held[slot.start]['frame'][0], 'slot': slot.slot}
        frame = PredictedFrame(t, held[t], {k: v[:, slot.slot] for k, v in window[1].items()})
        for i in [i for i in held if i < slot.start]:  # later frames' windows start at slot.start or later
            del held[i]
        return frame, window

    def metrics(self):
        return {'inference': {'mode': self.mode, 'clip_length': self.clip_length, 'window_length': self.window_length,
                              'n_windows': self.n_windows, 'per_frame': self.per_frame}}
