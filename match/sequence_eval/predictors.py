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
