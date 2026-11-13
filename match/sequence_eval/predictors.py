"""Frame predictors: how the sequence evaluation gets each frame's Gaussian texture.

A predictor reads the sequence's per-frame batches in order and emits one PredictedFrame per frame, also
in order. It may read several batches before it emits a frame (a texture filter needs the whole
sequence, a temporal model needs a window of frames around the one it emits); the evaluation scores and
renders each frame as soon as it is emitted. The default is PerFramePredictor: one forward pass per frame.
"""
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Protocol

import torch


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
