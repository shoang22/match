"""What panels and metric collectors get for each source frame of the sequence evaluation run."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # the runner pulls in the model, TF and diffusers; keep this module light to import
    from match.runner import MatchRunner
    from match.sequence_eval.frames import SequenceFrames
    from match.utils.data_util import MatchBatch


@dataclass(frozen=True)
class SequencePlan:
    """What the run evaluates and how the grid video shows it, fixed before the first forward pass.

    sequence: the frames to evaluate and the held-out cameras.
    schedule: index of the source frame shown in every grid-video frame (outputs.output_frame_schedule).
    fixed_view_camera: ID of the held-out camera used as the fixed novel viewpoint (closest to frontal).
    """
    sequence: 'SequenceFrames'
    schedule: list
    fixed_view_camera: str

    @property
    def heldout_camera_ids(self):
        return self.sequence.heldout_camera_ids

    @property
    def n_output_frames(self):
        return len(self.schedule)

    def output_frame_indices(self, frame_index):
        """Indices of the grid-video frames that show source frame `frame_index`."""
        return [k for k, s in enumerate(self.schedule) if s == frame_index]


@dataclass
class HeldOutView:
    """One held-out camera on one frame. Images are (3,H,W) float in [0,1] at the real image's crop and size."""
    camera_id: str
    pred: torch.Tensor  # MATCH render with the camera's real calibration, white background
    real: torch.Tensor  # real image, background as captured
    mask: torch.Tensor  # (1,H,W) foreground mask in [0,1]
    C2W: torch.Tensor  # (4,4)
    fxfycxcy: torch.Tensor  # (4,) normalised by the crop's W,H

    def real_on_background(self, bg=1.0):
        """The real image with the background outside the mask painted bg (white by default, like the renders)."""
        return self.real * self.mask + (1 - self.mask) * bg


@dataclass
class FrameContext:
    """Everything known about one source frame after its forward pass.

    batch: the MatchBatch on the device (B=1), including 'coarse_verts' (TEMPEH mesh, dataset coordinates),
      the 12 input views and the 'extra_*' held-out keys.
    gaussians: the masked Gaussians in dataset coordinates (not centred), dict of (1,1,C,L,1). Don't modify
      them in place; centre a copy for orbit renders.
    heldout: one HeldOutView per held-out camera, in config order.
    fixed_view_camera: ID of the held-out camera used as the fixed novel viewpoint (closest to frontal).
    output_frame_indices: indices of the grid-video frames that show this source frame.
    """
    runner: 'MatchRunner'
    frame_index: int
    frame_id: str
    batch: 'MatchBatch'
    gaussians: dict
    heldout: dict
    fixed_view_camera: str
    output_frame_indices: list

    @property
    def fixed_view(self) -> HeldOutView:
        return self.heldout[self.fixed_view_camera]

    @property
    def height(self):
        return self.fixed_view.real.shape[-2]

    @property
    def width(self):
        return self.fixed_view.real.shape[-1]

    def render(self, C2W, fxfycxcy, gaussians=None, bg_color=(1.0, 1.0, 1.0)):
        """Renders the frame's Gaussians (or the given ones) from cameras C2W (V,4,4), fxfycxcy (V,4) at the panel size.

        Returns (V,3,H,W) in [0,1].
        """
        gaussians = self.gaussians if gaussians is None else gaussians
        out = self.runner.render_gaussians(
            gaussians, C2W[None], fxfycxcy[None], height=self.height, width=self.width, bg_color=bg_color)
        return out['image'][0].float().clamp(0, 1)


def build_frame_context(runner, batch, gaussians, plan: SequencePlan, frame_index):
    """Renders every held-out camera with its real calibration and bundles the frame's data."""
    C2W = batch['extra_C2W'][0]
    fxfycxcy = batch['extra_fxfycxcy'][0]
    real = batch['extra_image'][0].float()
    mask = batch['extra_mask'][0].float()
    H, W = real.shape[-2:]
    pred = runner.render_gaussians(gaussians, C2W[None], fxfycxcy[None], height=H, width=W,
                                   bg_color=(1.0, 1.0, 1.0))['image'][0].float().clamp(0, 1)
    heldout = {
        camera: HeldOutView(camera, pred[e], real[e], mask[e], C2W[e], fxfycxcy[e])
        for e, camera in enumerate(plan.heldout_camera_ids)
    }
    return FrameContext(
        runner=runner, frame_index=frame_index, frame_id=batch['frame'][0], batch=batch, gaussians=gaussians,
        heldout=heldout, fixed_view_camera=plan.fixed_view_camera,
        output_frame_indices=plan.output_frame_indices(frame_index))
