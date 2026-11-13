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

    sequence: the frames to evaluate and the held-out cameras (none in a flicker-only run).
    schedule: index of the source frame shown in every grid-video frame (outputs.output_frame_schedule).
    fixed_view_camera: ID of the camera of the fixed view: the held-out camera closest to frontal, or, with no
      held-out cameras, the configured input camera.
    """
    sequence: 'SequenceFrames'
    schedule: list
    fixed_view_camera: str

    @property
    def heldout_camera_ids(self):
        return self.sequence.heldout_camera_ids

    @property
    def fixed_view_is_input(self):
        """True in a flicker-only run: the fixed view is an input camera, so it is no novel view."""
        return self.fixed_view_camera not in self.heldout_camera_ids

    @property
    def n_output_frames(self):
        return len(self.schedule)

    def output_frame_indices(self, frame_index):
        """Indices of the grid-video frames that show source frame `frame_index`."""
        return [k for k, s in enumerate(self.schedule) if s == frame_index]


@dataclass
class HeldOutView:
    """One held-out camera (or, in a flicker-only run, the fixed-view input camera) on one frame. Images are (3,H,W) float in [0,1] at the real image's crop and size."""
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
    heldout: one HeldOutView per held-out camera, in config order (empty in a flicker-only run).
    fixed_view_camera: ID of the camera of the fixed view (see SequencePlan).
    output_frame_indices: indices of the grid-video frames that show this source frame.
    input_view: the fixed view when it is an input camera (flicker-only run), else None.
    """
    runner: 'MatchRunner'
    frame_index: int
    frame_id: str
    batch: 'MatchBatch'
    gaussians: dict
    heldout: dict
    fixed_view_camera: str
    output_frame_indices: list
    input_view: HeldOutView = None

    @property
    def fixed_view(self) -> HeldOutView:
        return self.input_view if self.input_view is not None else self.heldout[self.fixed_view_camera]

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


def _render_views(runner, gaussians, C2W, fxfycxcy, real, mask, camera_ids):
    """Renders the Gaussians from each camera (V,4,4), (V,4) at the real images' size: one HeldOutView per camera."""
    H, W = real.shape[-2:]
    pred = runner.render_gaussians(gaussians, C2W[None], fxfycxcy[None], height=H, width=W,
                                   bg_color=(1.0, 1.0, 1.0))['image'][0].float().clamp(0, 1)
    return {camera: HeldOutView(camera, pred[v], real[v], mask[v], C2W[v], fxfycxcy[v])
            for v, camera in enumerate(camera_ids)}


def build_frame_context(runner, batch, gaussians, plan: SequencePlan, frame_index):
    """Renders every held-out camera (or, in a flicker-only run, the fixed-view input camera) with its real
    calibration and bundles the frame's data."""
    heldout, input_view = {}, None
    if plan.heldout_camera_ids:
        heldout = _render_views(runner, gaussians, batch['extra_C2W'][0], batch['extra_fxfycxcy'][0],
                                batch['extra_image'][0].float(), batch['extra_mask'][0].float(),
                                plan.heldout_camera_ids)
    if plan.fixed_view_is_input:
        v = [str(c) for c in batch['camera'][0].tolist()].index(plan.fixed_view_camera)
        views = slice(v, v + 1)
        (input_view,) = _render_views(runner, gaussians, batch['C2W'][0][views], batch['fxfycxcy'][0][views],
                                      batch['image'][0][views].float(), batch['mask'][0][views].float(),
                                      [plan.fixed_view_camera]).values()
    return FrameContext(
        runner=runner, frame_index=frame_index, frame_id=batch['frame'][0], batch=batch, gaussians=gaussians,
        heldout=heldout, fixed_view_camera=plan.fixed_view_camera,
        output_frame_indices=plan.output_frame_indices(frame_index), input_view=input_view)
