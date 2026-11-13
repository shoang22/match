"""Flicker-only sequence evaluation (#28): with no held-out cameras, a configured input camera is the fixed view.

CPU tests on a synthetic batch with no 'extra_*' keys (the dataset adds none when no extra cameras are configured).
"""
import torch

from match.sequence_eval.context import SequencePlan, build_frame_context

H, W = 8, 6
INPUT_CAMERAS = [401875, 401919, 402020]
FIXED = '401919'


class Sequence:
    heldout_camera_ids = []
    frame_ids = ['32885']


def input_batch():
    """Input view v has image value v/10 and mask value 1 - v/10, so the picked view is identifiable."""
    n = len(INPUT_CAMERAS)
    return {
        'frame': ['32885'],
        'camera': torch.tensor([INPUT_CAMERAS]),
        'image': torch.stack([torch.full((3, H, W), v / 10) for v in range(n)])[None],
        'mask': torch.stack([torch.full((1, H, W), 1 - v / 10) for v in range(n)])[None],
        'C2W': torch.stack([torch.eye(4) * (v + 1) for v in range(n)])[None],
        'fxfycxcy': torch.stack([torch.tensor([1.0, 1.0, 0.5, 0.5]) * (v + 1) for v in range(n)])[None],
    }


class Runner:
    """Renders an image whose value is the camera's C2W[0,0], and records the cameras it was asked for."""

    def __init__(self):
        self.calls = []

    def render_gaussians(self, gaussians, C2W, fxfycxcy, height, width, bg_color):
        self.calls.append((C2W, fxfycxcy, height, width))
        V = C2W.shape[1]
        return {'image': C2W[0, :, 0, 0].reshape(1, V, 1, 1, 1).expand(1, V, 3, height, width) / 10}


def test_the_fixed_view_is_the_configured_input_view_rendered_with_its_calibration():
    runner, batch = Runner(), input_batch()
    plan = SequencePlan(Sequence(), schedule=[0, 0], fixed_view_camera=FIXED)

    ctx = build_frame_context(runner, batch, {}, plan, frame_index=0)

    assert plan.fixed_view_is_input
    assert ctx.heldout == {}
    view = ctx.fixed_view
    assert view.camera_id == FIXED
    # input view 1 of the batch
    assert torch.allclose(view.real, torch.full((3, H, W), 0.1))
    assert torch.allclose(view.mask, torch.full((1, H, W), 0.9))
    assert torch.equal(view.C2W, torch.eye(4) * 2)
    assert torch.equal(view.fxfycxcy, torch.tensor([2.0, 2.0, 1.0, 1.0]))
    # the prediction is rendered from that camera at the real image's size
    ((C2W, fxfycxcy, height, width),) = runner.calls
    assert torch.equal(C2W, (torch.eye(4) * 2)[None, None])
    assert (height, width) == (H, W)
    assert torch.allclose(view.pred, torch.full((3, H, W), 0.2))
    assert (ctx.height, ctx.width) == (H, W)
