"""The sequence evaluation's frame loop with predictors that hold frames back: CPU, synthetic frames, no model.

Imports scripts/match/eval_sequence.py, which pulls in the runner (~12 s). Rendering, panels, collectors
and the video writers are stand-ins that record what they are given.
Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_eval_sequence_frame_loop.py
"""
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from match.sequence_eval.context import SequencePlan
from match.sequence_eval.predictors import PredictedFrame

H, W = 8, 8
CAMERA = '400001'
FRAMES = ['000010', '000022', '000034']
SCHEDULE = [0, 0, 1, 2, 2, 2]  # source frame shown in each grid-video frame


def load_script():
    spec = importlib.util.spec_from_file_location('eval_sequence', 'scripts/match/eval_sequence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_sequence = load_script()


class Sequence(torch.utils.data.Dataset):
    """Synthetic SequenceFrames: one held-out camera, and a 'value' that identifies the frame."""
    heldout_camera_ids = [CAMERA]
    frame_ids = FRAMES

    def __len__(self):
        return len(FRAMES)

    def __getitem__(self, i):
        return {'frame': FRAMES[i], 'camera': torch.arange(12) + 100 * i, 'value': torch.tensor(float(i)),
                'extra_C2W': torch.eye(4)[None], 'extra_fxfycxcy': torch.tensor([[1.0, 1.0, 0.5, 0.5]]),
                'extra_image': torch.zeros(1, 3, H, W), 'extra_mask': torch.ones(1, 1, H, W)}


def make_runner():
    def render_gaussians(gaussians, C2W, fxfycxcy, height, width, bg_color):
        return {'image': torch.zeros(1, C2W.shape[1], 3, height, width)}
    opt = SimpleNamespace(val_loader_params=[{'kwargs': {'dataloader_kwargs': {}}}])
    return SimpleNamespace(opt=opt, device=torch.device('cpu'), render_gaussians=render_gaussians)


class WholeSequencePredictor:
    """Reads every batch before it emits the first frame, like a filter over the whole sequence."""

    def __init__(self, order=None):
        self.order = order

    def predict(self, batches):
        held = [PredictedFrame(i, batch, {'xyz': batch['value'] * 10}) for i, batch in enumerate(batches)]
        for i in self.order or range(len(held)):
            yield held[i]

    def metrics(self):
        return {}


class Recorder:
    """Stands in for a metric collector, a panel and a video writer."""
    label = 'recorder'

    def __init__(self):
        self.frames, self.writes = [], []

    def add_frame(self, ctx):
        self.frames.append((ctx.frame_index, ctx.frame_id, ctx.gaussians['xyz'].item()))

    def render(self, ctx):
        return [torch.zeros(3, H, W) for _ in ctx.output_frame_indices]

    def write(self, k, grid):
        self.writes.append(k)


def run(predictor):
    plan = SequencePlan(Sequence(), schedule=SCHEDULE, fixed_view_camera=CAMERA)
    collector, panel, video, preview = Recorder(), Recorder(), Recorder(), Recorder()
    input_cameras = eval_sequence.run_frames(make_runner(), plan, predictor, [panel], [collector], video, preview)
    return input_cameras, collector, video, preview


def test_frames_held_back_by_the_predictor_are_scored_and_rendered_in_order():
    input_cameras, collector, video, preview = run(WholeSequencePredictor())

    assert collector.frames == [(0, '000010', 0.0), (1, '000022', 10.0), (2, '000034', 20.0)]
    assert video.writes == list(range(len(SCHEDULE)))
    assert preview.writes == [0, 1, 2]
    assert list(input_cameras) == FRAMES
    assert input_cameras['000022'] == [str(c) for c in range(100, 112)]


@pytest.mark.parametrize('order', [[1, 0, 2], [0, 2], [0, 1, 1, 2], [0, 1]])
def test_a_predictor_that_emits_frames_out_of_order_or_misses_one_fails_the_run(order):
    with pytest.raises(eval_sequence.FrameLoadError):
        run(WholeSequencePredictor(order))
