"""The sequence evaluation's frame loop with predictors that hold frames back: CPU, synthetic frames, no model.

Imports scripts/match/eval_sequence.py, which pulls in the runner (~12 s). Rendering, panels, collectors
and the video writers are stand-ins that record what they are given.
Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_eval_sequence_frame_loop.py
"""
from pathlib import Path
from types import SimpleNamespace

import gin
import pytest
import torch

from match.sequence_eval.context import SequencePlan
from match.sequence_eval.predictors import PredictedFrame, WindowCameraError
import eval_sequence_script

H, W = 8, 8
CAMERA = '400001'
FRAMES = ['000010', '000022', '000034']
SCHEDULE = [0, 0, 1, 2, 2, 2]  # source frame shown in each grid-video frame


eval_sequence = eval_sequence_script.load()


class Sequence(torch.utils.data.Dataset):
    """Synthetic SequenceFrames: one held-out camera, and a 'value' that identifies the frame. Frame i's input cameras
    are 100 * i .. 100 * i + 11, or the same on every frame with same_inputs."""
    heldout_camera_ids = [CAMERA]
    frame_ids = FRAMES

    def __init__(self, same_inputs=False):
        self.same_inputs = same_inputs

    def __len__(self):
        return len(FRAMES)

    def __getitem__(self, i):
        cameras = torch.arange(12) + (0 if self.same_inputs else 100 * i)
        return {'frame': FRAMES[i], 'camera': cameras, 'value': torch.tensor(float(i)),
                'extra_C2W': torch.eye(4)[None], 'extra_fxfycxcy': torch.tensor([[1.0, 1.0, 0.5, 0.5]]),
                'extra_image': torch.zeros(1, 3, H, W), 'extra_mask': torch.ones(1, 1, H, W)}


def make_runner():
    def render_gaussians(gaussians, C2W, fxfycxcy, height, width, bg_color):
        return {'image': torch.zeros(1, C2W.shape[1], 3, height, width)}
    opt = SimpleNamespace(val_loader_params=[{'kwargs': {'dataloader_kwargs': {}}}], temporal=False, clip_length=4)
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


SEQUENCE_CONFIG = 'configs/match/eval_sequence_SEN_a_good_morrow.gin'
needs_assets = pytest.mark.skipif(not Path('assets/pretrained_models/match/uv_res_1024/config_0.gin').exists(),
                                  reason='assets/ is missing; see the worktree section of CLAUDE.md')


def make_model_runner():
    """make_runner() plus a model whose texture for frame i has position 10 * i**2."""
    def forward(batch, func_name, **kwargs):
        assert func_name == 'forward_gaussians'
        i = batch['value'].reshape(1, 1, 1, 1, 1)
        return {'xyz': 10 * i * i, 'rgb': torch.full_like(i, 0.5), 'scale': torch.full_like(i, 0.01),
                'opacity': torch.full_like(i, 0.8), 'rotation': torch.tensor([1.0, 0, 0, 0]).reshape(1, 1, 4, 1, 1)}
    runner = make_runner()
    runner.model = forward
    runner.get_unwrapped_model = lambda: SimpleNamespace(mask_gaussians=lambda gaussians: gaussians)
    return runner


@pytest.fixture
def config():
    """Parses the sequence evaluation's config with extra bindings, as the script does."""
    def parse(*bindings):
        gin.clear_config()
        gin.parse_config_files_and_bindings(config_files=[SEQUENCE_CONFIG], bindings=list(bindings), skip_unknown=True)
    yield parse
    gin.clear_config()


def run_configured_predictor():
    runner = make_model_runner()
    predictor = eval_sequence.build_predictor(runner)
    plan = SequencePlan(Sequence(), schedule=SCHEDULE, fixed_view_camera=CAMERA)
    collector, panel, video, preview = Recorder(), Recorder(), Recorder(), Recorder()
    eval_sequence.run_frames(runner, plan, predictor, [panel], [collector], video, preview)
    return [value for _, _, value in collector.frames], predictor.metrics()


@needs_assets
def test_the_sequence_evaluation_scores_unfiltered_per_frame_textures_by_default(config):
    config()

    positions, metrics = run_configured_predictor()

    assert positions == [0.0, 10.0, 40.0]
    assert metrics == {}


@needs_assets
def test_the_config_turns_on_the_texture_filter_and_the_metrics_record_its_sigma(config):
    config('TEXTURE_FILTER_SIGMA = 1.0')

    positions, metrics = run_configured_predictor()

    # Middle frame: weights e^-1/2, 1, e^-1/2 over positions 0, 10, 40, i.e. (10 + 40 * 0.60653) / 2.21306.
    assert positions[1] == pytest.approx(15.4813, abs=1e-3)
    assert positions[0] < 10.0 and positions[2] < 40.0  # edge frames pulled toward their neighbours, not lagging
    assert metrics['texture_filter']['sigma'] == 1.0


@needs_assets
@pytest.mark.parametrize('sigma', [0.5, 2.0, 4.0])
def test_a_sigma_sweep_only_changes_the_config(config, sigma):
    config(f'TEXTURE_FILTER_SIGMA = {sigma}')

    _, metrics = run_configured_predictor()

    assert metrics['texture_filter']['sigma'] == sigma


def make_temporal_runner(clip_length=2):
    """A temporal checkpoint stand-in that only runs on clips: frame i of the window starting at frame s has
    position 100 * s + 10 * i, and TEMPEH part i when asked for it."""
    def forward(batch, func_name, return_masked_gaussians=False, return_tempeh_xyz=False, **kwargs):
        assert func_name == 'forward_gaussians'
        values = batch['value']
        assert values.ndim == 2, 'expected a clip batch, (1, T)'
        gaussians = {'xyz': (100 * values[0, 0] + 10 * values).reshape(1, -1, 1, 1, 1, 1)}
        if return_tempeh_xyz:
            gaussians['tempeh_xyz'] = values.reshape(1, -1, 1, 1, 1, 1)
        return (gaussians, gaussians) if return_masked_gaussians else gaussians
    runner = make_runner()
    runner.opt.temporal, runner.opt.clip_length = True, clip_length
    runner.model = forward
    runner.get_unwrapped_model = lambda: SimpleNamespace(mask_gaussians=lambda gaussians: gaussians)
    return runner


def run_temporal_predictor(runner, same_inputs=True):
    predictor = eval_sequence.build_predictor(runner)
    plan = SequencePlan(Sequence(same_inputs), schedule=SCHEDULE, fixed_view_camera=CAMERA)
    collector, panel, video, preview = Recorder(), Recorder(), Recorder(), Recorder()
    keys = []
    collector.add_frame = lambda ctx: (keys.append(set(ctx.gaussians)), Recorder.add_frame(collector, ctx))
    eval_sequence.run_frames(runner, plan, predictor, [panel], [collector], video, preview)
    return collector, keys, predictor.metrics()


@needs_assets
def test_a_temporal_checkpoint_runs_in_sliding_windows_by_default(config):
    config()

    collector, keys, metrics = run_temporal_predictor(make_temporal_runner(clip_length=2))

    # Windows of 2 over 3 frames: frames 0, 1 from the window at 0 (slots 0, 1), frame 2 from the window at 1 (slot 1).
    assert collector.frames == [(0, '000010', 0.0), (1, '000022', 10.0), (2, '000034', 120.0)]
    assert metrics['inference']['mode'] == 'sliding'
    assert metrics['inference']['per_frame'] == {
        '000010': {'window_start': 0, 'window_start_frame': '000010', 'slot': 0},
        '000022': {'window_start': 0, 'window_start_frame': '000010', 'slot': 1},
        '000034': {'window_start': 1, 'window_start_frame': '000022', 'slot': 1}}
    assert all('tempeh_xyz' in k for k in keys)  # the geometric jitter collector reads it


@needs_assets
def test_the_config_selects_block_mode_for_a_temporal_checkpoint(config):
    config("INFERENCE_MODE = 'block'")

    collector, _, metrics = run_temporal_predictor(make_temporal_runner(clip_length=2))

    # Blocks of 2 over 3 frames: frames 0, 1 from the block at 0, frame 2 from the last block shifted inward to 1.
    assert [value for _, _, value in collector.frames] == [0.0, 10.0, 120.0]
    assert metrics['inference']['mode'] == 'block'


@needs_assets
def test_a_sequence_shorter_than_the_temporal_window_runs_as_one_clip(config):
    config()

    collector, _, metrics = run_temporal_predictor(make_temporal_runner(clip_length=4))

    assert [value for _, _, value in collector.frames] == [0.0, 10.0, 20.0]
    assert metrics['inference']['window_length'] == 3


@needs_assets
def test_the_texture_filter_is_refused_for_a_temporal_checkpoint(config):
    config('TEXTURE_FILTER_SIGMA = 1.0')

    with pytest.raises(ValueError, match='texture filter'):
        eval_sequence.build_predictor(make_temporal_runner())


@needs_assets
def test_the_config_marks_the_checkpoint_as_temporal(config):
    from match.options import Options
    config()
    assert Options().temporal is False

    config('TEMPORAL = True')
    assert Options().temporal is True


@needs_assets
def test_a_windowed_run_fails_loudly_when_the_frames_of_a_window_have_different_input_cameras(config):
    config()

    with pytest.raises(WindowCameraError, match='different input cameras'):
        run_temporal_predictor(make_temporal_runner(), same_inputs=False)
