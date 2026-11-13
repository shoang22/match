"""Frame predictors of the sequence evaluation: CPU tests on synthetic batches, no model.

A predictor turns the sequence's per-frame batches (in order) into each frame's masked Gaussian texture.
Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_frame_predictors.py
"""
import math

import pytest
import torch

from match.sequence_eval.predictors import (PerFramePredictor, TextureFilterPredictor, WindowCameraError,
                                            WindowedPredictor)


def make_batches(n, cameras=None):
    """n frames' batches (B=1); cameras: each frame's input camera IDs (by default the same two on every frame)."""
    cameras = cameras or [[1, 2]] * n
    return [{'frame': [f'{i:06d}'], 'value': torch.full((1, 1), float(i)), 'camera': torch.tensor([cameras[i]])}
            for i in range(n)]


def test_per_frame_predictor_emits_every_frame_from_its_own_forward_pass_in_order():
    predictor = PerFramePredictor(forward=lambda batch: {'xyz': batch['value'] * 10})

    frames = list(predictor.predict(make_batches(3)))

    assert [f.frame_index for f in frames] == [0, 1, 2]
    assert [f.batch['frame'][0] for f in frames] == ['000000', '000001', '000002']
    assert [f.gaussians['xyz'].item() for f in frames] == [0.0, 10.0, 20.0]
    assert predictor.metrics() == {}


def test_per_frame_predictor_emits_each_frame_before_reading_the_next_batch():
    read = []

    def batches():
        for batch in make_batches(3):
            read.append(batch['frame'][0])
            yield batch

    predictor = PerFramePredictor(forward=lambda batch: {})
    for frame in predictor.predict(batches()):
        assert read[-1] == frame.batch['frame'][0]  # one batch in memory at a time, as before predictors


def per_frame_textures():
    """A PerFramePredictor whose texture for frame i has position i**2 (so filtering changes it)."""
    def forward(batch):
        i = batch['value'].item()
        return {'xyz': torch.full((1, 1, 3, 2, 1), i * i), 'rgb': torch.full((1, 1, 3, 2, 1), 0.5),
                'scale': torch.full((1, 1, 3, 2, 1), 0.01), 'opacity': torch.full((1, 1, 1, 2, 1), 0.8),
                'rotation': torch.tensor([1.0, 0, 0, 0]).reshape(1, 1, 4, 1, 1).expand(1, 1, 4, 2, 1),
                'tempeh_xyz': torch.full((1, 1, 3, 2, 1), i)}
    return PerFramePredictor(forward)


def test_texture_filter_predictor_emits_every_frame_in_order_with_its_batch_and_filtered_texture():
    frames = list(TextureFilterPredictor(per_frame_textures(), sigma=1.0).predict(make_batches(6)))

    assert [f.frame_index for f in frames] == list(range(6))
    assert [f.batch['frame'][0] for f in frames] == [f'{i:06d}' for i in range(6)]
    for t, frame in enumerate(frames):
        # Frame s has position s**2; a centred Gaussian of sigma 1 frame, cut at 4 sigma and at the ends, weights 1.
        weights = {s: math.exp(-0.5 * (s - t) ** 2) for s in range(max(0, t - 4), min(6, t + 5))}
        expected = sum(w * s * s for s, w in weights.items()) / sum(weights.values())
        assert frame.gaussians.keys() == {'xyz', 'rgb', 'scale', 'opacity', 'rotation', 'tempeh_xyz'}
        torch.testing.assert_close(frame.gaussians['xyz'], torch.full((1, 1, 3, 2, 1), expected))
        torch.testing.assert_close(frame.gaussians['tempeh_xyz'], torch.full((1, 1, 3, 2, 1), float(t)))  # unfiltered


def test_texture_filter_predictor_emits_each_frame_once_the_kernel_has_read_its_last_frame():
    read = []

    def batches():
        for batch in make_batches(7):
            read.append(int(batch['frame'][0]))
            yield batch

    for frame in TextureFilterPredictor(per_frame_textures(), sigma=0.5).predict(batches()):  # radius 2 frames
        assert read[-1] == min(frame.frame_index + 2, 6)


def test_texture_filter_predictor_records_that_the_filter_ran_and_its_sigma():
    predictor = TextureFilterPredictor(per_frame_textures(), sigma=1.5)
    list(predictor.predict(make_batches(3)))

    assert predictor.metrics() == {'texture_filter': {'sigma': 1.5, 'radius_frames': 6,
                                                      'filtered': ['xyz', 'rgb', 'scale', 'opacity', 'rotation']}}


class ClipModel:
    """forward_clip stand-in: slot k of a window holds 100 * window start + the frame's value. Records every window."""

    def __init__(self):
        self.windows = []

    def __call__(self, clip):
        values = clip['value'][0, :, 0]  # (T,)
        start = int(values[0])
        self.windows.append([int(v) for v in values])
        assert clip['frame'] == [[f'{v:06d}'] for v in self.windows[-1]]  # default_collate's clip layout, B=1
        xyz = (100 * start + values).reshape(1, -1, 1, 1, 1, 1)
        return {'xyz': xyz, 'tempeh_xyz': values.reshape(1, -1, 1, 1, 1, 1)}


def test_sliding_predictor_runs_one_window_per_frame_and_keeps_that_frames_slot():
    model = ClipModel()

    frames = list(WindowedPredictor(model, clip_length=4, mode='sliding').predict(make_batches(6)))

    assert model.windows == [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    assert [f.frame_index for f in frames] == list(range(6))
    assert [f.batch['frame'][0] for f in frames] == [f'{i:06d}' for i in range(6)]
    assert [f.gaussians['xyz'].item() for f in frames] == [0, 1, 102, 203, 204, 205]
    assert [f.gaussians['tempeh_xyz'].item() for f in frames] == [0, 1, 2, 3, 4, 5]
    assert all(f.gaussians['xyz'].shape == (1, 1, 1, 1, 1) for f in frames)  # a frame's masked Gaussians, no time axis


def test_block_predictor_keeps_every_frame_of_each_block_and_only_new_frames_of_the_last():
    model = ClipModel()

    frames = list(WindowedPredictor(model, clip_length=4, mode='block').predict(make_batches(6)))

    assert model.windows == [[0, 1, 2, 3], [2, 3, 4, 5]]
    assert [f.gaussians['xyz'].item() for f in frames] == [0, 1, 2, 3, 204, 205]


def test_a_sequence_shorter_than_the_window_runs_as_one_shorter_clip():
    model = ClipModel()

    frames = list(WindowedPredictor(model, clip_length=4, mode='sliding').predict(make_batches(2)))

    assert model.windows == [[0, 1]]
    assert [f.gaussians['xyz'].item() for f in frames] == [0, 1]


def test_windowed_predictor_emits_each_frame_once_its_window_has_been_read():
    read = []

    def batches():
        for batch in make_batches(7):
            read.append(int(batch['frame'][0]))
            yield batch

    for frame in WindowedPredictor(ClipModel(), clip_length=4, mode='sliding').predict(batches()):
        assert read[-1] == min(max(frame.frame_index + 2, 3), 6)


@pytest.mark.parametrize('mode', ['sliding', 'block'])
def test_windowed_predictor_records_the_mode_and_each_frames_window_start_and_slot(mode):
    predictor = WindowedPredictor(ClipModel(), clip_length=4, mode=mode)
    list(predictor.predict(make_batches(5)))

    inference = predictor.metrics()['inference']
    assert inference['mode'] == mode
    assert inference['clip_length'] == 4
    starts = {'sliding': [0, 0, 1, 1, 1], 'block': [0, 0, 0, 0, 1]}[mode]
    assert inference['per_frame'] == {
        f'{t:06d}': {'window_start': s, 'window_start_frame': f'{s:06d}', 'slot': t - s} for t, s in enumerate(starts)}


@pytest.mark.parametrize('mode', ['sliding', 'block'])
def test_a_window_whose_frames_have_different_input_cameras_fails_loudly(mode):
    # Image tokens are aligned by camera and pixel patch position across a clip, so every frame of a window must
    # have the same input cameras in the same order. Frame 2 lost camera 2 here.
    cameras = [[1, 2], [1, 2], [1, 3], [1, 2], [1, 2]]
    predictor = WindowedPredictor(ClipModel(), clip_length=4, mode=mode)

    with pytest.raises(WindowCameraError, match=r"000002.*\[1, 3\]"):
        list(predictor.predict(make_batches(5, cameras)))
