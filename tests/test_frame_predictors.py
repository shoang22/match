"""Frame predictors of the sequence evaluation: CPU tests on synthetic batches, no model.

A predictor turns the sequence's per-frame batches (in order) into each frame's masked Gaussian texture.
Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_frame_predictors.py
"""
import torch

from match.sequence_eval.predictors import PerFramePredictor, TextureFilterPredictor
from match.sequence_eval.texture_filter import filter_textures


def make_batches(n):
    return [{'frame': [f'{i:06d}'], 'value': torch.full((1, 1), float(i))} for i in range(n)]


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
    unfiltered = [f.gaussians for f in per_frame_textures().predict(make_batches(6))]

    assert [f.frame_index for f in frames] == list(range(6))
    assert [f.batch['frame'][0] for f in frames] == [f'{i:06d}' for i in range(6)]
    for frame, expected in zip(frames, filter_textures(unfiltered, sigma=1.0)):
        assert frame.gaussians.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(frame.gaussians[key], expected[key])


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
