"""Frame predictors of the sequence evaluation: CPU tests on synthetic batches, no model.

A predictor turns the sequence's per-frame batches (in order) into each frame's masked Gaussian texture.
Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_frame_predictors.py
"""
import torch

from match.sequence_eval.predictors import PerFramePredictor


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
