"""The single-frame novel-view benchmark loop: how each test frame is predicted (per frame, or from its sliding
window) and what the output reports. Model, data and rendering are replaced by fakes. CPU.

Run: conda run -n match python -m pytest -q tests/test_novel_view_benchmark.py
"""
import pytest
import torch

from match.novel_view.benchmark import run_benchmark
from match.novel_view.samples import FrameInputs, plan_samples

RUN = [str(100 + i) for i in range(6)]  # one run of 6 frames with complete inputs


def make_plan(context_length=4, frames=RUN):
    captures = {'S1': [FrameInputs(f, 'seq', ('a', 'b')) for f in frames]}
    return plan_samples(captures, lambda subject, frame: ['x'], n_samples=None, seed=0, context_length=context_length)


class FakeData:
    """Frames as batches (B=1) whose 'value' is the frame ID; records every load."""

    def __init__(self):
        self.loads = []

    def load_frame(self, subject, frame, target_camera):
        self.loads.append((subject, frame, target_camera))
        batch = {'frame': [frame], 'subject': [subject], 'camera': torch.tensor([[1, 2]]),
                 'value': torch.tensor([[float(frame)]])}
        if target_camera is not None:
            batch['extra_value'] = torch.tensor([[float(frame)]])
        return batch


class FakeModel:
    """Per-frame Gaussians carry the frame's value; clip Gaussians carry each slot's frame value."""

    def __init__(self):
        self.frame_calls, self.clip_calls = [], []

    def forward_frame(self, batch):
        self.frame_calls.append(batch['frame'][0])
        return {'xyz': batch['value'].reshape(1, 1, 1, 1, 1)}

    def forward_clip(self, clip):
        self.clip_calls.append([f[0] for f in clip['frame']])
        assert not any(k.startswith('extra_') for k in clip), 'the target view must not reach the model'
        return {'xyz': clip['value'].reshape(1, -1, 1, 1, 1, 1)}


def score(batch, gaussians):
    """The 'score' is which frame's Gaussians were kept, and which frame's target view they were scored on."""
    return {'psnr': gaussians['xyz'].item(), 'target_frame': batch['extra_value'].item()}


def run(mode, clip_length=4, plan=None):
    data, model = FakeData(), FakeModel()
    out = run_benchmark(plan or make_plan(), load_frame=data.load_frame, forward_frame=model.forward_frame,
                        forward_clip=model.forward_clip, clip_length=clip_length, mode=mode, score=score)
    return out, data, model


def test_per_frame_inference_predicts_each_test_frame_alone_and_scores_it_at_its_target():
    out, data, model = run('per_frame')

    assert model.frame_calls == RUN
    assert model.clip_calls == []
    assert data.loads == [('S1', f, 'x') for f in RUN]
    assert [(r['frame'], r['target_camera'], r['psnr'], r['target_frame']) for r in out['samples']] == [
        (f, 'x', float(f), float(f)) for f in RUN]


def test_a_per_frame_run_records_its_mode_and_no_windows():
    out, _, _ = run('per_frame')

    assert out['inference'] == {'mode': 'per_frame'}
    assert all('window' not in r for r in out['samples'])


def test_sliding_inference_predicts_each_test_frame_from_its_sliding_window_and_keeps_its_slot():
    out, _, model = run('sliding')

    # one frame before to two after, shifted inward at the run's ends
    windows = [RUN[0:4], RUN[0:4], RUN[1:5], RUN[2:6], RUN[2:6], RUN[2:6]]
    assert model.clip_calls == windows
    assert model.frame_calls == []
    assert [r['window'] for r in out['samples']] == [
        {'frames': w, 'slot': w.index(f)} for f, w in zip(RUN, windows)]
    assert [(r['psnr'], r['target_frame']) for r in out['samples']] == [(float(f), float(f)) for f in RUN]


def test_only_the_test_frame_of_a_window_is_loaded_with_its_target_camera():
    _, data, _ = run('sliding')

    first_window = data.loads[:4]
    assert first_window == [('S1', '100', 'x'), ('S1', '101', None), ('S1', '102', None), ('S1', '103', None)]


def test_a_windowed_run_records_its_mode_and_window_length():
    out, _, _ = run('sliding')

    assert out['inference'] == {'mode': 'sliding', 'clip_length': 4}


def test_the_summary_reports_the_inference_mode_alongside_the_mean_scores():
    out, _, _ = run('per_frame')

    assert out['summary']['inference_mode'] == 'per_frame'
    assert out['summary']['n_samples'] == len(RUN)
    assert out['summary']['psnr'] == pytest.approx(sum(float(f) for f in RUN) / len(RUN))


def test_each_sample_reports_its_subject_sequence_and_input_cameras():
    out, _, _ = run('per_frame')

    assert {(r['subject'], r['sequence']) for r in out['samples']} == {('S1', 'seq')}
    assert {tuple(r['input_cameras']) for r in out['samples']} == {('1', '2')}


@pytest.mark.parametrize('mode', ['block', 'nonsense'])
def test_only_per_frame_and_sliding_inference_are_benchmarked(mode):
    with pytest.raises(ValueError, match=mode):
        run(mode)


def test_a_window_longer_than_the_planned_context_is_refused():
    # the target camera was only kept off the inputs of the 4-frame context window
    with pytest.raises(ValueError, match='clip_length'):
        run('sliding', clip_length=5)
