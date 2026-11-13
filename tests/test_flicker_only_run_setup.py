"""Flicker-only sequence evaluation (#28): the panels and metrics a run builds, with and without held-out cameras.

Imports scripts/match/eval_sequence.py, which pulls in the runner (~12 s). CPU, no model, no data.
Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_flicker_only_run_setup.py
"""
from types import SimpleNamespace

import pytest

from match.sequence_eval import metrics as seq_metrics
from match.sequence_eval.context import SequencePlan
from tce049 import ASSETS
import eval_sequence_script

TEMPLATE_MESH = ASSETS / 'ava256/face_topology_cleaned.obj'


eval_sequence = eval_sequence_script.load()


def plan(heldout_camera_ids, fixed_view_camera):
    sequence = SimpleNamespace(heldout_camera_ids=heldout_camera_ids, frame_ids=['1', '2', '3'])
    return SequencePlan(sequence, schedule=[0, 1, 1, 2], fixed_view_camera=fixed_view_camera)


FLICKER_ONLY = plan([], '401875')
HELDOUT = plan(['402040', '401905'], '402040')


@pytest.mark.skipif(not TEMPLATE_MESH.is_file(), reason='assets/ava256 not on disk')
@pytest.mark.parametrize('run_plan, real_label', [
    (HELDOUT, 'Real held-out (cam 402040)'),
    (FLICKER_ONLY, 'Real input (cam 401875)'),
])
def test_the_same_four_panels_and_the_real_panel_says_which_kind_of_camera_it_shows(run_plan, real_label):
    camera = run_plan.fixed_view_camera
    panels = eval_sequence.build_panels(run_plan, SimpleNamespace(template_mesh_path=str(TEMPLATE_MESH)))
    assert [p.label for p in panels] == [f'MATCH prediction (cam {camera})', real_label,
                                         f'TEMPEH coarse mesh (cam {camera})', 'Orbit']


def test_a_flicker_only_run_scores_flicker_and_jitter_but_not_image_quality():
    kinds = [type(c) for c in eval_sequence.build_metric_collectors(FLICKER_ONLY)]
    assert kinds == [seq_metrics.TemporalMetrics, seq_metrics.GeometricJitter]


def test_a_run_with_heldout_cameras_also_scores_image_quality():
    kinds = [type(c) for c in eval_sequence.build_metric_collectors(HELDOUT)]
    assert kinds == [seq_metrics.ImageMetrics, seq_metrics.TemporalMetrics, seq_metrics.GeometricJitter]


def test_a_run_without_heldout_cameras_needs_a_fixed_view_input_camera(tmp_path):
    with pytest.raises(ValueError, match='fixed_view_input_camera'):
        eval_sequence.eval_sequence(None, out_dir=str(tmp_path / 'out'), heldout_camera_ids=[])
    assert not (tmp_path / 'out').exists()


def test_a_fixed_view_input_camera_is_only_for_runs_without_heldout_cameras(tmp_path):
    with pytest.raises(ValueError, match='fixed_view_input_camera'):
        eval_sequence.eval_sequence(None, out_dir=str(tmp_path / 'out'), heldout_camera_ids=['402040'],
                                    fixed_view_input_camera='401875')
    assert not (tmp_path / 'out').exists()
