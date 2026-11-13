"""TEMPEH predictions at stride 1 for the Ava-256 training subjects (#34).

Seam: the config's arguments, parsed the way scripts/tempeh/save_predictions.py parses them and turned into
datasets by TEMPEH's dataset registry. They must select every frame of every training subject on disk, never a
held-out subject, with the quickstart's cameras and sizes, and write to their own folder.

The run itself needs the training subjects on disk (none are, as of #34). Real-data tests skip without data/
and assets/:
  conda run -n match python -m pytest tests/test_tempeh_training_predictions.py
"""
import csv
import shlex
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tce049 import ASSETS, AVA_ROOT, DECODER_DIR, REPO_ROOT

CONFIG = Path('configs/tempeh/save_predictions_train_stride1.yml')  # as named in its header
OTHER_CONFIGS = [REPO_ROOT / 'configs/tempeh/save_predictions_quickstart.yml',
                 REPO_ROOT / 'configs/tempeh/save_predictions_seq_SEN_a_good_morrow.yml']
TEMPEH_CONFIG = ASSETS / 'pretrained_models/tempeh/config.json'

needs_data = pytest.mark.skipif(
    not (AVA_ROOT.exists() and TEMPEH_CONFIG.exists()),
    reason='needs data/ava-256 and the pretrained TEMPEH config (symlink data/ and assets/ into a worktree)',
)


def load_config(path, **overrides):
    config = OmegaConf.load(path)
    for key, value in overrides.items():
        assert key in config, key
        config[key] = value
    return OmegaConf.to_container(config, resolve=True)


def parse_script_args(monkeypatch, config):
    """Parses the config's arguments with the TEMPEH options parser, as the prediction script does."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / 'third_party/TEMPEH'))
    from option_handler.train_options_global import TrainOptions
    monkeypatch.setattr(sys, 'argv', ['save_predictions.py', *shlex.split(config['arguments'])])
    return TrainOptions().parse()


def dataset_specs(monkeypatch, config):
    from gtempeh_utils import dataset_registry
    return dataset_registry.dataset_specs_from_args(parse_script_args(monkeypatch, config))


def test_header_comment_gives_the_run_command():
    first_line = (REPO_ROOT / CONFIG).read_text().splitlines()[0]
    assert first_line.startswith('#') and f'condor/run.py {CONFIG.as_posix()}' in first_line


@needs_data
def test_config_writes_to_its_own_folder(monkeypatch):
    out = Path(parse_script_args(monkeypatch, load_config(REPO_ROOT / CONFIG)).outpath)
    for other in OTHER_CONFIGS:
        other_out = Path(parse_script_args(monkeypatch, load_config(other)).outpath)
        assert out != other_out and other_out not in out.parents and out not in other_out.parents, other


@needs_data
def test_config_keeps_the_quickstart_cameras_and_sizes_with_no_sample_cap(monkeypatch):
    args = parse_script_args(monkeypatch, load_config(REPO_ROOT / CONFIG))
    quickstart = parse_script_args(monkeypatch, load_config(OTHER_CONFIGS[0]))
    (spec,) = dataset_specs(monkeypatch, load_config(REPO_ROOT / CONFIG))
    assert spec['type'] == 'ava256'
    assert spec['camera_angles'] == [[0, 0], [0, -15], [20, 0], [20, 36], [20, -15], [-20, 0], [-20, 36],
                                     [-20, -15], [40, -5], [40, 25], [-40, -5], [-40, 25]]
    assert (spec['height'], spec['width']) == (256, 167)
    for name in ['uv_out_height', 'uv_out_width', 'config_filename']:
        assert getattr(args, name) == getattr(quickstart, name), name
    assert args.uv_n_samples == -1


@needs_data
def test_config_takes_every_frame_of_every_segment(monkeypatch):
    (spec,) = dataset_specs(monkeypatch, load_config(REPO_ROOT / CONFIG))
    assert spec['frame_stride'] == 1
    assert spec.get('segment') is None and not spec.get('only_subjects') and not spec.get('max_captures')


def ava_root_with_a_training_and_a_heldout_subject(tmp_path):
    """An Ava-256 root holding TCE049 (held out, in assets/ava256/validation_ids.json) and the same capture
    filed under AAN112, a training subject from assets/ava256/256_ids.csv."""
    for name in ['20230807--0839--TCE049', '20230405--1635--AAN112']:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'decoder').symlink_to(DECODER_DIR)
    return tmp_path


@needs_data
def test_config_selects_every_frame_of_the_training_subjects_and_no_heldout_subject(monkeypatch, tmp_path):
    root = ava_root_with_a_training_and_a_heldout_subject(tmp_path)
    specs = dataset_specs(monkeypatch, load_config(REPO_ROOT / CONFIG, data_dir=str(root)))
    from gtempeh_utils import dataset_registry

    (dataset,) = dataset_registry.build_datasets(specs)

    (capture,) = dataset.captures
    assert capture.sid == 'AAN112'
    with open(DECODER_DIR / 'frame_list.csv') as f:
        frames = [row['frame_id'] for row in csv.DictReader(f)]
    assert len(dataset) == len(frames) == 5024
    assert dataset.single_capture_datasets[capture].framelist.frame_id.tolist() == frames
