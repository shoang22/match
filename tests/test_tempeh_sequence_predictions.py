"""TEMPEH predictions for every frame of the SEN_a_good_morrow_to_you_my_boy segment of TCE049 (#6).

Two seams:
- the config: its arguments, parsed the way scripts/tempeh/save_predictions.py parses them, select the
  segment's 40 frames with the quickstart's cameras and sizes, and write to their own folder;
- the output folder after the run: 40 frames, each with verts.npy and one UV map per input camera.

Run from the repo root (real-data tests skip without data/ and assets/):
  conda run -n match python -m pytest tests/test_tempeh_sequence_predictions.py
"""
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

CONFIG = Path('configs/tempeh/save_predictions_seq_SEN_a_good_morrow.yml')
QUICKSTART_CONFIG = Path('configs/tempeh/save_predictions_quickstart.yml')
TEMPEH_CONFIG = Path('assets/pretrained_models/tempeh/config.json')
SEGMENT = 'SEN_a_good_morrow_to_you_my_boy'
# The segment's frames in the 4TB frame list (#3): 32885 + 12k for k = 0..39.
SEGMENT_FRAMES = [str(32885 + 12 * k) for k in range(40)]
# The 12 input cameras confirmed by the held-out-cameras ticket (#4), in camera-angle order.
INPUT_CAMERA_IDS = ['401875', '401919', '402020', '402012', '401920', '401965',
                    '401630', '401916', '401860', '402038', '401966', '401969']

needs_data = pytest.mark.skipif(
    not (Path('data/ava-256').exists() and TEMPEH_CONFIG.exists()),
    reason='needs data/ava-256 and the pretrained TEMPEH config (symlink data/ and assets/ into a worktree)',
)


def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def parse_script_args(monkeypatch, config):
    """Parses the config's arguments with the TEMPEH options parser, as the prediction script does."""
    monkeypatch.syspath_prepend('third_party/TEMPEH')
    from option_handler.train_options_global import TrainOptions
    monkeypatch.setattr(sys, 'argv', ['save_predictions.py', *shlex.split(config['arguments'])])
    return TrainOptions().parse()


def test_header_comment_gives_the_run_command():
    first_line = CONFIG.read_text().splitlines()[0]
    assert first_line.startswith('#') and f'condor/run.py {CONFIG.as_posix()}' in first_line


@needs_data
def test_config_writes_to_its_own_folder(monkeypatch):
    args = parse_script_args(monkeypatch, load_config(CONFIG))
    quickstart_args = parse_script_args(monkeypatch, load_config(QUICKSTART_CONFIG))
    out, quickstart_out = Path(args.outpath), Path(quickstart_args.outpath)
    assert out != quickstart_out
    assert quickstart_out not in out.parents and out not in quickstart_out.parents


@needs_data
def test_config_keeps_the_quickstart_cameras_and_sizes_with_no_sample_cap(monkeypatch):
    args = parse_script_args(monkeypatch, load_config(CONFIG))
    quickstart_args = parse_script_args(monkeypatch, load_config(QUICKSTART_CONFIG))
    for name in ['camera_angles', 'height', 'width', 'uv_out_height', 'uv_out_width', 'config_filename']:
        assert getattr(args, name) == getattr(quickstart_args, name), name
    assert args.uv_n_samples == -1


@needs_data
def test_config_selects_every_frame_of_the_segment(monkeypatch):
    args = parse_script_args(monkeypatch, load_config(CONFIG))
    assert args.segment == SEGMENT and args.frame_stride == 1
    from trainer.global_trainer import Trainer
    trainer = Trainer(args, torch.device('cpu'))
    trainer.register_dataset(init_full_dataset=True)
    (dataset,) = trainer.dataset_full.datasets
    (capture,) = dataset.single_capture_datasets.values()
    assert capture.framelist.seg_id.unique().tolist() == [SEGMENT]
    assert sorted(capture.framelist.frame_id.astype(str).tolist(), key=int) == SEGMENT_FRAMES


def segment_output_dir():
    if not CONFIG.exists():
        return Path('missing-config')
    out_dir = Path(load_config(CONFIG)['out_dir'])
    return out_dir / 'ava-256' / 'TCE049' / SEGMENT


needs_outputs = pytest.mark.skipif(
    not segment_output_dir().is_dir(),
    reason=f'run the TEMPEH predictions first: condor/run.py {CONFIG.as_posix()}',
)


@needs_outputs
def test_outputs_exist_for_every_frame_of_the_segment():
    frames = sorted((p.name for p in segment_output_dir().iterdir() if p.is_dir()), key=int)
    assert frames == SEGMENT_FRAMES


@needs_outputs
@pytest.mark.parametrize('frame', SEGMENT_FRAMES)
def test_each_frame_has_coarse_mesh_and_one_uv_map_per_input_camera(frame):
    frame_dir = segment_output_dir() / frame
    verts = np.load(frame_dir / 'verts.npy')
    assert verts.shape == (5741, 3) and np.isfinite(verts).all()
    uv_cameras = sorted(p.name[len('uv_cam'):-len('.png')] for p in frame_dir.glob('uv_cam*.png'))
    assert uv_cameras == sorted(INPUT_CAMERA_IDS)
