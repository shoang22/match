"""Tests for the camera-sweep helpers shared by the sweep and sequence-eval scripts.

Run from the repo root: conda run -n match python -m pytest tests/test_camera_sweep_util.py
"""
import json
import subprocess

import numpy as np
import torch
from PIL import Image

from match.utils import camera_sweep_util


def test_sweep_extrinsics_match_the_quickstart_sweep():
    # Expected values were produced by the original helper in
    # scripts/match/save_camera_sweeps.py, with the quickstart ranges.
    C2W = camera_sweep_util.generate_camera_sweep_extrinsics(
        vert_range_deg=[-15, 36], horiz_range_deg=[-40, 40], num_steps=200)

    assert C2W.shape == (200, 4, 4)
    np.testing.assert_allclose(C2W[0], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(C2W[50, :3], [
        [0.7661003592, 0.112683719, -0.6327658485, 0.6327658485],
        [0.0, 0.9845109813, 0.175322924, -0.175322924],
        [0.6427209656, -0.134314955, 0.7542342164, -0.7542342164]], atol=1e-9)
    np.testing.assert_allclose(C2W[199], C2W[0], atol=1e-9)


def test_sweep_cameras_look_at_the_origin_from_the_given_radius():
    C2W = camera_sweep_util.generate_camera_sweep_extrinsics(num_steps=30, radius=2.)

    rotations, positions = C2W[:, :3, :3], C2W[:, :3, 3]
    np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), np.broadcast_to(np.eye(3), (30, 3, 3)),
                               atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(positions, axis=-1), 2., atol=1e-9)
    # Camera +z (viewing direction) points from the camera towards the origin.
    np.testing.assert_allclose(C2W[:, :3, 2], -positions / 2., atol=1e-9)


def test_sweep_intrinsics_scale_the_first_input_view_focal_and_center_the_principal_point():
    # (B=2, V=2, 4) normalized fx, fy, cx, cy. Only sample 0, view 0 is used.
    fxfycxcy = torch.tensor([
        [[1.2, 1.6, 0.48, 0.52], [9., 9., 9., 9.]],
        [[7., 7., 7., 7.], [8., 8., 8., 8.]]], dtype=torch.float64)

    intrinsics = camera_sweep_util.sweep_intrinsics(fxfycxcy)

    assert intrinsics.dtype == torch.float32
    torch.testing.assert_close(intrinsics, torch.tensor([0.9, 1.2, 0.5, 0.5]))


def _probe_video(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames,pix_fmt', '-of', 'json',
          str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)['streams'][0]


def _write_frames(frames_dir, n=7):
    frames_dir.mkdir()
    for i in range(n):
        Image.new('RGB', (64, 48), color=(30 * i, 0, 0)).save(frames_dir / f'{i:03d}.jpg')


def test_frames_to_video_encodes_every_frame_at_the_given_fps_and_removes_the_frames(tmp_path):
    frames_dir = tmp_path / 'frames'
    _write_frames(frames_dir)
    video_path = tmp_path / 'video.mp4'

    camera_sweep_util.frames_to_video(frames_dir, video_path, fps=10)

    stream = _probe_video(video_path)
    assert (stream['width'], stream['height']) == (64, 48)
    assert stream['r_frame_rate'] == '10/1'
    assert stream['nb_read_frames'] == '7'
    # ffmpeg's default for JPEG input, as the quickstart camera sweep has always been encoded.
    assert stream['pix_fmt'] == 'yuvj420p'
    assert not frames_dir.exists()


def test_frames_to_video_can_encode_a_widely_playable_pixel_format(tmp_path):
    frames_dir = tmp_path / 'frames'
    _write_frames(frames_dir)
    video_path = tmp_path / 'video.mp4'

    camera_sweep_util.frames_to_video(frames_dir, video_path, fps=10, pix_fmt='yuv420p')

    stream = _probe_video(video_path)
    assert stream['pix_fmt'] == 'yuv420p'
    assert stream['nb_read_frames'] == '7'

