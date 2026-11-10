"""Tests for the camera-sweep helpers shared by the sweep and sequence-eval scripts.

Run from the repo root: conda run -n match python -m pytest tests/test_camera_sweep_util.py
"""
import numpy as np

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
  np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), np.broadcast_to(np.eye(3), (30, 3, 3)), atol=1e-9)
  np.testing.assert_allclose(np.linalg.norm(positions, axis=-1), 2., atol=1e-9)
  # Camera +z (viewing direction) points from the camera towards the origin.
  np.testing.assert_allclose(C2W[:, :3, 2], -positions / 2., atol=1e-9)
