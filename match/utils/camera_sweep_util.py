"""Camera-sweep helpers: sweep extrinsics, sweep intrinsics, and frames-to-video.

Shared by scripts/match/save_camera_sweeps.py and the sequence evaluation run.
"""
import math
import shutil

import ffmpeg
import numpy as np
import torch

# Default sweep path of scripts/match/save_camera_sweeps.py: angle ranges in degrees, within the views the
# model handles, and the focal-length scale of the sweep intrinsics.
SWEEP_HORIZ_RANGE_DEG = (-40, 40)
SWEEP_VERT_RANGE_DEG = (-15, 36)
SWEEP_FOCAL_SCALE = 0.75


def ellipse(xmin, xmax, ymin, ymax, sigma):
    """
    Returns a point (x, y) on the edge of an ellipse specified by
    x_range = (xmin, xmax)
    y_range = (ymin, ymax)
    sigma in [0, 1], where 0 and 1 correspond to the same position.
    """

    # Ellipse center
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0

    # Radii
    a = (xmax - xmin) / 2.0
    b = (ymax - ymin) / 2.0

    # Angle from sigma
    theta = 2 * np.pi * sigma

    # Edge point
    x = xc + a * np.cos(theta)
    y = yc + b * np.sin(theta)

    return x, y


def generate_camera_sweep_extrinsics(
    vert_range_deg=(-10, 10),   # (min, max) vertical tilt (x-axis rotation)
    horiz_range_deg=(-30, 30), # (min, max) horizontal pan (y-axis rotation)
    expand_periods = 0.5,
    reduce_periods = 0.5,
    full_periods = 1,
    num_steps=20,
    radius = 1.,
    center = [0., 0., 0.],
    up = [0., -1., 0.],

):
    """
    Produce a set of extrinsics for a camera sweep around the reference view.
    Returns camera-to-world matrices as a numpy array of shape (N, 4, 4).
    """
    all_periods = expand_periods + full_periods + reduce_periods
    steps_per_period = num_steps / all_periods
    expand_steps = round(steps_per_period*expand_periods)
    full_steps = round(steps_per_period*full_periods)
    reduce_steps = num_steps - expand_steps - full_steps
    center = np.array(center)
    up = np.array(up)

    radius_ratios = np.concatenate([
        np.sin(np.linspace(0, np.pi/2, expand_steps)),
        np.ones((full_steps)),
        np.cos(np.linspace(0, np.pi/2, reduce_steps))])

    t = np.linspace(0, all_periods, num_steps)
    angles_h, angles_v = ellipse(xmin=math.radians(horiz_range_deg[0])*radius_ratios,
                                 xmax=math.radians(horiz_range_deg[1])*radius_ratios,
                                 ymin=math.radians(vert_range_deg[0])*radius_ratios,
                                 ymax=math.radians(vert_range_deg[1])*radius_ratios,
                                 sigma=t,
                                 )

    # camera locations
    y_cam = -radius * np.sin(angles_v)
    rc = -radius * np.cos(angles_v)
    x_cam = rc * np.sin(angles_h)
    z_cam = rc * np.cos(angles_h)
    cam_pos = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # camera rotations:
    ax_z_cam = center[None] - cam_pos
    ax_x_cam = np.cross(ax_z_cam, up[None])
    ax_y_cam = np.cross(ax_z_cam, ax_x_cam)
    ax_x_cam /= np.linalg.norm(ax_x_cam, axis=-1, keepdims=True)
    ax_y_cam /= np.linalg.norm(ax_y_cam, axis=-1, keepdims=True)
    ax_z_cam /= np.linalg.norm(ax_z_cam, axis=-1, keepdims=True)

    C2W = np.repeat(np.eye(4)[None], num_steps, axis=0)
    C2W[:, :3, 0] = ax_x_cam
    C2W[:, :3, 1] = ax_y_cam
    C2W[:, :3, 2] = ax_z_cam
    C2W[:, :3, 3] = cam_pos
    return C2W


def sweep_intrinsics(fxfycxcy, focal_scale=SWEEP_FOCAL_SCALE):
    """
    Intrinsics for the camera sweep: the focal length of the first input view
    of the first sample, scaled by focal_scale, with the principal point at the
    image center.
    fxfycxcy: normalized intrinsics of shape (B, V, 4), e.g. batch['fxfycxcy'].
    Returns a float32 tensor [fx, fy, cx, cy] of shape (4,) on the same device.
    """
    fx = fxfycxcy[0, 0, 0].item() * focal_scale
    fy = fxfycxcy[0, 0, 1].item() * focal_scale
    return torch.tensor([fx, fy, 0.5, 0.5], dtype=torch.float32, device=fxfycxcy.device)


def frames_to_video(frames_dir, video_path, fps, pix_fmt=None):
    """
    Encodes all *.jpg files in frames_dir (in file-name order) into a video at
    video_path with the given frame rate, overwriting an existing file. Deletes
    frames_dir afterwards.
    pix_fmt: output pixel format, e.g. 'yuv420p' for players that don't handle
    the full-range yuvj420p that ffmpeg picks for JPEG input by default (None).
    """
    output_kwargs = {} if pix_fmt is None else {'pix_fmt': pix_fmt}
    (
      ffmpeg
      .input(f'{frames_dir}/*.jpg', pattern_type='glob', framerate=fps)
      .output(f'{video_path}', **output_kwargs)
      .overwrite_output()
      .run()
    )
    shutil.rmtree(frames_dir)
