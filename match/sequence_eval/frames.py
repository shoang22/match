"""Frame loading for the sequence evaluation run, with clear errors instead of silent skips.

The dataset's `__getitem__` silently moves on to the next frame when a frame fails to load, and
a held-out camera whose files are missing is silently replaced by the nearest other camera. Here every
frame of the configured frame list is loaded by ID, and the run stops with an error when a held-out
camera is incomplete, when the dataset picks other extra cameras than the configured held-out IDs, when
the fixed-view input camera of a flicker-only run (no held-out cameras) isn't an input, or when a frame
can't be loaded.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from match.utils import data_util


class HeldOutCameraError(RuntimeError):
    """A held-out camera's images, masks or calibration are missing, or the dataset didn't pick it."""


class FixedViewCameraError(RuntimeError):
    """The configured fixed-view input camera of a flicker-only run isn't one of the inputs on some frame."""


class FrameLoadError(RuntimeError):
    """A frame of the evaluated sequence can't be loaded."""


def single_capture(dataset):
    """The one capture of a multi-capture dataset (the run evaluates one capture's segment)."""
    captures = list(dataset.single_capture_datasets.values())
    if len(captures) != 1:
        names = [str(c.dir) for c in captures]
        raise ValueError(f'The sequence evaluation needs exactly one capture, the dataset has {len(captures)}: {names}. '
                         "Set 'only_subjects' in the dataset kwargs.")
    capture = captures[0]
    if len(capture) == 0:
        raise ValueError(f'The frame list of {capture.dir} is empty for the configured segment, stride and nframes.')
    return capture


def frame_ids(capture):
    """The frame IDs to evaluate, in order: the capture's frame list after the segment, stride and nframes filters."""
    return [str(capture.get_frameid(i)) for i in range(capture.num_frames())]


def segment_ids(capture):
    """The sequences (Ava-256 segments; a calibrated capture is one sequence) the evaluated frames belong to."""
    return sorted({str(capture.get_sequenceid(f)) for f in frame_ids(capture)})


def closest_to_frontal(capture, camera_ids):
    """The camera whose position is closest in angle to straight in front of the subject (az = el = 0).

    Dataset coordinates have frontal cameras at -z (see data_util.sample_cameras_from_angles).
    """
    def cos_to_frontal(camera):
        position = np.asarray(capture.campos[camera], dtype=np.float64)
        return -position[2] / np.linalg.norm(position)
    return max(camera_ids, key=cos_to_frontal)


def _zip_members(path):
    with zipfile.ZipFile(path) as z:
        return set(z.namelist())


def check_heldout_cameras(capture, heldout_camera_ids, frames):
    """Raises HeldOutCameraError unless every held-out camera has calibration, an image and a mask for every frame.

    Reads Ava-256's calibration and zips; with no held-out cameras (a flicker-only run) it reads nothing.
    """
    if not heldout_camera_ids:
        return
    with open(capture.krt_file_path) as f:
        calibrated = {str(c['cameraId']) for c in json.load(f)['KRT']}

    problems = []
    for camera in heldout_camera_ids:
        if camera not in calibrated:
            problems.append(f'camera {camera}: calibration missing (no entry in {capture.krt_file_path})')
        assets = [
            ('image', capture.camera_zip_path(camera), lambda frame: f'cam{camera}/{int(frame):06d}.avif'),
            ('mask', Path(capture.dir) / 'foreground_masks' / f'cam{camera}.zip', lambda frame: f'{int(frame):06d}.avif'),
        ]
        for asset, zip_path, member in assets:
            try:
                members = _zip_members(zip_path)
            except (FileNotFoundError, zipfile.BadZipFile) as e:
                problems.append(f'camera {camera}: {asset} zip missing or unreadable ({zip_path}: {e})')
                continue
            missing = [frame for frame in frames if member(frame) not in members]
            if missing:
                problems.append(f'camera {camera}: {asset} missing for frames {missing} in {zip_path}')
    if problems:
        raise HeldOutCameraError('Held-out cameras are incomplete:\n  ' + '\n  '.join(problems))


def check_fixed_view_input_camera(capture, camera, frames):
    """Raises FixedViewCameraError unless the dataset picks `camera` as an input on every frame."""
    camera = str(camera)
    if camera not in capture.cameras:
        raise FixedViewCameraError(f'The fixed-view input camera {camera} is not a camera of {capture.dir}')
    not_input = []
    for frame in frames:
        try:
            if camera not in capture.sample_cameras_from_camera_angles(frame):
                not_input.append(frame)
        except data_util.SkippableError as e:
            raise FrameLoadError(f'frame {frame} of {capture.dir} has no input cameras: {e}') from e
    if not_input:
        raise FixedViewCameraError(f'The fixed-view input camera {camera!r} is not an input on frames {not_input} '
                                   f'of {capture.dir}')


class SequenceFrames(torch.utils.data.Dataset):
    """Item i is the sample of the i-th evaluated frame, with exactly the configured held-out cameras as extra cameras.

    With no held-out cameras (a flicker-only run) samples have no 'extra_*' keys, and `fixed_view_input_camera`
    must be an input on every frame. Raises instead of retrying another frame, so it is safe to use in a DataLoader.
    """

    def __init__(self, capture, heldout_camera_ids, fixed_view_input_camera=None):
        self.capture = capture
        self.heldout_camera_ids = [str(c) for c in heldout_camera_ids]
        self.fixed_view_input_camera = None if fixed_view_input_camera is None else str(fixed_view_input_camera)
        self.frame_ids = frame_ids(capture)

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, i):
        capture, frame = self.capture, self.frame_ids[i]
        try:
            inputs = capture.sample_cameras_from_camera_angles(frame)
            used_as_input = [c for c in self.heldout_camera_ids if c in inputs]
            if used_as_input:
                raise HeldOutCameraError(f'frame {frame}: held-out cameras {used_as_input} were picked as inputs {inputs}')
            if self.fixed_view_input_camera is not None and self.fixed_view_input_camera not in inputs:
                raise FixedViewCameraError(f'frame {frame}: the fixed-view input camera {self.fixed_view_input_camera} '
                                           f'is not one of the inputs {inputs}')
            extras = capture.get_extra_camera_ids(frame, avoid_camera_ids=inputs)
            if extras != self.heldout_camera_ids:
                raise HeldOutCameraError(
                    f'frame {frame}: the dataset picked extra cameras {extras} for extra_camera_angles '
                    f'{capture.extra_camera_angles}, expected the held-out cameras {self.heldout_camera_ids}')
            self._check_input_uvs(frame, inputs)
            sample = capture.fetch_data_from_disk(frame_id=frame, camera_ids=inputs, extra_camera_ids=extras)
        except data_util.SkippableError as e:
            raise FrameLoadError(f'frame {frame} of {capture.dir} failed to load: {e}') from e
        extra_cameras = [str(c) for c in sample['extra_camera'].tolist()] if 'extra_camera' in sample else []
        if sample['frame'] != frame or extra_cameras != self.heldout_camera_ids:
            raise FrameLoadError(f'expected frame {frame} with extra cameras {self.heldout_camera_ids}, '
                                 f"got frame {sample['frame']} with {extra_cameras}")
        return sample

    def _check_input_uvs(self, frame, inputs):
        """The dataset silently zero-fills missing TEMPEH UV maps, which would corrupt the model input."""
        capture = self.capture
        if capture.uv_directory is None:
            return
        frame_dir = Path(capture.uv_directory, capture.sid, capture.get_sequenceid(frame), frame)
        missing = [c for c in inputs if not (frame_dir / f'uv_cam{c}.png').exists()]
        if missing:
            raise FrameLoadError(f'frame {frame}: TEMPEH UV maps missing for input cameras {missing} in {frame_dir}')
