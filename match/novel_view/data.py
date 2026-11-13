"""What the novel-view benchmark reads from a capture: each frame's input cameras when its inputs are complete,
the target cameras with data for a frame, and a frame's batch loaded by ID.

Frames are loaded by ID with errors instead of the dataset's silent retry of the next index, and a frame
without a TEMPEH prediction is refused rather than zero-filled.
"""
import json
from functools import lru_cache

import torch

from match.novel_view.samples import FrameInputs
from match.sequence_eval.frames import FrameLoadError, camera_frames_on_disk, frame_ids, missing_tempeh
from match.utils import data_util

__all__ = ['FrameLoadError', 'frame_inputs', 'target_cameras_on_disk', 'load_frame']


def frame_inputs(capture) -> list[FrameInputs]:
    """Every frame of the capture's frame list, with its input cameras, or None when its inputs are incomplete: no
    valid input cameras, or no TEMPEH prediction (coarse mesh or an input camera's UV map)."""
    out = []
    for frame in frame_ids(capture):
        sequence = str(capture.get_sequenceid(frame))
        inputs = None
        if 'ALL' not in capture.invalid_frame_cameras[frame]:
            try:
                cameras = [str(c) for c in capture.sample_cameras_from_camera_angles(frame)]
            except data_util.SkippableError:
                cameras = None
            if cameras is not None and not missing_tempeh(capture, frame, cameras):
                inputs = tuple(cameras)
        out.append(FrameInputs(frame=frame, sequence=sequence, inputs=inputs))
    return out


def target_cameras_on_disk(capture, camera_ids):
    """(subject, frame) -> the cameras of camera_ids (in that order) that have calibration, an image and a foreground
    mask for the frame and aren't known invalid on it (Ava-256 zips). Whether one is an input is left to the plan."""
    with open(capture.krt_file_path) as f:
        calibrated = {str(c['cameraId']) for c in json.load(f)['KRT']}
    cameras = [str(c) for c in camera_ids if str(c) in calibrated and str(c) in capture.cameras]
    all_frames = frame_ids(capture)

    @lru_cache(maxsize=None)
    def frames_of(camera):
        return camera_frames_on_disk(capture, camera, all_frames)

    def targets(subject, frame):
        invalid = capture.invalid_frame_cameras[frame]
        return [c for c in cameras if c not in invalid and frame in frames_of(c)]
    return targets


def load_frame(capture, frame, target_camera=None, device=None) -> data_util.MatchBatch:
    """The frame's batch (B=1) with its input cameras, and the target camera as the 'extra_*' view unless None.

    Raises FrameLoadError if the target camera is an input, the TEMPEH prediction is missing, or the frame fails to load.
    """
    try:
        inputs = [str(c) for c in capture.sample_cameras_from_camera_angles(frame)]
        if target_camera is not None and str(target_camera) in inputs:
            raise FrameLoadError(f'frame {frame}: the target camera {target_camera} is an input {inputs}')
        missing = missing_tempeh(capture, frame, inputs)
        if missing:
            raise FrameLoadError(f'frame {frame}: TEMPEH prediction missing: {missing}')
        extras = [] if target_camera is None else [str(target_camera)]
        sample = capture.fetch_data_from_disk(frame_id=frame, camera_ids=inputs, extra_camera_ids=extras)
    except data_util.SkippableError as e:
        raise FrameLoadError(f'frame {frame} of {capture.dir} failed to load: {e}') from e
    batch = data_util.MatchBatch(torch.utils.data.default_collate([sample]))
    if device is not None:
        batch.to_device(device)
    return batch
