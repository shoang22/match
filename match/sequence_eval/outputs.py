"""Output writing of the sequence evaluation run: video timing, grid video, metrics JSON and config copies."""
import json
import math
from fractions import Fraction
from pathlib import Path

import gin
import numpy as np
from PIL import Image

from match.utils.camera_sweep_util import frames_to_video

GRID_VIDEO = 'grid.mp4'
PREVIEW_VIDEO = 'preview.mp4'  # the grid with each source frame shown once, at the preview fps
METRICS_JSON = 'metrics.json'
VIDEO_PIX_FMT = 'yuv420p'  # limited-range 4:2:0, which every player decodes (ffmpeg would pick yuvj420p for JPEGs)


def source_frame_times(frame_ids, capture_fps):
    """Start time (s, exact Fraction) of every source frame relative to the first, and the hold of the last one.

    Frame IDs count recorded frames at capture_fps, so frame i starts at (id_i - id_0) / capture_fps. The last
    frame is held for the median spacing of the sequence.
    """
    ids = [int(f) for f in frame_ids]
    if len(ids) < 2:
        raise ValueError(f'Need at least 2 frames to time the video, got {frame_ids}')
    if any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError(f'Frame IDs must increase: {frame_ids}')
    capture_fps = Fraction(capture_fps)
    starts = [Fraction(i - ids[0]) / capture_fps for i in ids]
    last_hold = Fraction(int(np.median(np.diff(ids)))) / capture_fps
    return starts, last_hold


def output_frame_schedule(frame_ids, fps, capture_fps):
    """Index of the source frame shown in every output frame of a video at fps.

    Output frame k (at time k/fps) shows the latest source frame that started at or before that time, so each
    source frame is held for its true duration rounded to whole output frames, and the total duration is the
    sequence's true duration to within half an output frame. Nothing is interpolated.
    """
    starts, last_hold = source_frame_times(frame_ids, capture_fps)
    fps = Fraction(fps)
    n_out = round((starts[-1] + last_hold) * fps)
    schedule, i = [], 0
    for k in range(n_out):
        t = Fraction(k) / fps
        while i + 1 < len(starts) and starts[i + 1] <= t:
            i += 1
        schedule.append(i)
    shown = set(schedule)
    if len(shown) != len(starts):
        missing = [frame_ids[i] for i in range(len(starts)) if i not in shown]
        raise ValueError(f'Source frames {missing} are shorter than one output frame at {fps} fps')
    return schedule


class GridVideoWriter:
    """Collects grid frames as JPEGs and encodes them into a video. Frames must be written for every index."""

    def __init__(self, out_dir, n_frames, fps, name=GRID_VIDEO):
        self.video_path = Path(out_dir) / name
        self.frames_dir = Path(out_dir) / f'{Path(name).stem}_frames'
        self.n_frames = n_frames
        self.fps = fps
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def write(self, index, image):
        """image: (H,W,3) uint8."""
        Image.fromarray(image).save(self.frames_dir / f'{index:06d}.jpg', quality=95)

    def finish(self):
        written = sorted(self.frames_dir.glob('*.jpg'))
        if len(written) != self.n_frames:
            raise RuntimeError(f'{self.video_path}: expected {self.n_frames} frames, got {len(written)} in {self.frames_dir}')
        frames_to_video(self.frames_dir, self.video_path, fps=self.fps, pix_fmt=VIDEO_PIX_FMT)
        return self.video_path


def copy_configs(out_dir, config_files):
    """Copies the gin files passed on the command line as config_<i>.gin and every binding as config_all.gin."""
    out_dir = Path(out_dir)
    for i, path in enumerate(config_files):
        (out_dir / f'config_{i}.gin').write_text(Path(path).read_text())
    (out_dir / 'config_all.gin').write_text(gin.config_str())


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # 'inf' / 'nan': plain JSON has no infinities
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def write_metrics_json(out_dir, metrics):
    path = Path(out_dir) / METRICS_JSON
    path.write_text(json.dumps(_json_safe(metrics), indent=2))
    return path
