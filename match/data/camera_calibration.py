"""Reading per-camera calibrations of generic multi-view captures.

A capture folder holds one calibration file. `read_calibration` picks the
reader by file name; to support another format, add a reader to
`CALIBRATION_READERS`.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """One camera, in the calibration's own units.

    `K` uses OpenCV's pixel convention: the center of the top-left pixel is (0, 0).
    """
    K: np.ndarray       # (3, 3)
    dist: np.ndarray    # (5,) OpenCV k1, k2, p1, p2, k3
    extrin: np.ndarray  # (3, 4) [R | t], maps world to camera
    width: int
    height: int


def quaternion_to_rotation(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_opencv_ba(path) -> list[CameraCalibration]:
    """Reads `opencv_ba.txt`: one line per camera in camera-index order, `#` lines are comments.

    Each camera line: fu u0 v0 ar s | k1 k2 p1 p2 k3 | qw qx qy qz tx ty tz | width height.
    """
    cameras = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        values = line.split()
        if len(values) != 19:
            raise ValueError(f'{path}:{line_no}: expected 19 values per camera, got {len(values)}')
        fu, u0, v0, ar, skew, k1, k2, p1, p2, k3, qw, qx, qy, qz, tx, ty, tz = map(float, values[:17])
        width, height = int(values[17]), int(values[18])
        if skew != 0:
            raise ValueError(f'{path}:{line_no}: camera {len(cameras)} has nonzero skew {skew}; '
                             'skewed intrinsics are not supported')
        K = np.array([[fu, 0., u0], [0., fu * ar, v0], [0., 0., 1.]])
        extrin = np.concatenate([quaternion_to_rotation(qw, qx, qy, qz), [[tx], [ty], [tz]]], axis=1)
        cameras.append(CameraCalibration(K=K, dist=np.array([k1, k2, p1, p2, k3]), extrin=extrin,
                                         width=width, height=height))
    if not cameras:
        raise ValueError(f'{path}: no camera lines found')
    return cameras


CALIBRATION_READERS = {'opencv_ba.txt': read_opencv_ba}


def find_calibration_file(capture_dir) -> Path | None:
    for name in CALIBRATION_READERS:
        path = Path(capture_dir) / name
        if path.is_file():
            return path
    return None


def read_calibration(capture_dir) -> dict[int, CameraCalibration]:
    """Returns {camera index: calibration} for the calibration file in `capture_dir`."""
    path = find_calibration_file(capture_dir)
    if path is None:
        raise FileNotFoundError(f'No calibration file ({", ".join(CALIBRATION_READERS)}) in {capture_dir}')
    return dict(enumerate(CALIBRATION_READERS[path.name](path)))
