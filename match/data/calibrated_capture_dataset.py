"""Generic calibrated multi-view captures (registry name `calibrated`).

A capture folder contains:
    images/frame{FFFFFF}_cam{CCC}.png
    opencv_ba.txt          calibration, see `camera_calibration.read_opencv_ba`

Foreground masks come from a separate folder (`mask_directory`) with the same
file names as the images.
"""
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from match.data import base_dataset, camera_calibration

logger = logging.getLogger(__name__)

IMAGE_NAME_PATTERN = re.compile(r'^frame(\d+)_cam(\d+)\.png$')
MASK_DIRECTORY_REQUIRED = ('The calibrated-capture dataset requires `mask_directory`: '
                           'a folder of foreground masks named like the images.')


class CalibratedCapture(base_dataset.MugsyCapture):
    """One capture folder = one subject and sequence. `mcd` holds the sequence id."""

    def __init__(self, folder_name: str, mask_dir, subject_id: str | None = None, sequence_id: str | None = None):
        super().__init__(mcd=sequence_id or folder_name, mct='', sid=subject_id or folder_name)
        self.folder = folder_name
        self.mask_dir = None if mask_dir is None else Path(mask_dir)

    def folder_name(self) -> str:
        return self.folder


class CalibratedSingleCaptureDataset(base_dataset.BaseSingleCaptureDataset):
    """A single calibrated capture. Its masks are read from `capture.mask_dir`.

    World normalization (per instance): a point X in calibration units becomes
    `scene_scale * scene_rotation @ X - scene_center`, which puts the head
    center at the origin.

    Args (besides the base class's):
        scale_factor: calibration units to meters.
        scene_center: head center in calibration units. Defaults to the point
            closest to all cameras' optical axes.
        scene_rotation: 3x3 rotation into MATCH's world axes (x toward the
            subject's left, y down, z toward the back of the head). Identity by default.
    """

    NUM_VERTS = 5741  # same as the Ava-256 dataset, so batches can mix both

    def __init__(self, capture: CalibratedCapture, scale_factor: float = 1 / 1000,
                 scene_center: list | None = None, scene_rotation: list | None = None, **kwargs):
        if capture.mask_dir is None:
            raise ValueError(MASK_DIRECTORY_REQUIRED)
        self.mask_dir = capture.mask_dir
        self.scale_factor = scale_factor
        self.scene_center_override = scene_center
        self.scene_rotation_override = scene_rotation
        super().__init__(capture=capture, **kwargs)

    def load_metadata(self):
        self.calibration = {str(i): c for i, c in camera_calibration.read_calibration(self.dir).items()}
        self._check_resize_target()
        self._load_frame_list()
        self._set_scene_normalization()

    def _check_resize_target(self):
        sizes = {(c.width, c.height) for c in self.calibration.values()}
        if len(sizes) != 1:
            raise ValueError(f'All cameras of a capture must share one native size; {self.dir} has {sorted(sizes)}')
        native_width, native_height = sizes.pop()
        aspect_preserving_width = self.height * native_width / native_height
        if abs(self.width - aspect_preserving_width) > 1:
            raise ValueError(f'height={self.height}, width={self.width} would change the native aspect ratio '
                             f'of {self.dir} ({native_width}x{native_height}); at this height use '
                             f'width={round(aspect_preserving_width)}')

    def _load_frame_list(self):
        image_names = defaultdict(dict)  # frame index -> camera id -> file name
        for path in (self.dir / 'images').iterdir():
            match = IMAGE_NAME_PATTERN.match(path.name)
            if match is None:
                continue
            frame, camera = int(match.group(1)), int(match.group(2))
            image_names[frame][str(camera)] = path.name
        frames = sorted(image_names)[::self.frame_stride]
        if self.nframes is not None and self.nframes > 0:
            frames = frames[:self.nframes]
        self.frame_ids = [f'{f:06d}' for f in frames]
        self.image_names = {f'{f:06d}': image_names[f] for f in frames}

    def _set_scene_normalization(self):
        if self.scene_center_override is None:
            self.head_center = optical_axes_center([c.extrin for c in self.calibration.values()])
        else:
            self.head_center = np.asarray(self.scene_center_override, dtype=np.float64).reshape(3)
        rotation = np.eye(3) if self.scene_rotation_override is None else np.asarray(self.scene_rotation_override)
        rotation = rotation.reshape(3, 3)
        self.SCENE_ROTATION = rotation.astype(np.float32)
        self.SCALE_FACTOR = np.array(self.scale_factor, dtype=np.float32)
        self.SCENE_CENTER = (self.scale_factor * rotation @ self.head_center).astype(np.float32)

    def get_krt_dict(self):
        krt = {}
        for cam, c in self.calibration.items():
            intrin = c.K.copy()
            intrin[:2, 2] += 0.5  # OpenCV puts the top-left pixel center at 0, MATCH at 0.5
            krt[cam] = dict(intrin=intrin, extrin=c.extrin, dist=c.dist)
        return krt

    @property
    def height_original(self) -> int:
        return next(iter(self.calibration.values())).height

    @property
    def width_original(self) -> int:
        return next(iter(self.calibration.values())).width

    def get_frameid(self, idx: int) -> str:
        return self.frame_ids[idx]

    def __len__(self):
        return len(self.frame_ids)

    def get_sequenceid(self, frame_id: str) -> str:
        return self.capture.mcd

    def load_headpose_and_verts(self, frame_id: str):
        """A fixed head pose at the head center; no ground-truth vertices (zeros)."""
        headpose = np.eye(4, dtype=np.float32)
        headpose[:3, 3] = self.head_center
        return headpose, np.zeros((self.NUM_VERTS, 3), dtype=np.float32)

    def image_path(self, frame_id: str, camera_id: str) -> Path:
        name = self.image_names[frame_id].get(camera_id)
        if name is None:
            raise FileNotFoundError(f'No image for frame {frame_id}, camera {camera_id} in {self.dir / "images"}')
        return self.dir / 'images' / name

    def mask_path(self, frame_id: str, camera_id: str) -> Path:
        return self.mask_dir / self.image_path(frame_id, camera_id).name

    def load_image(self, frame_id: str, camera_id: str) -> np.ndarray:
        """sRGB uint8 RGB (HxWx3) at native size; no color correction, alpha dropped."""
        path = self.image_path(frame_id, camera_id)
        with Image.open(path) as img:
            img = np.asarray(img.convert('RGB'))
        return self.to_pinhole(self._check_native_size(img, path), camera_id)

    def load_fg_mask(self, frame_id: str, camera_id: str) -> np.ndarray:
        """uint8 HxWx1, 0..255. Uses the first channel of multi-channel masks."""
        path = self.mask_path(frame_id, camera_id)
        if not path.is_file():
            raise FileNotFoundError(f'Missing foreground mask {path}')
        with Image.open(path) as mask:
            mask = np.asarray(mask.convert('L') if mask.mode == '1' else mask)
        if mask.dtype != np.uint8:
            raise ValueError(f'Foreground mask {path} must be 8-bit, got {mask.dtype}')
        mask = mask[..., :1] if mask.ndim == 3 else mask[..., None]
        return self.to_pinhole(self._check_native_size(mask, path), camera_id)

    def load_sg(self, frame_id: str, camera_id: str):
        return None  # no Sapiens part maps; configs set require_segmentation=False

    def to_pinhole(self, image: np.ndarray, camera_id: str) -> np.ndarray:
        """Maps a raw native-size image or mask to the calibration's pinhole camera.

        Runs before the base class's resize. Distortion is not removed yet, so
        this returns the input unchanged.
        """
        return image

    def _check_native_size(self, image: np.ndarray, path: Path) -> np.ndarray:
        if image.shape[:2] != (self.height_original, self.width_original):
            raise ValueError(f'{path} is {image.shape[1]}x{image.shape[0]}, but the calibration says '
                             f'{self.width_original}x{self.height_original}')
        return image


def optical_axes_center(extrinsics) -> np.ndarray:
    """Least-squares point closest to all cameras' optical axes, in calibration units."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for extrin in extrinsics:
        R, t = extrin[:, :3], extrin[:, 3]
        center, axis = -R.T @ t, R[2]
        P = np.eye(3) - np.outer(axis, axis)  # projects onto the plane orthogonal to the axis
        A += P
        b += P @ center
    return np.linalg.solve(A, b)


class CalibratedMultiCaptureDataset(base_dataset.BaseMultiCaptureDataset):
    """All captures under `root_path`.

    `root_path` is either a capture itself, or a folder whose subfolders that
    hold a calibration file are captures. Masks of a root capture are read
    from `mask_directory`; masks of a subfolder capture from
    `mask_directory/<subfolder name>`.

    Args (besides the base class's):
        mask_directory: see above. Required.
        subject_id, sequence_id: override the ids, which default to the capture folder name.
    """
    SINGLE_CAPTURE_DATASET_CLS = CalibratedSingleCaptureDataset

    def __init__(self, root_path: str, mask_directory: str | None = None, subject_id: str | None = None,
                 sequence_id: str | None = None, **kwargs):
        if mask_directory is None:
            raise ValueError(MASK_DIRECTORY_REQUIRED)
        self.mask_directory = Path(mask_directory)
        self.subject_id = subject_id
        self.sequence_id = sequence_id
        super().__init__(root_path=root_path, **kwargs)

    def folder_parser(self, base_dir, training: bool | None = None):
        """Finds the captures. `training` is accepted for interface compatibility and ignored."""
        base_dir = Path(base_dir)
        if camera_calibration.find_calibration_file(base_dir) is not None:
            dirs = [base_dir]
        else:
            dirs = sorted(d for d in base_dir.iterdir()
                          if d.is_dir() and camera_calibration.find_calibration_file(d) is not None)
        if not dirs:
            raise FileNotFoundError(f'No capture found in {base_dir}: neither it nor any subfolder '
                                    f'contains a calibration file.')
        root_is_capture = dirs == [base_dir]
        captures = [CalibratedCapture(d.name,
                                      mask_dir=self.mask_directory if root_is_capture else self.mask_directory / d.name,
                                      subject_id=self.subject_id, sequence_id=self.sequence_id)
                    for d in dirs]
        return captures, [str(d) for d in dirs]


class CalibratedCaptureDataLoader(base_dataset.ParentDataLoader):
    DATASET_CLS = CalibratedMultiCaptureDataset
