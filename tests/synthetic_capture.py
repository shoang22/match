"""Writes a tiny calibrated multi-view capture to disk for dataset tests.

The rig copies the camera directions of the real 12-camera target capture: a
horizontal arc plus a frontal vertical column, every optical axis passing
through a known head center. All values here are ground truth, chosen
independently of the dataset code.

Pixel convention on disk: OpenCV, i.e. the center of the top-left pixel is at
(0, 0).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# (azimuth, elevation) in degrees of each camera as seen from the head, in
# MATCH's world convention: x toward the subject's left, y down, z toward the
# back of the head.
RIG_ANGLES_DEG = [
    (1.0, 0.2), (19.7, 0.5), (-17.2, -1.3), (32.3, 4.2), (33.4, -9.3), (-26.4, 1.1),
    (-26.1, -13.2), (-0.3, 18.3), (0.2, 9.4), (1.4, -12.4), (1.8, -21.9), (2.5, -30.0),
]
RIG_DISTANCES_MM = [966, 1028, 1047, 1146, 1181, 1100, 1123, 952, 965, 944, 891, 864]
HEAD_CENTER_MM = np.array([-17.0, 4.0, 966.0])
NATIVE_WIDTH = 200
NATIVE_HEIGHT = 160
# The red, green and blue image markers, in calibration units (mm).
MARKERS_MM = HEAD_CENTER_MM + np.array([[30.0, -20.0, -40.0], [-40.0, 30.0, 10.0], [10.0, 45.0, -60.0]])
MARKER_SIGMA_PX = 1.5

# The pretrained MATCH model's 12 target angles [az, el].
AVA_CAMERA_ANGLES = [[0, 0], [0, -15], [20, 0], [20, 36], [20, -15], [-20, 0], [-20, 36],
                     [-20, -15], [40, -5], [40, 25], [-40, -5], [-40, 25]]


@dataclass
class SyntheticCamera:
    K: np.ndarray       # 3x3, OpenCV pixel convention
    R: np.ndarray       # world -> camera rotation
    t: np.ndarray       # world -> camera translation (mm)
    aspect_ratio: float

    @property
    def center(self):
        return -self.R.T @ self.t

    def project(self, points_mm):
        cam = (self.R @ np.asarray(points_mm, dtype=np.float64).T).T + self.t
        uvw = (self.K @ cam.T).T
        return uvw[:, :2] / uvw[:, 2:]


@dataclass
class SyntheticCapture:
    root: Path
    mask_dir: Path
    cameras: dict = field(default_factory=dict)  # int camera index -> SyntheticCamera
    frames: list = field(default_factory=list)   # int frame indices written


def make_rig(skew_camera=None):
    cameras = {}
    for i, ((az, el), dist) in enumerate(zip(RIG_ANGLES_DEG, RIG_DISTANCES_MM)):
        az, el = np.deg2rad(az), np.deg2rad(el)
        direction = np.array([np.sin(az) * np.cos(el), -np.sin(el), -np.cos(az) * np.cos(el)])
        center = HEAD_CENTER_MM + dist * direction
        z = -direction
        y = np.array([0.0, 1.0, 0.0]) - z[1] * z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        R = np.stack([x, y, z])
        t = -R @ center
        fu = 470.0 + 5 * i
        ar = 1.0 + 0.002 * (i % 3 - 1)
        u0 = NATIVE_WIDTH / 2 + 6 * (i % 3 - 1) + 0.3
        v0 = NATIVE_HEIGHT / 2 + 4 * (i % 4 - 1.5)
        skew = 1.5 if i == skew_camera else 0.0
        K = np.array([[fu, skew, u0], [0, fu * ar, v0], [0, 0, 1]])
        cameras[i] = SyntheticCamera(K=K, R=R, t=t, aspect_ratio=ar)
    return cameras


def write_calibration(path, cameras):
    lines = ['# fu u0 v0 ar s | k1 k2 p1 p2 k3 | quaternion(scalar part first) translation | width height | ',
             '# reprojection error: 0.0']
    for i in sorted(cameras):
        cam = cameras[i]
        qx, qy, qz, qw = Rotation.from_matrix(cam.R).as_quat()
        values = [cam.K[0, 0], cam.K[0, 2], cam.K[1, 2], cam.aspect_ratio, cam.K[0, 1],
                  0, 0, 0, 0, 0,
                  qw, qx, qy, qz, *cam.t]
        lines.append(' '.join(f'{v:.9f}' for v in values) + f' {NATIVE_WIDTH} {NATIVE_HEIGHT}')
    Path(path).write_text('\n'.join(lines) + '\n')


def gaussian_blob(center_xy, sigma=MARKER_SIGMA_PX):
    xs = np.arange(NATIVE_WIDTH)[None, :]
    ys = np.arange(NATIVE_HEIGHT)[:, None]
    return np.exp(-((xs - center_xy[0]) ** 2 + (ys - center_xy[1]) ** 2) / (2 * sigma ** 2))


def marker_image(camera):
    uv = camera.project(MARKERS_MM)
    img = np.stack([gaussian_blob(p) for p in uv], axis=-1) * 255
    return np.round(img).astype(np.uint8)


def marker_mask(camera, mode):
    """Mask containing only the red marker, in one of several file formats."""
    blob = gaussian_blob(camera.project(MARKERS_MM[:1])[0])
    channel, kind = mode.split('-')
    first = np.round(blob * 255) if kind == 'soft' else np.where(blob > 0.2, 255, 0)
    first = first.astype(np.uint8)
    if channel == 'L':
        return Image.fromarray(first, mode='L')
    other = np.full_like(first, 255)  # other channels differ from the first one on purpose
    return Image.fromarray(np.stack([first, other, 255 - first], axis=-1), mode='RGB')


def write_capture(root, mask_dir, nframes=2, missing_images=(), missing_masks=(), skew_camera=None,
                  mask_mode='L-soft', flat_color=None, alpha=None, cameras=None):
    """Writes images, masks and `opencv_ba.txt`.

    Args:
        missing_images / missing_masks: (frame, camera) pairs to leave out.
        flat_color: RGB triple; if given, every image is that color with no markers.
        alpha: if given, images are RGBA with this constant alpha.
    """
    root, mask_dir = Path(root), Path(mask_dir)
    (root / 'images').mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    cameras = cameras if cameras is not None else make_rig(skew_camera=skew_camera)
    write_calibration(root / 'opencv_ba.txt', cameras)
    for frame in range(nframes):
        for i, cam in cameras.items():
            name = f'frame{frame:06d}_cam{i:03d}.png'
            if flat_color is not None:
                img = np.empty((NATIVE_HEIGHT, NATIVE_WIDTH, 3), np.uint8)
                img[:] = flat_color
            else:
                img = marker_image(cam)
            if alpha is not None:
                img = np.concatenate([img, np.full_like(img[..., :1], alpha)], axis=-1)
            if (frame, i) not in missing_images:
                Image.fromarray(img).save(root / 'images' / name)
            if (frame, i) not in missing_masks:
                marker_mask(cam, mask_mode).save(mask_dir / name)
    return SyntheticCapture(root=root, mask_dir=mask_dir, cameras=cameras, frames=list(range(nframes)))
