"""Dataset registration and per-sample output routing for TEMPEH's `save_predictions`.

A dataset spec is a dict: `type` names a MATCH dataset (`match.data.get_dataset_cls`), every other key is
passed to that dataset's constructor (root_path, camera_angles, height, width, head_crop*, frame_stride, ...).
"""
import ast
from pathlib import Path

import numpy as np

# Constructor kwargs a dataset type gets unless its spec sets them.
TYPE_DEFAULTS = {
    'nersemble': dict(invalid_captures_path='assets/nersemble/invalid_captures.txt'),
}

def _literal(value, default):
    if value is None:
        return default
    return ast.literal_eval(value) if isinstance(value, str) else value


def dataset_specs_from_args(args) -> list:
    """Dataset specs configured in TEMPEH's options.

    Without a `datasets` list, this is the legacy setup: Ava-256 from the `dataset_directory` options, plus
    NeRSemble when `nersemble_dataset_directory` is set.
    """
    configured = _literal(getattr(args, 'datasets', None), None)
    if configured:
        return [{**TYPE_DEFAULTS.get(spec['type'], {}), **spec} for spec in configured]

    specs = [dict(
        type='ava256',
        root_path=args.dataset_directory,
        camera_angles=_literal(args.camera_angles, None),
        height=args.height, width=args.width,
        frame_stride=args.frame_stride,
        max_captures=args.max_captures,
        only_subjects=_literal(args.subjects, []),
    )]
    nersemble_root = getattr(args, 'nersemble_dataset_directory', None)
    if nersemble_root is not None:
        specs.append(dict(
            **TYPE_DEFAULTS['nersemble'],
            type='nersemble',
            root_path=nersemble_root,
            camera_angles=_literal(getattr(args, 'nersemble_camera_angles', None), []),
            height=getattr(args, 'nersemble_height', None), width=getattr(args, 'nersemble_width', None),
            frame_stride=getattr(args, 'nersemble_frame_stride', None),
            max_captures=getattr(args, 'nersemble_max_captures', None),
            only_subjects=_literal(getattr(args, 'nersemble_subjects', None), []),
        ))
    return specs


def build_datasets(specs, process_idx=0, world_size=1) -> list:
    """One MATCH multi-capture dataset per spec, set up for inference (no GT vertices or segmentations)."""
    from match.data import get_dataset_cls
    datasets = list()
    for spec in specs:
        kwargs = dict(deterministic_shuffle=False, require_verts=False, require_segmentation=False)
        kwargs.update({k: v for k, v in spec.items() if k != 'type'})
        kwargs.update(training=None, process_idx=process_idx, world_size=world_size)
        datasets.append(get_dataset_cls(spec['type'])(**kwargs))
    return datasets


def to_dataset_frame(points_mm, batch):
    """TEMPEH predictions (B, N, 3), in mm of MATCH's normalized frame, mapped back to the frame of the dataset
    each sample came from, using that sample's `scene_rotation`, `scene_center` and `scene_scale`."""
    from match.utils import geo_util

    def as_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

    return geo_util.apply_inv_rotation_scale_center_to_points(
        np.asarray(points_mm) / 1000.0,
        rotation=as_numpy(batch['scene_rotation']),
        scale=as_numpy(batch['scene_scale']),
        center=as_numpy(batch['scene_center']))


def _producing_datasets(batch, datasets) -> list:
    """For each sample of a collated `MultiDataset` batch, the dataset that produced it."""
    return [datasets[int(i)] for i in batch['dataset_idx']]


def sample_output_dirs(batch, datasets, outdir) -> list:
    """`<outdir>/<dataset type>/<subject>/<sequence>/<frame>` for each sample of the batch."""
    return [Path(outdir) / ds.DATASET_TYPE / str(subject) / str(sequence) / str(frame)
            for ds, subject, sequence, frame in zip(
                _producing_datasets(batch, datasets), batch['subject'], batch['sequence'], batch['frame'])]



def crop_boxes_in_full_image(batch, datasets):
    """Where each view's (head-cropped) input image lies in the full (resized, uncropped) image of its dataset:
    (B, V, 4) as left, top, right, bottom, normalized by the full image size."""
    bboxs = np.asarray(batch['bboxs'], dtype=np.float64)  # in full-image pixels
    full = np.array([(ds.width, ds.height) for ds in _producing_datasets(batch, datasets)], dtype=np.float64)
    return bboxs / np.tile(full, 2)[:, None, :]


def uv_frame_fxfycxcy(batch, datasets):
    """Intrinsics to render each sample's per-camera UV maps with, or None to keep TEMPEH's input frame.

    MATCH loads a UV map by resizing it to the full (resized, uncropped) image of its dataset, then applies its own
    head crop. A head-cropped TEMPEH input therefore needs its UV maps rendered in the full image: this returns the
    full-image intrinsics (B, V, 4) as normalized fx, fy, cx, cy. Without any head crop in the batch, TEMPEH's input
    is already the full image, so None.
    """
    crop_h, crop_w = batch['image'].shape[-2:]
    if np.all(np.asarray(batch['bboxs']) == [0, 0, crop_w, crop_h]):
        return None
    fxfycxcy = np.asarray(batch['fxfycxcy'], dtype=np.float64)  # normalized by the input (crop) size
    left, top, right, bottom = np.moveaxis(crop_boxes_in_full_image(batch, datasets), -1, 0)
    width, height = right - left, bottom - top
    return np.stack([
        fxfycxcy[..., 0] * width,
        fxfycxcy[..., 1] * height,
        fxfycxcy[..., 2] * width + left,
        fxfycxcy[..., 3] * height + top,
    ], axis=-1).astype(np.float32)
