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

# Output folder of each dataset type, where it differs from the type name (existing Ava-256 output paths)
OUTPUT_FOLDERS = {
    'ava256': 'ava-256',
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
        segment=getattr(args, 'segment', None),
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


def sample_output_dirs(batch, specs, outdir) -> list:
    """`<outdir>/<dataset folder>/<subject>/<sequence>/<frame>` for each sample of a collated `MultiDataset` batch,
    where the folder is named after the type of the spec whose dataset produced the sample."""
    folders = [OUTPUT_FOLDERS.get(specs[int(i)]['type'], specs[int(i)]['type']) for i in batch['dataset_idx']]
    return [Path(outdir) / folder / str(subject) / str(sequence) / str(frame)
            for folder, subject, sequence, frame in zip(folders, batch['subject'], batch['sequence'], batch['frame'])]
