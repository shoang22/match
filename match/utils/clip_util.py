"""Clips: consecutive frames of one sequence stacked on a time axis, for clip training and windowed inference."""
from typing import Any, Dict, List

import torch


def stack_frames(frames: List[Dict[str, Any]], time_dim: int) -> Dict[str, Any]:
    """Combines consecutive frames into one clip: tensors stacked on a new time axis at `time_dim`, `frame` as the
    list of the frames' values (in time order), other values (subject, sequence) taken from the first frame.

    time_dim 0 for single-frame dataset samples (time axis first, `frame` a list of IDs); 1 for batches (B, ...),
    which gives the layout `default_collate` makes of clip samples (tensors (B, T, ...), `frame` T lists of B IDs).
    The result has the first frame's type (dict, MatchBatch).
    """
    first = frames[0]
    clip = type(first)()
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            clip[k] = torch.stack([f[k] for f in frames], dim=time_dim)
        elif k == 'frame':
            clip[k] = [f[k] for f in frames]
        else:
            clip[k] = v
    return clip
