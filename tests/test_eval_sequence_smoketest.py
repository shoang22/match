"""What the smoke run's videos show (#8 to #12): panel timing, the orbit, the TEMPEH mesh's alignment and the preview.

The output contract shared with the full run (timing, metrics, plots, config copies) is in
test_eval_sequence_outputs.py. Skipped until the smoke run has been done (run command in the header of
configs/match/eval_sequence_smoketest.gin); EVAL_SEQUENCE_SMOKE_DIR checks a smoke run written elsewhere.
"""
import numpy as np
import pytest

from eval_sequence_runs import GRID_COLS, LABEL_H, PANEL_H, PANEL_W, PANELS, SMOKE, decode_video, load_metrics

PREDICTION_LABEL, REAL_LABEL, TEMPEH_LABEL, ORBIT_LABEL = PANELS
STATIC_LABELS = [PREDICTION_LABEL, REAL_LABEL, TEMPEH_LABEL]


@pytest.fixture(scope='module')
def metrics():
    return load_metrics(SMOKE)


@pytest.fixture(scope='module')
def video_frames(metrics):
    return decode_video(SMOKE.out_dir / metrics['video']['file'])


@pytest.fixture(scope='module')
def preview_frames(metrics):
    return decode_video(SMOKE.out_dir / metrics['preview']['file'])


def panel_frames(metrics, video_frames, label):
    """The labelled tile (label band + image) of the named panel in every video frame, by its grid position."""
    r, c = divmod(metrics['video']['panels'].index(label), GRID_COLS)
    th = LABEL_H + PANEL_H
    return video_frames[:, r * th:(r + 1) * th, c * PANEL_W:(c + 1) * PANEL_W]


def frame_changes(tile):
    """Mean absolute change of the panel image into each next video frame, in 0-255 levels."""
    return np.abs(np.diff(tile[:, LABEL_H:], axis=0)).mean(axis=(1, 2, 3))


def codec_noise(metrics, video_frames):
    """The largest change of the prediction panel within a source frame's hold, where the image repeats."""
    source = metrics['output_frames']
    change = frame_changes(panel_frames(metrics, video_frames, PREDICTION_LABEL))
    return max(c for k, c in enumerate(change) if source[k + 1] == source[k])


def silhouette(image):
    """Foreground of a panel image on a white background."""
    return image.min(axis=-1) < 200


def test_every_panel_has_its_label_drawn(metrics, video_frames):
    for label in metrics['video']['panels']:
        assert panel_frames(metrics, video_frames, label)[:, :LABEL_H].max() > 200, f"panel '{label}' has no label text"


def test_static_panels_hold_still_between_source_frames(metrics, video_frames):
    source = metrics['output_frames']
    for label in STATIC_LABELS:
        change = frame_changes(panel_frames(metrics, video_frames, label))  # change into frame k+1
        within_hold = [c for k, c in enumerate(change) if source[k + 1] == source[k]]
        at_new_frame = [c for k, c in enumerate(change) if source[k + 1] != source[k]]
        # Held frames are repeats (codec noise only), never blends; static panels all switch at the same output frames.
        assert max(within_hold) < 0.5, (label, max(within_hold))
        assert min(at_new_frame) > 2 * max(within_hold), (label, at_new_frame, max(within_hold))


def test_orbit_camera_moves_smoothly_on_every_output_frame(metrics, video_frames):
    change = frame_changes(panel_frames(metrics, video_frames, ORBIT_LABEL))
    # The orbit never holds still, even within a source frame's hold ...
    assert min(change) > 4 * codec_noise(metrics, video_frames), change
    # ... and moves in small steps without jumps.
    assert max(change) < 3 * np.median(change), change


def test_orbit_goes_round_once_over_the_whole_video(metrics, video_frames):
    tile = panel_frames(metrics, video_frames, ORBIT_LABEL)
    orbit, steps = tile[:, LABEL_H:], frame_changes(tile)
    from_start = np.abs(orbit - orbit[0]).mean(axis=(1, 2, 3))
    # The last frame is one step before the frontal start pose, so the orbit closes when the video loops ...
    assert from_start[-1] < 1.5 * np.median(steps), (from_start[-1], np.median(steps))
    # ... and never gets back to the start pose within the video, where only the face would differ (~2.5 levels),
    # so the orbit isn't repeated or cut mid-loop.
    assert min(from_start[1:]) > 0.5 * np.median(steps), (from_start, np.median(steps))


def test_tempeh_mesh_lines_up_with_the_head_in_the_prediction_and_real_panels(metrics, video_frames):
    source = metrics['output_frames']
    first_of_each_frame = [source.index(f) for f in SMOKE.frames]
    for k in first_of_each_frame:
        mesh, pred, real = (silhouette(panel_frames(metrics, video_frames, label)[k, LABEL_H:])
                            for label in [TEMPEH_LABEL, PREDICTION_LABEL, REAL_LABEL])
        assert 0.05 < mesh.mean() < 0.8, f'output frame {k}: mesh covers {mesh.mean():.2f} of the panel'
        # The coarse mesh has no hair or shoulders, so it sits inside the head rather than matching its outline.
        # Aligned, 99.7% of it lies on the head; shifting it by 20 px drops that to ~97%.
        for name, head in [('prediction', pred), ('real', real)]:
            inside = (mesh & head).sum() / mesh.sum()
            assert inside > 0.98, f'output frame {k}: {inside:.3f} of the mesh lies on the {name} head'


def test_preview_frames_are_grid_frames_not_blends(metrics, video_frames, preview_frames):
    first_shown = [metrics['output_frames'].index(f) for f in SMOKE.frames]  # grid frame where each source frame starts
    for i, k in enumerate(first_shown):
        same = np.abs(preview_frames[i] - video_frames[k]).mean()
        others = [np.abs(preview_frames[i] - video_frames[j]).mean() for j in first_shown if j != k]
        # Codec noise only: ~1.05 levels with all 4 panels (both videos are h264-encoded from the same image),
        # against ~6-8 between different source frames; a 50/50 blend of neighbours would sit near 3.
        assert same < 2.0, (i, same)
        assert same < 0.25 * min(others), (i, same, others)
