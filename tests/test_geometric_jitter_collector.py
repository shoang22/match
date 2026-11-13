"""The sequence run's geometric-jitter collector: CPU tests on synthetic per-texel positions.

The collector sees one FrameContext per source frame, whose masked Gaussians carry the final texel positions
('xyz') and the TEMPEH part of them ('tempeh_xyz'), in metres. It returns the 'jitter' section of
metrics.json, in mm, for the TEMPEH part, the offset (final minus TEMPEH part) and the final position.
"""
import math

import pytest
import torch

from match.sequence_eval.context import FrameContext
from match.sequence_eval.metrics import GeometricJitter, TexelRegion

L = 300  # masked texels
FRAMES = ['32885', '32897', '32909', '32921']
SOURCES = ['tempeh', 'offset', 'final']


def surface(seed=0):
    """(3, L) texel positions of a head-sized surface, in metres."""
    return 0.1 * torch.randn(3, L, generator=torch.Generator().manual_seed(seed))


def unit_directions(seed=1):
    d = torch.randn(3, L, generator=torch.Generator().manual_seed(seed))
    return d / d.norm(dim=0, keepdim=True)


def masked(x):
    """(3, L) -> the masked-Gaussian layout (1, 1, 3, L, 1)."""
    return x[None, None, :, :, None]


def contexts(tempeh, final, frames=FRAMES):
    """One FrameContext per frame with the TEMPEH part tempeh[i] and the final positions final[i], (3, L) each."""
    return [FrameContext(runner=None, frame_index=i, frame_id=frame, batch=None,
                         gaussians={'xyz': masked(final[i]), 'tempeh_xyz': masked(tempeh[i])},
                         heldout={}, fixed_view_camera=None, output_frame_indices=[i])
            for i, frame in enumerate(frames)]


def collect(ctxs, region=None):
    collector = GeometricJitter(region)
    for ctx in ctxs:
        collector.add_frame(ctx)
    return collector.finish()


def tempeh_jumping_in_frame_1(final_offset=None):
    """TEMPEH surface still except frame 1, where every texel is 1 mm off; final = TEMPEH + offset."""
    tempeh = [surface().clone() for _ in FRAMES]
    tempeh[1] += 0.001 * unit_directions()
    offset = 0.005 * unit_directions(seed=2) if final_offset is None else final_offset
    return tempeh, [t + offset for t in tempeh]


def test_jitter_section_has_every_source_per_middle_frame_in_mm():
    sections = collect(contexts(*tempeh_jumping_in_frame_1()))

    assert list(sections) == ['jitter']
    jitter = sections['jitter']
    assert jitter['units'] == 'mm'
    # Each triple of consecutive frames is keyed by its middle frame and names the frames around it.
    assert list(jitter['per_frame']) == FRAMES[1:-1]
    for previous, frame, following in zip(FRAMES, FRAMES[1:], FRAMES[2:]):
        entry = jitter['per_frame'][frame]
        assert entry['previous_frame'] == previous and entry['next_frame'] == following
        assert all(isinstance(entry[s], float) and math.isfinite(entry[s]) for s in SOURCES), entry


def test_a_constant_offset_carries_the_tempeh_jitter_into_the_final_position():
    # Frame 1 is 1 mm off: the triple centred on it sees 2 mm, the one centred on frame 2 sees 1 mm.
    per_frame = collect(contexts(*tempeh_jumping_in_frame_1()))['jitter']['per_frame']

    for frame, expected in [('32897', 2.0), ('32909', 1.0)]:
        assert per_frame[frame]['tempeh'] == pytest.approx(expected, rel=1e-3)
        assert per_frame[frame]['final'] == pytest.approx(expected, rel=1e-3)
        assert per_frame[frame]['offset'] == pytest.approx(0.0, abs=1e-3)


def test_an_offset_that_cancels_the_tempeh_jitter_leaves_a_still_final_position():
    tempeh, _ = tempeh_jumping_in_frame_1()
    final = [surface(seed=5)] * len(FRAMES)
    per_frame = collect(contexts(tempeh, final))['jitter']['per_frame']

    for frame, expected in [('32897', 2.0), ('32909', 1.0)]:
        assert per_frame[frame]['tempeh'] == pytest.approx(expected, rel=1e-3)
        assert per_frame[frame]['offset'] == pytest.approx(expected, rel=1e-3)
        assert per_frame[frame]['final'] == 0.0


def test_summary_has_the_mean_and_the_worst_frames_of_every_source():
    jitter = collect(contexts(*tempeh_jumping_in_frame_1()))['jitter']

    for source in SOURCES:
        summary = jitter['summary'][source]
        values = {f: jitter['per_frame'][f][source] for f in FRAMES[1:-1]}
        assert summary['mean'] == pytest.approx(sum(values.values()) / 2)
        assert [w['value'] for w in summary['worst_frames']] == sorted(values.values(), reverse=True)
    assert [w['frame'] for w in jitter['summary']['tempeh']['worst_frames']] == ['32897', '32909']


def test_two_frames_have_no_triples():
    jitter = collect(contexts(*tempeh_jumping_in_frame_1(), frames=FRAMES[:2]))['jitter']
    assert jitter['per_frame'] == {}
    for source in SOURCES:
        assert math.isnan(jitter['summary'][source]['mean'])
        assert jitter['summary'][source]['worst_frames'] == []


def test_jitter_outside_the_texel_region_does_not_count_and_the_region_is_recorded():
    # The first 100 texels are the region (e.g. face skin); only texels outside it jump in frame 1.
    in_region = torch.zeros(L, dtype=torch.bool)
    in_region[:100] = True
    tempeh = [surface().clone() for _ in FRAMES]
    tempeh[1][:, ~in_region] += 0.001 * unit_directions()[:, ~in_region]
    final = [t + 0.005 for t in tempeh]

    jitter = collect(contexts(tempeh, final), region=TexelRegion('fitting_region_skin', in_region))['jitter']

    assert all(entry[s] == pytest.approx(0., abs=1e-4) for entry in jitter['per_frame'].values() for s in SOURCES)
    assert jitter['texels'] == {'region': 'fitting_region_skin', 'n_texels': 100, 'n_masked_texels': L}


def test_without_a_region_every_masked_texel_counts():
    jitter = collect(contexts(*tempeh_jumping_in_frame_1()))['jitter']
    assert jitter['texels'] == {'region': 'all', 'n_texels': L, 'n_masked_texels': L}
