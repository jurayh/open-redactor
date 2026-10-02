import numpy as np

from open_redactor.masks import (
    build_per_frame_masks,
    carry_forward,
    fill_single_frame_gaps,
    merge_masks,
    pad_mask,
    smooth_mask_temporal,
)


def test_pad_mask_expands():
    m = np.zeros((20, 20), dtype=bool)
    m[10, 10] = True
    padded = pad_mask(m, margin=2)
    assert padded[10, 10]
    assert padded[10, 12]
    assert padded[12, 10]
    # Diagonal corners are filled by the separable dilation
    assert padded[12, 12]
    assert not padded[10, 13]


def test_carry_forward_extends_after_last():
    frames = {0: np.ones((4, 4), dtype=bool)}
    out = carry_forward(frames, total_frames=10, carry_frames=3)
    assert 1 in out and 2 in out and 3 in out
    assert 4 not in out


def test_fill_single_frame_gap():
    a = np.zeros((4, 4), dtype=bool)
    a[1, 1] = True
    frames = {0: a, 2: a.copy()}
    filled = fill_single_frame_gaps(frames, max_gap=1)
    assert 1 in filled
    assert np.array_equal(filled[1], a)


def test_fill_does_not_fill_long_gap():
    a = np.ones((4, 4), dtype=bool)
    frames = {0: a, 5: a.copy()}
    filled = fill_single_frame_gaps(frames, max_gap=1)
    assert 1 not in filled


def test_smooth_keeps_coverage():
    m0 = np.zeros((5, 5), dtype=bool)
    m0[2, 2] = True
    masks = [m0, m0.copy(), m0.copy()]
    smoothed = smooth_mask_temporal(masks, radius=1)
    assert smoothed[1][2, 2]


def test_build_per_frame_zero_tracks():
    merged = build_per_frame_masks(tracks={}, total_frames=3, shape=(8, 8), margin=2, carry_frames=2)
    assert len(merged) == 3
    assert not np.any(merged[0])


def test_merge_masks_empty_needs_shape():
    merged = merge_masks([], shape=(4, 4))
    assert merged.shape == (4, 4)
    assert not np.any(merged)
