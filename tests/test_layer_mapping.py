"""Direct tests for the layer-mapping helpers (no napari viewer needed)."""

from __future__ import annotations

import numpy as np
from medh5.meta import SpatialMeta

from napari_medh5._layers import (
    _nnunet_labels,
    _seg_display_name,
    _spatial_to_affine,
)


def test_affine_identity_falls_back() -> None:
    s = SpatialMeta(
        spacing=[1.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        direction=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert _spatial_to_affine(s, 3) is None


def test_affine_non_identity_returns_matrix() -> None:
    direction = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    s = SpatialMeta(
        spacing=[2.0, 1.0, 1.0],
        origin=[10.0, 20.0, 30.0],
        direction=direction,
    )
    affine = _spatial_to_affine(s, 3)
    assert affine is not None
    assert affine.shape == (4, 4)
    assert np.allclose(affine[:3, 3], [10.0, 20.0, 30.0])
    # First column should be the spacing-scaled first column of direction.
    assert np.allclose(affine[:3, 0], np.array([0.0, 2.0, 0.0]))


def test_nnunet_label_extraction() -> None:
    extra = {"nnunetv2": {"labels": {"background": 0, "tumor": 1, "vessel": "2"}}}
    mapping = _nnunet_labels(extra)
    assert mapping == {0: "background", 1: "tumor", 2: "vessel"}


def test_seg_display_name_prefers_nnunet_class() -> None:
    mapping = {0: "background", 1: "tumor"}
    assert _seg_display_name("1", mapping) == "tumor"
    assert _seg_display_name("tumor", mapping) == "tumor"
    assert _seg_display_name("other", mapping) == "other"


def test_no_extra_returns_empty_map() -> None:
    assert _nnunet_labels(None) == {}
    assert _nnunet_labels({}) == {}
    assert _nnunet_labels({"nnunetv2": "not-a-dict"}) == {}
