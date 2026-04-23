"""Edge-case coverage for the layer-mapping helpers."""

from __future__ import annotations

from medh5.meta import SpatialMeta

from napari_medh5._layers import (
    _nnunet_labels,
    _spatial_kwargs,
    _spatial_to_affine,
)


def test_affine_none_when_direction_shape_mismatches() -> None:
    s = SpatialMeta(direction=[[1.0, 0.0], [0.0, 1.0]])  # 2x2 but ndim=3
    assert _spatial_to_affine(s, 3) is None


def test_spatial_kwargs_with_non_identity_affine() -> None:
    s = SpatialMeta(
        spacing=[2.0, 1.0, 1.0],
        origin=[1.0, 1.0, 1.0],
        direction=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    kwargs = _spatial_kwargs(s, 3)
    assert "affine" in kwargs
    assert "scale" not in kwargs


def test_spatial_kwargs_empty_when_no_metadata() -> None:
    assert _spatial_kwargs(SpatialMeta(), 3) == {}


def test_nnunet_labels_skips_non_dict_labels() -> None:
    assert _nnunet_labels({"nnunetv2": {"labels": "not-a-dict"}}) == {}


def test_nnunet_labels_drops_non_integer_values() -> None:
    mapping = _nnunet_labels(
        {"nnunetv2": {"labels": {"a": 0, "b": "not-int", "c": None}}}
    )
    assert mapping == {0: "a"}
