"""Edge-case coverage for the bbox <-> Shapes translation."""

from __future__ import annotations

import numpy as np
import pytest

from napari_medh5._bbox import (
    _cuboid_wires,
    _rectangle_in_plane,
    arrays_to_shapes,
    shapes_to_arrays,
)


def test_empty_bboxes_yields_no_layers() -> None:
    assert (
        arrays_to_shapes(
            bboxes=None,
            scores=None,
            labels=None,
            spatial_kwargs={},
            sample_shape=[8, 8, 8],
            path="x.medh5",
            stem="x",
        )
        == []
    )
    assert (
        arrays_to_shapes(
            bboxes=np.empty((0, 3, 2)),
            scores=None,
            labels=None,
            spatial_kwargs={},
            sample_shape=[8, 8, 8],
            path="x.medh5",
            stem="x",
        )
        == []
    )


def test_rectangle_in_plane_rejects_too_few_axes() -> None:
    with pytest.raises(ValueError):
        _rectangle_in_plane(np.array([[0.0, 1.0]]), ndim=1, depth_axis=0, depth_value=0)


def test_cuboid_wires_returns_empty_for_non_3d() -> None:
    assert _cuboid_wires(np.array([[0.0, 1.0], [0.0, 1.0]]), ndim=2) == []


def test_shapes_to_arrays_empty_input() -> None:
    assert shapes_to_arrays([], "rectangle", None, ndim=3, sample_shape=[8, 8, 8]) == (
        None,
        None,
        None,
    )


def test_shapes_to_arrays_ignores_invalid_shapes() -> None:
    # A 1-point "shape" and an alien shape type are both ignored.
    shapes = [np.array([[0.0, 0.0, 0.0]]), np.zeros((4, 2))]
    bboxes, scores, labels = shapes_to_arrays(
        shapes, ["rectangle", "ellipse"], None, ndim=3, sample_shape=[8, 8, 8]
    )
    assert bboxes is None and scores is None and labels is None


def test_shapes_to_arrays_uses_feature_depth(tmp_path) -> None:
    shape = np.array(
        [[0.0, 1.0, 2.0], [0.0, 5.0, 2.0], [0.0, 5.0, 6.0], [0.0, 1.0, 6.0]]
    )
    features = {
        "depth_axis": np.array([0]),
        "depth_lo": np.array([2.0]),
        "depth_hi": np.array([4.0]),
        "score": np.array([0.8]),
        "label": np.array(["x"], dtype=object),
    }
    bboxes, scores, labels = shapes_to_arrays(
        [shape], "rectangle", features, ndim=3, sample_shape=[10, 10, 10]
    )
    assert bboxes is not None
    assert np.allclose(bboxes[0], [[2.0, 4.0], [1.0, 5.0], [2.0, 6.0]])
    assert scores is not None and np.allclose(scores, [0.8])
    assert labels == ["x"]


def test_shapes_to_arrays_clips_to_sample_shape() -> None:
    # Rectangle that extends beyond the volume should get clipped to [0, shape].
    shape = np.array(
        [[0.0, -5.0, 0.0], [0.0, 20.0, 0.0], [0.0, 20.0, 20.0], [0.0, -5.0, 20.0]]
    )
    bboxes, _, _ = shapes_to_arrays(
        [shape], "rectangle", None, ndim=3, sample_shape=[10, 10, 10]
    )
    assert bboxes is not None
    assert bboxes[0, 1, 0] == 0.0
    assert bboxes[0, 1, 1] == 10.0


def test_shapes_to_arrays_single_shape_type_string() -> None:
    shape = np.array(
        [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 4.0]]
    )
    bboxes, _, _ = shapes_to_arrays(
        [shape], "rectangle", None, ndim=3, sample_shape=None
    )
    assert bboxes is not None and bboxes.shape == (1, 3, 2)


def test_shapes_to_arrays_bad_feature_index_returns_none_score() -> None:
    shape = np.array(
        [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 4.0]]
    )
    features = {
        "depth_axis": np.array([0]),
        "depth_lo": np.array([np.nan]),  # triggers default fallback
        "depth_hi": np.array([np.nan]),
        "score": np.array([np.nan]),
    }
    _, scores, labels = shapes_to_arrays(
        [shape], ["rectangle"], features, ndim=3, sample_shape=None
    )
    # All-NaN score column → None; no label column → None.
    assert scores is None
    assert labels is None


def test_arrays_to_shapes_triggers_wireframe() -> None:
    bboxes = np.array(
        [
            [[0, 4], [0, 5], [0, 5]],  # hi-lo=4 on axis 0 → wireframe
        ],
        dtype=np.float64,
    )
    layers = arrays_to_shapes(
        bboxes=bboxes,
        scores=None,
        labels=None,
        spatial_kwargs={},
        sample_shape=[8, 8, 8],
        path="x.medh5",
        stem="x",
    )
    roles = [kw["metadata"]["medh5_role"] for _, kw, _ in layers]
    assert "bbox_rect" in roles
    assert "bbox_wire" in roles
