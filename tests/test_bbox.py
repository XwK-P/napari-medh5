"""Boxes: the half-voxel convention, the wireframe, and the round trip.

medh5 stores box corners at voxel *edges*; napari draws rectangles at voxel
*centres*.  Half the tests here exist because a round trip that forgets that
moves every box half a voxel per save, and nothing raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import medh5
import numpy as np
import pytest

from napari_medh5._bbox import (
    EDGE_TO_CENTRE,
    _cuboid_wires,
    _rectangle_in_plane,
    boxes_to_shapes,
    shapes_to_boxes,
)
from napari_medh5._layers import sample_to_layers


def _boxes_annotation(path: Path) -> Any:
    sample = medh5.open(path)
    return sample, sample.annotations["boxes"]


class TestRead:
    def test_a_box_becomes_a_rectangle_and_keeps_its_depth(self, tiny_medh5):
        sample, annotation = _boxes_annotation(tiny_medh5)
        try:
            layers = boxes_to_shapes(
                annotation,
                names={1: "Tumor"},
                layer_kwargs={},
                path=str(tiny_medh5),
                stem="tiny",
            )
        finally:
            sample.close()
        data, kwargs, kind = layers[0]
        assert kind == "shapes"
        assert len(data) == 2
        features = kwargs["features"]
        assert list(features["label"]) == ["Tumor", "2"]
        assert list(features["class_id"]) == [1, 2]
        assert features["depth_axis"][0] == 0

    def test_S8_1_corners_shift_by_half_a_voxel_into_napari(self, tiny_medh5):
        """[1.5, 4.5] at voxel edges is [2.0, 5.0] at voxel centres."""
        sample, annotation = _boxes_annotation(tiny_medh5)
        try:
            stored = np.asarray(annotation.boxes)
            layers = boxes_to_shapes(
                annotation, names={}, layer_kwargs={}, path="p", stem="s"
            )
        finally:
            sample.close()
        features = layers[0][1]["features"]
        assert stored[0][0].tolist() == [1.5, 4.5]
        assert features["depth_lo"][0] == pytest.approx(2.0)
        assert features["depth_hi"][0] == pytest.approx(5.0)
        assert EDGE_TO_CENTRE == 0.5

    def test_an_empty_box_annotation_yields_no_layers(self, tmp_path):
        class _Empty:
            boxes = np.zeros((0, 3, 2), dtype=np.float32)

        assert (
            boxes_to_shapes(_Empty(), names={}, layer_kwargs={}, path="p", stem="s")
            == []
        )

    def test_a_deep_box_gets_a_wireframe(self, deep_bbox_medh5):
        sample, annotation = _boxes_annotation(deep_bbox_medh5)
        try:
            layers = boxes_to_shapes(
                annotation, names={}, layer_kwargs={}, path="p", stem="s"
            )
        finally:
            sample.close()
        assert len(layers) == 2
        assert layers[1][1]["metadata"]["medh5_role"] == "bbox_wire"
        assert len(layers[1][0]) == 12

    def test_a_single_slice_box_gets_no_wireframe(self, tmp_path):
        """A box one voxel deep is fully described by its rectangle."""

        class _Flat:
            ann_id = "boxes"
            grid_id = "g"
            boxes = np.array([[[1.5, 2.5], [3.5, 9.5], [3.5, 9.5]]], np.float32)
            class_ids = np.array([1])
            scores = None
            instance_ids = None

            class grid:
                spatial_shape = (8, 16, 16)

        layers = boxes_to_shapes(_Flat(), names={}, layer_kwargs={}, path="p", stem="s")
        assert [k["metadata"]["medh5_role"] for _, k, _ in layers] == ["bbox_rect"]

    def test_the_layer_carries_the_grid_shape(self, tiny_medh5):
        sample, annotation = _boxes_annotation(tiny_medh5)
        try:
            layers = boxes_to_shapes(
                annotation, names={}, layer_kwargs={}, path="p", stem="s"
            )
        finally:
            sample.close()
        assert layers[0][1]["metadata"]["sample_shape"] == [8, 16, 16]


class TestGeometryHelpers:
    def test_a_rectangle_needs_two_in_plane_axes(self):
        with pytest.raises(ValueError, match="at least 2-D"):
            _rectangle_in_plane(np.zeros((2, 2)), 2, 0, 0.0)

    def test_wires_are_only_defined_in_3d(self):
        assert _cuboid_wires(np.zeros((2, 2)), 2) == []
        assert len(_cuboid_wires(np.zeros((3, 2)), 3)) == 12


class TestWrite:
    def test_an_empty_layer_writes_nothing(self):
        assert shapes_to_boxes([], "rectangle", None, ndim=3) == (
            None,
            None,
            None,
            None,
        )

    def test_shapes_that_are_not_boxes_are_skipped(self):
        assert shapes_to_boxes([np.zeros((4, 3))], ["polygon"], None, ndim=3) == (
            None,
            None,
            None,
            None,
        )

    def test_a_shape_with_the_wrong_rank_is_skipped(self):
        assert shapes_to_boxes([np.zeros((4, 2))], "rectangle", None, ndim=3) == (
            None,
            None,
            None,
            None,
        )

    def test_a_single_shape_type_string_applies_to_all(self):
        rect = np.array([[0.0, 1, 1], [0, 5, 1], [0, 5, 5], [0, 1, 5]])
        boxes, *_ = shapes_to_boxes([rect, rect], "rectangle", None, ndim=3)
        assert boxes is not None and boxes.shape == (2, 3, 2)

    def test_depth_comes_from_the_features_not_the_rectangle(self):
        """A rectangle is flat; the depth it stands for is in the features."""
        rect = np.array([[3.0, 1, 1], [3, 5, 1], [3, 5, 5], [3, 1, 5]])
        features = {
            "depth_axis": [0],
            "depth_lo": [2.0],
            "depth_hi": [6.0],
            "class_id": [1],
            "score": [0.5],
            "instance_id": [-1],
        }
        boxes, class_ids, scores, instances = shapes_to_boxes(
            [rect], "rectangle", features, ndim=3
        )
        assert boxes is not None
        assert boxes[0][0].tolist() == [1.5, 5.5]  # centres -> edges
        assert class_ids == [1]
        assert scores is not None and scores[0] == 0.5
        assert instances is None  # -1 means "not carried"

    def test_a_missing_feature_column_degrades_to_none(self):
        rect = np.array([[0.0, 1, 1], [0, 5, 1], [0, 5, 5], [0, 1, 5]])
        boxes, class_ids, scores, _ = shapes_to_boxes(
            [rect], "rectangle", {"depth_axis": [0]}, ndim=3
        )
        assert boxes is not None
        assert class_ids is None and scores is None

    def test_a_short_feature_column_does_not_raise(self):
        rect = np.array([[0.0, 1, 1], [0, 5, 1], [0, 5, 5], [0, 1, 5]])
        boxes, _, scores, _ = shapes_to_boxes(
            [rect, rect], "rectangle", {"score": [0.4]}, ndim=3
        )
        assert boxes is not None and scores is None

    def test_all_nan_scores_are_dropped(self):
        rect = np.array([[0.0, 1, 1], [0, 5, 1], [0, 5, 5], [0, 1, 5]])
        _, _, scores, _ = shapes_to_boxes(
            [rect], "rectangle", {"score": [float("nan")]}, ndim=3
        )
        assert scores is None


class TestRoundTrip:
    def test_S8_1_a_box_survives_read_and_write_unchanged(self, tiny_medh5):
        """The half-voxel shift must cancel exactly, or boxes drift per save."""
        with medh5.open(tiny_medh5) as sample:
            before = np.asarray(sample.annotations["boxes"].boxes).copy()

        layers = sample_to_layers(medh5.open(tiny_medh5), tiny_medh5)
        rect = next(
            (d, k)
            for d, k, t in layers
            if t == "shapes" and k["metadata"]["medh5_role"] == "bbox_rect"
        )
        after, class_ids, scores, _ = shapes_to_boxes(
            list(rect[0]), rect[1]["shape_type"], rect[1]["features"], ndim=3
        )
        assert after is not None
        assert np.allclose(after, before)
        assert class_ids == [1, 2]
        assert scores is not None and np.allclose(scores, [0.9, 0.5])

    def test_a_box_between_two_voxels_is_not_snapped(self):
        """0.x rounded to integers here; 1.0 boxes are float and must stay so."""
        rect = np.array(
            [[0.0, 1.25, 1.25], [0, 5.75, 1.25], [0, 5.75, 5.75], [0, 1.25, 5.75]]
        )
        boxes, *_ = shapes_to_boxes([rect], "rectangle", None, ndim=3)
        assert boxes is not None
        assert boxes[0][1].tolist() == [0.75, 5.25]
