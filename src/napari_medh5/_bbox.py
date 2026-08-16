"""Translate between medh5 boxes and napari Shapes layers.

**The half voxel is the whole problem.**  A medh5 box is ``float32`` at voxel
*edges*: ``[a, b]`` covers the numpy slice ``a+0.5 : b+0.5`` (spec §8.1).  A
napari rectangle is drawn at voxel *centres*, because that is where the pixels
are.  So every corner shifts by half a voxel on the way in and back on the way
out, and a round trip that forgets it moves every box by half a voxel per
save --- the classic silent drift.

napari has no native 3-D box, so:

* **Read** --- each box is projected onto its shallowest axis and drawn as a
  rectangle on that axis's centre slice.  The full extent survives in
  ``features["depth_axis"|"depth_lo"|"depth_hi"]``.  A box deeper than one
  voxel also gets a 12-segment wireframe in a companion layer.
* **Write** --- the rectangle layer is authoritative and the wireframe is
  ignored.  Depth comes from the feature columns, never from the drawn
  rectangle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._types import LayerDataTuple

_SHAPE_TYPES = ("rectangle", "line")

EDGE_TO_CENTRE = 0.5
"""medh5 stores box corners at voxel edges; napari draws at voxel centres."""


def _depth_axis(box: npt.NDArray[Any]) -> int:
    extents = box[:, 1] - box[:, 0]
    return int(np.argmin(extents))


def _rectangle_in_plane(
    box: npt.NDArray[Any], ndim: int, depth_axis: int, depth_value: float
) -> npt.NDArray[Any]:
    """A ``(4, ndim)`` rectangle in napari coordinate order."""
    corners = np.zeros((4, ndim), dtype=np.float64)
    plane_axes = [a for a in range(ndim) if a != depth_axis]
    if len(plane_axes) < 2:
        raise ValueError("a box must be at least 2-D to draw")
    a0, a1 = plane_axes[0], plane_axes[1]
    lo0, hi0 = float(box[a0, 0]), float(box[a0, 1])
    lo1, hi1 = float(box[a1, 0]), float(box[a1, 1])
    corners[0, a0], corners[0, a1] = lo0, lo1
    corners[1, a0], corners[1, a1] = hi0, lo1
    corners[2, a0], corners[2, a1] = hi0, hi1
    corners[3, a0], corners[3, a1] = lo0, hi1
    for corner in corners:
        corner[depth_axis] = depth_value
    return corners


def _cuboid_wires(box: npt.NDArray[Any], ndim: int) -> list[npt.NDArray[Any]]:
    """Twelve ``(2, ndim)`` line segments around a 3-D box."""
    if ndim != 3:
        return []
    lo = box[:, 0].astype(np.float64)
    hi = box[:, 1].astype(np.float64)
    pts = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]  # fmt: skip
    return [np.stack([pts[a], pts[b]]) for a, b in edges]


def boxes_to_shapes(
    annotation: Any,
    *,
    names: dict[int, str],
    layer_kwargs: dict[str, Any],
    path: str,
    stem: str,
    suffix: str = "",
) -> list[LayerDataTuple]:
    """Build one or two Shapes layers from a `boxes` annotation."""
    boxes = np.asarray(annotation.boxes, dtype=np.float64)
    if boxes.size == 0:
        return []
    # Edges -> centres.  Do it once, here, so every downstream coordinate in
    # this module is in napari's convention and nothing has to remember.
    boxes = boxes + EDGE_TO_CENTRE
    count, ndim = boxes.shape[0], boxes.shape[1]

    class_ids = [int(c) for c in np.asarray(annotation.class_ids).ravel()]
    scores = getattr(annotation, "scores", None)
    instance_ids = getattr(annotation, "instance_ids", None)

    features: dict[str, Any] = {
        "depth_axis": np.zeros(count, dtype=np.int64),
        "depth_lo": np.zeros(count, dtype=np.float64),
        "depth_hi": np.zeros(count, dtype=np.float64),
        "class_id": np.asarray(class_ids, dtype=np.int64),
        "label": np.asarray([names.get(c, str(c)) for c in class_ids], dtype=object),
        "score": np.asarray(
            scores if scores is not None else [np.nan] * count, dtype=np.float64
        ),
        "instance_id": np.asarray(
            instance_ids if instance_ids is not None else [-1] * count, dtype=np.int64
        ),
    }

    rectangles: list[npt.NDArray[Any]] = []
    wires: list[npt.NDArray[Any]] = []
    needs_wire = False
    for index, box in enumerate(boxes):
        axis = _depth_axis(box)
        lo, hi = float(box[axis, 0]), float(box[axis, 1])
        rectangles.append(_rectangle_in_plane(box, ndim, axis, 0.5 * (lo + hi)))
        features["depth_axis"][index] = axis
        features["depth_lo"][index] = lo
        features["depth_hi"][index] = hi
        if hi - lo > 1:
            needs_wire = True
            wires.extend(_cuboid_wires(box, ndim))

    rect_kwargs: dict[str, Any] = {
        "name": f"{stem}:{annotation.ann_id}{suffix}",
        "shape_type": ["rectangle"] * count,
        "features": features,
        "edge_color": "yellow",
        "face_color": "transparent",
        "edge_width": 2,
        "text": {"string": "{label}", "color": "yellow", "size": 10},
        "metadata": {
            "medh5_path": path,
            "medh5_role": "bbox_rect",
            "medh5_name": annotation.ann_id,
            "medh5_grid": annotation.grid_id,
            "medh5_classes": dict(names),
            "sample_shape": [int(s) for s in annotation.grid.spatial_shape],
        },
        **layer_kwargs,
    }
    layers: list[LayerDataTuple] = [(rectangles, rect_kwargs, "shapes")]

    if needs_wire and wires:
        layers.append(
            (
                wires,
                {
                    "name": f"{stem}:{annotation.ann_id}:wire{suffix}",
                    "shape_type": ["line"] * len(wires),
                    "edge_color": "yellow",
                    "edge_width": 1,
                    "opacity": 0.5,
                    "metadata": {
                        "medh5_path": path,
                        "medh5_role": "bbox_wire",
                        "medh5_name": annotation.ann_id,
                    },
                    **layer_kwargs,
                },
                "shapes",
            )
        )
    return layers


def shapes_to_boxes(
    shapes_data: list[npt.NDArray[Any]],
    shape_types: list[str] | str,
    features: dict[str, Any] | None,
    ndim: int,
) -> tuple[
    npt.NDArray[Any] | None, list[int] | None, npt.NDArray[Any] | None, list[int] | None
]:
    """Convert a rectangles layer back to medh5 boxes, class ids, scores, ids.

    Returns coordinates at **voxel edges**, ready to write.  The wireframe
    companion is ignored: the rectangle layer carries the depth extent in its
    features, and the wires are a rendering of it.
    """
    if not shapes_data:
        return None, None, None, None

    if isinstance(shape_types, str):
        shape_types = [shape_types] * len(shapes_data)

    boxes: list[npt.NDArray[Any]] = []
    used: list[int] = []
    for index, (shape, kind) in enumerate(zip(shapes_data, shape_types, strict=False)):
        if kind not in _SHAPE_TYPES:
            continue
        corners = np.asarray(shape, dtype=np.float64)
        if corners.shape[0] < 2 or corners.shape[1] != ndim:
            continue
        box = np.stack([corners.min(axis=0), corners.max(axis=0)], axis=1)

        axis = _feature_value(features, "depth_axis", index)
        lo = _feature_value(features, "depth_lo", index)
        hi = _feature_value(features, "depth_hi", index)
        if axis is not None and lo is not None and hi is not None:
            position = int(axis)
            if 0 <= position < ndim:
                box[position, 0] = float(lo)
                box[position, 1] = float(hi)

        boxes.append(box)
        used.append(index)

    if not boxes:
        return None, None, None, None

    # Centres -> edges.  1.0 boxes are float at voxel edges, so nothing is
    # rounded here: a box drawn between two voxels stays between them, instead
    # of snapping to a voxel and moving the annotation.
    out = np.stack(boxes, axis=0) - EDGE_TO_CENTRE

    class_ids = _collect(features, "class_id", used, int)
    scores = _collect(features, "score", used, float)
    instances = _collect(features, "instance_id", used, int)
    return (
        out.astype(np.float32),
        [int(c) for c in class_ids] if class_ids is not None else None,
        scores,
        [int(i) for i in instances] if instances is not None else None,
    )


def _feature_value(features: dict[str, Any] | None, key: str, index: int) -> Any:
    if features is None or key not in features:
        return None
    try:
        value = features[key][index]
    except (IndexError, KeyError, TypeError):
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _collect(
    features: dict[str, Any] | None, key: str, indices: list[int], dtype: Any
) -> npt.NDArray[Any] | None:
    if features is None or key not in features:
        return None
    column = features[key]
    try:
        values = [column[i] for i in indices]
    except (IndexError, KeyError, TypeError):
        return None
    if dtype is float:
        array = np.asarray(values, dtype=np.float64)
        return None if np.all(np.isnan(array)) else array
    array = np.asarray(values, dtype=np.int64)
    return None if np.all(array < 0) else array


__all__ = ["EDGE_TO_CENTRE", "boxes_to_shapes", "shapes_to_boxes"]
