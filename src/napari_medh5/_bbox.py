"""Translate between medh5 ``(n, ndim, 2)`` bbox arrays and napari Shapes.

The round-trip strategy:

- For rendering, each bbox is projected onto its "depth" axis (the axis with
  the smallest extent — usually Z for axial CT).  A single rectangle is drawn
  on the centre slice of that axis.  If any box spans more than one voxel
  along the depth axis, a second Shapes layer with 12 line segments per box
  renders the full 3D wireframe.
- For saving, the rectangles layer is authoritative.  Each rectangle's depth
  extent is taken from the ``features["depth"]`` column (populated on read,
  editable through napari's shape features).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import LayerDataTuple

_SHAPE_TYPES = ("rectangle", "line")


def _depth_axis(box: np.ndarray) -> int:
    extents = box[:, 1] - box[:, 0]
    return int(np.argmin(extents))


def _rectangle_in_plane(
    box: np.ndarray, ndim: int, depth_axis: int, depth_value: float
) -> np.ndarray:
    """Return a ``(4, ndim)`` rectangle in napari coord order."""
    corners = np.zeros((4, ndim), dtype=np.float64)
    plane_axes = [a for a in range(ndim) if a != depth_axis]
    if len(plane_axes) < 2:
        raise ValueError("bbox must be at least 2D")
    a0, a1 = plane_axes[0], plane_axes[1]
    lo0, hi0 = float(box[a0, 0]), float(box[a0, 1])
    lo1, hi1 = float(box[a1, 0]), float(box[a1, 1])
    corners[0, a0], corners[0, a1] = lo0, lo1
    corners[1, a0], corners[1, a1] = hi0, lo1
    corners[2, a0], corners[2, a1] = hi0, hi1
    corners[3, a0], corners[3, a1] = lo0, hi1
    for c in corners:
        c[depth_axis] = depth_value
    return corners


def _cuboid_wires(box: np.ndarray, ndim: int) -> list[np.ndarray]:
    """Return 12 line segments (each as a ``(2, ndim)`` array) around a 3D box."""
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
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [np.stack([pts[a], pts[b]]) for a, b in edges]


def arrays_to_shapes(
    *,
    bboxes: np.ndarray | None,
    scores: np.ndarray | None,
    labels: list[str] | None,
    spatial_kwargs: dict[str, Any],
    sample_shape: list[int],
    path: str,
    stem: str,
) -> list[LayerDataTuple]:
    """Build one or two Shapes layers from a ``(n, ndim, 2)`` bbox array."""
    if bboxes is None or len(bboxes) == 0:
        return []
    bboxes = np.asarray(bboxes, dtype=np.float64)
    n = bboxes.shape[0]
    ndim = bboxes.shape[1]

    rectangles: list[np.ndarray] = []
    rect_features = {
        "depth_axis": np.zeros(n, dtype=np.int64),
        "depth_lo": np.zeros(n, dtype=np.float64),
        "depth_hi": np.zeros(n, dtype=np.float64),
        "label": np.asarray(labels or [""] * n, dtype=object),
        "score": np.asarray(
            scores if scores is not None else [np.nan] * n, dtype=np.float64
        ),
    }
    wire_segments: list[np.ndarray] = []
    needs_wire = False

    for i, box in enumerate(bboxes):
        axis = _depth_axis(box)
        lo, hi = float(box[axis, 0]), float(box[axis, 1])
        centre = 0.5 * (lo + hi)
        rectangles.append(_rectangle_in_plane(box, ndim, axis, centre))
        rect_features["depth_axis"][i] = axis
        rect_features["depth_lo"][i] = lo
        rect_features["depth_hi"][i] = hi
        if hi - lo > 1:
            needs_wire = True
            wire_segments.extend(_cuboid_wires(box, ndim))

    rect_kwargs: dict[str, Any] = {
        "name": f"{stem}:bboxes",
        "shape_type": ["rectangle"] * n,
        "features": rect_features,
        "edge_color": "yellow",
        "face_color": "transparent",
        "edge_width": 2,
        "metadata": {
            "medh5_path": path,
            "medh5_role": "bbox_rect",
            "sample_shape": list(sample_shape),
        },
        **spatial_kwargs,
    }
    if labels is not None:
        rect_kwargs["text"] = {"string": "{label}", "color": "yellow", "size": 10}

    layers: list[LayerDataTuple] = [(rectangles, rect_kwargs, "shapes")]

    if needs_wire and wire_segments:
        wire_kwargs: dict[str, Any] = {
            "name": f"{stem}:bboxes:wire",
            "shape_type": ["line"] * len(wire_segments),
            "edge_color": "yellow",
            "edge_width": 1,
            "opacity": 0.5,
            "metadata": {
                "medh5_path": path,
                "medh5_role": "bbox_wire",
            },
            **spatial_kwargs,
        }
        layers.append((wire_segments, wire_kwargs, "shapes"))

    return layers


def shapes_to_arrays(
    shapes_data: list[np.ndarray],
    shape_types: list[str] | str,
    features: dict[str, Any] | None,
    ndim: int,
    sample_shape: list[int] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
    """Convert a napari Shapes ``rect`` layer back to ``(n, ndim, 2)`` bboxes.

    The wireframe companion layer is ignored — ``rect`` carries the full truth
    through the ``depth_lo`` / ``depth_hi`` / ``depth_axis`` feature columns.
    """
    if not shapes_data:
        return None, None, None

    if isinstance(shape_types, str):
        shape_types = [shape_types] * len(shapes_data)

    boxes: list[np.ndarray] = []
    used_indices: list[int] = []
    for idx, (shape, stype) in enumerate(zip(shapes_data, shape_types, strict=False)):
        if stype not in _SHAPE_TYPES:
            continue
        arr = np.asarray(shape, dtype=np.float64)
        if arr.shape[0] < 2 or arr.shape[1] != ndim:
            continue
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        box = np.stack([mins, maxs], axis=1)

        if features is not None and "depth_axis" in features:
            ax = _feature_value(features, "depth_axis", idx, default=None)
            lo = _feature_value(features, "depth_lo", idx, default=None)
            hi = _feature_value(features, "depth_hi", idx, default=None)
            if ax is not None and lo is not None and hi is not None:
                axi = int(ax)
                if 0 <= axi < ndim:
                    box[axi, 0] = float(lo)
                    box[axi, 1] = float(hi)

        if sample_shape is not None:
            for a in range(ndim):
                upper = float(sample_shape[a])
                box[a, 0] = max(0.0, min(box[a, 0], upper))
                box[a, 1] = max(0.0, min(box[a, 1], upper))

        boxes.append(box)
        used_indices.append(idx)

    if not boxes:
        return None, None, None

    bboxes = np.stack(boxes, axis=0)
    scores = _collect_feature(features, "score", used_indices, dtype=float)
    labels_raw = _collect_feature(features, "label", used_indices, dtype=object)
    labels: list[str] | None = (
        [str(v) for v in labels_raw] if labels_raw is not None else None
    )
    return bboxes, scores, labels


def _feature_value(
    features: dict[str, Any] | None, key: str, idx: int, default: Any = None
) -> Any:
    if features is None or key not in features:
        return default
    col = features[key]
    try:
        value = col[idx]
    except (IndexError, KeyError, TypeError):
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    return value


def _collect_feature(
    features: dict[str, Any] | None,
    key: str,
    indices: list[int],
    dtype: Any,
) -> np.ndarray | None:
    if features is None or key not in features:
        return None
    col = features[key]
    try:
        values = [col[i] for i in indices]
    except (IndexError, KeyError, TypeError):
        return None
    if dtype is float:
        arr = np.asarray(values, dtype=np.float64)
        if np.all(np.isnan(arr)):
            return None
        return arr
    return np.asarray(values, dtype=dtype)
