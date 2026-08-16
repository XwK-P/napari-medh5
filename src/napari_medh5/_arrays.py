"""Lazy dask views onto a medh5 sample's images and voxel annotations.

An image is one HDF5 dataset, so a dask view of it is direct.  A **voxel
annotation is not**: five encodings sit behind one read contract, and only
one of them (`labelmap`) stores anything a viewer could map to colours
directly.  Materialising the whole labelmap would work and would also load
a 25 MB array per annotation before the first pixel is drawn.

So the labels layer is built with ``map_blocks`` over
``VoxelAnnotation.labelmap(roi=...)``: napari asks for the slice it is about
to draw, medh5 decodes exactly that window out of whatever encoding the file
uses, and the encoding stays a storage decision the viewer never sees.
"""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt

LABEL_DTYPE = np.uint16
"""Class ids are uint16 (§5.1); 65535 is the reserved `ignore` id."""


def image_array(image: Any) -> Any:
    """A lazy dask view of one image dataset."""
    dataset = image.dataset
    return da.from_array(dataset, chunks=dataset.chunks or "auto")


def annotation_array(annotation: Any) -> Any:
    """A lazy labelmap over a voxel annotation, whatever its encoding.

    Overlapping classes have to collapse here --- a napari ``Labels`` layer
    holds one id per voxel and cannot show two --- so the order they collapse
    in is a real decision, and :func:`draw_priority` makes it.  The overlap is
    untouched in the file; reading the annotation directly still returns both.
    """
    shape = tuple(int(s) for s in annotation.spatial_shape)
    chunks = _chunks_for(annotation, shape)
    priority = draw_priority(annotation)

    def block(block_info: dict[Any, Any] | None = None) -> npt.NDArray[Any]:
        if block_info is None:  # pragma: no cover - dask meta probe
            return np.zeros((0,) * len(shape), dtype=LABEL_DTYPE)
        roi = tuple(
            slice(int(a), int(b)) for a, b in block_info[None]["array-location"]
        )
        return np.asarray(
            annotation.labelmap(roi=roi, priority=priority), dtype=LABEL_DTYPE
        )

    return da.map_blocks(block, dtype=LABEL_DTYPE, chunks=chunks)


def draw_priority(annotation: Any) -> list[int]:
    """Precedence for `labelmap`: the most specific class wins the voxel.

    ``labelmap(priority=...)`` takes highest precedence **first**.  A lesion
    *inside* an organ is a child of it in the label set's DAG (§5.1), so it is
    deeper --- and without a priority the organ can simply overwrite it, hiding
    the structure the reader opened the file to look at.  Ordering by depth,
    deepest first, keeps the specific class visible.  A flat label set falls
    back to id order, which is what `labelmap` would have done anyway.
    """
    label_set = getattr(annotation, "label_set", None)
    ids = [int(c) for c in annotation.class_ids]
    if label_set is None:
        return ids

    def depth(class_id: int, seen: frozenset[int] = frozenset()) -> int:
        if class_id in seen:  # pragma: no cover - a DAG has no cycles
            return 0
        try:
            parents = [int(p) for p in label_set[class_id].parents]
        except (KeyError, IndexError):
            return 0
        if not parents:
            return 0
        return 1 + max(depth(p, seen | {class_id}) for p in parents)

    return sorted(ids, key=lambda c: (-depth(c), c))


def _chunks_for(annotation: Any, shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Block the annotation the way it is stored, so a read is one decode.

    Falls back to whole-axis blocks, which is correct if slower --- never to a
    guess that cuts across the stored chunk grid and decodes each block twice.
    """
    stored = _stored_chunks(annotation, len(shape))
    out: list[tuple[int, ...]] = []
    for axis, extent in enumerate(shape):
        step = stored[axis] if stored else extent
        step = max(1, min(int(step), extent))
        full, tail = divmod(extent, step)
        out.append(tuple([step] * full + ([tail] if tail else [])))
    return tuple(out)


def _stored_chunks(annotation: Any, ndim: int) -> tuple[int, ...] | None:
    """The spatial part of the stored chunk shape, when there is one."""
    group = getattr(annotation, "group", None)
    if group is None:
        return None
    for name in group:
        node = group[name]
        chunks = getattr(node, "chunks", None)
        if chunks and len(chunks) >= ndim:
            return tuple(int(c) for c in chunks[-ndim:])
    return None


__all__ = ["LABEL_DTYPE", "annotation_array", "draw_priority", "image_array"]
