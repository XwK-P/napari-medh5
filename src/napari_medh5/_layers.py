"""Translate a ``MEDH5File`` into napari layer-data tuples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
from medh5 import MEDH5File
from medh5.meta import SpatialMeta

from napari_medh5._bbox import arrays_to_shapes
from napari_medh5._types import LayerDataTuple


def _spatial_to_affine(spatial: SpatialMeta, ndim: int) -> np.ndarray | None:
    """Return a ``(ndim+1, ndim+1)`` homogeneous affine, or ``None`` if the
    spatial metadata collapses to pure ``scale`` + ``translate`` (in which
    case the caller should pass those directly instead).
    """
    direction = spatial.direction
    if direction is None:
        return None
    rot = np.asarray(direction, dtype=np.float64)
    if rot.shape != (ndim, ndim):
        return None
    if np.allclose(rot, np.eye(ndim)):
        return None

    spacing = np.asarray(spatial.spacing or [1.0] * ndim, dtype=np.float64)
    origin = np.asarray(spatial.origin or [0.0] * ndim, dtype=np.float64)

    affine = np.eye(ndim + 1, dtype=np.float64)
    affine[:ndim, :ndim] = rot * spacing[np.newaxis, :]
    affine[:ndim, ndim] = origin
    return affine


def _spatial_kwargs(spatial: SpatialMeta, ndim: int) -> dict[str, Any]:
    """Build ``scale`` / ``translate`` / ``affine`` kwargs for a layer."""
    kwargs: dict[str, Any] = {}
    affine = _spatial_to_affine(spatial, ndim)
    if affine is not None:
        kwargs["affine"] = affine
        return kwargs
    if spatial.spacing is not None:
        kwargs["scale"] = list(spatial.spacing)
    if spatial.origin is not None:
        kwargs["translate"] = list(spatial.origin)
    return kwargs


def _nnunet_labels(meta_extra: dict[str, Any] | None) -> dict[int, str]:
    """Extract ``{value: class_name}`` from ``extra["nnunetv2"]["labels"]``."""
    if not meta_extra:
        return {}
    nn = meta_extra.get("nnunetv2")
    if not isinstance(nn, dict):
        return {}
    labels = nn.get("labels")
    if not isinstance(labels, dict):
        return {}
    out: dict[int, str] = {}
    for name, value in labels.items():
        try:
            out[int(value)] = str(name)
        except (TypeError, ValueError):
            continue
    return out


def _seg_display_name(raw_name: str, nnunet_map: dict[int, str]) -> str:
    """Prefer an nnU-Net class name when the seg key is a bare integer."""
    if raw_name in set(nnunet_map.values()):
        return raw_name
    try:
        idx = int(raw_name)
    except ValueError:
        return raw_name
    return nnunet_map.get(idx, raw_name)


def file_to_layers(f: MEDH5File, path: str | Path) -> list[LayerDataTuple]:
    """Convert an open :class:`MEDH5File` into a list of napari layer tuples."""
    path = Path(path)
    stem = path.stem
    meta = f.meta
    spatial = meta.spatial
    first_name = next(iter(f.images))
    first_ds = f.images[first_name]
    ndim = len(first_ds.shape)
    spatial_kwargs = _spatial_kwargs(spatial, ndim)

    layers: list[LayerDataTuple] = []

    for name in f.images:
        ds = f.images[name]
        arr = da.from_array(ds, chunks=ds.chunks or "auto")
        layer_kwargs: dict[str, Any] = {
            "name": f"{stem}:{name}",
            "metadata": {
                "medh5_path": str(path),
                "medh5_role": "image",
                "medh5_name": name,
            },
            **spatial_kwargs,
        }
        layers.append((arr, layer_kwargs, "image"))

    seg_group = f.seg
    nnunet_map = _nnunet_labels(meta.extra)
    if seg_group is not None:
        for name in seg_group:
            ds = seg_group[name]
            arr = da.from_array(ds, chunks=ds.chunks or "auto").astype(np.uint8)
            display = _seg_display_name(name, nnunet_map)
            layer_kwargs = {
                "name": f"{stem}:seg:{display}",
                "metadata": {
                    "medh5_path": str(path),
                    "medh5_role": "seg",
                    "medh5_name": name,
                },
                **spatial_kwargs,
            }
            layers.append((arr, layer_kwargs, "labels"))

    bboxes, scores, labels = f.bbox_arrays()
    shape_list = list(first_ds.shape)
    bbox_layers = arrays_to_shapes(
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        spatial_kwargs=spatial_kwargs,
        sample_shape=shape_list,
        path=str(path),
        stem=stem,
    )
    layers.extend(bbox_layers)

    return layers
