"""napari writer for ``.medh5`` files.

Operates in two modes:

- **In-place update** (fast): when the set of image modalities in the layer
  list matches the source file, only seg, bbox, and metadata changes are
  pushed via :func:`medh5.MEDH5File.update`.  Preserves compression.
- **Save As** (full rewrite): when the destination file is different from the
  source, or the image set diverges, a full :func:`medh5.MEDH5File.write`
  call is made.  Dask image arrays are materialised as numpy here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from medh5 import MEDH5File

from napari_medh5._bbox import shapes_to_arrays
from napari_medh5._handles import REGISTRY
from napari_medh5._types import LayerDataTuple


@dataclass
class _Bundle:
    images: dict[str, np.ndarray]
    seg: dict[str, np.ndarray]
    bboxes: np.ndarray | None
    bbox_scores: np.ndarray | None
    bbox_labels: list[str] | None
    sample_shape: list[int] | None
    source_path: str | None


def write_sample(path: str, layer_data: list[LayerDataTuple]) -> list[str]:
    """Entry point for napari's multi-layer writer contribution."""
    dest = Path(path)
    if dest.suffix != ".medh5":
        dest = dest.with_suffix(".medh5")

    bundle = _collect(layer_data)
    if not bundle.images:
        raise ValueError("No image layers tagged medh5_role='image' to save")

    source = bundle.source_path
    if source and Path(source).resolve() == dest.resolve():
        _save_inplace(dest, bundle)
    else:
        _save_full(dest, bundle)
    return [str(dest)]


def _collect(layer_data: list[LayerDataTuple]) -> _Bundle:
    images: dict[str, np.ndarray] = {}
    seg: dict[str, np.ndarray] = {}
    rect_payload: tuple[Any, dict[str, Any]] | None = None
    source_paths: set[str] = set()
    sample_shape: list[int] | None = None

    for data, meta_kwargs, layer_type in layer_data:
        m = _meta_dict(meta_kwargs)
        role = m.get("medh5_role")
        src = m.get("medh5_path")
        if src:
            source_paths.add(str(src))
        if role == "image" and layer_type == "image":
            name = str(m.get("medh5_name") or meta_kwargs.get("name") or "image")
            arr = np.asarray(data)
            images[name] = arr
            if sample_shape is None:
                sample_shape = list(arr.shape)
        elif role == "seg" and layer_type == "labels":
            name = str(m.get("medh5_name") or meta_kwargs.get("name") or "seg")
            seg[name] = np.asarray(data).astype(bool)
        elif role == "bbox_rect" and layer_type == "shapes":
            rect_payload = (data, meta_kwargs)
        elif role == "bbox_wire":
            continue

    if len(source_paths) > 1:
        raise ValueError(
            "Cannot save layers originating from multiple .medh5 files in one pass; "
            f"got paths: {sorted(source_paths)}"
        )

    bboxes: np.ndarray | None = None
    scores: np.ndarray | None = None
    labels: list[str] | None = None
    if rect_payload is not None and sample_shape is not None:
        data, meta_kwargs = rect_payload
        shape_types = meta_kwargs.get("shape_type", "rectangle")
        features = _features_as_dict(meta_kwargs.get("features"))
        shape_sample = (
            list(meta_kwargs.get("metadata", {}).get("sample_shape", sample_shape))
            if isinstance(meta_kwargs.get("metadata"), dict)
            else sample_shape
        )
        bboxes, scores, labels = shapes_to_arrays(
            list(data) if data is not None else [],
            shape_types,
            features,
            ndim=len(sample_shape),
            sample_shape=shape_sample,
        )

    return _Bundle(
        images=images,
        seg=seg,
        bboxes=bboxes,
        bbox_scores=scores,
        bbox_labels=labels,
        sample_shape=sample_shape,
        source_path=next(iter(source_paths), None),
    )


def _meta_dict(layer_kwargs: dict[str, Any]) -> dict[str, Any]:
    raw = layer_kwargs.get("metadata") or {}
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    return {}


def _features_as_dict(features: Any) -> dict[str, Any] | None:
    if features is None:
        return None
    if isinstance(features, dict):
        return cast(dict[str, Any], features)
    to_dict = getattr(features, "to_dict", None)
    if callable(to_dict):
        converted = to_dict(orient="list")
        if isinstance(converted, dict):
            return cast(dict[str, Any], converted)
    return None


def _save_inplace(dest: Path, bundle: _Bundle) -> None:
    # HDF5 forbids opening the same file twice in one process.  Close any
    # registry handle before mutating; callers must re-read after saving.
    REGISTRY.drop(dest)
    sample = MEDH5File.read(dest)
    source_images = set(sample.images.keys())
    new_images = set(bundle.images.keys())
    if source_images != new_images:
        raise ValueError(
            "Image modalities differ from source file "
            f"({sorted(source_images)} vs {sorted(new_images)}); "
            "use Save As to write a new file."
        )
    for name, arr in bundle.images.items():
        if arr.shape != sample.images[name].shape:
            raise ValueError(
                f"Image '{name}' shape {arr.shape} differs from source "
                f"{sample.images[name].shape}; use Save As to write a new file."
            )

    prev_seg = dict(sample.seg or {})
    add: dict[str, np.ndarray] = {}
    replace: dict[str, np.ndarray] = {}
    for name, mask in bundle.seg.items():
        if name in prev_seg:
            if not np.array_equal(mask, prev_seg[name]):
                replace[name] = mask
        else:
            add[name] = mask
    remove = [n for n in prev_seg if n not in bundle.seg]
    seg_ops: dict[str, Any] = {}
    if add:
        seg_ops["add"] = add
    if replace:
        seg_ops["replace"] = replace
    if remove:
        seg_ops["remove"] = remove

    bbox_ops: dict[str, Any] = {}
    if bundle.bboxes is None:
        if sample.bboxes is not None:
            bbox_ops["clear"] = True
    else:
        bbox_ops["bboxes"] = bundle.bboxes
        bbox_ops["bbox_scores"] = bundle.bbox_scores
        bbox_ops["bbox_labels"] = bundle.bbox_labels

    if not seg_ops and not bbox_ops:
        return

    MEDH5File.update(
        dest,
        seg_ops=seg_ops or None,
        bbox_ops=bbox_ops or None,
    )


def _save_full(dest: Path, bundle: _Bundle) -> None:
    meta_kwargs: dict[str, Any] = {}
    if bundle.source_path and Path(bundle.source_path).exists():
        meta = MEDH5File.read_meta(bundle.source_path)
        if meta.label is not None:
            meta_kwargs["label"] = meta.label
        if meta.label_name is not None:
            meta_kwargs["label_name"] = meta.label_name
        if meta.patch_size is not None:
            meta_kwargs["patch_size"] = meta.patch_size
        if meta.extra is not None:
            meta_kwargs["extra"] = meta.extra
        s = meta.spatial
        if s.spacing is not None:
            meta_kwargs["spacing"] = s.spacing
        if s.origin is not None:
            meta_kwargs["origin"] = s.origin
        if s.direction is not None:
            meta_kwargs["direction"] = s.direction
        if s.axis_labels is not None:
            meta_kwargs["axis_labels"] = s.axis_labels
        if s.coord_system is not None:
            meta_kwargs["coord_system"] = s.coord_system

    MEDH5File.write(
        dest,
        images=bundle.images,
        seg=bundle.seg or None,
        bboxes=bundle.bboxes,
        bbox_scores=bundle.bbox_scores,
        bbox_labels=bundle.bbox_labels,
        checksum=True,
        **meta_kwargs,
    )
