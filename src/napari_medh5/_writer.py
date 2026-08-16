"""napari writer for ``.medh5`` files.

Two modes, chosen by whether the destination is the file the layers came
from:

* **Amend** --- same path, same images.  ``medh5.amend`` is copy-on-write: it
  rebuilds the file from the old one and replaces it atomically, so unknown
  objects (including ones written by a future minor version) are copied
  through untouched.  Only the annotations that changed are re-encoded.
* **Full write** --- Save As, or a different set of images.  ``medh5.create``
  from scratch, carrying the source's identity, timepoints, label set and
  geometry across where there is a source to carry them from.

Both record a provenance activity naming napari as the agent, because an
annotation edited in a viewer and one produced by a model should not be
indistinguishable a year later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import medh5
import numpy as np
import numpy.typing as npt

from napari_medh5._bbox import shapes_to_boxes
from napari_medh5._handles import REGISTRY, rebind_viewer_layers
from napari_medh5._types import LayerDataTuple

AGENT = "napari-medh5"


@dataclass
class _Bundle:
    images: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    labelmaps: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    seg_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    boxes: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    source_path: str | None = None
    image_meta: dict[str, dict[str, Any]] = field(default_factory=dict)


def write_sample(path: str, layer_data: list[LayerDataTuple]) -> list[str]:
    """Entry point for napari's multi-layer writer contribution."""
    dest = Path(path)
    if dest.suffix != ".medh5":
        dest = dest.with_suffix(".medh5")

    bundle = _collect(layer_data)
    if not bundle.images:
        raise ValueError("no image layers tagged medh5_role='image' to save")

    source = bundle.source_path
    if source and Path(source).resolve() == dest.resolve():
        _amend(dest, bundle)
    else:
        _write_new(dest, bundle)
    return [str(dest)]


def _collect(layer_data: list[LayerDataTuple]) -> _Bundle:
    bundle = _Bundle()
    sources: set[str] = set()

    for data, kwargs, layer_type in layer_data:
        meta = _meta_dict(kwargs)
        role = meta.get("medh5_role")
        source = meta.get("medh5_path")
        if source:
            sources.add(str(source))

        if role == "image" and layer_type == "image":
            name = str(meta.get("medh5_name") or kwargs.get("name") or "image")
            bundle.images[name] = np.asarray(data)
            bundle.image_meta[name] = meta
        elif role == "seg" and layer_type == "labels":
            name = str(meta.get("medh5_name") or kwargs.get("name") or "seg")
            bundle.labelmaps[name] = np.asarray(data)
            bundle.seg_meta[name] = meta
        elif role == "bbox_rect" and layer_type == "shapes":
            name = str(meta.get("medh5_name") or kwargs.get("name") or "boxes")
            bundle.boxes[name] = (data, kwargs, meta)
        elif role == "bbox_wire":
            continue  # a rendering of the rectangles, never a source of truth

    if len(sources) > 1:
        raise ValueError(
            "cannot save layers from more than one .medh5 file in one pass; got "
            f"{sorted(sources)}"
        )
    bundle.source_path = next(iter(sources), None)
    return bundle


def _meta_dict(kwargs: dict[str, Any]) -> dict[str, Any]:
    raw = kwargs.get("metadata") or {}
    return cast(dict[str, Any], raw) if isinstance(raw, dict) else {}


def _features(features: Any) -> dict[str, Any] | None:
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


def _masks_from(
    labelmap: npt.NDArray[Any], meta: dict[str, Any]
) -> dict[int, npt.NDArray[Any]]:
    """Split an edited labelmap back into per-class masks.

    Every class the annotation *declared* gets a mask, including ones the
    editor left empty: a class that was examined and is now absent is a
    verified negative (§11.3), and dropping it would turn that into "nobody
    looked".
    """
    declared = [int(c) for c in meta.get("medh5_annotated") or ()]
    present = [int(v) for v in np.unique(labelmap) if int(v) not in (0, 65535)]
    for value in present:
        if value not in declared:
            declared.append(value)
    return {class_id: labelmap == class_id for class_id in sorted(declared)}


def _annotated(meta: dict[str, Any], masks: dict[int, npt.NDArray[Any]]) -> list[int]:
    declared = [int(c) for c in meta.get("medh5_annotated") or ()]
    return declared or sorted(masks)


def _amend(dest: Path, bundle: _Bundle) -> None:
    """Rewrite the annotations that changed, leaving everything else alone."""
    # `amend` replaces the file, so a handle held across it would keep serving
    # the old inode.  Drop first, rebind after.
    with medh5.open(dest) as sample:
        source_images = set(sample.images)
        source_shapes = {k: tuple(v.shape) for k, v in sample.images.items()}
        grids = {k: v.grid_id for k, v in sample.annotations.items()}
        ann_grids = dict(grids)
        image_grids = {k: v.grid_id for k, v in sample.images.items()}
        ann_timepoints = {k: list(v.timepoints) for k, v in sample.annotations.items()}
        existing = set(sample.annotations)

    if source_images != set(bundle.images):
        raise ValueError(
            f"image set differs from the source ({sorted(source_images)} vs "
            f"{sorted(bundle.images)}); use Save As to write a new file"
        )
    for name, array in bundle.images.items():
        if tuple(array.shape) != source_shapes[name]:
            raise ValueError(
                f"image {name!r} is {array.shape}, the source is "
                f"{source_shapes[name]}; use Save As to write a new file"
            )

    REGISTRY.drop(dest)
    with medh5.amend(dest) as writer:
        agent = writer.software(AGENT, _version())
        activity = writer.activity("annotate", agent=agent, tool="napari")

        for name, labelmap in bundle.labelmaps.items():
            meta = bundle.seg_meta[name]
            grid = ann_grids.get(name) or meta.get("medh5_grid")
            if grid is None:
                grid = next(iter(image_grids.values()))
            masks = _masks_from(labelmap, meta)
            if name in existing:
                writer.remove_annotation(name)
            writer.add_segmentation(
                name,
                grid=grid,
                masks=masks,
                annotated_classes=_annotated(meta, masks),
                timepoints=ann_timepoints.get(name) or None,
                prov=activity,
            )

        for name, (data, kwargs, meta) in bundle.boxes.items():
            grid = ann_grids.get(name) or meta.get("medh5_grid")
            if grid is None:
                grid = next(iter(image_grids.values()))
            _write_boxes(writer, name, data, kwargs, grid, activity, existing)

    rebind_viewer_layers(dest)


def _write_boxes(
    writer: Any,
    name: str,
    data: Any,
    kwargs: dict[str, Any],
    grid: str,
    activity: Any,
    existing: set[str],
) -> None:
    meta = _meta_dict(kwargs)
    shape = meta.get("sample_shape") or []
    boxes, class_ids, scores, instance_ids = shapes_to_boxes(
        list(data) if data is not None else [],
        kwargs.get("shape_type", "rectangle"),
        _features(kwargs.get("features")),
        ndim=len(shape) if shape else 3,
    )
    if name in existing:
        writer.remove_annotation(name)
    if boxes is None or class_ids is None:
        return  # every box was deleted in the viewer
    writer.add_boxes(
        name,
        boxes=boxes,
        class_ids=class_ids,
        grid=grid,
        space="index",
        scores=scores,
        instance_ids=instance_ids,
        prov=activity,
    )


def _write_new(dest: Path, bundle: _Bundle) -> None:
    """Write a fresh sample, carrying what the source can supply."""
    source = bundle.source_path
    document = None
    grids: dict[str, Any] = {}
    if source and Path(source).exists():
        with medh5.open(source) as sample:
            document = sample.document
            grids = {k: v for k, v in sample.grids.items()}

    first = next(iter(bundle.images.values()))
    with medh5.create(
        dest,
        sample_id=document.identity.sample_id if document else dest.stem,
        subject_id=document.identity.subject_id if document else dest.stem,
    ) as writer:
        if document is not None:
            writer.identity(**document.identity.to_json())
            writer.cohort(**document.cohort.to_json())
            for timepoint in document.timepoints:
                fields = timepoint.to_json()
                writer.add_timepoint(str(fields.pop("id")), **fields)
            if document.label_set is not None:
                writer.label_set(document.label_set)
            for namespace, value in document.extra.items():
                writer.extra(namespace, value)
        else:
            writer.add_timepoint("tp0")

        agent = writer.software(AGENT, _version())
        activity = writer.activity("annotate", agent=agent, tool="napari")

        written_grids = _declare_grids(writer, bundle, grids, first)

        for name, array in bundle.images.items():
            meta = bundle.image_meta.get(name, {})
            writer.add_image(
                name,
                array,
                grid=written_grids[name],
                modality=str(meta.get("modality") or "OT"),
                value_units=meta.get("value_units"),
                prov=activity,
            )

        for name, labelmap in bundle.labelmaps.items():
            meta = bundle.seg_meta[name]
            masks = _masks_from(labelmap, meta)
            writer.add_segmentation(
                name,
                grid=_grid_for(meta, written_grids),
                masks=masks,
                annotated_classes=_annotated(meta, masks),
                prov=activity,
            )

        for name, (data, kwargs, meta) in bundle.boxes.items():
            _write_boxes(
                writer,
                name,
                data,
                kwargs,
                _grid_for(meta, written_grids),
                activity,
                set(),
            )


def _declare_grids(
    writer: Any, bundle: _Bundle, source_grids: dict[str, Any], first: npt.NDArray[Any]
) -> dict[str, str]:
    """Declare one grid per distinct source grid, reusing its geometry.

    Geometry is never invented: without a source grid the fallback is unit
    spacing at the origin, which is what an image with no stated geometry
    actually means.
    """
    out: dict[str, str] = {}
    declared: set[str] = set()
    for name, array in bundle.images.items():
        grid_id = str(bundle.image_meta.get(name, {}).get("medh5_grid") or "grid")
        if grid_id not in declared:
            source = source_grids.get(grid_id)
            if source is not None and tuple(source.shape) == tuple(array.shape):
                writer.add_grid(
                    grid_id,
                    shape=source.shape,
                    spacing=source.spacing,
                    origin=source.origin,
                    direction=source.direction,
                    coord_system=source.coord_system,
                    units=source.units,
                    timepoint=source.timepoint,
                    frame_uid=source.frame_uid,
                )
            else:
                writer.add_grid(grid_id, shape=array.shape, spacing=(1.0,) * array.ndim)
            declared.add(grid_id)
        out[name] = grid_id
    return out


def _grid_for(meta: dict[str, Any], written: dict[str, str]) -> str:
    grid = meta.get("medh5_grid")
    if grid and grid in set(written.values()):
        return str(grid)
    return next(iter(written.values()))


def _version() -> str:
    from napari_medh5 import __version__

    return __version__


__all__ = ["write_sample"]
