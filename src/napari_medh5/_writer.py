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

from napari_medh5._arrays import LABEL_DTYPE, draw_priority
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
    # Annotation names the reader produced layers for.  Empty means "no layer
    # said", which is why an unrecorded annotation is never removed.
    opened: set[str] = field(default_factory=set)


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
        recorded = meta.get("medh5_opened")
        if isinstance(recorded, (list, tuple, set)):
            bundle.opened.update(str(one) for one in recorded)

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


#: Encodings a single labelmap cannot carry back: a voxel holds one id, and
#: these hold more than that per voxel, so no merge recovers them.
RICH_KINDS = ("probmap", "instances")

#: Encodings that are per-class masks, and so survive a merge unchanged.
MASK_KINDS = ("labelmap", "layers", "bitmask")

#: The reserved id napari shows an ignore region as.
IGNORE_ID = 65535


@dataclass
class _Source:
    """What the file being written *from* says about the annotations on screen.

    Read before the file is closed, because both write paths rewrite it.
    """

    kinds: dict[str, str] = field(default_factory=dict)
    maps: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    masks: dict[str, dict[int, npt.NDArray[Any]]] = field(default_factory=dict)
    ignores: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    #: Annotations whose ignore region the encoding will not hand back.
    opaque: set[str] = field(default_factory=set)
    timepoints: dict[str, list[str]] = field(default_factory=dict)
    provs: dict[str, str] = field(default_factory=dict)
    qualities: dict[str, str] = field(default_factory=dict)

    def edited(self, name: str, current: npt.NDArray[Any]) -> bool:
        stored = self.maps.get(name)
        return stored is None or not np.array_equal(current, stored)


def _capture(sample: Any, bundle: _Bundle, *, always: bool) -> _Source:
    """The stored labelmaps, and the masks behind them where they are needed.

    *always* for a full write, which reproduces every annotation; an amend only
    needs the masks of the ones that changed, and they can be large.
    """
    out = _Source()
    for name, annotation in sample.annotations.items():
        if name not in bundle.labelmaps or not hasattr(annotation, "labelmap"):
            continue
        stored = np.asarray(
            annotation.labelmap(priority=draw_priority(annotation)), dtype=LABEL_DTYPE
        )
        out.kinds[name] = str(annotation.kind)
        out.maps[name] = stored
        out.timepoints[name] = [str(t) for t in annotation.timepoints]
        if annotation.header.prov:
            out.provs[name] = str(annotation.header.prov)
        if annotation.header.quality:
            out.qualities[name] = str(annotation.header.quality)
        if annotation.has_ignore_region:
            region = _source_ignore(annotation)
            if region is None:
                out.opaque.add(name)
            else:
                out.ignores[name] = region
        current = np.asarray(bundle.labelmaps[name], dtype=LABEL_DTYPE)
        if always or not np.array_equal(current, stored):
            out.masks[name] = {
                int(c): np.asarray(annotation.dense([int(c)])[0])
                for c in annotation.class_ids
            }
    return out


def _resolve(
    source: _Source, name: str, current: npt.NDArray[Any], meta: dict[str, Any]
) -> tuple[dict[int, npt.NDArray[Any]], npt.NDArray[Any] | None, str]:
    """The masks, ignore region and encoding to write for one annotation.

    Both write paths go through here.  Fixing the amend loop and leaving Save
    As to rebuild everything from the collapsed labelmap was the same data loss
    by a second route.
    """
    kind = source.kinds.get(name)
    stored = source.maps.get(name)
    masks = source.masks.get(name)
    if stored is None or masks is None:
        return _masks_from(current, meta), _ignore_of(current), "auto"
    changed = current != stored
    merged = _merged_masks(current, stored, masks, meta, changed)
    # The stored region, not `stored == IGNORE_ID`: `labelmap()` does not
    # surface the ignore id, so napari never showed the region and the
    # displayed map cannot be its source.  A voxel *painted* 65535 in the
    # viewer is an edit, and joins it.
    ignore = source.ignores.get(name)
    ignore = np.zeros(current.shape, dtype=bool) if ignore is None else ignore.copy()
    ignore[changed] = current[changed] == IGNORE_ID
    encoding = kind if kind in MASK_KINDS else "auto"
    return merged, (ignore if bool(ignore.any()) else None), encoding


def _ignore_of(current: npt.NDArray[Any]) -> npt.NDArray[Any] | None:
    """An ignore region painted into a labelmap with no source to merge with."""
    ignore = current == IGNORE_ID
    return ignore if bool(ignore.any()) else None


def _source_ignore(annotation: Any) -> npt.NDArray[Any] | None:
    """The stored ignore region, where the encoding hands it back.

    Only `labelmap` exposes `ignore_mask()`.  For the rest the region is real
    --- `has_ignore_region` says so --- and there is no public way to read it as
    a mask, which is why rewriting one is refused rather than attempted.
    """
    reader = getattr(annotation, "ignore_mask", None)
    if not callable(reader):
        return None
    mask = np.asarray(reader(), dtype=bool)
    return mask if bool(mask.any()) else None


def _refuse_lossy(name: str, kind: str | None, opaque_ignore: bool = False) -> None:
    """Refuse only what nothing can reconstruct.

    A labelmap holds one id per voxel, so it cannot carry a probability or an
    instance identity back at all --- there is no merge that recovers those,
    and the file being overwritten is the only copy.  Overlapping *classes* are
    a different case and are handled by :func:`_merged_masks`.
    """
    if opaque_ignore:
        raise ValueError(
            f"annotation {name!r} marks an ignore region and its {kind!r} "
            "encoding does not expose it as a mask, so rewriting the annotation "
            "would drop it --- and napari never displayed it to begin with, so "
            "nothing on screen could put it back. Edit it with a tool that can "
            "write that encoding, or copy the file directly."
        )
    if kind in RICH_KINDS:
        raise ValueError(
            f"annotation {name!r} is a {kind!r} annotation, and napari edits it "
            "as a labelmap of one id per voxel; saving would replace the "
            f"{kind} with a hard segmentation. Edit it with a tool that can "
            "write that encoding, or copy the file directly."
        )


def _merged_masks(
    current: npt.NDArray[Any],
    stored: npt.NDArray[Any],
    masks: dict[int, npt.NDArray[Any]],
    meta: dict[str, Any],
    changed: npt.NDArray[Any],
) -> dict[int, npt.NDArray[Any]]:
    """Take the edit where the labelmap changed; keep the file everywhere else.

    napari holds one id per voxel, so what it hands back cannot express two
    classes claiming the same voxel.  Rebuilding the whole annotation from it
    therefore deleted every overlap in the file --- including all the ones the
    editor never went near.  Only the voxels that actually changed take their
    class from the viewer; the rest keep exactly what was stored, overlaps
    intact.  The ignore region rides along the same way, in :func:`_resolve`:
    it is not a class, so clearing every mask at an ignored voxel turned it
    into background instead.
    """
    out = {int(c): np.asarray(m, dtype=bool).copy() for c, m in masks.items()}
    for value in np.unique(current[changed]) if changed.any() else ():
        if int(value) not in (0, IGNORE_ID):
            out.setdefault(int(value), np.zeros(current.shape, dtype=bool))
    for class_id in [int(c) for c in meta.get("medh5_annotated") or ()]:
        out.setdefault(class_id, np.zeros(current.shape, dtype=bool))
    for class_id, mask in out.items():
        mask[changed] = current[changed] == class_id
    return dict(sorted(out.items()))


def _annotated(meta: dict[str, Any], masks: dict[int, npt.NDArray[Any]]) -> list[int]:
    """Every class this annotation claims to have been examined for.

    A class somebody painted was examined by definition, so it joins the
    declared list rather than being left out of it.  Writing voxels for a class
    that `annotated_classes` says nobody looked at is a contradiction --- it
    turns a positive finding into "not examined", which is the one distinction
    §11.3 exists to keep --- and the format rejects it outright.
    """
    declared = {int(c) for c in meta.get("medh5_annotated") or ()}
    return sorted(declared | {int(c) for c in masks})


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
        src = _capture(sample, bundle, always=False)

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

    # Before the handle is dropped: a refusal after that point leaves every
    # layer backed by a closed file, and the user asked for a save, not a
    # broken viewer.
    for name, labelmap in bundle.labelmaps.items():
        if src.edited(name, np.asarray(labelmap, dtype=LABEL_DTYPE)):
            _refuse_lossy(name, src.kinds.get(name), name in src.opaque)

    REGISTRY.drop(dest)
    with medh5.amend(dest) as writer:
        agent = writer.software(AGENT, _version())
        activity = writer.activity("annotate", agent=agent, tool="napari")

        for name, labelmap in bundle.labelmaps.items():
            current = np.asarray(labelmap, dtype=LABEL_DTYPE)
            if not src.edited(name, current):
                # Untouched.  `amend` is copy-on-write and carries it through
                # exactly as it was --- encoding, overlaps and all.  Re-encoding
                # it from what the viewer displayed is how merely *saving* used
                # to delete every voxel where two classes overlapped.
                continue
            meta = bundle.seg_meta[name]
            grid = ann_grids.get(name) or meta.get("medh5_grid")
            if grid is None:
                grid = next(iter(image_grids.values()))
            masks, ignore, encoding = _resolve(src, name, current, meta)
            if name in existing:
                writer.remove_annotation(name)
            writer.add_segmentation(
                name,
                grid=grid,
                masks=masks,
                ignore=ignore,
                encoding=encoding,
                annotated_classes=_annotated(meta, masks),
                timepoints=ann_timepoints.get(name) or None,
                prov=activity,
            )

        for name, (data, kwargs, meta) in bundle.boxes.items():
            grid = ann_grids.get(name) or meta.get("medh5_grid")
            if grid is None:
                grid = next(iter(image_grids.values()))
            _write_boxes(writer, name, data, kwargs, grid, activity, existing)

        for name in _deleted(bundle, existing):
            writer.remove_annotation(name)

    rebind_viewer_layers(dest)


def _deleted(bundle: _Bundle, existing: set[str]) -> list[str]:
    """Annotations the user opened and then removed from the viewer.

    `amend` is copy-on-write, so anything not rewritten is carried through
    untouched --- which is what protects an *unedited* annotation and,
    until this, protected a deleted one too.

    The set has to be the reader's record of what it produced, never the
    file's own contents.  Reconciling against the file would read every
    annotation the user did not have on screen as a deletion: a subset load, a
    layer closed to tidy the list, or a kind napari cannot show as a layer at
    all.  A save would then destroy annotations the user never saw. Layers
    carrying no record contribute nothing, so the fallback is to delete
    nothing.
    """
    present = set(bundle.labelmaps) | set(bundle.boxes)
    return sorted((bundle.opened & existing) - present)


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


# The annotation kinds napari has no layer for.  `sample_to_layers` renders
# every voxel kind and `boxes`; these six can never appear on screen, so their
# absence from the layer list is never something the user decided.  That is
# what makes copying them safe where re-deriving a missing *segmentation* would
# not be: a deleted Labels layer is a decision, a missing mesh is not.
CARRY_KINDS = ("classification", "contours", "keypoints", "mesh", "obb", "points")


@dataclass
class _Carried:
    """One source annotation to reproduce verbatim in the destination."""

    name: str
    kind: str
    grid: str | None
    kwargs: dict[str, Any]


def _header_kwargs(annotation: Any) -> dict[str, Any]:
    """The §6.2 header fields every `add_*` takes, read off the source.

    `annotated_classes` is passed explicitly rather than left to default:
    §11.3 distinguishes what an annotation *contains* from what was *looked
    for*, and `all_given` would quietly promote every unexamined class to an
    examined one.
    """
    header = annotation.header
    return {
        "annotated_classes": [int(c) for c in annotation.annotated_class_ids],
        "closure": str(annotation.closure),
        "timepoints": list(header.timepoints) if header.timepoints else None,
        "prov": header.prov or None,
        "quality": header.quality or None,
        "derived_from": tuple(header.derived_from),
    }


def _dataset_present(annotation: Any, name: str) -> bool:
    """Whether the annotation stores *name*, as opposed to defaulting it.

    Some readers substitute a value when a dataset is absent --- `mesh`
    reports the header's class ids when it has no `mesh_class_ids`.  Writing
    that back would materialise a dataset the source did not have, so the copy
    asks the group rather than the property.
    """
    group = getattr(annotation, "group", None)
    return group is not None and name in group


def _carry_kwargs(annotation: Any) -> dict[str, Any]:
    """Everything `add_<kind>` needs to rebuild *annotation* exactly."""
    kind = str(annotation.kind)
    out = _header_kwargs(annotation)
    if kind == "classification":
        out.update(
            # `labels` keys come back as class *names* where a label set
            # resolves them and as digit strings where none does; the writer
            # re-resolves names, so a digit string has to go back as an int or
            # it is looked up as a name that was never declared.
            labels={
                (int(k) if str(k).lstrip("-").isdigit() else k): float(v)
                for k, v in annotation.labels.items()
            },
            scope=str(annotation.scope),
            multilabel=bool(annotation.multilabel),
            scope_ids=(
                [int(v) for v in annotation.scope_ids]
                if annotation.scope_ids is not None
                else None
            ),
            schemes=list(annotation.schemes) if annotation.schemes else None,
            scheme_values=(
                list(annotation.scheme_values) if annotation.scheme_values else None
            ),
        )
        return out

    # Everything below is geometric, and geometry is stated in a space and a
    # frame of reference.  Dropping either would move the annotation.
    out.update(
        task=str(annotation.task),
        space=str(annotation.space),
        frame_uid=annotation.frame_uid or None,
    )
    scores = annotation.scores
    instances = annotation.instance_ids
    common = {
        "instance_ids": [int(v) for v in instances] if instances is not None else None,
        "scores": [float(v) for v in scores] if scores is not None else None,
    }
    if kind == "obb":
        out.update(
            centers=np.asarray(annotation.centers),
            sizes=np.asarray(annotation.sizes),
            rotations=np.asarray(annotation.rotations),
            class_ids=[int(c) for c in annotation.object_class_ids],
            attributes=annotation.attributes,
            **common,
        )
    elif kind == "keypoints":
        out.update(
            points=np.asarray(annotation.points),
            keypoint_classes=[int(c) for c in annotation.keypoint_class_ids],
            class_ids=[int(c) for c in annotation.object_class_ids],
            visibility=(
                np.asarray(annotation.visibility)
                if annotation.visibility is not None
                else None
            ),
            skeleton=annotation.skeleton_id or None,
            **common,
        )
    elif kind == "points":
        out.update(
            points=np.asarray(annotation.points),
            class_ids=[int(c) for c in annotation.object_class_ids],
            names=list(annotation.names) if annotation.names is not None else None,
            weights=(
                [float(w) for w in annotation.weights]
                if annotation.weights is not None
                else None
            ),
            correspondence=annotation.correspondence or None,
        )
    elif kind == "contours":
        out.update(polygons=list(annotation.polygons()))
    elif kind == "mesh":
        offsets = annotation.group.get("mesh_offsets") if annotation.group else None
        out.update(
            vertices=np.asarray(annotation.vertices),
            faces=np.asarray(annotation.faces),
            normals=(
                np.asarray(annotation.normals)
                if annotation.normals is not None
                else None
            ),
            vertex_class_ids=(
                [int(c) for c in annotation.vertex_class_ids]
                if annotation.vertex_class_ids is not None
                else None
            ),
            mesh_offsets=[int(v) for v in offsets[...]]
            if offsets is not None
            else None,
            mesh_class_ids=(
                [int(c) for c in annotation.object_class_ids]
                if _dataset_present(annotation, "mesh_class_ids")
                else None
            ),
        )
    else:  # pragma: no cover - guarded by the caller's CARRY_KINDS filter
        raise ValueError(f"no copy path for annotation kind {kind!r}")
    return out


def _carry(sample: Any, bundle: _Bundle) -> list[_Carried]:
    """Source annotations the write would otherwise lose.

    Both write paths build the destination from what is on screen, and only
    voxel and box annotations ever become layers.  Everything else was dropped
    by Save As without a warning --- silently, and in the operation a user
    reaches for to make a *copy*.

    Reproduced-or-decided annotations are excluded: the ones this write
    rebuilds from layers, and the ones the reader opened, which the user may
    have deliberately deleted (that is #6's contract, and it is the user's
    call). What is left is only ever a kind napari cannot show.
    """
    decided = set(bundle.labelmaps) | set(bundle.boxes) | bundle.opened
    out: list[_Carried] = []
    for name, annotation in sample.annotations.items():
        if name in decided or str(annotation.kind) not in CARRY_KINDS:
            continue
        out.append(
            _Carried(
                name=name,
                kind=str(annotation.kind),
                grid=annotation.grid_id,
                kwargs=_carry_kwargs(annotation),
            )
        )
    return out


def _write_carried(writer: Any, carried: list[_Carried], declared: set[str]) -> None:
    """Write each carried annotation back through its own `add_*`."""
    for one in carried:
        add = getattr(writer, f"add_{one.kind}")
        grid = one.grid if one.grid and one.grid in declared else None
        add(one.name, grid=grid, **one.kwargs)


def _write_new(dest: Path, bundle: _Bundle) -> None:
    """Write a fresh sample, carrying what the source can supply."""
    source = bundle.source_path
    document = None
    grids: dict[str, Any] = {}
    src = _Source()
    carried: list[_Carried] = []
    if source and Path(source).exists():
        with medh5.open(source) as sample:
            document = sample.document
            grids = {k: v for k, v in sample.grids.items()}
            # Save As reproduces every annotation, so it needs every mask.
            src = _capture(sample, bundle, always=True)
            # Read now, while the source is open: the payloads are geometry and
            # metadata rather than voxels, so holding them is cheap and beats
            # keeping a second handle open across `medh5.create`.
            carried = _carry(sample, bundle)
    for name in bundle.labelmaps:
        _refuse_lossy(name, src.kinds.get(name), name in src.opaque)

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
            # A copy of a reviewed sample is still reviewed.  Carrying only the
            # identity left the destination with no quality records and no
            # provenance but napari's own, so a sample somebody had approved
            # came out looking untouched --- and the audit trail that says who
            # drew what, a year later, is the thing this format is for.
            for agent_record in document.provenance.agents:
                writer.document.provenance.add_agent(agent_record)
            for prior in document.provenance.activities:
                writer.document.provenance.add_activity(prior)
            writer.document.quality.update(document.quality)
        else:
            writer.add_timepoint("tp0")

        agent = writer.software(AGENT, _version())
        activity = writer.activity("annotate", agent=agent, tool="napari")

        written_grids, declared_grids = _declare_grids(
            writer, bundle, grids, first, carried
        )

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
            current = np.asarray(labelmap, dtype=LABEL_DTYPE)
            masks, ignore, encoding = _resolve(src, name, current, meta)
            writer.add_segmentation(
                name,
                grid=_grid_for(meta, written_grids, declared_grids),
                masks=masks,
                ignore=ignore,
                encoding=encoding,
                annotated_classes=_annotated(meta, masks),
                # A copy carries the source's association and its review; the
                # napari activity only claims the ones napari actually touched.
                timepoints=src.timepoints.get(name) or None,
                quality=src.qualities.get(name),
                prov=(
                    activity
                    if src.edited(name, current)
                    else src.provs.get(name) or activity
                ),
            )

        for name, (data, kwargs, meta) in bundle.boxes.items():
            _write_boxes(
                writer,
                name,
                data,
                kwargs,
                _grid_for(meta, written_grids, declared_grids),
                activity,
                set(),
            )

        _write_carried(writer, carried, declared_grids)


def _declare_grids(
    writer: Any,
    bundle: _Bundle,
    source_grids: dict[str, Any],
    first: npt.NDArray[Any],
    carried: list[_Carried] | None = None,
) -> tuple[dict[str, str], set[str]]:
    """Declare one grid per distinct source grid, reusing its geometry.

    Geometry is never invented.  Without a source grid the fallback is unit
    spacing at the origin, which is what an image with no stated geometry
    actually means --- but a layer that *came* from a grid and no longer
    matches its shape is a different case, and it is refused.  See
    :func:`_refuse_reshaped`.
    """
    out: dict[str, str] = {}
    declared: set[str] = set()
    # Shape -> the grid id unclaimed layers of that shape share.  Layers napari
    # created carry no `medh5_grid`, so they all used to claim the literal id
    # `"grid"`: the first one won and the rest were assigned to it, which fails
    # outright on a shape mismatch and merges unrelated images where the shapes
    # happen to agree.  Unclaimed grids are unit spacing at the origin, so two
    # of the same shape genuinely are the same grid --- and two of different
    # shapes never are.
    unclaimed: dict[tuple[int, ...], str] = {}

    def unclaimed_id(name: str, shape: tuple[int, ...]) -> str:
        if shape in unclaimed:
            return unclaimed[shape]
        # The first one keeps the historical name, so a single-image write is
        # byte-for-byte what it was.
        candidate = "grid" if not unclaimed else f"grid_{_safe_id(name)}"
        taken = set(declared) | set(source_grids) | set(unclaimed.values())
        base, suffix = candidate, 2
        while candidate in taken:
            candidate = f"{base}_{suffix}"
            suffix += 1
        unclaimed[shape] = candidate
        return candidate

    def declare(
        grid_id: str, shape: tuple[int, ...] | None, *, claimed: bool = True
    ) -> None:
        if grid_id in declared:
            return
        source = source_grids.get(grid_id)
        if not claimed:
            # The layer did not name this grid --- `grid_id` is the fallback
            # name, and any source grid answering to it is a coincidence.  A
            # layer napari created has no geometry to lose, so there is nothing
            # here to refuse over.
            source = None
        if source is not None and (shape is None or tuple(source.shape) == shape):
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
        elif source is not None and shape is not None:
            _refuse_reshaped(grid_id, source, shape)
        elif shape is not None:
            writer.add_grid(grid_id, shape=shape, spacing=(1.0,) * len(shape))
        else:
            return
        declared.add(grid_id)

    for name, array in bundle.images.items():
        referenced = bundle.image_meta.get(name, {}).get("medh5_grid")
        shape = tuple(array.shape)
        grid_id = str(referenced) if referenced else unclaimed_id(name, shape)
        declare(grid_id, shape, claimed=bool(referenced))
        out[name] = grid_id

    # Annotations may sit on a grid no image uses --- a segmentation at its own
    # spacing, a detection on the grid it was run against.  Declaring only the
    # image grids left `_grid_for` to substitute the first image grid, which
    # either fails outright on a shape mismatch or, worse, succeeds and gives
    # the annotation somebody else's spacing, origin and frame of reference.
    for meta in (*bundle.seg_meta.values(), *(m for _, _, m in bundle.boxes.values())):
        referenced = meta.get("medh5_grid")
        if referenced:
            declare(str(referenced), None)
    # An annotation carried across whole may sit on a grid nothing on screen
    # uses --- a mesh at its own spacing, a classification scoped to a visit
    # whose images were not loaded.  Without this its grid would be missing and
    # the write would fail on a dangling reference.
    for one in carried or ():
        if one.grid:
            declare(str(one.grid), None)
    return out, declared


def _safe_id(name: str) -> str:
    """*name* reduced to the §2.3 identifier alphabet, `[A-Za-z0-9_.-]{1,128}`.

    A layer name reaches here from napari and is not an identifier, so it is
    sanitised rather than trusted; uniqueness is settled by the caller.
    """
    cleaned = "".join(
        c if c.isascii() and (c.isalnum() or c in "_.-") else "_" for c in name
    )
    return (cleaned[:120] or "image").lstrip(".")


def _refuse_reshaped(grid_id: str, source: Any, shape: tuple[int, ...]) -> None:
    """Refuse a layer whose shape no longer matches the grid it came from.

    The fallback for an unknown grid is unit spacing at the origin, and for a
    layer that never had geometry that is honest.  For a *cropped* one it is
    not: the source spacing, direction, coordinate system and frame of
    reference are all known, and replacing them with defaults leaves an image
    that opens cleanly and sits in the wrong place --- misregistered against
    the annotations drawn on it, with nothing to indicate anything was lost.

    Carrying the geometry across is no better, because a crop is not only a
    shape change: it moves the origin, and napari does not report which corner
    was cropped to.  There is no derivable answer here, and inventing one is
    what `medh5`'s own converters refuse to do (§3).
    """
    raise ValueError(
        f"layer on grid {grid_id!r} is {tuple(shape)}, but that grid is "
        f"{tuple(source.shape)}. A reshaped layer has no derivable geometry: "
        "cropping moves the origin and napari does not say which corner it "
        "kept, so the spacing, direction and frame of reference cannot be "
        "carried across and unit spacing would silently misregister the "
        "image. Save the layer at its original shape, or clear "
        "`layer.metadata['medh5_grid']` to write it as a new image with no "
        "stated geometry."
    )


def _grid_for(meta: dict[str, Any], written: dict[str, str], declared: set[str]) -> str:
    """The grid an annotation belongs on --- its own wherever that exists.

    Falling back to an image grid is a last resort for a layer naming a grid
    the source does not have, and it is a substitution: the annotation keeps
    its voxels and takes somebody else's spacing, origin and frame.
    """
    grid = meta.get("medh5_grid")
    if grid and str(grid) in declared:
        return str(grid)
    return next(iter(written.values()))


def _version() -> str:
    from napari_medh5 import __version__

    return __version__


__all__ = ["write_sample"]
