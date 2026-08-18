"""Translate a medh5 :class:`~medh5.Sample` into napari layer-data tuples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from napari_medh5._arrays import annotation_array, image_array
from napari_medh5._bbox import boxes_to_shapes
from napari_medh5._types import LayerDataTuple

VOXEL_KINDS = frozenset(
    {"labelmap", "layers", "bitmask", "instances", "probmap", "mask"}
)
BOX_KINDS = frozenset({"boxes"})


def grid_kwargs(grid: Any) -> dict[str, Any]:
    """``scale``/``translate``, or a single ``affine`` when there is rotation.

    napari composes ``scale`` and ``translate`` in array order and applies no
    rotation, so a grid with a non-identity ``direction`` has to go through the
    homogeneous affine instead.  Mixing the two --- passing ``scale`` *and* an
    oblique ``affine`` --- silently double-applies the spacing.
    """
    direction = np.asarray(grid.direction, dtype=np.float64)
    spatial = tuple(grid.spatial_axes)
    if not np.allclose(direction, np.eye(direction.shape[0])):
        return {"affine": np.asarray(grid.affine, dtype=np.float64)}
    scale = [1.0] * grid.ndim
    translate = [0.0] * grid.ndim
    for position, axis in enumerate(spatial):
        scale[axis] = float(grid.spacing[position])
        translate[axis] = float(grid.origin[position])
    return {"scale": scale, "translate": translate}


def _class_names(sample: Any, annotation: Any) -> dict[int, str]:
    """``{class_id: display name}`` from the sample's label set."""
    label_set = sample.label_set
    if label_set is None:
        return {}
    out: dict[int, str] = {}
    for class_id in annotation.class_ids:
        try:
            entry = label_set[int(class_id)]
        except (KeyError, IndexError):
            continue
        out[int(class_id)] = entry.name or entry.key
    return out


def _suffix(timepoint: str | None, multi: bool) -> str:
    """Tag a layer with its visit, but only when the sample has more than one."""
    return f" [{timepoint}]" if multi and timepoint else ""


def sample_to_layers(sample: Any, path: str | Path) -> list[LayerDataTuple]:
    """Convert an open sample into napari layer tuples."""
    path = Path(path)
    stem = path.stem
    multi = len(sample.timepoints) > 1
    layers: list[LayerDataTuple] = []

    for name, image in sample.images.items():
        grid = image.grid
        layers.append(
            (
                image_array(image),
                {
                    "name": f"{stem}:{name}{_suffix(image.timepoint, multi)}",
                    "metadata": {
                        "medh5_path": str(path),
                        "medh5_role": "image",
                        "medh5_name": name,
                        "medh5_grid": image.grid_id,
                        "medh5_timepoint": image.timepoint,
                        "modality": image.modality,
                        "value_units": image.value_units,
                    },
                    **grid_kwargs(grid),
                },
                "image",
            )
        )

    for name, annotation in sample.annotations.items():
        if annotation.kind in VOXEL_KINDS:
            timepoint = next(iter(annotation.timepoints), None)
            layers.append(
                (
                    annotation_array(annotation),
                    {
                        "name": f"{stem}:seg:{name}{_suffix(timepoint, multi)}",
                        "metadata": {
                            "medh5_path": str(path),
                            "medh5_role": "seg",
                            "medh5_name": name,
                            "medh5_grid": annotation.grid_id,
                            "medh5_timepoint": timepoint,
                            "medh5_kind": annotation.kind,
                            "medh5_classes": _class_names(sample, annotation),
                            # What was *looked for*, not just what is here: a
                            # reviewer editing a mask needs to know whether a
                            # missing class was searched for (§11.3).
                            "medh5_annotated": [
                                int(c) for c in annotation.annotated_class_ids
                            ],
                        },
                        **grid_kwargs(annotation.grid),
                    },
                    "labels",
                )
            )
        elif annotation.kind in BOX_KINDS:
            layers.extend(
                boxes_to_shapes(
                    annotation,
                    names=_class_names(sample, annotation),
                    layer_kwargs=grid_kwargs(annotation.grid),
                    path=str(path),
                    stem=stem,
                    suffix=_suffix(next(iter(annotation.timepoints), None), multi),
                )
            )

    return layers


__all__ = ["BOX_KINDS", "VOXEL_KINDS", "grid_kwargs", "sample_to_layers"]
