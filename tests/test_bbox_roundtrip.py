"""Round-trip: write → read into napari layers → save back → read numerically."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from medh5 import MEDH5File

from napari_medh5._handles import REGISTRY
from napari_medh5._reader import napari_get_reader
from napari_medh5._writer import write_sample


def _load(path: Path) -> list[tuple]:
    reader = napari_get_reader(str(path))
    assert reader is not None
    return reader(str(path))


def test_in_place_bbox_roundtrip(tiny_medh5: Path) -> None:
    layers = _load(tiny_medh5)
    try:
        write_sample(str(tiny_medh5), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))

    sample = MEDH5File.read(tiny_medh5)
    assert sample.bboxes is not None
    assert sample.bboxes.shape == (2, 3, 2)
    expected = np.array(
        [
            [[2, 5], [4, 10], [4, 10]],
            [[1, 3], [1, 5], [1, 5]],
        ],
        dtype=np.float64,
    )
    assert np.allclose(sample.bboxes, expected)
    assert sample.bbox_labels == ["tumor", "incidental"]


def test_save_as_creates_new_file(tiny_medh5: Path, tmp_path: Path) -> None:
    dest = tmp_path / "copy.medh5"
    layers = _load(tiny_medh5)
    try:
        out = write_sample(str(dest), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))

    assert out == [str(dest)]
    copied = MEDH5File.read(dest)
    original = MEDH5File.read(tiny_medh5)
    assert sorted(copied.images) == sorted(original.images)
    assert copied.bboxes is not None and original.bboxes is not None
    assert np.allclose(copied.bboxes, original.bboxes)
    assert copied.meta.spatial.spacing == original.meta.spatial.spacing


def test_save_blocks_on_modality_rename(tiny_medh5: Path) -> None:
    layers = _load(tiny_medh5)
    # Rename the first image layer's tagged modality to an unknown name.
    for _, kwargs, layer_type in layers:
        if layer_type == "image":
            kwargs["metadata"]["medh5_name"] = "CT_renamed"
            break
    try:
        try:
            write_sample(str(tiny_medh5), layers)
        except ValueError as exc:
            assert "Image modalities differ" in str(exc)
        else:
            raise AssertionError("Expected ValueError for modality rename")
    finally:
        REGISTRY.release(str(tiny_medh5))
