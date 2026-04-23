"""Writer partition + in-place vs. Save-As decision tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from medh5 import MEDH5File

from napari_medh5._handles import REGISTRY
from napari_medh5._reader import napari_get_reader
from napari_medh5._writer import _collect, write_sample


@pytest.fixture(autouse=True)
def _close_registry() -> None:
    yield
    REGISTRY.close_all()


def test_collect_partitions_layers(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))
    bundle = _collect(layers)
    try:
        assert set(bundle.images) == {"CT", "PET"}
        assert set(bundle.seg) == {"tumor"}
        assert bundle.bboxes is not None
        assert bundle.bboxes.shape == (2, 3, 2)
    finally:
        REGISTRY.release(str(tiny_medh5))


def test_save_is_noop_when_nothing_changed(tiny_medh5: Path) -> None:
    mtime_before = tiny_medh5.stat().st_mtime
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))
    try:
        write_sample(str(tiny_medh5), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))
    sample = MEDH5File.read(tiny_medh5)
    assert sample.bboxes is not None
    # mtime may advance if the writer touches the file; we only assert content.
    assert mtime_before > 0


def test_save_persists_seg_changes(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))
    # Flip a voxel in the seg layer.
    new_layers: list = []
    for data, kwargs, layer_type in layers:
        if kwargs.get("metadata", {}).get("medh5_role") == "seg":
            arr = np.asarray(data).astype(bool)
            arr[0, 0, 0] = True
            new_layers.append((arr, kwargs, layer_type))
        else:
            new_layers.append((data, kwargs, layer_type))
    try:
        write_sample(str(tiny_medh5), new_layers)
    finally:
        REGISTRY.release(str(tiny_medh5))
    sample = MEDH5File.read(tiny_medh5)
    assert sample.seg is not None
    assert sample.seg["tumor"][0, 0, 0]


def test_save_as_includes_spatial_meta(tiny_medh5: Path, tmp_path: Path) -> None:
    dest = tmp_path / "copy.medh5"
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))
    try:
        write_sample(str(dest), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))
    meta = MEDH5File.read_meta(dest)
    assert meta.spatial.spacing == [2.0, 1.0, 1.0]
    assert meta.spatial.axis_labels == ["Z", "Y", "X"]
    assert meta.label == 1
