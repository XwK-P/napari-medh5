"""Edge-case coverage for the writer module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from medh5 import MEDH5File

from napari_medh5._handles import REGISTRY
from napari_medh5._reader import napari_get_reader
from napari_medh5._writer import (
    _features_as_dict,
    _meta_dict,
    _save_full,
    write_sample,
)


@pytest.fixture(autouse=True)
def _close_registry() -> None:
    yield
    REGISTRY.close_all()


def test_write_sample_appends_missing_extension(
    tiny_medh5: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "no_suffix"
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))
    try:
        out = write_sample(str(dest), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))
    assert out == [str(dest.with_suffix(".medh5"))]
    assert dest.with_suffix(".medh5").exists()


def test_write_sample_rejects_empty_image_set(tmp_path: Path) -> None:
    dest = tmp_path / "out.medh5"
    with pytest.raises(ValueError, match="No image layers"):
        write_sample(str(dest), [])


def test_write_sample_rejects_multi_source(
    tiny_medh5: Path, nnunet_medh5: Path, tmp_path: Path
) -> None:
    reader1 = napari_get_reader(str(tiny_medh5))
    reader2 = napari_get_reader(str(nnunet_medh5))
    assert reader1 is not None and reader2 is not None
    layers = reader1(str(tiny_medh5)) + reader2(str(nnunet_medh5))
    try:
        with pytest.raises(ValueError, match="multiple .medh5"):
            write_sample(str(tmp_path / "out.medh5"), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))
        REGISTRY.release(str(nnunet_medh5))


def test_in_place_clears_bboxes_when_layer_removed(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = [
        tpl
        for tpl in reader(str(tiny_medh5))
        if tpl[1].get("metadata", {}).get("medh5_role") != "bbox_rect"
        and tpl[1].get("metadata", {}).get("medh5_role") != "bbox_wire"
    ]
    try:
        write_sample(str(tiny_medh5), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))
    sample = MEDH5File.read(tiny_medh5)
    assert sample.bboxes is None


def test_in_place_adds_new_seg_class(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = list(reader(str(tiny_medh5)))
    new_mask = np.zeros((8, 16, 16), dtype=bool)
    new_mask[3, 3, 3] = True
    layers.append(
        (
            new_mask,
            {
                "metadata": {
                    "medh5_path": str(tiny_medh5),
                    "medh5_role": "seg",
                    "medh5_name": "new_class",
                },
            },
            "labels",
        )
    )
    try:
        write_sample(str(tiny_medh5), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))
    sample = MEDH5File.read(tiny_medh5)
    assert sample.seg is not None
    assert "new_class" in sample.seg


def test_in_place_removes_dropped_seg_class(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = [
        tpl
        for tpl in reader(str(tiny_medh5))
        if tpl[1].get("metadata", {}).get("medh5_role") != "seg"
    ]
    try:
        write_sample(str(tiny_medh5), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))
    sample = MEDH5File.read(tiny_medh5)
    assert sample.seg is None or "tumor" not in sample.seg


def test_in_place_shape_mismatch_raises(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = list(reader(str(tiny_medh5)))
    # Swap one image layer's array for a differently-shaped numpy array.
    for i, (_data, kwargs, layer_type) in enumerate(layers):
        if kwargs.get("metadata", {}).get("medh5_role") == "image":
            layers[i] = (np.zeros((4, 4, 4), dtype=np.float32), kwargs, layer_type)
            break
    try:
        with pytest.raises(ValueError, match="shape"):
            write_sample(str(tiny_medh5), layers)
    finally:
        REGISTRY.release(str(tiny_medh5))


def test_meta_dict_returns_empty_for_non_dict() -> None:
    assert _meta_dict({"metadata": "not a dict"}) == {}
    assert _meta_dict({}) == {}


def test_features_as_dict_handles_pandas_like() -> None:
    class _Frame:
        def to_dict(self, orient: str = "list") -> dict[str, Any]:
            assert orient == "list"
            return {"score": [0.1, 0.2]}

    assert _features_as_dict(_Frame()) == {"score": [0.1, 0.2]}
    assert _features_as_dict(None) is None
    assert _features_as_dict(42) is None


def test_save_full_without_source_writes_minimal(tmp_path: Path) -> None:
    from napari_medh5._writer import _Bundle

    dest = tmp_path / "fresh.medh5"
    bundle = _Bundle(
        images={"CT": np.zeros((4, 4, 4), dtype=np.float32)},
        seg={},
        bboxes=None,
        bbox_scores=None,
        bbox_labels=None,
        sample_shape=[4, 4, 4],
        source_path=None,
    )
    _save_full(dest, bundle)
    assert dest.exists()
    meta = MEDH5File.read_meta(dest)
    assert meta.image_names == ["CT"]
    assert meta.spatial.spacing is None
