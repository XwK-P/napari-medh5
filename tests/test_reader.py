"""Reader contribution tests."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from napari_medh5._handles import REGISTRY
from napari_medh5._reader import napari_get_reader


@pytest.fixture(autouse=True)
def _close_registry() -> None:
    yield
    REGISTRY.close_all()


def test_returns_none_for_non_medh5(tmp_path: Path) -> None:
    other = tmp_path / "foo.nii.gz"
    other.write_bytes(b"")
    assert napari_get_reader(str(other)) is None


def test_reader_emits_layers(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))

    roles = [kwargs["metadata"]["medh5_role"] for _, kwargs, _ in layers]
    types = [t for _, _, t in layers]

    assert roles.count("image") == 2
    assert roles.count("seg") == 1
    assert roles.count("bbox_rect") == 1
    assert types.count("image") == 2
    assert types.count("labels") == 1
    assert types.count("shapes") >= 1


def test_images_are_dask_and_scaled(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))
    for data, kwargs, layer_type in layers:
        if layer_type == "image":
            assert isinstance(data, da.Array)
            assert kwargs["scale"] == [2.0, 1.0, 1.0]


def test_nnunet_labels_rename_seg_layer(nnunet_medh5: Path) -> None:
    reader = napari_get_reader(str(nnunet_medh5))
    assert reader is not None
    layers = reader(str(nnunet_medh5))
    seg_names = [
        kwargs["name"] for _, kwargs, layer_type in layers if layer_type == "labels"
    ]
    assert any(name.endswith(":tumor") for name in seg_names), seg_names


def test_fallback_layer_name_without_nnunet(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))
    seg_names = [
        kwargs["name"] for _, kwargs, layer_type in layers if layer_type == "labels"
    ]
    assert any(name.endswith(":tumor") for name in seg_names), seg_names


def test_handle_refcount_closes_file(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    reader(str(tiny_medh5))
    REGISTRY.release(str(tiny_medh5))
    assert REGISTRY.get(str(tiny_medh5)) is None


def test_attach_viewer_drops_handle_on_last_layer_removal(tiny_medh5: Path) -> None:
    from napari_medh5._handles import attach_viewer

    class _Event:
        def __init__(self) -> None:
            self._callbacks: list = []

        def connect(self, cb) -> None:  # type: ignore[no-untyped-def]
            self._callbacks.append(cb)

        def emit(self, value) -> None:  # type: ignore[no-untyped-def]
            class _Ev:
                pass

            ev = _Ev()
            ev.value = value  # type: ignore[attr-defined]
            for cb in self._callbacks:
                cb(ev)

    class _Events:
        def __init__(self) -> None:
            self.removed = _Event()

    class _Layer:
        def __init__(self, path: str) -> None:
            self.metadata = {"medh5_path": path}

    class _Layers(list):  # type: ignore[type-arg]
        def __init__(self) -> None:
            super().__init__()
            self.events = _Events()

    class _Viewer:
        def __init__(self) -> None:
            self.layers = _Layers()

    viewer = _Viewer()
    attach_viewer(viewer)

    path = str(tiny_medh5)
    REGISTRY.acquire(path)
    layer_a = _Layer(path)
    layer_b = _Layer(path)
    viewer.layers.extend([layer_a, layer_b])

    viewer.layers.remove(layer_a)
    viewer.layers.events.removed.emit(layer_a)
    assert REGISTRY.get(path) is not None, "handle should survive while a layer remains"

    viewer.layers.remove(layer_b)
    viewer.layers.events.removed.emit(layer_b)
    assert REGISTRY.get(path) is None, "handle should drop after last layer is removed"


def test_shapes_layer_has_features(tiny_medh5: Path) -> None:
    reader = napari_get_reader(str(tiny_medh5))
    assert reader is not None
    layers = reader(str(tiny_medh5))
    rect = next(
        (
            kwargs
            for _, kwargs, t in layers
            if t == "shapes" and kwargs["metadata"]["medh5_role"] == "bbox_rect"
        ),
        None,
    )
    assert rect is not None
    feats = rect["features"]
    assert "depth_axis" in feats
    assert "depth_lo" in feats
    assert "depth_hi" in feats
    assert np.array_equal(feats["depth_lo"], np.array([2.0, 1.0]))
    assert np.array_equal(feats["depth_hi"], np.array([5.0, 3.0]))
