"""The reader contribution and the shared handle registry."""

from __future__ import annotations

from typing import Any

import dask.array as da
import medh5
import numpy as np
import pytest

from napari_medh5._handles import REGISTRY, attach_viewer, rebind_viewer_layers
from napari_medh5._reader import napari_get_reader


class _Layer:
    def __init__(self, metadata: dict[str, Any], data: Any = None) -> None:
        self.metadata = metadata
        self.data = data


class _Event:
    def __init__(self) -> None:
        self._callbacks: list[Any] = []

    def connect(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def emit(self, value: Any) -> None:
        for callback in self._callbacks:
            callback(type("E", (), {"value": value})())


class _Layers(list):
    def __init__(self) -> None:
        super().__init__()
        self.events = type("Ev", (), {"removed": _Event(), "inserted": _Event()})()


class _Viewer:
    def __init__(self) -> None:
        self.layers = _Layers()


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    REGISTRY.close_all()


class TestDispatch:
    def test_a_non_medh5_path_is_declined(self, tmp_path):
        assert napari_get_reader(str(tmp_path / "x.nii.gz")) is None

    def test_a_medh5_path_is_accepted(self, tiny_medh5):
        assert callable(napari_get_reader(str(tiny_medh5)))

    def test_a_collection_is_accepted(self, tmp_path, tiny_medh5):
        shard = tmp_path / "shard.medh5c"
        medh5.pack([tiny_medh5], shard)
        assert callable(napari_get_reader(str(shard)))

    def test_an_empty_list_is_declined(self):
        assert napari_get_reader([]) is None

    def test_a_mixed_list_is_declined(self, tiny_medh5, tmp_path):
        assert napari_get_reader([str(tiny_medh5), str(tmp_path / "x.nii")]) is None


class TestRead:
    def test_it_emits_image_seg_and_box_layers(self, tiny_medh5):
        layers = napari_get_reader(str(tiny_medh5))(str(tiny_medh5))
        roles = [k["metadata"]["medh5_role"] for _, k, _ in layers]
        assert roles.count("image") == 2
        assert roles.count("seg") == 1
        assert roles.count("bbox_rect") == 1

    def test_images_are_lazy_and_carry_geometry(self, tiny_medh5):
        layers = napari_get_reader(str(tiny_medh5))(str(tiny_medh5))
        data, kwargs, kind = next(
            (d, k, t) for d, k, t in layers if k["metadata"]["medh5_role"] == "image"
        )
        assert kind == "image"
        assert isinstance(data, da.Array)
        assert kwargs["scale"] == [2.0, 1.0, 1.0]

    def test_reading_two_files_keeps_them_apart(self, tiny_medh5, rotated_medh5):
        layers = napari_get_reader([str(tiny_medh5)])(
            [str(tiny_medh5), str(rotated_medh5)]
        )
        sources = {k["metadata"]["medh5_path"] for _, k, _ in layers}
        assert sources == {str(tiny_medh5), str(rotated_medh5)}


class TestRegistry:
    def test_a_handle_is_shared_and_closed_on_the_last_release(self, tiny_medh5):
        first = REGISTRY.acquire(tiny_medh5)
        second = REGISTRY.acquire(tiny_medh5)
        assert first is second
        REGISTRY.release(tiny_medh5)
        assert REGISTRY.get(tiny_medh5) is not None
        REGISTRY.release(tiny_medh5)
        assert REGISTRY.get(tiny_medh5) is None

    def test_releasing_an_unknown_path_is_harmless(self, tmp_path):
        REGISTRY.release(tmp_path / "nope.medh5")

    def test_dropping_ignores_the_refcount(self, tiny_medh5):
        REGISTRY.acquire(tiny_medh5)
        REGISTRY.acquire(tiny_medh5)
        REGISTRY.drop(tiny_medh5)
        assert REGISTRY.get(tiny_medh5) is None

    def test_removing_the_last_layer_drops_the_handle(self, tiny_medh5):
        viewer = _Viewer()
        attach_viewer(viewer)
        REGISTRY.acquire(tiny_medh5)
        layer = _Layer({"medh5_path": str(tiny_medh5), "medh5_role": "image"})
        viewer.layers.append(layer)
        viewer.layers.remove(layer)
        viewer.layers.events.removed.emit(layer)
        assert REGISTRY.get(tiny_medh5) is None

    def test_another_layer_of_the_same_file_holds_it_open(self, tiny_medh5):
        viewer = _Viewer()
        attach_viewer(viewer)
        REGISTRY.acquire(tiny_medh5)
        first = _Layer({"medh5_path": str(tiny_medh5), "medh5_role": "image"})
        second = _Layer({"medh5_path": str(tiny_medh5), "medh5_role": "seg"})
        viewer.layers.extend([first, second])
        viewer.layers.remove(first)
        viewer.layers.events.removed.emit(first)
        assert REGISTRY.get(tiny_medh5) is not None

    def test_a_relative_path_resolves_to_the_same_entry(self, tiny_medh5, monkeypatch):
        """The reader stores what it was handed; the registry keys on resolved."""
        monkeypatch.chdir(tiny_medh5.parent)
        viewer = _Viewer()
        attach_viewer(viewer)
        REGISTRY.acquire(tiny_medh5)
        layer = _Layer({"medh5_path": tiny_medh5.name, "medh5_role": "image"})
        viewer.layers.append(layer)
        viewer.layers.remove(layer)
        viewer.layers.events.removed.emit(layer)
        assert REGISTRY.get(tiny_medh5) is None

    def test_a_second_viewer_of_the_same_file_holds_it_open(self, tiny_medh5):
        """The registry is process-global; the predicate was per-viewer.

        Two windows on one `.medh5` share one handle.  Closing the last layer
        in one dropped it while the other's lazy arrays were still bound to it,
        so those failed on the next slice --- in a window the user never
        touched.
        """
        first, second = _Viewer(), _Viewer()
        attach_viewer(first)
        attach_viewer(second)
        REGISTRY.acquire(tiny_medh5)
        here = _Layer({"medh5_path": str(tiny_medh5), "medh5_role": "image"})
        there = _Layer({"medh5_path": str(tiny_medh5), "medh5_role": "image"})
        first.layers.append(here)
        second.layers.append(there)

        first.layers.remove(here)
        first.layers.events.removed.emit(here)
        assert REGISTRY.get(tiny_medh5) is not None

        second.layers.remove(there)
        second.layers.events.removed.emit(there)
        assert REGISTRY.get(tiny_medh5) is None

    def test_attach_is_idempotent(self, tiny_medh5):
        viewer = _Viewer()
        attach_viewer(viewer)
        attach_viewer(viewer)
        assert len(viewer.layers.events.removed._callbacks) == 1

    def test_attach_tolerates_something_that_is_not_a_viewer(self):
        attach_viewer(None)
        attach_viewer(object())


class TestRebind:
    def test_it_swaps_data_on_matching_layers(self, tiny_medh5):
        viewer = _Viewer()
        attach_viewer(viewer)
        layer = _Layer(
            {
                "medh5_path": str(tiny_medh5),
                "medh5_role": "image",
                "medh5_name": "CT",
            },
            data=None,
        )
        viewer.layers.append(layer)
        rebind_viewer_layers(tiny_medh5, viewer)
        assert isinstance(layer.data, da.Array)

    def test_it_rebinds_segmentation_layers_too(self, tiny_medh5):
        viewer = _Viewer()
        attach_viewer(viewer)
        layer = _Layer(
            {
                "medh5_path": str(tiny_medh5),
                "medh5_role": "seg",
                "medh5_name": "seg",
            }
        )
        viewer.layers.append(layer)
        rebind_viewer_layers(tiny_medh5, viewer)
        assert np.asarray(layer.data).max() > 0

    def test_it_reaches_every_attached_viewer(self, tiny_medh5):
        """One drop can strand layers in several windows of one process."""
        first, second = _Viewer(), _Viewer()
        attach_viewer(first)
        attach_viewer(second)
        layers = []
        for viewer in (first, second):
            layer = _Layer(
                {
                    "medh5_path": str(tiny_medh5),
                    "medh5_role": "image",
                    "medh5_name": "CT",
                }
            )
            viewer.layers.append(layer)
            layers.append(layer)
        rebind_viewer_layers(tiny_medh5)
        assert all(isinstance(one.data, da.Array) for one in layers)

    def test_a_relative_medh5_path_still_matches(self, tiny_medh5, monkeypatch):
        monkeypatch.chdir(tiny_medh5.parent)
        viewer = _Viewer()
        attach_viewer(viewer)
        layer = _Layer(
            {
                "medh5_path": tiny_medh5.name,
                "medh5_role": "image",
                "medh5_name": "CT",
            }
        )
        viewer.layers.append(layer)
        rebind_viewer_layers(tiny_medh5, viewer)
        assert isinstance(layer.data, da.Array)

    def test_no_viewers_is_a_no_op(self, tiny_medh5):
        rebind_viewer_layers(tiny_medh5)

    def test_an_unknown_object_name_is_skipped(self, tiny_medh5):
        viewer = _Viewer()
        attach_viewer(viewer)
        layer = _Layer(
            {
                "medh5_path": str(tiny_medh5),
                "medh5_role": "image",
                "medh5_name": "GONE",
            }
        )
        viewer.layers.append(layer)
        rebind_viewer_layers(tiny_medh5, viewer)
        assert layer.data is None
