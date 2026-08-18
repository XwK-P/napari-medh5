"""End-to-end through a real ``napari.Viewer``.

The unit tests check the translation; these check that napari accepts it ---
that the reader contribution resolves, the geometry survives, a save round
trips, and a lazy layer still reads after the file underneath it is replaced.

Headless via ``QT_QPA_PLATFORM=offscreen`` (set in ``conftest.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

import dask.array as da  # noqa: E402
import medh5  # noqa: E402

from napari_medh5._handles import REGISTRY  # noqa: E402
from napari_medh5._widget import MEDH5Widget  # noqa: E402
from napari_medh5._writer import write_sample  # noqa: E402

pytestmark = pytest.mark.integration

PLUGIN = "napari-medh5"


@pytest.fixture(autouse=True)
def _registry_reset() -> Any:
    yield
    REGISTRY.close_all()


def _roles(viewer: Any) -> list[str]:
    return [(layer.metadata or {}).get("medh5_role", "") for layer in viewer.layers]


def _by_role(viewer: Any, role: str) -> Any:
    return next(
        layer for layer in viewer.layers if layer.metadata.get("medh5_role") == role
    )


def _layer_tuples(viewer: Any) -> list[Any]:
    """The layer list in the shape napari hands a writer."""
    out = []
    for layer in viewer.layers:
        kind = type(layer).__name__.lower()
        kwargs: dict[str, Any] = {"name": layer.name, "metadata": dict(layer.metadata)}
        if kind == "shapes":
            kwargs["shape_type"] = list(layer.shape_type)
            kwargs["features"] = {
                key: list(values) for key, values in layer.features.items()
            }
            data = list(layer.data)
        else:
            data = np.asarray(layer.data)
        out.append((data, kwargs, kind))
    return out


class TestReader:
    def test_opening_populates_every_layer_kind(self, real_viewer, tiny_medh5):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        roles = _roles(real_viewer)
        assert roles.count("image") == 2
        assert roles.count("seg") == 1
        assert roles.count("bbox_rect") == 1

    def test_images_stay_lazy_in_the_viewer(self, real_viewer, tiny_medh5):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        assert isinstance(_by_role(real_viewer, "image").data, da.Array)

    def test_scale_reaches_the_layer(self, real_viewer, tiny_medh5):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        assert list(_by_role(real_viewer, "image").scale) == [2.0, 1.0, 1.0]

    def test_an_oblique_grid_arrives_as_an_affine(self, real_viewer, rotated_medh5):
        real_viewer.open(str(rotated_medh5), plugin=PLUGIN)
        affine = np.asarray(_by_role(real_viewer, "image").affine.affine_matrix)
        assert not np.allclose(affine[:3, :3], np.eye(3) * affine[0, 0])

    def test_a_deep_box_brings_its_wireframe(self, real_viewer, deep_bbox_medh5):
        real_viewer.open(str(deep_bbox_medh5), plugin=PLUGIN)
        assert "bbox_wire" in _roles(real_viewer)

    def test_a_longitudinal_sample_shows_every_visit(
        self, real_viewer, longitudinal_medh5
    ):
        real_viewer.open(str(longitudinal_medh5), plugin=PLUGIN)
        names = [layer.name for layer in real_viewer.layers]
        assert any("[tp0]" in n for n in names)
        assert any("[tp1]" in n for n in names)

    def test_the_labels_layer_shows_class_ids(self, real_viewer, tiny_medh5):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        data = np.asarray(_by_role(real_viewer, "seg").data)
        assert set(np.unique(data)) == {0, 1, 3}
        # The tumor is entirely inside the organ; if the organ were painted on
        # top it would vanish, which is what draw_priority prevents.
        assert int(data[3, 6, 6]) == 1

    def test_removing_every_layer_drops_the_handle(self, real_viewer, tiny_medh5):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        while real_viewer.layers:
            real_viewer.layers.pop()
        assert REGISTRY.get(tiny_medh5) is None

    def test_opening_the_same_file_twice_does_not_break(self, real_viewer, tiny_medh5):
        before = len(real_viewer.layers)
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        opened = len(real_viewer.layers) - before
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        assert len(real_viewer.layers) == before + 2 * opened


class TestWriter:
    def test_a_seg_edit_survives_a_save(self, real_viewer, tiny_medh5):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        layer = _by_role(real_viewer, "seg")
        edited = np.asarray(layer.data)
        edited[0, 0, 0] = 1
        layer.data = edited
        write_sample(str(tiny_medh5), _layer_tuples(real_viewer))
        with medh5.open(tiny_medh5) as sample:
            assert sample.annotations["seg"].contains(1, (0, 0, 0))

    def test_a_lazy_layer_reads_after_the_file_is_replaced(
        self, real_viewer, tiny_medh5
    ):
        """`amend` swaps the inode; without a rebind the layer serves stale data."""
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        image = _by_role(real_viewer, "image")
        write_sample(str(tiny_medh5), _layer_tuples(real_viewer))
        assert np.asarray(image.data[0:2]).shape == (2, 16, 16)

    def test_save_as_writes_a_second_file(self, real_viewer, tiny_medh5, tmp_path):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), _layer_tuples(real_viewer))
        with medh5.open(dest) as sample:
            assert set(sample.images) == {"CT", "PET"}
            assert sample.grids["g_tp0"].spacing == (2.0, 1.0, 1.0)

    def test_boxes_round_trip_through_the_viewer(self, real_viewer, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            before = np.asarray(sample.annotations["boxes"].boxes).copy()
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        write_sample(str(tiny_medh5), _layer_tuples(real_viewer))
        with medh5.open(tiny_medh5) as sample:
            assert np.allclose(np.asarray(sample.annotations["boxes"].boxes), before)

    def test_layers_from_two_files_are_refused(
        self, real_viewer, tiny_medh5, rotated_medh5
    ):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        real_viewer.open(str(rotated_medh5), plugin=PLUGIN)
        with pytest.raises(ValueError, match="more than one"):
            write_sample(str(tiny_medh5), _layer_tuples(real_viewer))

    def test_a_renamed_modality_is_refused(self, real_viewer, tiny_medh5):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        layers = _layer_tuples(real_viewer)
        for _, kwargs, _ in layers:
            if kwargs["metadata"].get("medh5_name") == "PET":
                kwargs["metadata"]["medh5_name"] = "MR"
        with pytest.raises(ValueError, match="image set differs"):
            write_sample(str(tiny_medh5), layers)

    def test_the_saved_file_still_validates(self, real_viewer, tiny_medh5):
        from medh5.validate import validate_file

        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        write_sample(str(tiny_medh5), _layer_tuples(real_viewer))
        assert not validate_file(tiny_medh5).errors


class TestWidget:
    def test_it_picks_up_the_open_sample(self, real_viewer, tiny_medh5):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        widget = MEDH5Widget(real_viewer)
        assert widget._active_path == str(tiny_medh5)
        assert widget._sample_picker.count() == 1

    def test_it_switches_between_two_samples(
        self, real_viewer, tiny_medh5, rotated_medh5
    ):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        real_viewer.open(str(rotated_medh5), plugin=PLUGIN)
        widget = MEDH5Widget(real_viewer)
        assert widget._sample_picker.count() == 2
        widget._sample_picker.setCurrentText(str(rotated_medh5))
        assert widget._active_path == str(rotated_medh5)

    def test_a_tampered_file_is_reported(self, real_viewer, corrupt_medh5):
        real_viewer.open(str(corrupt_medh5), plugin=PLUGIN)
        widget = MEDH5Widget(real_viewer)
        widget._verify()
        assert "MISMATCH" in widget._integrity_label.text()

    def test_a_quality_record_round_trips_through_the_widget(
        self, real_viewer, tiny_medh5
    ):
        real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
        widget = MEDH5Widget(real_viewer)
        widget._annotation_picker.setCurrentText("seg")
        widget._status_picker.setCurrentText("approved")
        widget._save_quality()

        with medh5.open(tiny_medh5) as sample:
            assert sample.document.quality["seg"].status == "approved"
        # The write replaced the file; the layers must still read.
        assert np.asarray(_by_role(real_viewer, "image").data[0:1]).shape == (
            1,
            16,
            16,
        )
