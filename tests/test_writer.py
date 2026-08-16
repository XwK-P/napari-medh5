"""The writer contribution: amend in place, or write a new sample."""

from __future__ import annotations

from typing import Any

import medh5
import numpy as np
import pytest

from napari_medh5._handles import REGISTRY
from napari_medh5._reader import napari_get_reader
from napari_medh5._writer import _collect, _masks_from, _meta_dict, write_sample


def read_layers(path) -> list[Any]:
    return napari_get_reader(str(path))(str(path))


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    REGISTRY.close_all()


def materialise(layers: list[Any]) -> list[Any]:
    """Turn lazy layer data into numpy, the way napari hands it to a writer."""
    out = []
    for data, kwargs, kind in layers:
        out.append(
            (np.asarray(data) if hasattr(data, "compute") else data, kwargs, kind)
        )
    return out


class TestCollect:
    def test_layers_are_partitioned_by_role(self, tiny_medh5):
        bundle = _collect(materialise(read_layers(tiny_medh5)))
        assert set(bundle.images) == {"CT", "PET"}
        assert set(bundle.labelmaps) == {"seg"}
        assert set(bundle.boxes) == {"boxes"}
        assert bundle.source_path == str(tiny_medh5)

    def test_a_wireframe_layer_is_never_a_source_of_truth(self, deep_bbox_medh5):
        bundle = _collect(materialise(read_layers(deep_bbox_medh5)))
        assert set(bundle.boxes) == {"boxes"}

    def test_untagged_layers_are_ignored(self):
        bundle = _collect([(np.zeros((2, 2)), {"name": "scratch"}, "image")])
        assert not bundle.images

    def test_two_source_files_are_refused(self, tiny_medh5, rotated_medh5):
        layers = materialise(read_layers(tiny_medh5)) + materialise(
            read_layers(rotated_medh5)
        )
        with pytest.raises(ValueError, match="more than one"):
            _collect(layers)

    def test_a_non_dict_metadata_is_tolerated(self):
        assert _meta_dict({"metadata": "nope"}) == {}


class TestMasksFromLabelmap:
    def test_S11_3_an_emptied_class_stays_declared(self):
        """Erasing a mask must not turn "examined and absent" into "unexamined"."""
        labelmap = np.zeros((4, 4, 4), np.uint16)
        labelmap[1:3, 1:3, 1:3] = 1
        masks = _masks_from(labelmap, {"medh5_annotated": [1, 2, 3]})
        assert sorted(masks) == [1, 2, 3]
        assert masks[2].sum() == 0

    def test_a_class_painted_in_the_viewer_is_picked_up(self):
        labelmap = np.zeros((4, 4, 4), np.uint16)
        labelmap[0, 0, 0] = 7
        masks = _masks_from(labelmap, {"medh5_annotated": [1]})
        assert sorted(masks) == [1, 7]

    def test_background_and_ignore_are_not_classes(self):
        labelmap = np.zeros((4, 4, 4), np.uint16)
        labelmap[0, 0, 0] = 65535
        assert _masks_from(labelmap, {}) == {}


class TestAmend:
    def test_a_seg_edit_persists(self, tiny_medh5):
        layers = materialise(read_layers(tiny_medh5))
        for data, kwargs, _kind in layers:
            if kwargs["metadata"].get("medh5_role") == "seg":
                data[0, 0, 0] = 1
        write_sample(str(tiny_medh5), layers)
        with medh5.open(tiny_medh5) as sample:
            assert sample.annotations["seg"].contains(1, (0, 0, 0))

    def test_everything_else_survives_the_amend(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            before = sample.document.identity.subject_id
            images = {k: v.read() for k, v in sample.images.items()}
        write_sample(str(tiny_medh5), materialise(read_layers(tiny_medh5)))
        with medh5.open(tiny_medh5) as sample:
            assert sample.document.identity.subject_id == before
            for name, array in images.items():
                assert np.array_equal(sample.images[name].read(), array)

    def test_the_edit_is_recorded_in_provenance(self, tiny_medh5):
        write_sample(str(tiny_medh5), materialise(read_layers(tiny_medh5)))
        with medh5.open(tiny_medh5) as sample:
            tools = [a.tool for a in sample.document.provenance.activities]
            assert "napari" in tools

    def test_boxes_round_trip_through_the_viewer(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            before = np.asarray(sample.annotations["boxes"].boxes).copy()
        write_sample(str(tiny_medh5), materialise(read_layers(tiny_medh5)))
        with medh5.open(tiny_medh5) as sample:
            assert np.allclose(np.asarray(sample.annotations["boxes"].boxes), before)

    def test_deleting_every_box_removes_the_annotation(self, tiny_medh5):
        layers = materialise(read_layers(tiny_medh5))
        emptied = [
            ([], k, t) if k["metadata"].get("medh5_role") == "bbox_rect" else (d, k, t)
            for d, k, t in layers
        ]
        write_sample(str(tiny_medh5), emptied)
        with medh5.open(tiny_medh5) as sample:
            assert "boxes" not in sample.annotations

    def test_a_changed_image_set_is_refused(self, tiny_medh5):
        layers = [
            (d, k, t)
            for d, k, t in materialise(read_layers(tiny_medh5))
            if k["metadata"].get("medh5_name") != "PET"
        ]
        with pytest.raises(ValueError, match="image set differs"):
            write_sample(str(tiny_medh5), layers)

    def test_a_changed_image_shape_is_refused(self, tiny_medh5):
        layers = []
        for data, kwargs, kind in materialise(read_layers(tiny_medh5)):
            if kwargs["metadata"].get("medh5_name") == "CT":
                data = data[:4]
            layers.append((data, kwargs, kind))
        with pytest.raises(ValueError, match="use Save As"):
            write_sample(str(tiny_medh5), layers)

    def test_saving_with_no_images_is_refused(self):
        with pytest.raises(ValueError, match="no image layers"):
            write_sample("out.medh5", [])

    def test_the_result_still_validates(self, tiny_medh5):
        from medh5.validate import validate_file

        write_sample(str(tiny_medh5), materialise(read_layers(tiny_medh5)))
        assert not validate_file(tiny_medh5).errors


class TestSaveAs:
    def test_it_writes_a_new_file(self, tiny_medh5, tmp_path):
        dest = tmp_path / "copy.medh5"
        assert write_sample(str(dest), materialise(read_layers(tiny_medh5))) == [
            str(dest)
        ]
        assert dest.exists()

    def test_a_missing_extension_is_added(self, tiny_medh5, tmp_path):
        written = write_sample(
            str(tmp_path / "copy"), materialise(read_layers(tiny_medh5))
        )
        assert written[0].endswith(".medh5")

    def test_geometry_and_identity_come_from_the_source(self, tiny_medh5, tmp_path):
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(tiny_medh5)))
        with medh5.open(tiny_medh5) as before, medh5.open(dest) as after:
            assert after.identity.subject_id == before.identity.subject_id
            assert after.grids["g_tp0"].spacing == before.grids["g_tp0"].spacing
            assert after.grids["g_tp0"].origin == before.grids["g_tp0"].origin
            assert after.label_set.digest() == before.label_set.digest()

    def test_an_oblique_direction_survives(self, rotated_medh5, tmp_path):
        dest = tmp_path / "rot-copy.medh5"
        write_sample(str(dest), materialise(read_layers(rotated_medh5)))
        with medh5.open(rotated_medh5) as before, medh5.open(dest) as after:
            assert np.allclose(
                after.grids["g_tp0"].direction, before.grids["g_tp0"].direction
            )

    def test_a_sample_written_from_nothing_still_validates(self, tmp_path):
        from medh5.validate import validate_file

        dest = tmp_path / "scratch.medh5"
        layers = [
            (
                np.zeros((4, 6, 6), np.int16),
                {
                    "name": "CT",
                    "metadata": {"medh5_role": "image", "medh5_name": "CT"},
                },
                "image",
            )
        ]
        write_sample(str(dest), layers)
        assert not validate_file(dest).errors
        with medh5.open(dest) as sample:
            # Geometry is never invented: no source grid means unit spacing,
            # which is what an image with no stated geometry actually means.
            assert sample.grids["grid"].spacing == (1.0, 1.0, 1.0)
