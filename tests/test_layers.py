"""Sample -> napari layers: geometry, naming, and the metadata the writer needs."""

from __future__ import annotations

import medh5
import numpy as np

from napari_medh5._layers import grid_kwargs, sample_to_layers


def _by_role(layers, role):
    return [(d, k) for d, k, _ in layers if k["metadata"].get("medh5_role") == role]


class TestGeometry:
    def test_an_axis_aligned_grid_gives_scale_and_translate(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            kwargs = grid_kwargs(sample.grids["g_tp0"])
        assert kwargs["scale"] == [2.0, 1.0, 1.0]
        assert kwargs["translate"] == [0.0, 0.0, 0.0]
        assert "affine" not in kwargs

    def test_an_oblique_grid_gives_a_single_affine(self, rotated_medh5):
        """Passing scale *and* an oblique affine double-applies the spacing."""
        with medh5.open(rotated_medh5) as sample:
            kwargs = grid_kwargs(sample.grids["g_tp0"])
        assert "affine" in kwargs
        assert "scale" not in kwargs and "translate" not in kwargs
        assert np.asarray(kwargs["affine"]).shape == (4, 4)

    def test_the_affine_is_the_grids_own(self, rotated_medh5):
        with medh5.open(rotated_medh5) as sample:
            grid = sample.grids["g_tp0"]
            assert np.allclose(grid_kwargs(grid)["affine"], grid.affine)


class TestLayers:
    def test_every_image_becomes_a_layer(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            layers = sample_to_layers(sample, tiny_medh5)
        images = _by_role(layers, "image")
        assert {k["metadata"]["medh5_name"] for _, k in images} == {"CT", "PET"}
        assert all(k["name"].startswith("tiny:") for _, k in images)

    def test_an_image_layer_carries_its_modality_and_units(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            layers = sample_to_layers(sample, tiny_medh5)
        meta = next(
            k["metadata"]
            for _, k in _by_role(layers, "image")
            if k["metadata"]["medh5_name"] == "CT"
        )
        assert meta["modality"] == "CT"
        assert meta["medh5_grid"] == "g_tp0"
        assert meta["medh5_timepoint"] == "tp0"

    def test_a_voxel_annotation_becomes_one_labels_layer(self, tiny_medh5):
        """One layer per annotation, not one per class: the classes are ids."""
        with medh5.open(tiny_medh5) as sample:
            layers = sample_to_layers(sample, tiny_medh5)
        segs = _by_role(layers, "seg")
        assert len(segs) == 1
        assert segs[0][1]["name"] == "tiny:seg:seg"
        # Every *declared* class, including one examined and not found --- the
        # names are what the editor shows, and a missing class needs a name too.
        assert segs[0][1]["metadata"]["medh5_classes"] == {
            1: "Tumor",
            2: "Incidental finding",
            3: "Organ",
        }

    def test_S11_3_the_layer_records_what_was_examined(self, tiny_medh5):
        """A reviewer editing a mask has to know which classes were searched for."""
        with medh5.open(tiny_medh5) as sample:
            layers = sample_to_layers(sample, tiny_medh5)
        meta = _by_role(layers, "seg")[0][1]["metadata"]
        assert meta["medh5_annotated"] == [1, 2, 3]

    def test_boxes_become_a_shapes_layer(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            layers = sample_to_layers(sample, tiny_medh5)
        rects = _by_role(layers, "bbox_rect")
        assert len(rects) == 1
        assert "features" in rects[0][1]

    def test_a_sample_with_one_visit_is_not_tagged(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            layers = sample_to_layers(sample, tiny_medh5)
        assert all("[tp" not in k["name"] for _, k, _ in layers)

    def test_a_longitudinal_sample_tags_each_layer_with_its_visit(
        self, longitudinal_medh5
    ):
        with medh5.open(longitudinal_medh5) as sample:
            layers = sample_to_layers(sample, longitudinal_medh5)
        names = [k["name"] for _, k, _ in layers]
        assert any(n.endswith("[tp0]") for n in names)
        assert any(n.endswith("[tp1]") for n in names)
        assert len(_by_role(layers, "image")) == 4

    def test_a_sample_without_a_label_set_still_reads(self, bare_medh5):
        with medh5.open(bare_medh5) as sample:
            layers = sample_to_layers(sample, bare_medh5)
        assert len(_by_role(layers, "image")) == 1
        assert _by_role(layers, "seg") == []

    def test_every_layer_names_its_source_file(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            layers = sample_to_layers(sample, tiny_medh5)
        assert all(k["metadata"]["medh5_path"] == str(tiny_medh5) for _, k, _ in layers)
