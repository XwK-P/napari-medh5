"""The writer contribution: amend in place, or write a new sample."""

from __future__ import annotations

from pathlib import Path
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


def _overlapping(
    tmp_path: Path,
    *,
    encoding: str = "layers",
    classes: tuple[int, ...] = (1, 3),
    ignore: bool = False,
) -> Path:
    """A sample whose lesion sits inside its liver, in an encoding that allows it."""
    from medh5.labels.labelset import LabelClass, LabelSet

    shape = (8, 16, 16)
    liver = np.zeros(shape, dtype=bool)
    liver[2:6, 2:10, 2:10] = True
    lesion = np.zeros(shape, dtype=bool)
    lesion[3:5, 4:7, 4:7] = True
    unexamined = np.zeros(shape, dtype=bool)
    unexamined[7, 12:15, 12:15] = True
    masks = {1: liver, 3: lesion}
    path = tmp_path / f"overlap-{encoding}-{ignore}.medh5"
    with medh5.create(path, sample_id="s", subject_id="subj", codec="portable") as w:
        w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
        w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="g", modality="CT")
        w.label_set(
            LabelSet(
                "t",
                version="1.0.0",
                classes=[
                    LabelClass(1, "liver", "Liver"),
                    LabelClass(3, "lesion", "Lesion", parents=[1]),
                ],
            )
        )
        w.add_segmentation(
            "organs",
            grid="g",
            masks={c: masks[c] for c in classes},
            encoding=encoding,
            annotated_classes=list(classes),
            ignore=unexamined if ignore else None,
        )
    return path


def _reviewed(tmp_path: Path) -> Path:
    """A sample somebody signed off, associated with two timepoints."""
    from medh5.labels.labelset import LabelClass, LabelSet

    shape = (8, 16, 16)
    liver = np.zeros(shape, dtype=bool)
    liver[2:6, 2:10, 2:10] = True
    path = tmp_path / "reviewed.medh5"
    with medh5.create(path, sample_id="s", subject_id="j", codec="portable") as w:
        w.add_timepoint("tp0", days_from_baseline=0)
        w.add_timepoint("tp1", days_from_baseline=90)
        w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
        w.label_set(
            LabelSet(
                "t",
                version="1.0.0",
                classes=[
                    LabelClass(1, "liver", "Liver"),
                    LabelClass(3, "lesion", "Lesion"),
                ],
            )
        )
        w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="g", modality="CT")
        activity = w.activity("annotate", agent=w.person("Dr Reviewer"), tool="manual")
        w.add_segmentation(
            "organs",
            grid="g",
            masks={1: liver},
            annotated_classes=[1],
            timepoints=["tp0", "tp1"],
            quality={"status": "approved"},
            prov=activity,
        )
    return path


def _overlap_state(path: Path) -> tuple[str, int, int]:
    """``(kind, overlapping voxels, liver voxels)``."""
    with medh5.open(path) as sample:
        annotation = sample.annotations["organs"]
        liver = annotation.dense([1])[0]
        lesion = annotation.dense([3])[0]
        return str(annotation.kind), int((liver & lesion).sum()), int(liver.sum())


class TestAmend:
    def test_a_seg_edit_persists(self, tiny_medh5):
        layers = materialise(read_layers(tiny_medh5))
        for data, kwargs, _kind in layers:
            if kwargs["metadata"].get("medh5_role") == "seg":
                data[0, 0, 0] = 1
        write_sample(str(tiny_medh5), layers)
        with medh5.open(tiny_medh5) as sample:
            assert sample.annotations["seg"].contains(1, (0, 0, 0))

    def test_S7_saving_without_editing_changes_nothing(self, tmp_path):
        """Opening a file and pressing save must not cost the user data.

        napari collapses an annotation to one id per voxel to display it.
        Re-encoding every annotation from that view on every amend meant a save
        with no edits at all deleted each voxel where two classes overlapped,
        and demoted `layers` to `labelmap` on the way past.
        """
        path = _overlapping(tmp_path)
        before = _overlap_state(path)
        assert before == ("layers", 18, 256), "the fixture overlaps to begin with"

        write_sample(str(path), materialise(read_layers(path)))

        assert _overlap_state(path) == before

    def test_S7_an_edit_keeps_the_overlaps_it_did_not_touch(self, tmp_path):
        """The edit wins where it landed; the file wins everywhere else.

        A labelmap cannot express two classes on one voxel, so rebuilding the
        annotation from it discards every overlap --- including all the ones
        the editor never went near.  Only the changed voxels take their class
        from the viewer.
        """
        path = _overlapping(tmp_path)
        layers = materialise(read_layers(path))
        for data, kwargs, _kind in layers:
            if kwargs["metadata"].get("medh5_role") == "seg":
                data[0, 0, 0] = 1  # somewhere far from the overlap

        write_sample(str(path), layers)

        kind, overlap, liver = _overlap_state(path)
        assert (kind, overlap, liver) == ("layers", 18, 257), "overlap kept, edit added"
        with medh5.open(path) as sample:
            assert sample.annotations["organs"].contains(1, (0, 0, 0))

    def test_S6_4_an_ignore_region_survives_an_edit(self, tmp_path):
        """An ignore region is not a class, and clearing it makes it background.

        `_merged_masks` writes a class per voxel, so an ignored voxel came back
        as `0` --- "examined and empty" instead of "not examined". The region
        is merged separately, from the annotation rather than from the
        labelmap, because `labelmap()` never surfaces the ignore id and napari
        therefore never displayed it.
        """
        path = _overlapping(tmp_path, encoding="labelmap", classes=(1,), ignore=True)
        with medh5.open(path) as sample:
            assert sample.annotations["organs"].has_ignore_region

        layers = materialise(read_layers(path))
        for data, kwargs, _kind in layers:
            if kwargs["metadata"].get("medh5_role") == "seg":
                data[0, 0, 0] = 1
        write_sample(str(path), layers)

        with medh5.open(path) as sample:
            annotation = sample.annotations["organs"]
            assert annotation.has_ignore_region, "the region outlived the edit"
            assert annotation.contains(1, (0, 0, 0))

    def test_S6_4_an_unreadable_ignore_region_is_refused(self, tmp_path):
        """`ignore_mask()` is a `labelmap` accessor; the others do not expose one.

        Rewriting such an annotation would drop the region, and nothing on
        screen could put it back --- so the save is refused and the file is left
        exactly as it was.
        """
        path = _overlapping(tmp_path, encoding="layers", ignore=True)
        before = _overlap_state(path)

        layers = materialise(read_layers(path))
        for data, kwargs, _kind in layers:
            if kwargs["metadata"].get("medh5_role") == "seg":
                data[0, 0, 0] = 1
        with pytest.raises(ValueError, match="ignore region"):
            write_sample(str(path), layers)

        assert _overlap_state(path) == before
        with medh5.open(path) as sample:
            assert sample.annotations["organs"].has_ignore_region

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


class TestDeletedLayers:
    """Removing a layer and saving in place.

    `amend` is copy-on-write, so anything not rewritten is carried through
    untouched.  That is what protects an unedited annotation, and it protected
    a deleted one too.
    """

    def _without(self, source, role, name=None):
        out = []
        for data, kwargs, kind in materialise(read_layers(source)):
            meta = kwargs["metadata"]
            if meta.get("medh5_role") == role and (
                name is None or meta.get("medh5_name") == name
            ):
                continue
            out.append((data, kwargs, kind))
        return out

    def test_deleting_a_labels_layer_removes_the_annotation(self, tiny_medh5):
        write_sample(str(tiny_medh5), self._without(tiny_medh5, "seg"))
        REGISTRY.close_all()
        with medh5.open(tiny_medh5) as sample:
            assert "seg" not in sample.annotations
            assert "boxes" in sample.annotations  # still on screen

    def test_deleting_a_shapes_layer_removes_the_annotation(self, tiny_medh5):
        """The boxes loop had the same hole: emptying a Shapes layer worked,
        because `shapes_to_boxes` returns `None`, but deleting the whole layer
        never visited it."""
        layers = [
            one
            for one in materialise(read_layers(tiny_medh5))
            if not str(one[1]["metadata"].get("medh5_role", "")).startswith("bbox")
        ]
        write_sample(str(tiny_medh5), layers)
        REGISTRY.close_all()
        with medh5.open(tiny_medh5) as sample:
            assert "boxes" not in sample.annotations
            assert "seg" in sample.annotations

    def test_an_annotation_that_was_never_loaded_is_never_deleted(self, tiny_medh5):
        """The dangerous version of this fix.

        Reconciling against the *file's* annotations rather than the reader's
        record deletes whatever the user did not have on screen. Layers built
        by hand carry no record, so the fallback has to be to delete nothing.
        """
        layers = []
        for data, kwargs, kind in materialise(read_layers(tiny_medh5)):
            kwargs = dict(kwargs)
            kwargs["metadata"] = {
                k: v for k, v in kwargs["metadata"].items() if k != "medh5_opened"
            }
            layers.append((data, kwargs, kind))
        layers = [
            one for one in layers if one[1]["metadata"].get("medh5_role") != "seg"
        ]

        write_sample(str(tiny_medh5), layers)
        REGISTRY.close_all()
        with medh5.open(tiny_medh5) as sample:
            assert "seg" in sample.annotations

    def test_a_partial_load_cannot_delete_what_it_did_not_open(self, tiny_medh5):
        """A record naming one annotation says nothing about the other."""
        layers = []
        for data, kwargs, kind in materialise(read_layers(tiny_medh5)):
            if kwargs["metadata"].get("medh5_role") == "seg":
                continue
            kwargs = dict(kwargs)
            kwargs["metadata"] = {**kwargs["metadata"], "medh5_opened": ["boxes"]}
            layers.append((data, kwargs, kind))

        write_sample(str(tiny_medh5), layers)
        REGISTRY.close_all()
        with medh5.open(tiny_medh5) as sample:
            assert "seg" in sample.annotations

    def test_keeping_every_layer_deletes_nothing(self, tiny_medh5):
        write_sample(str(tiny_medh5), materialise(read_layers(tiny_medh5)))
        REGISTRY.close_all()
        with medh5.open(tiny_medh5) as sample:
            assert set(sample.annotations) == {"seg", "boxes"}


class TestCarriedAnnotations:
    """Save As and the annotations napari cannot put on screen.

    Both write paths build the destination from the layer list, and only voxel
    and box annotations ever become layers.  Everything else was dropped
    without a warning --- in the operation a user reaches for to make a *copy*.
    """

    NEVER_RENDERED = {
        "grade": "classification",
        "landmarks": "points",
        "pose": "keypoints",
        "oriented": "obb",
        "outline": "contours",
        "surface": "mesh",
    }

    def test_every_kind_that_is_not_a_layer_survives_save_as(
        self, rich_medh5, tmp_path
    ):
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(rich_medh5)))
        REGISTRY.close_all()
        with medh5.open(dest) as sample:
            kinds = {k: v.kind for k, v in sample.annotations.items()}
        for name, kind in self.NEVER_RENDERED.items():
            assert kinds.get(name) == kind, f"{name} ({kind}) was lost"

    def test_a_carried_annotation_keeps_its_payload(self, rich_medh5, tmp_path):
        """Presence is not fidelity. A copy path degrades quietly or not at all."""
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(rich_medh5)))
        REGISTRY.close_all()
        with medh5.open(rich_medh5) as before, medh5.open(dest) as after:
            old_points = before.annotations["landmarks"]
            new_points = after.annotations["landmarks"]
            assert np.allclose(new_points.points, old_points.points)
            assert new_points.names == old_points.names
            assert np.allclose(new_points.weights, old_points.weights)

            old_kp = before.annotations["pose"]
            new_kp = after.annotations["pose"]
            assert np.allclose(new_kp.visibility, old_kp.visibility)
            assert np.allclose(new_kp.scores, old_kp.scores)

            old_mesh = before.annotations["surface"]
            new_mesh = after.annotations["surface"]
            assert np.allclose(new_mesh.vertices, old_mesh.vertices)
            assert np.allclose(new_mesh.faces, old_mesh.faces)
            assert np.allclose(new_mesh.normals, old_mesh.normals)
            # Submeshes are an offsets dataset, not a count to recompute.
            assert new_mesh.n_submeshes == old_mesh.n_submeshes == 2

            old_poly = next(before.annotations["outline"].polygons())
            new_poly = next(after.annotations["outline"].polygons())
            assert np.allclose(new_poly.vertices, old_poly.vertices)
            assert (new_poly.plane, new_poly.role) == (old_poly.plane, old_poly.role)

            assert dict(after.annotations["grade"].labels) == dict(
                before.annotations["grade"].labels
            )

    def test_S11_3_a_carried_annotation_keeps_its_coverage(self, rich_medh5, tmp_path):
        """`annotated_class_ids` is what was *looked for*, not what is present.

        Letting `add_*` default it to `all_given` would promote every
        unexamined class to an examined one, turning "nobody checked" into a
        usable negative -- the §11.3 distinction this format exists to keep.
        """
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(rich_medh5)))
        REGISTRY.close_all()
        with medh5.open(rich_medh5) as before, medh5.open(dest) as after:
            for name in self.NEVER_RENDERED:
                assert (
                    after.annotations[name].annotated_class_ids
                    == before.annotations[name].annotated_class_ids
                ), name
            assert after.annotations["outline"].annotated_class_ids == (1,)

    def test_a_carried_annotation_is_not_attributed_to_napari(
        self, rich_medh5, tmp_path
    ):
        """napari did not touch these -- it cannot even display them.

        Stamping its own activity on them would put a viewer's name on work it
        never saw, which is the opposite of what the provenance graph is for.
        """
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(rich_medh5)))
        REGISTRY.close_all()
        with medh5.open(dest) as sample:
            for name in self.NEVER_RENDERED:
                assert sample.annotations[name].header.prov is None, name

    def test_a_grid_only_a_carried_annotation_uses_is_declared(
        self, rich_medh5, tmp_path
    ):
        """The mesh sits on a grid no image references."""
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(rich_medh5)))
        REGISTRY.close_all()
        with medh5.open(rich_medh5) as before, medh5.open(dest) as after:
            assert after.annotations["surface"].grid_id == "mesh_g"
            assert after.grids["mesh_g"].frame_uid == before.grids["mesh_g"].frame_uid
            assert after.grids["mesh_g"].spacing == before.grids["mesh_g"].spacing

    def test_a_carried_copy_still_validates(self, rich_medh5, tmp_path):
        from medh5.validate import validate_file

        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(rich_medh5)))
        REGISTRY.close_all()
        assert not validate_file(dest).errors

    def test_a_deleted_layer_is_still_deleted(self, rich_medh5, tmp_path):
        """The copy must not resurrect what the user removed.

        A voxel or box annotation absent from the layer list is a decision the
        user made (#6). Only the kinds napari cannot show are copied, because
        their absence is never a decision -- if the carry set were "everything
        missing from the bundle" instead, deleting a segmentation would put it
        straight back.
        """
        layers = [
            one
            for one in materialise(read_layers(rich_medh5))
            if one[1]["metadata"].get("medh5_role") != "seg"
        ]
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), layers)
        REGISTRY.close_all()
        with medh5.open(dest) as sample:
            assert "seg" not in sample.annotations
            assert "surface" in sample.annotations  # never a layer, never a decision


class TestFallbackGridNames:
    """Layers napari created carry no `medh5_grid`, so they all claimed `"grid"`."""

    def _layer(self, name, shape):
        return (
            np.zeros(shape, np.float32),
            {"name": name, "metadata": {"medh5_role": "image", "medh5_name": name}},
            "image",
        )

    def test_new_layers_of_different_shapes_get_their_own_grids(self, tmp_path):
        """The first one won and the rest were assigned to it, which fails."""
        dest = tmp_path / "two.medh5"
        write_sample(
            str(dest), [self._layer("A", (4, 8, 8)), self._layer("B", (2, 4, 4))]
        )
        with medh5.open(dest) as sample:
            assert sample.images["A"].grid_id != sample.images["B"].grid_id
            assert sample.grids[sample.images["A"].grid_id].shape == (4, 8, 8)
            assert sample.grids[sample.images["B"].grid_id].shape == (2, 4, 4)

    def test_new_layers_of_one_shape_still_share_a_grid(self, tmp_path):
        """Unclaimed grids are unit spacing at the origin, so these are the
        same grid -- merging them is deduplication, not a claim."""
        dest = tmp_path / "same.medh5"
        write_sample(
            str(dest), [self._layer("A", (4, 8, 8)), self._layer("B", (4, 8, 8))]
        )
        with medh5.open(dest) as sample:
            assert sample.images["A"].grid_id == sample.images["B"].grid_id
            assert sorted(sample.grids) == ["grid"]

    def test_a_new_layer_does_not_adopt_a_source_grid_by_name(self, tmp_path):
        """A source grid answering to `"grid"` is a coincidence, not a match."""
        source = tmp_path / "named.medh5"
        with medh5.create(source, sample_id="s", subject_id="j") as w:
            w.add_timepoint("tp0")
            w.add_grid("grid", shape=(8, 16, 16), spacing=(2.0, 1.0, 1.0))
            w.add_image(
                "CT", np.zeros((8, 16, 16), np.int16), grid="grid", modality="CT"
            )

        dest = tmp_path / "out.medh5"
        layers = [
            one
            for one in materialise(read_layers(source))
            if one[1]["metadata"].get("medh5_role") == "image"
        ]
        write_sample(str(dest), [*layers, self._layer("NEW", (4, 4, 4))])
        REGISTRY.close_all()
        with medh5.open(dest) as sample:
            assert sample.grids[sample.images["CT"].grid_id].spacing == (2.0, 1.0, 1.0)
            assert sample.grids[sample.images["NEW"].grid_id].shape == (4, 4, 4)


class TestSaveAs:
    def test_it_writes_a_new_file(self, tiny_medh5, tmp_path):
        dest = tmp_path / "copy.medh5"
        assert write_sample(str(dest), materialise(read_layers(tiny_medh5))) == [
            str(dest)
        ]
        assert dest.exists()

    def _cropped(self, source):
        """Every image layer cropped, the way a napari crop leaves them."""
        out = []
        for data, kwargs, kind in materialise(read_layers(source)):
            if kwargs["metadata"].get("medh5_role") != "image":
                continue
            out.append((np.asarray(data)[:, :8, :8], kwargs, kind))
        return out

    def test_S3_a_reshaped_layer_is_refused_rather_than_given_unit_spacing(
        self, tiny_medh5, tmp_path
    ):
        """Geometry is never invented (§3), and a crop moves the origin.

        The fallback for an unknown grid is unit spacing at the origin, which
        is honest for a layer that never had geometry.  For a cropped one it
        replaced a known spacing, direction, coordinate system and frame of
        reference with defaults --- so the image opened cleanly, sat in the
        wrong place, and nothing said anything had been lost.
        """
        dest = tmp_path / "cropped.medh5"
        with pytest.raises(ValueError, match="no derivable geometry"):
            write_sample(str(dest), self._cropped(tiny_medh5))
        # A refused Save As must not leave a half-written file behind.
        assert not dest.exists()

    def test_S3_the_refusal_names_a_way_out_that_works(self, tiny_medh5, tmp_path):
        """A message advertising an escape hatch has to be tested for one.

        Clearing `medh5_grid` says the layer no longer claims the source
        geometry, which makes unit spacing the honest answer rather than a
        substitution for something known.
        """
        layers = []
        for data, kwargs, kind in self._cropped(tiny_medh5):
            kwargs = dict(kwargs)
            kwargs["metadata"] = {
                k: v for k, v in kwargs["metadata"].items() if k != "medh5_grid"
            }
            layers.append((data, kwargs, kind))

        dest = tmp_path / "nogrid.medh5"
        write_sample(str(dest), layers)
        with medh5.open(dest) as sample:
            grid = sample.grids[sample.images["CT"].grid_id]
            assert grid.shape == (8, 8, 8)
            assert grid.spacing == (1.0, 1.0, 1.0)

    def test_S3_an_unchanged_shape_still_carries_the_geometry_across(
        self, tiny_medh5, tmp_path
    ):
        """The refusal must not catch an ordinary Save As."""
        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(tiny_medh5)))
        with medh5.open(tiny_medh5) as before, medh5.open(dest) as after:
            assert (
                after.grids[after.images["CT"].grid_id].spacing
                == before.grids[before.images["CT"].grid_id].spacing
            )

    def test_S3_an_annotation_keeps_the_grid_it_was_stored_on(self, tmp_path):
        """A segmentation may sit on its own grid, at its own spacing.

        Only the grids images referenced were declared, so `_grid_for` had
        nothing to match and substituted the first image grid --- which fails
        outright on a shape mismatch and, where the shapes happen to agree,
        quietly hands the annotation somebody else's spacing, origin and frame
        of reference.
        """
        shape = (8, 16, 16)
        mask = np.zeros(shape, dtype=bool)
        mask[2:6, 2:10, 2:10] = True
        source = tmp_path / "src.medh5"
        with medh5.create(source, sample_id="s", subject_id="j", codec="portable") as w:
            w.add_grid(
                "ct", shape=shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
            )
            w.add_grid(
                "seg", shape=shape, spacing=(2.5, 0.8, 0.8), origin=(-4.0, -6.0, -6.0)
            )
            w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="ct", modality="CT")
            w.add_segmentation("organs", grid="seg", masks={1: mask})

        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(source)))

        with medh5.open(dest) as sample:
            grid = sample.grids[sample.annotations["organs"].grid_id]
            assert grid.grid_id == "seg"
            assert list(grid.spacing) == [2.5, 0.8, 0.8]
            assert list(grid.origin) == [-4.0, -6.0, -6.0]
            assert sorted(sample.grids) == ["ct", "seg"]

    def test_S7_save_as_preserves_overlaps_too(self, tmp_path):
        """The amend fix left Save As rebuilding from the collapsed labelmap.

        Same data loss, second route: a `layers` annotation copied to a new
        path came out `labelmap` with every overlap gone.  Both paths resolve
        their masks through one function now.
        """
        source = _overlapping(tmp_path)
        assert _overlap_state(source) == ("layers", 18, 256)

        dest = tmp_path / "copy.medh5"
        write_sample(str(dest), materialise(read_layers(source)))

        assert _overlap_state(dest) == ("layers", 18, 256)

    def test_S7_save_as_carries_an_edit_without_flattening_the_rest(self, tmp_path):
        source = _overlapping(tmp_path)
        layers = materialise(read_layers(source))
        for data, kwargs, _kind in layers:
            if kwargs["metadata"].get("medh5_role") == "seg":
                data[0, 0, 0] = 1

        dest = tmp_path / "edited.medh5"
        write_sample(str(dest), layers)

        assert _overlap_state(dest) == ("layers", 18, 257)
        with medh5.open(dest) as sample:
            assert sample.annotations["organs"].contains(1, (0, 0, 0))

    def test_S11_a_copy_of_a_reviewed_sample_is_still_reviewed(self, tmp_path):
        """Save As carried the identity and left the audit trail behind.

        A sample somebody had approved came out with no quality record and no
        provenance but napari's own --- looking untouched.  Who drew what, and
        whether anybody signed it off, is the thing §11 is for.
        """
        source = _reviewed(tmp_path)
        dest = tmp_path / "copy.medh5"

        write_sample(str(dest), materialise(read_layers(source)))

        with medh5.open(dest) as sample:
            annotation = sample.annotations["organs"]
            record = sample.document.quality_of(annotation.header.quality)
            assert record is not None and record.status == "approved"
            agents = {a.name for a in sample.document.provenance.agents}
            assert "Dr Reviewer" in agents, "the reviewer survived the copy"
            assert "napari-medh5" in agents, "and napari recorded its own pass"
            assert list(annotation.timepoints) == ["tp0", "tp1"]

    def test_S11_3_a_painted_class_counts_as_examined(self, tmp_path):
        """Voxels for a class `annotated_classes` excludes is a contradiction.

        `annotated_classes` says what was *looked for*; a class somebody just
        painted was looked for by definition.  Declaring otherwise turns a
        positive finding into "nobody examined this".
        """
        source = _reviewed(tmp_path)
        layers = materialise(read_layers(source))
        for data, kwargs, _kind in layers:
            if kwargs["metadata"].get("medh5_role") == "seg":
                data[0, 0, 0] = 3  # a class the source never declared

        dest = tmp_path / "painted.medh5"
        write_sample(str(dest), layers)

        with medh5.open(dest) as sample:
            annotation = sample.annotations["organs"]
            assert list(annotation.annotated_class_ids) == [1, 3]
            assert annotation.contains(3, (0, 0, 0))

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
