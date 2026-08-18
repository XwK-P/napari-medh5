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


class TestSaveAs:
    def test_it_writes_a_new_file(self, tiny_medh5, tmp_path):
        dest = tmp_path / "copy.medh5"
        assert write_sample(str(dest), materialise(read_layers(tiny_medh5))) == [
            str(dest)
        ]
        assert dest.exists()

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
