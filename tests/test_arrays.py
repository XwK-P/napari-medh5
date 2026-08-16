"""Lazy dask views over images and voxel annotations.

The annotation view is the interesting one: five encodings sit behind one read
contract, and napari must not know which is in the file.
"""

from __future__ import annotations

import dask.array as da
import medh5
import numpy as np
import pytest

from napari_medh5._arrays import (
    LABEL_DTYPE,
    annotation_array,
    draw_priority,
    image_array,
)


class TestImages:
    def test_an_image_view_is_lazy_and_matches_the_file(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            array = image_array(sample.images["CT"])
            assert isinstance(array, da.Array)
            assert array.shape == sample.images["CT"].shape
            assert np.array_equal(np.asarray(array), sample.images["CT"].read())

    def test_the_view_follows_the_stored_chunking(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            image = sample.images["CT"]
            array = image_array(image)
            if image.chunks:
                assert array.chunksize == image.chunks


class TestAnnotations:
    def test_a_labelmap_view_is_lazy_and_typed(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            array = annotation_array(sample.annotations["seg"])
            assert isinstance(array, da.Array)
            assert array.dtype == LABEL_DTYPE
            assert array.shape == sample.annotations["seg"].spatial_shape

    def test_it_agrees_with_the_eager_labelmap(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            annotation = sample.annotations["seg"]
            assert np.array_equal(
                np.asarray(annotation_array(annotation)),
                annotation.labelmap(priority=draw_priority(annotation)),
            )

    def test_a_slice_reads_only_that_window(self, tiny_medh5):
        """The point of the map_blocks path: napari asks for what it draws."""
        with medh5.open(tiny_medh5) as sample:
            annotation = sample.annotations["seg"]
            lazy = annotation_array(annotation)
            window = (slice(2, 5), slice(4, 10), slice(4, 10))
            assert np.array_equal(
                np.asarray(lazy[window]),
                annotation.labelmap(roi=window, priority=draw_priority(annotation)),
            )

    @pytest.mark.parametrize("encoding", ["labelmap", "layers", "bitmask", "instances"])
    def test_every_encoding_reads_the_same(self, tmp_path, encoding, label_set):
        """The viewer must not be able to tell which encoding is in the file."""
        # Disjoint on purpose: `labelmap` cannot hold overlapping classes and
        # says so.  Overlap is covered by the next test, on encodings that can.
        shape = (6, 10, 10)
        masks = {1: np.zeros(shape, bool), 3: np.zeros(shape, bool)}
        masks[1][1:4, 2:8, 2:8] = True
        masks[3][4:5, 3:5, 3:5] = True
        path = tmp_path / f"{encoding}.medh5"
        with medh5.create(path, sample_id="s", subject_id="p") as writer:
            writer.label_set(label_set)
            writer.add_timepoint("tp0")
            writer.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            writer.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            writer.add_segmentation("seg", grid="g", masks=masks, encoding=encoding)
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert annotation.kind == encoding
            assert np.array_equal(
                np.asarray(annotation_array(annotation)),
                annotation.labelmap(priority=draw_priority(annotation)),
            )

    def test_overlap_collapses_the_way_labelmap_says(self, tmp_path, label_set):
        """A napari Labels layer holds one id per voxel; the file still holds both."""
        shape = (6, 10, 10)
        organ = np.zeros(shape, bool)
        organ[1:5, 1:9, 1:9] = True
        lesion = np.zeros(shape, bool)
        lesion[2:4, 3:6, 3:6] = True
        path = tmp_path / "overlap.medh5"
        with medh5.create(path, sample_id="s", subject_id="p") as writer:
            writer.label_set(label_set)
            writer.add_timepoint("tp0")
            writer.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            writer.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            writer.add_segmentation("seg", grid="g", masks={3: organ, 1: lesion})
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            view = np.asarray(annotation_array(annotation))
            # The lesion is inside the organ, and the label set says so, so the
            # lesion --- the thing the reader opened the file for --- wins the
            # overlapping voxels rather than vanishing under the organ.
            assert view[2, 4, 4] == 1
            assert np.array_equal(
                view, annotation.labelmap(priority=draw_priority(annotation))
            )
            # And the overlap is untouched in the file, whatever the view shows.
            assert annotation.contains(3, (2, 4, 4))
            assert annotation.contains(1, (2, 4, 4))


class TestDrawPriority:
    def test_the_deepest_class_takes_precedence(self, tiny_medh5):
        with medh5.open(tiny_medh5) as sample:
            annotation = sample.annotations["seg"]
            # 1 is a child of 3 in the fixture's label set.
            assert draw_priority(annotation)[0] == 1

    def test_it_changes_what_the_viewer_shows(self, tiny_medh5):
        """Without it the organ buries the tumor entirely."""
        with medh5.open(tiny_medh5) as sample:
            annotation = sample.annotations["seg"]
            naive = annotation.labelmap()
            ranked = annotation.labelmap(priority=draw_priority(annotation))
        assert set(np.unique(naive)) == {0, 3}
        assert set(np.unique(ranked)) == {0, 1, 3}

    def test_a_flat_label_set_falls_back_to_id_order(self, tmp_path, label_set):
        from medh5.labels.labelset import LabelClass, LabelSet

        flat = LabelSet(
            "flat",
            version="1.0.0",
            classes=[LabelClass(1, "a", "A"), LabelClass(3, "b", "B")],
        )
        shape = (4, 8, 8)
        masks = {1: np.zeros(shape, bool), 3: np.zeros(shape, bool)}
        masks[1][0:2, 0:4, 0:4] = True
        masks[3][2:4, 4:8, 4:8] = True
        path = tmp_path / "flat.medh5"
        with medh5.create(path, sample_id="s", subject_id="p") as writer:
            writer.label_set(flat)
            writer.add_timepoint("tp0")
            writer.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            writer.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            writer.add_segmentation("seg", grid="g", masks=masks)
        with medh5.open(path) as sample:
            assert draw_priority(sample.annotations["seg"]) == [1, 3]

    def test_no_label_set_is_not_an_error(self, bare_medh5):
        class _Bare:
            class_ids = (5, 2)
            label_set = None

        assert draw_priority(_Bare()) == [5, 2]
