"""The dock widget: validation, integrity, quality, label set, document."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import medh5
import pytest

pytest.importorskip("qtpy")

from napari_medh5._handles import REGISTRY  # noqa: E402
from napari_medh5._reader import napari_get_reader  # noqa: E402
from napari_medh5._widget import MEDH5Widget, _tree_item  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    REGISTRY.close_all()


def load(viewer: Any, path: Path) -> None:
    """Push a file's layers into the mock viewer, as the reader would."""
    for data, kwargs, _kind in napari_get_reader(str(path))(str(path)):
        viewer.layers.append(
            type(
                "L",
                (),
                {"metadata": kwargs["metadata"], "data": data, "name": kwargs["name"]},
            )()
        )


def rows(tree: Any) -> list[list[str]]:
    return [
        [tree.topLevelItem(i).text(c) for c in range(tree.columnCount())]
        for i in range(tree.topLevelItemCount())
    ]


class TestSampleDiscovery:
    def test_no_viewer_is_a_valid_state(self, qtbot):
        widget = MEDH5Widget(None)
        qtbot.addWidget(widget)
        assert widget._active_path is None

    def test_an_empty_viewer_shows_nothing(self, qtbot, make_widget_app):
        widget, _ = make_widget_app()
        qtbot.addWidget(widget)
        assert widget._sample_picker.count() == 0
        assert widget._validation_tree.topLevelItemCount() == 0

    def test_loading_a_sample_populates_the_picker(
        self, qtbot, make_widget_app, tiny_medh5
    ):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        assert widget._sample_picker.count() == 1
        assert widget._active_path == str(tiny_medh5)

    def test_two_samples_can_be_switched(
        self, qtbot, make_widget_app, tiny_medh5, rotated_medh5
    ):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        load(viewer, rotated_medh5)
        widget._refresh_samples()
        assert widget._sample_picker.count() == 2
        widget._sample_picker.setCurrentText(str(rotated_medh5))
        assert widget._active_path == str(rotated_medh5)

    def test_the_selection_survives_a_refresh(
        self, qtbot, make_widget_app, tiny_medh5, rotated_medh5
    ):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        load(viewer, rotated_medh5)
        widget._refresh_samples()
        widget._sample_picker.setCurrentText(str(rotated_medh5))
        widget._refresh_samples()
        assert widget._active_path == str(rotated_medh5)


class TestValidation:
    def test_a_clean_sample_reports_no_issues(self, qtbot, make_widget_app, tiny_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        codes = {row[1] for row in rows(widget._validation_tree)}
        assert "validate_failed" not in codes

    def test_the_level_can_be_changed(self, qtbot, make_widget_app, tiny_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        widget._level_picker.setCurrentText("strict")
        assert widget._level_picker.currentText() == "strict"

    def test_a_broken_file_is_surfaced_not_raised(
        self, qtbot, make_widget_app, tmp_path
    ):
        widget, _ = make_widget_app()
        qtbot.addWidget(widget)
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        widget._active_path = str(broken)
        widget._refresh_validation()
        assert widget._validation_tree.topLevelItemCount() >= 1


class TestIntegrity:
    def test_a_clean_file_verifies(self, qtbot, make_widget_app, tiny_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        widget._verify()
        assert "OK" in widget._integrity_label.text()

    def test_a_tampered_file_reports_the_object(
        self, qtbot, make_widget_app, corrupt_medh5
    ):
        widget, _ = make_widget_app()
        qtbot.addWidget(widget)
        widget._active_path = str(corrupt_medh5)
        widget._verify()
        text = widget._integrity_label.text()
        assert "MISMATCH" in text and "images/CT" in text

    def test_an_unreadable_file_shows_the_error(self, qtbot, make_widget_app, tmp_path):
        widget, _ = make_widget_app()
        qtbot.addWidget(widget)
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        widget._active_path = str(broken)
        widget._verify()
        assert widget._integrity_label.text().startswith("Error:")


class TestQuality:
    def test_the_annotations_are_listed(self, qtbot, make_widget_app, tiny_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        listed = {
            widget._annotation_picker.itemText(i)
            for i in range(widget._annotation_picker.count())
        }
        assert listed == {"seg", "boxes"}

    def test_S11_2_a_record_is_written_per_annotation(
        self, qtbot, make_widget_app, tiny_medh5
    ):
        """0.x had one status per file; 1.0 can say which mask was reviewed."""
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        widget._annotation_picker.setCurrentText("seg")
        widget._status_picker.setCurrentText("approved")
        widget._reviewer_edit.setText("RAD-07")
        widget._notes_edit.setPlainText("inferior edge")
        widget._save_quality()

        with medh5.open(tiny_medh5) as sample:
            record = sample.document.quality["seg"]
            assert record.status == "approved"
            assert record.reviewed_by
            assert [i.note for i in record.issues] == ["inferior edge"]
            assert "boxes" not in sample.document.quality

    def test_the_review_lands_in_provenance(self, qtbot, make_widget_app, tiny_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        widget._annotation_picker.setCurrentText("seg")
        widget._status_picker.setCurrentText("reviewed")
        widget._save_quality()
        with medh5.open(tiny_medh5) as sample:
            types = [a.type for a in sample.document.provenance.activities]
            assert "review" in types

    def test_the_saved_status_comes_back(self, qtbot, make_widget_app, tiny_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        widget._annotation_picker.setCurrentText("seg")
        widget._status_picker.setCurrentText("rejected")
        widget._save_quality()
        widget._on_annotation_changed("seg")
        assert widget._status_picker.currentText() == "rejected"

    def test_saving_without_a_sample_is_a_no_op(self, qtbot, make_widget_app):
        widget, _ = make_widget_app()
        qtbot.addWidget(widget)
        widget._save_quality()

    def test_a_write_error_is_surfaced(self, qtbot, make_widget_app, tmp_path):
        widget, _ = make_widget_app()
        qtbot.addWidget(widget)
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        widget._active_path = str(broken)
        widget._annotation_picker.addItem("seg")
        widget._annotation_picker.setCurrentText("seg")
        widget._save_quality()
        assert "Error" in widget._notes_edit.placeholderText()


class TestLabelSet:
    def test_the_classes_are_listed(self, qtbot, make_widget_app, tiny_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        listed = {row[1] for row in rows(widget._labels_tree)}
        assert listed == {"tumor", "incidental", "organ"}

    def test_a_sample_without_one_says_so(self, qtbot, make_widget_app, bare_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, bare_medh5)
        widget._refresh_samples()
        assert any("no label set" in row[2] for row in rows(widget._labels_tree))


class TestDocument:
    def test_the_summary_is_rendered(self, qtbot, make_widget_app, tiny_medh5):
        widget, viewer = make_widget_app()
        qtbot.addWidget(widget)
        load(viewer, tiny_medh5)
        widget._refresh_samples()
        keys = {row[0] for row in rows(widget._document_tree)}
        assert {"sample_id", "subject_id", "grids", "images", "annotations"} <= keys

    def test_an_unreadable_file_shows_the_error(self, qtbot, make_widget_app, tmp_path):
        widget, _ = make_widget_app()
        qtbot.addWidget(widget)
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        widget._active_path = str(broken)
        widget._refresh_document()
        assert rows(widget._document_tree)[0][0] == "error"


class TestTreeFormatting:
    def test_a_nested_list_becomes_children(self):
        item = _tree_item("grids", [{"id": "g"}, {"id": "h"}])
        assert item.childCount() == 2

    def test_a_flat_list_becomes_json(self):
        assert _tree_item("shape", [1, 2, 3]).text(1) == "[1, 2, 3]"

    def test_none_renders_empty(self):
        assert _tree_item("x", None).text(1) == ""

    def test_a_scalar_renders_as_text(self):
        assert _tree_item("n", 7).text(1) == "7"
