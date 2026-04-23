"""Dock-widget integration tests (Qt headless)."""

from __future__ import annotations

from pathlib import Path

import pytest
from medh5.review import get_review_status

from napari_medh5._handles import REGISTRY

pytest.importorskip("qtpy")


@pytest.fixture(autouse=True)
def _close_registry() -> None:
    yield
    REGISTRY.close_all()


def _register_sample(viewer, path: Path) -> None:
    class _Layer:
        def __init__(self, medh5_path: str) -> None:
            self.metadata = {"medh5_path": medh5_path}

    viewer.layers.append(_Layer(str(path)))


def test_validation_panel_populates(qtbot, make_widget_app, tiny_medh5: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)

    assert widget._sample_picker.currentText() == str(tiny_medh5)
    # "ok" row when no errors/warnings
    items = [
        widget._validation_tree.topLevelItem(i).text(0)
        for i in range(widget._validation_tree.topLevelItemCount())
    ]
    assert items, "validation tree should never be empty after loading"


def test_save_review_persists(qtbot, make_widget_app, tiny_medh5: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)

    widget._status_buttons["reviewed"].setChecked(True)
    widget._notes_edit.setPlainText("LGTM")
    widget._annotator_edit.setText("alice")
    widget._save_review()

    status = get_review_status(tiny_medh5)
    assert status.status == "reviewed"
    assert status.annotator == "alice"
    assert status.notes == "LGTM"


def test_nnunet_labels_render(qtbot, make_widget_app, nnunet_medh5: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, nnunet_medh5)

    count = widget._nnunet_tree.topLevelItemCount()
    names = [widget._nnunet_tree.topLevelItem(i).text(0) for i in range(count)]
    assert "background" in names
    assert "tumor" in names


def test_metadata_tree_has_spatial_branch(
    qtbot, make_widget_app, tiny_medh5: Path
) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)

    top = [
        widget._meta_tree.topLevelItem(i).text(0)
        for i in range(widget._meta_tree.topLevelItemCount())
    ]
    assert "spatial" in top
    assert "shape" in top
    assert "extra" in top


def test_checksum_verify_ok(qtbot, make_widget_app, tiny_medh5: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)

    widget._verify_checksum()
    assert widget._checksum_label.text().startswith("OK @")
