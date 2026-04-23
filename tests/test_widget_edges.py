"""Error-path and empty-state coverage for the dock widget."""

from __future__ import annotations

from pathlib import Path

import pytest
from medh5 import MEDH5File
from medh5.review import set_review_status

from napari_medh5._handles import REGISTRY
from napari_medh5._widget import _tree_item

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


def test_widget_without_viewer_noop(qtbot) -> None:
    from napari_medh5._widget import MEDH5Widget

    widget = MEDH5Widget(None)
    qtbot.addWidget(widget)
    assert widget._active_path is None


def test_empty_viewer_clears_panels(qtbot, make_widget_app) -> None:
    widget, _ = make_widget_app()
    qtbot.addWidget(widget)

    widget._refresh_validation()
    widget._verify_checksum()
    widget._refresh_review()
    widget._refresh_metadata_tree()
    widget._refresh_nnunet_table()
    widget._save_review()

    assert widget._validation_tree.topLevelItemCount() == 0
    assert widget._checksum_label.text() == "—"


def test_validation_error_path(qtbot, make_widget_app, tmp_path: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    bogus = tmp_path / "missing.medh5"
    _register_sample(viewer, bogus)

    assert widget._validation_tree.topLevelItemCount() >= 1
    row = widget._validation_tree.topLevelItem(0)
    assert row.text(0) == "error"


def test_checksum_error_path(qtbot, make_widget_app, tmp_path: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tmp_path / "missing.medh5")
    widget._verify_checksum()
    assert widget._checksum_label.text().startswith("Error:")


def test_review_error_path(qtbot, make_widget_app, tmp_path: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tmp_path / "missing.medh5")
    # placeholder text reflects the failure.
    assert "Error:" in widget._annotator_edit.placeholderText()


def test_save_review_without_selection_is_noop(
    qtbot, make_widget_app, tiny_medh5: Path
) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)

    for btn in widget._status_buttons.values():
        btn.setChecked(False)
    widget._save_review()  # no selection → should not persist anything


def test_save_review_error_surfaces(
    qtbot, make_widget_app, tiny_medh5: Path, monkeypatch
) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)

    widget._status_buttons["reviewed"].setChecked(True)

    def _boom(*args, **kwargs):
        raise RuntimeError("persist failed")

    monkeypatch.setattr("napari_medh5._widget.set_review_status", _boom)
    widget._save_review()
    assert "Error:" in widget._annotator_edit.placeholderText()


def test_metadata_tree_error_path(qtbot, make_widget_app, tmp_path: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tmp_path / "missing.medh5")

    assert widget._meta_tree.topLevelItemCount() >= 1
    item = widget._meta_tree.topLevelItem(0)
    assert item.text(0) == "error"


def test_nnunet_ignores_non_dict_labels(
    qtbot, make_widget_app, tiny_medh5: Path
) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)
    # The tiny sample has no nnunetv2 extras, so tree must be empty.
    assert widget._nnunet_tree.topLevelItemCount() == 0


def test_nnunet_read_meta_failure(qtbot, make_widget_app, tmp_path: Path) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tmp_path / "missing.medh5")
    # read_meta raised; the tree must be cleared and stay empty.
    assert widget._nnunet_tree.topLevelItemCount() == 0


def test_nnunet_extra_without_labels(qtbot, make_widget_app, tmp_path: Path) -> None:
    import numpy as np

    path = tmp_path / "no_labels.medh5"
    MEDH5File.write(
        path,
        images={"0": np.zeros((4, 4, 4), dtype=np.float32)},
        extra={"nnunetv2": {"labels": "not-a-dict"}},
    )
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, path)
    assert widget._nnunet_tree.topLevelItemCount() == 0


def test_populate_review_renders_history(
    qtbot, make_widget_app, tiny_medh5: Path
) -> None:
    set_review_status(tiny_medh5, status="pending", annotator="alice", notes="first")
    set_review_status(tiny_medh5, status="reviewed", annotator="bob", notes="second")
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)
    assert widget._history_tree.topLevelItemCount() >= 1


def test_sample_picker_preserves_selection(
    qtbot, make_widget_app, tiny_medh5: Path
) -> None:
    widget, viewer = make_widget_app()
    qtbot.addWidget(widget)
    _register_sample(viewer, tiny_medh5)
    # Simulate an unrelated refresh that re-emits with the same list.
    widget._refresh_samples()
    assert widget._active_path == str(tiny_medh5)


def test_tree_item_formats_list_of_lists() -> None:
    item = _tree_item("matrix", [[1.0, 0.0], [0.0, 1.0]])
    assert item.text(0) == "matrix"
    assert item.childCount() == 2


def test_tree_item_formats_scalar_and_none() -> None:
    assert _tree_item("empty", None).text(1) == ""
    assert _tree_item("num", 3).text(1) == "3"
    assert _tree_item("vec", [1, 2, 3]).text(1) == "[1, 2, 3]"
