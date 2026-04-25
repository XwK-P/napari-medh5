"""End-to-end tests that drive every user-facing feature of napari-medh5
through a real ``napari.Viewer``.

Run headless via ``QT_QPA_PLATFORM=offscreen`` (set in ``conftest.py``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

import dask.array as da  # noqa: E402
from medh5 import MEDH5File  # noqa: E402

from napari_medh5._handles import REGISTRY  # noqa: E402

pytestmark = pytest.mark.integration

PLUGIN = "napari-medh5"


@pytest.fixture(autouse=True)
def _registry_reset() -> Any:
    """Close any handles left behind by the previous test."""
    yield
    REGISTRY.close_all()


def _roles(viewer: Any) -> list[str]:
    return [(layer.metadata or {}).get("medh5_role", "") for layer in viewer.layers]


def _layer_by_role(viewer: Any, role: str) -> Any:
    return next(
        layer for layer in viewer.layers if layer.metadata.get("medh5_role") == role
    )


# ======================================================================
# Reader
# ======================================================================


def test_open_through_viewer_populates_layers(
    real_viewer: Any, tiny_medh5: Path
) -> None:
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    roles = _roles(real_viewer)
    assert roles.count("image") == 2
    assert roles.count("seg") == 1
    assert roles.count("bbox_rect") == 1

    for layer in real_viewer.layers:
        meta = layer.metadata
        assert meta["medh5_path"] == str(tiny_medh5)
        if meta["medh5_role"] == "image":
            assert isinstance(layer.data, da.Array)
            assert list(layer.scale) == [2.0, 1.0, 1.0]


def test_open_applies_affine_when_rotated(
    real_viewer: Any, rotated_medh5: Path
) -> None:
    real_viewer.open(str(rotated_medh5), plugin=PLUGIN)
    img = _layer_by_role(real_viewer, "image")
    affine = np.asarray(img.affine.affine_matrix)
    assert not np.allclose(affine, np.eye(4)), (
        "affine should be non-identity for rotated sample"
    )


def test_open_wireframe_appears_for_deep_box(
    real_viewer: Any, deep_bbox_medh5: Path
) -> None:
    real_viewer.open(str(deep_bbox_medh5), plugin=PLUGIN)
    roles = _roles(real_viewer)
    assert "bbox_wire" in roles
    assert "bbox_rect" in roles


def test_open_nnunet_renames_seg(real_viewer: Any, nnunet_medh5: Path) -> None:
    real_viewer.open(str(nnunet_medh5), plugin=PLUGIN)
    assert any(layer.name.endswith(":tumor") for layer in real_viewer.layers)


# ======================================================================
# Writer
# ======================================================================


def test_inplace_save_persists_seg_edit(real_viewer: Any, tiny_medh5: Path) -> None:
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    seg = _layer_by_role(real_viewer, "seg")
    seg.data = np.asarray(seg.data).copy()  # materialise dask → numpy so edit sticks
    seg.data[0, 0, 0] = 1

    real_viewer.layers.save(str(tiny_medh5), plugin=PLUGIN)

    sample = MEDH5File.read(tiny_medh5)
    try:
        assert bool(sample.seg["tumor"][0, 0, 0]) is True
    finally:
        sample.close() if hasattr(sample, "close") else None


def test_saveas_full_rewrite_preserves_spatial(
    real_viewer: Any, tiny_medh5: Path, tmp_path: Path
) -> None:
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    dest = tmp_path / "copy.medh5"
    real_viewer.layers.save(str(dest), plugin=PLUGIN)

    meta = MEDH5File.read_meta(dest)
    assert list(meta.spatial.spacing or []) == [2.0, 1.0, 1.0]
    assert meta.label == 1
    assert meta.label_name == "present"


def test_multi_source_rejected(
    real_viewer: Any, tiny_medh5: Path, rotated_medh5: Path, tmp_path: Path
) -> None:
    """Writer raises ValueError; napari 0.7 converts it into an empty result
    plus a UserWarning. Verify napari's observable behavior."""
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    real_viewer.open(str(rotated_medh5), plugin=PLUGIN)
    dest = tmp_path / "merged.medh5"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        written = real_viewer.layers.save(str(dest), plugin=PLUGIN)
    assert written == []
    assert not dest.exists()
    assert any("not a valid writer" in str(w.message) for w in captured)


def test_modality_rename_rejected(real_viewer: Any, tiny_medh5: Path) -> None:
    """Same swallow-to-warning pattern as multi-source rejection."""
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    ct = next(
        layer
        for layer in real_viewer.layers
        if layer.metadata.get("medh5_role") == "image"
        and layer.metadata.get("medh5_name") == "CT"
    )
    ct.metadata["medh5_name"] = "MR"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        written = real_viewer.layers.save(str(tiny_medh5), plugin=PLUGIN)
    assert written == []
    assert any("not a valid writer" in str(w.message) for w in captured)


def test_bbox_roundtrip_through_viewer(
    real_viewer: Any, tiny_medh5: Path, tmp_path: Path
) -> None:
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    dest = tmp_path / "roundtrip.medh5"
    real_viewer.layers.save(str(dest), plugin=PLUGIN)

    with MEDH5File(dest) as f:
        bboxes, _scores, _labels = f.bbox_arrays()
    expected = np.array(
        [[[2, 5], [4, 10], [4, 10]], [[1, 3], [1, 5], [1, 5]]], dtype=np.float64
    )
    np.testing.assert_allclose(bboxes, expected, atol=1e-6)


def test_wireframe_ignored_on_save(
    real_viewer: Any, deep_bbox_medh5: Path, tmp_path: Path
) -> None:
    """The ``bbox_wire`` companion layer must be dropped by ``_collect``."""
    real_viewer.open(str(deep_bbox_medh5), plugin=PLUGIN)
    dest = tmp_path / "deep_copy.medh5"
    real_viewer.layers.save(str(dest), plugin=PLUGIN)

    with MEDH5File(dest) as f:
        bboxes, _, _ = f.bbox_arrays()
    assert bboxes is not None
    np.testing.assert_allclose(
        bboxes, np.array([[[2, 7], [4, 10], [4, 10]]]), atol=1e-6
    )


# ======================================================================
# Widget
# ======================================================================


def test_widget_on_real_viewer_populates_picker(
    qtbot: Any, real_viewer: Any, tiny_medh5: Path
) -> None:
    from napari_medh5._widget import MEDH5Widget

    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    widget = MEDH5Widget(real_viewer)
    qtbot.addWidget(widget)
    assert widget._sample_picker.currentText() == str(tiny_medh5)
    assert widget._validation_tree.topLevelItemCount() >= 1
    assert widget._meta_tree.topLevelItemCount() >= 1


def test_two_samples_switch_panels(
    qtbot: Any, real_viewer: Any, tiny_medh5: Path, nnunet_medh5: Path
) -> None:
    from napari_medh5._widget import MEDH5Widget

    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    real_viewer.open(str(nnunet_medh5), plugin=PLUGIN)
    widget = MEDH5Widget(real_viewer)
    qtbot.addWidget(widget)

    paths = {
        widget._sample_picker.itemText(i) for i in range(widget._sample_picker.count())
    }
    assert {str(tiny_medh5), str(nnunet_medh5)} <= paths

    widget._sample_picker.setCurrentText(str(nnunet_medh5))
    names = {
        widget._nnunet_tree.topLevelItem(i).text(0)
        for i in range(widget._nnunet_tree.topLevelItemCount())
    }
    assert "tumor" in names


def test_checksum_mismatch_after_tamper(
    qtbot: Any, real_viewer: Any, corrupt_medh5: Path
) -> None:
    from napari_medh5._widget import MEDH5Widget

    real_viewer.open(str(corrupt_medh5), plugin=PLUGIN)
    widget = MEDH5Widget(real_viewer)
    qtbot.addWidget(widget)
    widget._verify_checksum()
    assert widget._checksum_label.text().startswith("MISMATCH @")


def test_review_roundtrips_via_reopen(
    qtbot: Any, real_viewer: Any, tiny_medh5: Path
) -> None:
    from napari_medh5._widget import MEDH5Widget

    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    widget = MEDH5Widget(real_viewer)
    qtbot.addWidget(widget)
    widget._status_buttons["flagged"].setChecked(True)
    widget._annotator_edit.setText("bob")
    widget._notes_edit.setPlainText("needs revisit")
    widget._save_review()

    # Close everything, reopen, rebuild widget, confirm persisted state
    real_viewer.layers.clear()
    REGISTRY.close_all()

    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    widget2 = MEDH5Widget(real_viewer)
    qtbot.addWidget(widget2)
    assert widget2._status_buttons["flagged"].isChecked()
    assert widget2._annotator_edit.text() == "bob"


# ======================================================================
# Handles
# ======================================================================


def test_remove_all_layers_drops_registry(real_viewer: Any, tiny_medh5: Path) -> None:
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    assert REGISTRY.get(str(tiny_medh5)) is not None
    real_viewer.layers.clear()
    assert REGISTRY.get(str(tiny_medh5)) is None


def test_double_open_no_h5_crash(real_viewer: Any, tiny_medh5: Path) -> None:
    """HDF5 forbids re-opening; second open must reuse the registry handle."""
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    first_handle = REGISTRY.get(str(tiny_medh5))
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    assert REGISTRY.get(str(tiny_medh5)) is first_handle


def test_inplace_save_drops_and_reopens(real_viewer: Any, tiny_medh5: Path) -> None:
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    real_viewer.layers.save(str(tiny_medh5), plugin=PLUGIN)
    # Re-opening must succeed (HDF5 would refuse if a stale handle were still open)
    real_viewer.layers.clear()
    REGISTRY.close_all()
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    assert any(
        layer.metadata.get("medh5_role") == "image" for layer in real_viewer.layers
    )


def test_lazy_slice_after_inplace_save(real_viewer: Any, tiny_medh5: Path) -> None:
    """Exposing test: after in-place save, existing lazy image layers must
    still be readable (i.e. be auto-rebound to the reopened file)."""
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    img = _layer_by_role(real_viewer, "image")
    real_viewer.layers.save(str(tiny_medh5), plugin=PLUGIN)
    # If rebinding did not happen, the underlying h5py.Dataset is closed and
    # this raises. The auto-rebind in _writer.py is what makes it pass.
    first_slice = np.asarray(img.data[0])
    assert first_slice.shape == (16, 16)


# ======================================================================
# Bbox
# ======================================================================


def test_edit_depth_features_roundtrip(
    real_viewer: Any, tiny_medh5: Path, tmp_path: Path
) -> None:
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    rect = _layer_by_role(real_viewer, "bbox_rect")
    feats = {k: np.asarray(v).copy() for k, v in rect.features.items()}
    feats["depth_lo"] = np.array([1.0, 1.0])
    feats["depth_hi"] = np.array([6.0, 4.0])
    rect.features = feats

    dest = tmp_path / "depth.medh5"
    real_viewer.layers.save(str(dest), plugin=PLUGIN)
    with MEDH5File(dest) as f:
        bboxes, _, _ = f.bbox_arrays()
    # depth_axis for both boxes is axis 0; depth_lo/hi drives box[0, :]
    np.testing.assert_allclose(bboxes[0, 0], [1.0, 6.0])
    np.testing.assert_allclose(bboxes[1, 0], [1.0, 4.0])


def test_out_of_bounds_clamped(
    real_viewer: Any, tiny_medh5: Path, tmp_path: Path
) -> None:
    real_viewer.open(str(tiny_medh5), plugin=PLUGIN)
    rect = _layer_by_role(real_viewer, "bbox_rect")
    feats = {k: np.asarray(v).copy() for k, v in rect.features.items()}
    feats["depth_hi"] = np.array([9999.0, 3.0])
    rect.features = feats

    dest = tmp_path / "clamp.medh5"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        real_viewer.layers.save(str(dest), plugin=PLUGIN)

    with MEDH5File(dest) as f:
        bboxes, _, _ = f.bbox_arrays()
    # sample_shape[0] == 8 → depth_hi must be clamped to ≤ 8
    assert bboxes[0, 0, 1] <= 8.0
    assert any(issubclass(w.category, UserWarning) for w in captured)
