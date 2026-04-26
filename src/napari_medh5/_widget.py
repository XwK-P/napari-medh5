"""Dock widget: validation report, checksum verify, review status, metadata tree."""

from __future__ import annotations

import getpass
import json
from datetime import datetime, timezone
from typing import Any

from medh5 import MEDH5File, VerifyResult
from medh5.review import ReviewStatus, get_review_status, set_review_status
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_medh5._handles import REGISTRY, attach_viewer, rebind_viewer_layers

_STATUSES = ("pending", "reviewed", "flagged", "rejected")


class MEDH5Widget(QWidget):
    """Dock widget reflecting the active ``.medh5`` sample."""

    def __init__(self, napari_viewer: Any | None = None) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._active_path: str | None = None

        root = QVBoxLayout(self)

        self._sample_picker = QComboBox(self)
        self._sample_picker.currentTextChanged.connect(self._on_sample_changed)
        root.addWidget(_titled("Active sample", self._sample_picker))

        root.addWidget(self._build_validation_box())
        root.addWidget(self._build_checksum_box())
        root.addWidget(self._build_review_box())
        root.addWidget(self._build_metadata_box())
        root.addWidget(self._build_nnunet_box())
        root.addStretch(1)

        if napari_viewer is not None:
            attach_viewer(napari_viewer)
            napari_viewer.layers.events.inserted.connect(self._refresh_samples)
            napari_viewer.layers.events.removed.connect(self._refresh_samples)
            self._refresh_samples()

    # ------------------------------------------------------------------
    # Sample discovery & switching
    # ------------------------------------------------------------------

    def _refresh_samples(self, event: Any = None) -> None:
        paths: list[str] = []
        if self._viewer is not None:
            for layer in self._viewer.layers:
                meta = getattr(layer, "metadata", None) or {}
                p = meta.get("medh5_path") if isinstance(meta, dict) else None
                if isinstance(p, str) and p not in paths:
                    paths.append(p)
        prev = self._active_path
        self._sample_picker.blockSignals(True)
        self._sample_picker.clear()
        self._sample_picker.addItems(paths)
        if prev in paths:
            self._sample_picker.setCurrentText(prev)
        self._sample_picker.blockSignals(False)
        current = self._sample_picker.currentText() or None
        if current != self._active_path:
            self._on_sample_changed(current or "")

    def _on_sample_changed(self, path: str) -> None:
        self._active_path = path or None
        self._reload_all()

    def _reload_all(self) -> None:
        self._refresh_validation()
        self._refresh_checksum_label()
        self._refresh_review()
        self._refresh_metadata_tree()
        self._refresh_nnunet_table()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _build_validation_box(self) -> QGroupBox:
        box = QGroupBox("Validation", self)
        layout = QVBoxLayout(box)
        self._validation_tree = QTreeWidget(box)
        self._validation_tree.setHeaderLabels(
            ["Severity", "Code", "Location", "Message"]
        )
        self._validation_tree.setRootIsDecorated(False)
        layout.addWidget(self._validation_tree)
        btn_row = QHBoxLayout()
        rerun = QPushButton("Re-run validation", box)
        rerun.clicked.connect(self._refresh_validation)
        btn_row.addWidget(rerun)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        return box

    def _refresh_validation(self) -> None:
        self._validation_tree.clear()
        if not self._active_path:
            return
        try:
            report = MEDH5File.validate(self._active_path)
        except Exception as exc:  # noqa: BLE001 — surface anything to the user
            item = QTreeWidgetItem(["error", "validate_failed", "", str(exc)])
            self._validation_tree.addTopLevelItem(item)
            return
        for issue in report.errors:
            item = QTreeWidgetItem(
                ["error", issue.code, issue.location or "", issue.message]
            )
            self._validation_tree.addTopLevelItem(item)
        for issue in report.warnings:
            item = QTreeWidgetItem(
                ["warning", issue.code, issue.location or "", issue.message]
            )
            self._validation_tree.addTopLevelItem(item)
        if not report.errors and not report.warnings:
            self._validation_tree.addTopLevelItem(
                QTreeWidgetItem(["ok", "", "", "No issues reported"])
            )

    # ------------------------------------------------------------------
    # Checksum
    # ------------------------------------------------------------------

    def _build_checksum_box(self) -> QGroupBox:
        box = QGroupBox("Checksum", self)
        layout = QVBoxLayout(box)
        self._checksum_label = QLabel("—", box)
        layout.addWidget(self._checksum_label)
        btn = QPushButton("Verify checksum", box)
        btn.clicked.connect(self._verify_checksum)
        layout.addWidget(btn)
        return box

    def _refresh_checksum_label(self) -> None:
        self._checksum_label.setText("Not verified yet" if self._active_path else "—")

    def _verify_checksum(self) -> None:
        if not self._active_path:
            return
        try:
            result = MEDH5File.verify(self._active_path)
        except Exception as exc:  # noqa: BLE001
            self._checksum_label.setText(f"Error: {exc}")
            return
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # medh5 0.6.0 returns a tri-state ``VerifyResult`` so audit UIs
        # can distinguish "no checksum was ever stored" from "verified
        # good" — both used to render as a bare green tick.
        if result is VerifyResult.OK:
            text = f"OK @ {stamp}"
        elif result is VerifyResult.MISSING:
            text = f"No checksum stored @ {stamp}"
        else:
            text = f"MISMATCH @ {stamp}"
        self._checksum_label.setText(text)

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def _build_review_box(self) -> QGroupBox:
        box = QGroupBox("Review", self)
        layout = QVBoxLayout(box)

        self._status_group = QButtonGroup(box)
        radio_row = QHBoxLayout()
        self._status_buttons: dict[str, QRadioButton] = {}
        for status in _STATUSES:
            btn = QRadioButton(status, box)
            self._status_group.addButton(btn)
            self._status_buttons[status] = btn
            radio_row.addWidget(btn)
        layout.addLayout(radio_row)

        form = QFormLayout()
        self._annotator_edit = QLineEdit(box)
        self._annotator_edit.setText(getpass.getuser())
        form.addRow("Annotator", self._annotator_edit)
        self._notes_edit = QPlainTextEdit(box)
        self._notes_edit.setMaximumHeight(80)
        form.addRow("Notes", self._notes_edit)
        layout.addLayout(form)

        btn = QPushButton("Save review", box)
        btn.clicked.connect(self._save_review)
        layout.addWidget(btn)

        self._history_tree = QTreeWidget(box)
        self._history_tree.setHeaderLabels(
            ["Status", "Annotator", "Timestamp", "Notes"]
        )
        self._history_tree.setRootIsDecorated(False)
        layout.addWidget(QLabel("History", box))
        layout.addWidget(self._history_tree)

        return box

    def _refresh_review(self) -> None:
        self._history_tree.clear()
        for btn in self._status_buttons.values():
            btn.setChecked(False)
        self._notes_edit.clear()
        if not self._active_path:
            return
        try:
            status = get_review_status(self._active_path)
        except Exception as exc:  # noqa: BLE001
            self._annotator_edit.setPlaceholderText(f"Error: {exc}")
            return
        self._populate_review(status)

    def _populate_review(self, status: ReviewStatus) -> None:
        if status.status in self._status_buttons:
            self._status_buttons[status.status].setChecked(True)
        if status.annotator:
            self._annotator_edit.setText(status.annotator)
        if status.notes:
            self._notes_edit.setPlainText(status.notes)
        for entry in status.history or []:
            self._history_tree.addTopLevelItem(
                QTreeWidgetItem(
                    [
                        str(entry.get("status", "")),
                        str(entry.get("annotator", "") or ""),
                        str(entry.get("timestamp", "") or ""),
                        str(entry.get("notes", "") or ""),
                    ]
                )
            )

    def _save_review(self) -> None:
        if not self._active_path:
            return
        checked = self._status_group.checkedButton()
        if checked is None:
            return
        status = checked.text()
        # ``set_review_status`` opens the file in append mode; HDF5 forbids a
        # second open while the registry still holds the read handle, so we
        # drop first. We rely on medh5 0.6.0's ``on_reopened`` callback to
        # rebind lazy layers on success, and an explicit ``finally`` rebind
        # to recover from validation/IO errors that abort before the write.
        REGISTRY.drop(self._active_path)
        new_status: ReviewStatus | None = None
        try:
            new_status = set_review_status(
                self._active_path,
                status=status,
                annotator=self._annotator_edit.text() or None,
                notes=self._notes_edit.toPlainText() or None,
                on_reopened=lambda p: rebind_viewer_layers(p, self._viewer),
            )
        except Exception as exc:  # noqa: BLE001
            self._annotator_edit.setPlaceholderText(f"Error: {exc}")
            rebind_viewer_layers(self._active_path, self._viewer)
            return
        # 0.6.0 returns the freshly persisted ReviewStatus, so we can refresh
        # the UI without reopening the file.
        self._history_tree.clear()
        for btn in self._status_buttons.values():
            btn.setChecked(False)
        self._populate_review(new_status)

    # ------------------------------------------------------------------
    # Metadata tree
    # ------------------------------------------------------------------

    def _build_metadata_box(self) -> QGroupBox:
        box = QGroupBox("Metadata", self)
        layout = QVBoxLayout(box)
        self._meta_tree = QTreeWidget(box)
        self._meta_tree.setHeaderLabels(["Key", "Value"])
        self._meta_tree.setAlternatingRowColors(True)
        layout.addWidget(self._meta_tree)
        return box

    def _refresh_metadata_tree(self) -> None:
        self._meta_tree.clear()
        if not self._active_path:
            return
        try:
            meta = MEDH5File.read_meta(self._active_path)
        except Exception as exc:  # noqa: BLE001
            self._meta_tree.addTopLevelItem(QTreeWidgetItem(["error", str(exc)]))
            return
        data: dict[str, Any] = {
            "schema_version": meta.schema_version,
            "image_names": meta.image_names,
            "label": meta.label,
            "label_name": meta.label_name,
            "shape": meta.shape,
            "has_seg": meta.has_seg,
            "seg_names": meta.seg_names,
            "has_bbox": meta.has_bbox,
            "patch_size": meta.patch_size,
            "spatial": {
                "spacing": meta.spatial.spacing,
                "origin": meta.spatial.origin,
                "direction": meta.spatial.direction,
                "axis_labels": meta.spatial.axis_labels,
                "coord_system": meta.spatial.coord_system,
            },
            "extra": meta.extra or {},
        }
        for key, value in data.items():
            self._meta_tree.addTopLevelItem(_tree_item(key, value))

    # ------------------------------------------------------------------
    # nnU-Net class surfacing
    # ------------------------------------------------------------------

    def _build_nnunet_box(self) -> QGroupBox:
        box = QGroupBox("nnU-Net v2 classes", self)
        layout = QVBoxLayout(box)
        self._nnunet_tree = QTreeWidget(box)
        self._nnunet_tree.setHeaderLabels(["Class name", "Value"])
        self._nnunet_tree.setRootIsDecorated(False)
        layout.addWidget(self._nnunet_tree)
        return box

    def _refresh_nnunet_table(self) -> None:
        self._nnunet_tree.clear()
        if not self._active_path:
            return
        try:
            meta = MEDH5File.read_meta(self._active_path)
        except Exception:  # noqa: BLE001
            return
        extra = meta.extra or {}
        nn = extra.get("nnunetv2") if isinstance(extra, dict) else None
        if not isinstance(nn, dict):
            return
        labels = nn.get("labels")
        if not isinstance(labels, dict):
            return
        for name, value in labels.items():
            self._nnunet_tree.addTopLevelItem(QTreeWidgetItem([str(name), str(value)]))


def _titled(title: str, widget: QWidget) -> QWidget:
    wrap = QWidget()
    layout = QVBoxLayout(wrap)
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel(title)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(label)
    layout.addWidget(widget)
    return wrap


def _tree_item(key: str, value: Any) -> QTreeWidgetItem:
    if isinstance(value, dict):
        item = QTreeWidgetItem([str(key), ""])
        for k, v in value.items():
            item.addChild(_tree_item(k, v))
        return item
    if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
        item = QTreeWidgetItem([str(key), ""])
        for i, v in enumerate(value):
            item.addChild(_tree_item(f"[{i}]", v))
        return item
    if isinstance(value, (list, tuple)):
        return QTreeWidgetItem([str(key), json.dumps(list(value))])
    if value is None:
        return QTreeWidgetItem([str(key), ""])
    return QTreeWidgetItem([str(key), str(value)])
