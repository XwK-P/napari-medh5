"""Dock widget: validation, integrity, per-annotation quality, and the document.

The 0.x widget had one review status per *file*, because that is what the
format could hold.  1.0 records quality **per annotation** (§11.2) with a
provenance activity behind it, so the widget follows: pick an annotation,
set its status, and the file records who said so and when --- rather than a
single flag that cannot say which of three masks was reviewed.
"""

from __future__ import annotations

import getpass
import json
from typing import Any

import medh5
from medh5.validate import validate_file
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_medh5._handles import REGISTRY, attach_viewer, rebind_viewer_layers

QUALITY_STATUS = (
    "draft",
    "submitted",
    "reviewed",
    "approved",
    "rejected",
    "deprecated",
)
LEVELS = ("structural", "semantic", "integrity", "strict")


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
        root.addWidget(self._build_integrity_box())
        root.addWidget(self._build_quality_box())
        root.addWidget(self._build_labels_box())
        root.addWidget(self._build_document_box())
        root.addStretch(1)

        if napari_viewer is not None:
            attach_viewer(napari_viewer)
            napari_viewer.layers.events.inserted.connect(self._refresh_samples)
            napari_viewer.layers.events.removed.connect(self._refresh_samples)
            self._refresh_samples()

    # -- sample discovery --------------------------------------------------

    def _refresh_samples(self, event: Any = None) -> None:
        paths: list[str] = []
        if self._viewer is not None:
            for layer in self._viewer.layers:
                meta = getattr(layer, "metadata", None) or {}
                path = meta.get("medh5_path") if isinstance(meta, dict) else None
                if isinstance(path, str) and path not in paths:
                    paths.append(path)
        previous = self._active_path
        self._sample_picker.blockSignals(True)
        self._sample_picker.clear()
        self._sample_picker.addItems(paths)
        if previous in paths:
            self._sample_picker.setCurrentText(previous)
        self._sample_picker.blockSignals(False)
        current = self._sample_picker.currentText() or None
        if current != self._active_path:
            self._on_sample_changed(current or "")

    def _on_sample_changed(self, path: str) -> None:
        self._active_path = path or None
        self._reload_all()

    def _reload_all(self) -> None:
        self._refresh_validation()
        self._refresh_integrity_label()
        self._refresh_quality()
        self._refresh_labels()
        self._refresh_document()

    # -- validation --------------------------------------------------------

    def _build_validation_box(self) -> QGroupBox:
        box = QGroupBox("Validation", self)
        layout = QVBoxLayout(box)
        self._validation_tree = QTreeWidget(box)
        self._validation_tree.setHeaderLabels(
            ["Severity", "Code", "Location", "Message"]
        )
        self._validation_tree.setRootIsDecorated(False)
        layout.addWidget(self._validation_tree)

        row = QHBoxLayout()
        self._level_picker = QComboBox(box)
        self._level_picker.addItems(LEVELS)
        self._level_picker.setCurrentText("semantic")
        self._level_picker.currentTextChanged.connect(self._refresh_validation)
        row.addWidget(QLabel("Level", box))
        row.addWidget(self._level_picker)
        rerun = QPushButton("Re-run", box)
        rerun.clicked.connect(self._refresh_validation)
        row.addWidget(rerun)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _refresh_validation(self, *_: Any) -> None:
        self._validation_tree.clear()
        if not self._active_path:
            return
        try:
            report = validate_file(
                self._active_path, level=self._level_picker.currentText()
            )
        except Exception as exc:  # noqa: BLE001 - surface anything to the user
            self._validation_tree.addTopLevelItem(
                QTreeWidgetItem(["error", "validate_failed", "", str(exc)])
            )
            return
        for diagnostic in report.diagnostics:
            self._validation_tree.addTopLevelItem(
                QTreeWidgetItem(
                    [
                        diagnostic.severity,
                        diagnostic.code,
                        diagnostic.location or "",
                        diagnostic.message,
                    ]
                )
            )
        if not report.diagnostics:
            self._validation_tree.addTopLevelItem(
                QTreeWidgetItem(["ok", "", "", "No issues reported"])
            )

    # -- integrity ---------------------------------------------------------

    def _build_integrity_box(self) -> QGroupBox:
        box = QGroupBox("Integrity", self)
        layout = QVBoxLayout(box)
        self._integrity_label = QLabel("—", box)
        self._integrity_label.setWordWrap(True)
        layout.addWidget(self._integrity_label)
        button = QPushButton("Verify digests", box)
        button.clicked.connect(self._verify)
        layout.addWidget(button)
        return box

    def _refresh_integrity_label(self) -> None:
        self._integrity_label.setText("Not verified yet" if self._active_path else "—")

    def _verify(self) -> None:
        if not self._active_path:
            return
        try:
            with medh5.open(self._active_path) as sample:
                result = sample.verify()
                content_id = sample.content_id
        except Exception as exc:  # noqa: BLE001
            self._integrity_label.setText(f"Error: {exc}")
            return
        lines = [f"{len(result.checked)} object(s) checked"]
        if result.mismatched:
            lines.append(f"MISMATCH: {', '.join(result.mismatched)}")
        if result.stale_index:
            lines.append(
                f"stale index: {', '.join(result.stale_index)} "
                "(rebuild with `medh5 fix --rebuild-index`)"
            )
        # `None` is not "fine": it means the file declares no content_id, so
        # nothing was compared.  Saying "OK" there would be a claim about a
        # check that never ran.
        if result.content_id_ok is None:
            lines.append("no content_id declared — nothing to compare")
        elif result.content_id_ok:
            lines.append(f"content_id OK: {content_id}")
        else:
            lines.append("content_id does NOT match the file's contents")
        self._integrity_label.setText("\n".join(lines))

    # -- quality -----------------------------------------------------------

    def _build_quality_box(self) -> QGroupBox:
        box = QGroupBox("Annotation quality", self)
        layout = QVBoxLayout(box)

        form = QFormLayout()
        self._annotation_picker = QComboBox(box)
        self._annotation_picker.currentTextChanged.connect(self._on_annotation_changed)
        form.addRow("Annotation", self._annotation_picker)
        self._status_picker = QComboBox(box)
        self._status_picker.addItems(QUALITY_STATUS)
        form.addRow("Status", self._status_picker)
        self._reviewer_edit = QLineEdit(box)
        self._reviewer_edit.setText(getpass.getuser())
        form.addRow("Reviewer", self._reviewer_edit)
        self._notes_edit = QPlainTextEdit(box)
        self._notes_edit.setMaximumHeight(60)
        form.addRow("Note", self._notes_edit)
        layout.addLayout(form)

        button = QPushButton("Save quality record", box)
        button.clicked.connect(self._save_quality)
        layout.addWidget(button)

        self._provenance_tree = QTreeWidget(box)
        self._provenance_tree.setHeaderLabels(["Activity", "Agent", "Tool", "When"])
        self._provenance_tree.setRootIsDecorated(False)
        layout.addWidget(QLabel("Provenance", box))
        layout.addWidget(self._provenance_tree)
        return box

    def _refresh_quality(self) -> None:
        self._annotation_picker.blockSignals(True)
        self._annotation_picker.clear()
        self._provenance_tree.clear()
        if not self._active_path:
            self._annotation_picker.blockSignals(False)
            return
        try:
            with medh5.open(self._active_path) as sample:
                self._annotation_picker.addItems(sorted(sample.annotations))
                document = sample.document
                agents = {a.id: a.name or a.id for a in document.provenance.agents}
                for activity in document.provenance.activities:
                    self._provenance_tree.addTopLevelItem(
                        QTreeWidgetItem(
                            [
                                activity.type,
                                agents.get(activity.agent or "", activity.agent or ""),
                                str(activity.tool or ""),
                                str(activity.ended or activity.started or ""),
                            ]
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            self._notes_edit.setPlaceholderText(f"Error: {exc}")
        finally:
            self._annotation_picker.blockSignals(False)
        self._on_annotation_changed(self._annotation_picker.currentText())

    def _on_annotation_changed(self, name: str) -> None:
        if not name or not self._active_path:
            return
        try:
            with medh5.open(self._active_path) as sample:
                record = sample.document.quality.get(name)
        except Exception:  # noqa: BLE001
            return
        if record is None:
            self._status_picker.setCurrentText("draft")
            self._notes_edit.clear()
            return
        self._status_picker.setCurrentText(record.status)
        notes = [i.note for i in record.issues if i.note]
        self._notes_edit.setPlainText("\n".join(notes))

    def _save_quality(self) -> None:
        path = self._active_path
        name = self._annotation_picker.currentText()
        if not path or not name:
            return
        reviewer = self._reviewer_edit.text().strip() or None
        note = self._notes_edit.toPlainText().strip()

        # `amend` replaces the file, so the read handle has to go first or
        # every lazy layer keeps serving the pre-edit inode.
        REGISTRY.drop(path)
        try:
            with medh5.amend(path) as writer:
                agent = (
                    writer.person(reviewer)
                    if reviewer
                    else writer.software("napari-medh5")
                )
                writer.activity(
                    "review",
                    agent=agent,
                    tool="napari-medh5",
                    outputs=[f"annotations/{name}"],
                    params={"status": self._status_picker.currentText()},
                )
                writer.set_quality(
                    name,
                    status=self._status_picker.currentText(),
                    reviewed_by=[agent.id],
                    issues=(
                        [{"code": "reviewer_note", "severity": "info", "note": note}]
                        if note
                        else []
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            self._notes_edit.setPlaceholderText(f"Error: {exc}")
        finally:
            rebind_viewer_layers(path, self._viewer)
        self._refresh_quality()

    # -- label set ---------------------------------------------------------

    def _build_labels_box(self) -> QGroupBox:
        box = QGroupBox("Label set", self)
        layout = QVBoxLayout(box)
        self._labels_tree = QTreeWidget(box)
        self._labels_tree.setHeaderLabels(["id", "key", "name", "parents"])
        self._labels_tree.setRootIsDecorated(False)
        layout.addWidget(self._labels_tree)
        return box

    def _refresh_labels(self) -> None:
        self._labels_tree.clear()
        if not self._active_path:
            return
        try:
            with medh5.open(self._active_path) as sample:
                label_set = sample.label_set
                if label_set is None:
                    self._labels_tree.addTopLevelItem(
                        QTreeWidgetItem(["", "", "no label set declared", ""])
                    )
                    return
                for entry in label_set:
                    self._labels_tree.addTopLevelItem(
                        QTreeWidgetItem(
                            [
                                str(entry.id),
                                entry.key,
                                entry.name or "",
                                ", ".join(str(p) for p in entry.parents),
                            ]
                        )
                    )
        except Exception:  # noqa: BLE001
            return

    # -- document ----------------------------------------------------------

    def _build_document_box(self) -> QGroupBox:
        box = QGroupBox("Document", self)
        layout = QVBoxLayout(box)
        self._document_tree = QTreeWidget(box)
        self._document_tree.setHeaderLabels(["Key", "Value"])
        self._document_tree.setAlternatingRowColors(True)
        layout.addWidget(self._document_tree)
        return box

    def _refresh_document(self) -> None:
        self._document_tree.clear()
        if not self._active_path:
            return
        try:
            with medh5.open(self._active_path) as sample:
                data = sample.summary()
        except Exception as exc:  # noqa: BLE001
            self._document_tree.addTopLevelItem(QTreeWidgetItem(["error", str(exc)]))
            return
        for key, value in data.items():
            self._document_tree.addTopLevelItem(_tree_item(key, value))


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
        for child_key, child in value.items():
            item.addChild(_tree_item(child_key, child))
        return item
    if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
        item = QTreeWidgetItem([str(key), ""])
        for index, child in enumerate(value):
            item.addChild(_tree_item(f"[{index}]", child))
        return item
    if isinstance(value, (list, tuple)):
        return QTreeWidgetItem([str(key), json.dumps(list(value))])
    if value is None:
        return QTreeWidgetItem([str(key), ""])
    return QTreeWidgetItem([str(key), str(value)])
