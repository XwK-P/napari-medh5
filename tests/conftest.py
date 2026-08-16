"""Shared fixtures for napari-medh5 tests."""

from __future__ import annotations

import os
import sys

# napari integration tests rely on ``napari.Viewer(show=False)``. On macOS the
# ``offscreen`` Qt platform can't provide the OpenGL context napari's vispy
# canvas needs, so a ``QT_QPA_PLATFORM=offscreen`` hint causes a segfault.
# Only set it on non-Darwin platforms (Linux CI); let macOS use the default
# native platform and suppress the window via ``show=False``.
if sys.platform != "darwin":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Pin both qtpy and pytest-qt to PyQt5, the binding listed in napari-medh5's
# dev extras (pyproject.toml). Without this, pytest-qt silently picks PyQt6
# when both are installed, which can destabilise the test session because
# napari may not be built against the selected binding.
os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("PYTEST_QT_API", "pyqt5")

from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import medh5
import numpy as np
import pytest
from medh5 import LabelClass, LabelSet

LABELS = LabelSet(
    "napari-test-v1",
    version="1.0.0",
    classes=[
        # The tumor is *inside* the organ, and the label set says so (§5.1).
        # That is what lets the viewer paint the specific class on top instead
        # of burying it under the one that contains it.
        LabelClass(1, "tumor", "Tumor", parents=[3], category="lesion"),
        LabelClass(2, "incidental", "Incidental finding"),
        LabelClass(3, "organ", "Organ", category="organ"),
    ],
)


def build_sample(
    path: Path,
    *,
    shape: tuple[int, ...] = (8, 16, 16),
    spacing: tuple[float, ...] = (2.0, 1.0, 1.0),
    origin: tuple[float, ...] = (0.0, 0.0, 0.0),
    direction: Any | None = None,
    masks: dict[Any, Any] | None = None,
    boxes: Any = None,
    class_ids: list[int] | None = None,
    scores: list[float] | None = None,
    timepoints: tuple[str, ...] = ("tp0",),
    label_set: LabelSet | None = LABELS,
    images: dict[str, Any] | None = None,
) -> Path:
    """A 1.0 sample built through the public writer.

    Every fixture goes through this, so a test that reads one is also a test
    that the writer produced something readable.
    """
    rng = np.random.default_rng(0)
    if images is None:
        images = {
            "CT": rng.integers(-100, 300, size=shape, dtype=np.int16),
            "PET": rng.random(size=shape, dtype=np.float32),
        }
    with medh5.create(path, sample_id=path.stem, subject_id="SUBJ-1") as writer:
        if label_set is not None:
            writer.label_set(label_set)
        for index, timepoint in enumerate(timepoints):
            writer.add_timepoint(timepoint, index=index, days_from_baseline=90 * index)
        tool = writer.software("test-suite", "1.0")
        activity = writer.activity("import", agent=tool, tool="conftest")
        for index, timepoint in enumerate(timepoints):
            grid = f"g_{timepoint}"
            writer.add_grid(
                grid,
                shape=shape,
                spacing=spacing,
                origin=origin,
                direction=direction,
                timepoint=timepoint,
                frame_uid=f"pseudo:frame-{timepoint}",
            )
            for name, array in images.items():
                writer.add_image(
                    f"{name}_{timepoint}" if len(timepoints) > 1 else name,
                    array,
                    grid=grid,
                    modality="CT" if name == "CT" else "PT",
                    prov=activity,
                )
            if masks:
                writer.add_segmentation(
                    f"seg_{timepoint}" if len(timepoints) > 1 else "seg",
                    grid=grid,
                    masks=masks,
                    annotated_classes=[1, 2, 3]
                    if label_set is not None
                    else "all_given",
                    prov=activity,
                )
            if boxes is not None and index == 0:
                writer.add_boxes(
                    "boxes",
                    boxes=np.asarray(boxes, dtype=np.float32),
                    class_ids=class_ids or [1] * len(boxes),
                    grid=grid,
                    space="index",
                    scores=scores,
                    prov=activity,
                )
    return path


@pytest.fixture
def label_set() -> LabelSet:
    return LABELS


@pytest.fixture
def tiny_medh5(tmp_path: Path) -> Path:
    """Two modalities, one segmentation with two classes, two boxes."""
    shape = (8, 16, 16)
    tumor = np.zeros(shape, dtype=bool)
    tumor[2:5, 4:10, 4:10] = True
    organ = np.zeros(shape, dtype=bool)
    organ[1:7, 2:14, 2:14] = True
    return build_sample(
        tmp_path / "tiny.medh5",
        shape=shape,
        masks={1: tumor, 3: organ},
        # Voxel edges (§8.1): [1.5, 4.5] is exactly slice(2, 5).
        boxes=[
            [[1.5, 4.5], [3.5, 9.5], [3.5, 9.5]],
            [[0.5, 2.5], [0.5, 4.5], [0.5, 4.5]],
        ],
        class_ids=[1, 2],
        scores=[0.9, 0.5],
    )


@pytest.fixture
def longitudinal_medh5(tmp_path: Path) -> Path:
    """Two visits, so layer names and timepoint tagging are exercised."""
    shape = (8, 16, 16)
    tumor = np.zeros(shape, dtype=bool)
    tumor[2:5, 4:10, 4:10] = True
    return build_sample(
        tmp_path / "long.medh5",
        shape=shape,
        masks={1: tumor},
        timepoints=("tp0", "tp1"),
    )


@pytest.fixture
def bare_medh5(tmp_path: Path) -> Path:
    """One image, no label set, no annotations."""
    return build_sample(
        tmp_path / "bare.medh5",
        shape=(4, 8, 8),
        label_set=None,
        images={"CT": np.zeros((4, 8, 8), dtype=np.float32)},
    )


@pytest.fixture
def rotated_medh5(tmp_path: Path) -> Path:
    """A non-identity ``direction`` --- exercises the affine path."""
    shape = (6, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[1:4, 2:6, 2:6] = True
    return build_sample(
        tmp_path / "rot.medh5",
        shape=shape,
        spacing=(1.0, 1.0, 1.0),
        direction=[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        masks={1: mask},
        images={"CT": np.zeros(shape, dtype=np.float32)},
    )


@pytest.fixture
def deep_bbox_medh5(tmp_path: Path) -> Path:
    """A box deeper than one voxel on its shallowest axis --- triggers the wireframe."""
    shape = (12, 16, 16)
    return build_sample(
        tmp_path / "deep.medh5",
        shape=shape,
        spacing=(1.0, 1.0, 1.0),
        images={"CT": np.zeros(shape, dtype=np.float32)},
        boxes=[[[1.5, 6.5], [3.5, 9.5], [3.5, 9.5]]],
        class_ids=[1],
        scores=[0.8],
    )


@pytest.fixture
def corrupt_medh5(tiny_medh5: Path) -> Path:
    """``tiny_medh5`` with one voxel flipped --- the file is valid, the digest is not.

    Exactly what an external tool editing an array does, and the reason
    ``verify`` reports per object rather than one yes/no.
    """
    with h5py.File(tiny_medh5, "r+") as handle:
        dataset = handle["images/CT"]
        original = int(dataset[0, 0, 0])
        dataset[0, 0, 0] = np.int16(original ^ 0xFF)
    return tiny_medh5


@pytest.fixture
def make_widget_app() -> Callable[[], tuple[Any, Any]]:
    """Factory returning ``(widget, mock_viewer)`` with no real napari viewer."""
    pytest.importorskip("qtpy")
    from napari_medh5._widget import MEDH5Widget

    class _MockEvent:
        def __init__(self) -> None:
            self._callbacks: list[Callable[..., Any]] = []

        def connect(self, cb: Callable[..., Any]) -> None:
            self._callbacks.append(cb)

        def emit(self, *args: Any, **kwargs: Any) -> None:
            for cb in self._callbacks:
                cb(*args, **kwargs)

    class _MockEvents:
        def __init__(self) -> None:
            self.inserted = _MockEvent()
            self.removed = _MockEvent()

    class _MockLayers(list[Any]):
        def __init__(self) -> None:
            super().__init__()
            self.events = _MockEvents()

        def append(self, layer: Any) -> None:
            super().append(layer)
            self.events.inserted.emit()

    class _MockViewer:
        def __init__(self) -> None:
            self.layers = _MockLayers()

    def factory() -> tuple[Any, Any]:
        viewer = _MockViewer()
        widget = MEDH5Widget(viewer)
        return widget, viewer

    return factory


@pytest.fixture
def real_viewer(make_napari_viewer: Callable[..., Any]) -> Any:
    """Thin wrapper around napari's ``make_napari_viewer`` for readability."""
    return make_napari_viewer()
