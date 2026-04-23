"""Shared fixtures for napari-medh5 tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from medh5 import MEDH5File


@pytest.fixture
def tiny_medh5(tmp_path: Path) -> Path:
    """Build a small two-modality sample with one seg class and two bboxes."""
    rng = np.random.default_rng(0)
    shape = (8, 16, 16)
    images = {
        "CT": rng.integers(-100, 300, size=shape, dtype=np.int16),
        "PET": rng.random(size=shape, dtype=np.float32),
    }
    seg = {"tumor": np.zeros(shape, dtype=bool)}
    seg["tumor"][2:5, 4:10, 4:10] = True
    bboxes = np.array(
        [
            [[2, 5], [4, 10], [4, 10]],
            [[1, 3], [1, 5], [1, 5]],
        ],
        dtype=np.float64,
    )
    bbox_scores = np.array([0.9, 0.5], dtype=np.float32)
    bbox_labels = ["tumor", "incidental"]

    path = tmp_path / "tiny.medh5"
    MEDH5File.write(
        path,
        images=images,
        seg=seg,
        bboxes=bboxes,
        bbox_scores=bbox_scores,
        bbox_labels=bbox_labels,
        label=1,
        label_name="present",
        spacing=[2.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        axis_labels=["Z", "Y", "X"],
        coord_system="RAS",
        checksum=True,
    )
    return path


@pytest.fixture
def nnunet_medh5(tmp_path: Path) -> Path:
    """Tiny sample with nnUNet v2 metadata in ``extra["nnunetv2"]``."""
    shape = (4, 8, 8)
    images = {"0": np.zeros(shape, dtype=np.float32)}
    seg = {"1": np.zeros(shape, dtype=bool)}
    path = tmp_path / "nnu.medh5"
    MEDH5File.write(
        path,
        images=images,
        seg=seg,
        extra={
            "nnunetv2": {
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, "tumor": 1},
                "numTraining": 1,
                "file_ending": ".nii.gz",
            }
        },
    )
    return path


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
