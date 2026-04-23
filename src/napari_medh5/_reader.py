"""napari reader contribution for ``.medh5`` files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from napari_medh5._handles import REGISTRY
from napari_medh5._layers import file_to_layers
from napari_medh5._types import LayerDataTuple

_SUFFIX = ".medh5"


def napari_get_reader(
    path: str | list[str],
) -> Callable[[str | list[str]], list[LayerDataTuple]] | None:
    """Return a reader callable if *path* can be opened."""
    paths = [path] if isinstance(path, str) else list(path)
    if not paths or not all(Path(p).suffix == _SUFFIX for p in paths):
        return None
    return _read


def _read(path: str | list[str]) -> list[LayerDataTuple]:
    paths = [path] if isinstance(path, str) else list(path)
    layers: list[LayerDataTuple] = []
    for p in paths:
        handle = REGISTRY.acquire(p)
        layers.extend(file_to_layers(handle, p))
    return layers
