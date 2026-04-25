"""Shared registry of open ``MEDH5File`` handles.

Lazy napari layers need the underlying ``h5py.File`` to stay open for the
lifetime of the layer.  We keep one :class:`medh5.MEDH5File` per path and
reference-count against the number of napari layers that still depend on it.

:func:`attach_viewer` wires a viewer's ``layers.events.removed`` signal so
the registry drops a file's handle once the last napari layer backed by
that file is removed.

:func:`rebind_viewer_layers` re-attaches lazy ``Image`` / ``Labels`` layers
to a freshly reopened file after an in-place mutation — HDF5 forbids
opening the same file twice, so callers must ``REGISTRY.drop(path)`` before
writing, which invalidates the dask arrays pointing at the now-closed
datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from weakref import WeakSet

import dask.array as da
from medh5 import MEDH5File


@dataclass
class _Entry:
    handle: MEDH5File
    refcount: int = 0


@dataclass
class _Registry:
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def acquire(self, path: str | Path) -> MEDH5File:
        """Open ``path`` (or reuse an existing handle) and bump the refcount."""
        key = str(Path(path).resolve())
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(handle=MEDH5File(key, mode="r"))
                self._entries[key] = entry
            entry.refcount += 1
            return entry.handle

    def release(self, path: str | Path) -> None:
        """Decrement the refcount for ``path`` and close when it hits zero."""
        key = str(Path(path).resolve())
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.refcount -= 1
            if entry.refcount <= 0:
                entry.handle.close()
                del self._entries[key]

    def get(self, path: str | Path) -> MEDH5File | None:
        key = str(Path(path).resolve())
        with self._lock:
            entry = self._entries.get(key)
            return entry.handle if entry else None

    def drop(self, path: str | Path) -> None:
        """Close and forget the handle for ``path`` regardless of refcount.

        Use this before an in-place mutation (``MEDH5File.update``) so the
        file can be reopened in append mode — HDF5 refuses to open the same
        file twice in a single process.  Any lazy layer arrays backed by the
        closed handle will raise on next access; callers are expected to
        re-read the file afterwards.
        """
        key = str(Path(path).resolve())
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                entry.handle.close()

    def close_all(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                entry.handle.close()
            self._entries.clear()


REGISTRY = _Registry()

_attached_viewers: WeakSet[Any] = WeakSet()


def _layer_medh5_path(layer: Any) -> str | None:
    meta = getattr(layer, "metadata", None)
    if not isinstance(meta, dict):
        return None
    path = meta.get("medh5_path")
    return path if isinstance(path, str) else None


def attach_viewer(viewer: Any) -> None:
    """Hook *viewer* so removing the last layer of a file drops its handle.

    Idempotent per viewer — safe to call on every reader invocation and
    from the widget constructor.  The viewer is tracked via a ``WeakSet``
    so garbage-collected viewers don't leak ids across sessions.
    """
    if viewer is None or viewer in _attached_viewers:
        return
    layers = getattr(viewer, "layers", None)
    events = getattr(layers, "events", None)
    removed = getattr(events, "removed", None)
    if removed is None:
        return
    _attached_viewers.add(viewer)

    def _on_removed(event: Any) -> None:
        removed_layer = getattr(event, "value", None)
        path = _layer_medh5_path(removed_layer)
        if path is None:
            return
        if any(_layer_medh5_path(layer) == path for layer in layers):
            return
        REGISTRY.drop(path)

    removed.connect(_on_removed)


def rebind_viewer_layers(path: str | Path, viewer: Any | None = None) -> None:
    """Rebind lazy layer arrays after an in-place mutation of ``path``.

    An in-place write requires :func:`REGISTRY.drop` first — the closed
    ``h5py.File`` invalidates every lazy ``dask.array`` that napari layers
    hold, so any subsequent slice read would raise.  This helper:

    1. Re-acquires a fresh handle via :func:`REGISTRY.acquire`.
    2. Walks *viewer*'s layers and swaps ``.data`` on every ``Image`` /
       ``Labels`` layer tagged with ``medh5_path == path`` to a new dask
       view of the matching dataset in the reopened file.

    ``Shapes`` layers (bbox rect / wire) are pure numpy and need no rebind.
    If *viewer* is ``None`` the function falls back to
    ``napari.current_viewer()``; if that is also ``None`` it silently
    no-ops — callers invoke it best-effort (e.g. the writer may run
    without any viewer in unit tests).
    """
    if viewer is None:
        try:
            import napari
        except ImportError:
            return
        viewer = napari.current_viewer()
    if viewer is None:
        return
    layers = getattr(viewer, "layers", None)
    if layers is None:
        return
    key = str(Path(path).resolve())
    targets = [
        layer
        for layer in layers
        if _layer_medh5_path(layer) == key
        and (layer.metadata or {}).get("medh5_role") in {"image", "seg"}
    ]
    if not targets:
        return
    handle = REGISTRY.acquire(key)
    for layer in targets:
        meta = layer.metadata
        name = meta.get("medh5_name")
        role = meta.get("medh5_role")
        group = handle.images if role == "image" else handle.seg
        if group is None or name not in group:
            continue
        ds = group[name]
        arr = da.from_array(ds, chunks=ds.chunks or "auto")
        if role == "seg":
            import numpy as np

            arr = arr.astype(np.uint8)
        layer.data = arr
