"""Shared registry of open ``medh5.Sample`` handles.

Lazy napari layers need the underlying ``h5py.File`` to stay open for the
lifetime of the layer.  The registry keeps one :class:`medh5.Sample` per
resolved path and reference-counts it against the number of napari layers
that still depend on it; :func:`attach_viewer` wires a viewer's
``layers.events.removed`` so the handle closes once the last layer backed by
that file goes away.

**Why a write still has to drop the handle.**  In 0.x the reason was that
HDF5 refuses to open one file twice in a process.  In 1.0 the reason is
better: ``medh5.amend`` is copy-on-write --- it builds a new file and
``os.replace``\\ s it into position --- so a handle opened before the write
keeps serving the *old inode* indefinitely.  Nothing raises; the viewer just
shows pre-edit data forever.  Dropping and rebinding is what makes the edit
visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from weakref import WeakSet

import medh5

from napari_medh5._arrays import annotation_array, image_array


@dataclass
class _Entry:
    handle: Any  # medh5.Sample
    refcount: int = 0


@dataclass
class _Registry:
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def acquire(self, path: str | Path) -> Any:
        """Open ``path`` (or reuse an existing handle) and bump the refcount."""
        key = str(Path(path).resolve())
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(handle=medh5.open(key))
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

    def get(self, path: str | Path) -> Any | None:
        key = str(Path(path).resolve())
        with self._lock:
            entry = self._entries.get(key)
            return entry.handle if entry else None

    def drop(self, path: str | Path) -> None:
        """Close and forget the handle for ``path`` regardless of refcount.

        Call this before any write.  ``medh5.amend`` replaces the file, so a
        handle held across the write reads the pre-edit inode and shows stale
        data with no error at all --- the failure mode a raise would at least
        have made obvious.
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


def _resolved_layer_medh5_path(layer: Any) -> str | None:
    """The layer's ``medh5_path``, resolved the way the registry keys entries.

    The reader stores the path it was handed, which may be relative or contain
    symlinks.  Comparing those raw strings against a resolved registry key
    would leave lazy layers stranded after :meth:`_Registry.drop`.
    """
    raw = _layer_medh5_path(layer)
    if raw is None:
        return None
    return str(Path(raw).resolve())


def attach_viewer(viewer: Any) -> None:
    """Hook *viewer* so removing the last layer of a file drops its handle.

    Idempotent per viewer --- safe to call on every reader invocation and from
    the widget constructor.  Viewers are tracked in a ``WeakSet`` so
    garbage-collected ones do not leak ids across sessions.
    """
    if viewer is None or viewer in _attached_viewers:
        return
    layers: Any = getattr(viewer, "layers", None)
    events = getattr(layers, "events", None)
    removed = getattr(events, "removed", None)
    if removed is None or layers is None:
        return
    _attached_viewers.add(viewer)

    def _on_removed(event: Any) -> None:
        removed_layer = getattr(event, "value", None)
        key = _resolved_layer_medh5_path(removed_layer)
        if key is None:
            return
        if any(_resolved_layer_medh5_path(layer) == key for layer in layers):
            return
        REGISTRY.drop(key)

    removed.connect(_on_removed)


def rebind_viewer_layers(path: str | Path, viewer: Any | None = None) -> None:
    """Rebind lazy layer arrays after a write to ``path``.

    A write requires :meth:`_Registry.drop` first, which leaves every lazy
    array in every viewer pointing at a closed --- or, worse, replaced ---
    file.  This re-acquires a handle and swaps ``.data`` on every ``Image``
    and ``Labels`` layer tagged with a matching ``medh5_path``.

    The registry is process-global, so one drop can invalidate layers across
    several viewers in a multi-window session; this rebinds across all of them
    rather than only the one passed.  ``Shapes`` layers (boxes) are numpy and
    need no rebind.
    """
    candidates: list[Any] = list(_attached_viewers)
    if viewer is not None and viewer not in candidates:
        candidates.append(viewer)
    if not candidates:
        try:
            import napari
        except ImportError:
            return
        current = napari.current_viewer()
        if current is None:
            return
        candidates.append(current)

    key = str(Path(path).resolve())
    targets: list[Any] = []
    for one in candidates:
        layers = getattr(one, "layers", None)
        if layers is None:
            continue
        for layer in list(layers):
            if _resolved_layer_medh5_path(layer) == key and (layer.metadata or {}).get(
                "medh5_role"
            ) in {"image", "seg"}:
                targets.append(layer)
    if not targets:
        return

    sample = REGISTRY.acquire(key)
    for layer in targets:
        meta = layer.metadata
        name = meta.get("medh5_name")
        role = meta.get("medh5_role")
        if role == "image" and name in sample.images:
            layer.data = image_array(sample.images[name])
        elif role == "seg" and name in sample.annotations:
            layer.data = annotation_array(sample.annotations[name])
