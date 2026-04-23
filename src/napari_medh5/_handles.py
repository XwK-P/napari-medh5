"""Shared registry of open ``MEDH5File`` handles.

Lazy napari layers need the underlying ``h5py.File`` to stay open for the
lifetime of the layer.  We keep one :class:`medh5.MEDH5File` per path and
reference-count against the number of napari layers that still depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

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
