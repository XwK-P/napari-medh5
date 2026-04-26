# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Full check (ruff + mypy + pytest with coverage floor):

```bash
ruff check . && ruff format --check . && mypy src && pytest -v
```

Pytest runs with `--cov=napari_medh5 --cov-fail-under=90` by default (see `pyproject.toml`); the suite fails if coverage drops below 90%. To run a single test file or case, use the standard pytest selectors, e.g. `pytest tests/test_writer.py -v` or `pytest tests/test_writer.py::test_name -v`.

Install for development (the core `medh5` library lives in a **sibling repo** and is not on PyPI — install it editable first):

```bash
pip install -e "../medh5"
pip install -e ".[dev]"
```

Launch napari with the plugin: `napari --plugin napari-medh5 path/to/sample.medh5`. Dock widget is registered under **Plugins → Metadata & Review**.

## Architecture

This is a napari plugin that wraps the external `medh5` library (sibling repo). The plugin exposes three contributions via `src/napari_medh5/napari.yaml`: a reader, a writer, and a dock widget. Several cross-file invariants are load-bearing and not obvious from any single file.

### Shared handle registry (`_handles.py`)

Lazy `dask.array.from_array(h5py_dataset)` layers require the backing `h5py.File` to stay open for the lifetime of the napari layer. `REGISTRY` is a reference-counted, thread-locked map `{resolved_path: MEDH5File}`. The reader calls `REGISTRY.acquire(path)` per opened file; `attach_viewer()` wires `viewer.layers.events.removed` so the handle is released (and closed) when the last layer backed by a file is removed.

**Critical**: HDF5 forbids opening the same file twice in a single process. Before any in-place mutation (`MEDH5File.update`), callers must invoke `REGISTRY.drop(path)` to close the open handle — the writer does this in `_save_inplace`. After save, any still-attached lazy layer will raise on next access; callers re-read the file.

### Layer role tagging

Every layer produced by the reader carries a `metadata` dict with:
- `medh5_path`: resolved source-file path (used by the widget for sample discovery and by the writer to detect the source).
- `medh5_role`: one of `"image"`, `"seg"`, `"bbox_rect"`, `"bbox_wire"`.
- `medh5_name`: original key inside the medh5 file (preserves round-trip when display names are remapped, e.g. by nnU-Net labels).

The writer's `_collect()` relies entirely on these tags — layers without `medh5_role` are silently ignored. If you add new layer kinds, extend both the reader tagging and the writer's dispatch.

### Bbox round-trip (`_bbox.py`)

Medh5 stores bboxes as `(n, ndim, 2)` arrays. napari doesn't have a native 3D box primitive, so:
- **Read**: each box projects onto its smallest-extent axis ("depth axis"); a rectangle is drawn on that axis's centre slice. Depth info is preserved in `features["depth_axis"|"depth_lo"|"depth_hi"]`. If any box spans more than one voxel along depth, a companion `bbox_wire` Shapes layer renders the full 3D cuboid as 12 line segments.
- **Write**: the `bbox_rect` layer is authoritative. The wireframe companion (`bbox_wire`) is skipped by the writer. Depth extents come from the `features` columns, not from rectangle geometry.

### Writer modes (`_writer.py`)

`write_sample` chooses between two paths:
- **In-place** (`_save_inplace`): destination resolves to the same path as `medh5_path` AND image modality set + shapes match the source. Uses `MEDH5File.update(seg_ops=..., bbox_ops=...)` which preserves compression and only rewrites the changed sub-datasets.
- **Full rewrite** (`_save_full`): any other case (Save As, added/removed modalities). Uses `MEDH5File.write()`; spatial/label metadata is copied from the source via `MEDH5File.read_meta(source)` if available. Dask arrays are materialised to numpy here.

Multi-source saves (layers from different `medh5_path`s) are rejected.

### Spatial transforms (`_layers.py`)

If the source `direction` matrix is identity (or absent), layers receive napari `scale` + `translate` kwargs. If direction is a non-identity rotation, layers receive a single `(ndim+1, ndim+1)` homogeneous `affine` instead — mixing `scale`/`translate` with rotation produces wrong geometry in napari.

### nnU-Net label surfacing

If `meta.extra["nnunetv2"]["labels"]` is present (dict of `{class_name: value}`), seg layer names whose raw key is a bare integer are displayed as the class name (`:seg:tumor` instead of `:seg:1`). The widget also renders this mapping in its "nnU-Net v2 classes" group box. The `medh5_name` metadata always preserves the original raw key so writes round-trip correctly.

## Tooling notes

- `mypy` runs in `strict` mode on `src/` only; `disable_error_code = ["misc"]` and `h5py`/`napari`/`qtpy`/`dask`/`medh5` have `ignore_missing_imports = true, follow_imports = "skip"` — don't add type stubs for those.
- Ruff selects `E, F, I, UP, B, SIM`. Target is `py310`.
- Tests avoid booting a real napari viewer: `conftest.py` exposes `make_widget_app` which returns the widget paired with a `_MockViewer` that implements the minimal `layers.events.inserted/removed` signal surface the widget consumes. Use this pattern instead of `pytest-qt`-only fixtures when adding widget tests.
- The `build/` directory is a stale `pip install` artefact (same tree as `src/`); ignore it — the package is `src/`-layout per `pyproject.toml`'s `tool.setuptools.packages.find`.
