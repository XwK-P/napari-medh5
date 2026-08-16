# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Full check (ruff + mypy + pytest with coverage floor):

```bash
ruff check . && ruff format --check . && mypy src && pytest -v
```

Pytest runs with `--cov=napari_medh5 --cov-fail-under=90` by default (see `pyproject.toml`); the suite fails if coverage drops below 90%. To run a single test file or case, use the standard pytest selectors, e.g. `pytest tests/test_writer.py -v` or `pytest tests/test_writer.py::test_name -v`.

Install for development (`medh5` 1.0 is on PyPI; use the sibling repo when working on both):

```bash
pip install -e ".[dev]"
pip install -e "../medh5"   # only when developing medh5 alongside
```

Launch napari with the plugin: `napari --plugin napari-medh5 path/to/sample.medh5`. Dock widget is registered under **Plugins → Metadata & Review**.

## Architecture

A napari plugin wrapping the external `medh5` library (sibling repo), against
**format 1.0**. Three contributions in `src/napari_medh5/napari.yaml`: a
reader, a writer, and a dock widget. Several cross-file invariants are
load-bearing and not obvious from any single file.

### Shared handle registry (`_handles.py`)

Lazy `dask` layers need the backing `h5py.File` open for the layer's lifetime.
`REGISTRY` is a reference-counted, thread-locked map `{resolved_path:
medh5.Sample}`. The reader calls `REGISTRY.acquire(path)`; `attach_viewer()`
wires `viewer.layers.events.removed` so the handle closes when the last layer
backed by a file goes away.

**Critical, and for a different reason than in 0.x.** Any write must
`REGISTRY.drop(path)` first. In 0.x that was because HDF5 refuses to open one
file twice. In 1.0 `medh5.amend` is copy-on-write — it builds a new file and
`os.replace`s it — so a handle held across the write keeps serving the *old
inode* with no error at all. `rebind_viewer_layers` re-acquires and swaps
`.data` on every affected layer, across every attached viewer, since the
registry is process-global.

### Lazy arrays (`_arrays.py`)

An image is one dataset, so `image_array` is a direct `da.from_array`. A voxel
annotation is not: five encodings sit behind one read contract and only
`labelmap` stores anything a viewer can colour. `annotation_array` uses
`map_blocks` over `VoxelAnnotation.labelmap(roi=...)`, so napari asks for the
slice it is about to draw and medh5 decodes exactly that window out of
whatever encoding is in the file.

`draw_priority` decides which class wins an overlapping voxel. A napari
`Labels` layer holds one id per voxel, and `labelmap(priority=...)` takes
**highest precedence first**. A lesion inside an organ is a child of it in the
label set DAG (§5.1); ordering by depth, deepest first, keeps the specific
class visible instead of buried under the one containing it.

### Layer role tagging

Every layer carries `metadata`:
- `medh5_path` — resolved source file
- `medh5_role` — `"image"`, `"seg"`, `"bbox_rect"`, `"bbox_wire"`
- `medh5_name` — the object id inside the file
- `medh5_grid`, `medh5_timepoint` — so the writer puts it back on the right grid
- `medh5_classes` — `{class_id: display name}` from the label set
- `medh5_annotated` — the classes that were *examined* (§11.3), so erasing a
  mask does not turn "examined and absent" into "nobody looked"

The writer's `_collect()` relies entirely on these; layers without
`medh5_role` are ignored. New layer kinds need both the reader tagging and the
writer dispatch.

### Boxes (`_bbox.py`)

medh5 boxes are `float32` at voxel **edges** (§8.1); napari rectangles are at
voxel **centres**. Every corner shifts by ±0.5 across the boundary, and a
round trip that forgets it moves every box half a voxel per save with nothing
raising. `EDGE_TO_CENTRE` is applied once on each side, and
`TestRoundTrip::test_S8_1_a_box_survives_read_and_write_unchanged` holds it.

napari has no 3-D box primitive, so a box is drawn as a rectangle on its
shallowest axis's centre slice with the depth extent in
`features["depth_axis"|"depth_lo"|"depth_hi"]`; a box deeper than one voxel
also gets a 12-segment wireframe companion. On write the rectangle layer is
authoritative and the wireframe is skipped. Nothing is rounded — 1.0 boxes are
float, so a box drawn between two voxels stays between them.

### Writer modes (`_writer.py`)

- **Amend** (`_amend`): destination is the source, and the image set and shapes
  match. `medh5.amend` copies unknown objects through untouched, so only the
  edited annotations are re-encoded.
- **Full write** (`_write_new`): Save As, or a changed image set.
  `medh5.create`, carrying identity, timepoints, label set and grid geometry
  from the source where there is one. Geometry is never invented: with no
  source grid the fallback is unit spacing at the origin.

Both record an `annotate` provenance activity naming napari, so an edit made
in a viewer is distinguishable from a model prediction later.

Multi-source saves are rejected.

### Widget (`_widget.py`)

Quality is recorded **per annotation** (§11.2), not per file — 0.x had one
review status for the whole sample and could not say which of three masks was
reviewed. Saving writes a `review` activity plus a quality record. The widget
also shows the validator at a selectable level, per-object digest results, the
label set, and the sample document.

## Tooling notes

- `mypy` runs in `strict` mode on `src/` only; `disable_error_code = ["misc"]` and `h5py`/`napari`/`qtpy`/`dask`/`medh5` have `ignore_missing_imports = true, follow_imports = "skip"` — don't add type stubs for those.
- Ruff selects `E, F, I, UP, B, SIM`. Target is `py310`.
- Tests avoid booting a real napari viewer: `conftest.py` exposes `make_widget_app` which returns the widget paired with a `_MockViewer` that implements the minimal `layers.events.inserted/removed` signal surface the widget consumes. Use this pattern instead of `pytest-qt`-only fixtures when adding widget tests.
- The `build/` directory is a stale `pip install` artefact (same tree as `src/`); ignore it — the package is `src/`-layout per `pyproject.toml`'s `tool.setuptools.packages.find`.
