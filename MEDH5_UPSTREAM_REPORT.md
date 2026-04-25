# Upstream `medh5` — Review & Improvement Recommendations

**Author:** napari-medh5 plugin maintainer
**Date:** 2026-04-24
**Scope:** Observations gathered while building and hardening the `napari-medh5`
plugin with a headless end-to-end integration test suite driving a real
`napari.Viewer`. All findings are grounded in bugs that had to be worked around
downstream.

---

## 1. Background

`medh5` is the storage/format library that defines the `.medh5` container
(HDF5-backed) and ships the core read/write/update/verify surface
(`MEDH5File`), metadata dataclasses (`SampleMeta`, `SpatialMeta`,
`ValidationReport`), and review-status helpers.

`napari-medh5` is a thin plugin that exposes `medh5` to napari via three
npe2 contributions:

- **Reader**: opens a `.medh5`, returns lazy `dask.array` layers backed by
  open `h5py.Dataset` handles.
- **Writer**: saves layers back to a `.medh5` either in-place
  (`MEDH5File.update`) or as a full rewrite (`MEDH5File.write`).
- **Dock widget**: surfaces metadata, nnU-Net label map, validation, checksum
  verification, and review-status editing — backed by `MEDH5File.read_meta`,
  `MEDH5File.verify`, and `set_review_status`.

During integration testing against a live napari viewer we observed four
concrete bugs in the plugin that all trace back to the same shared root
cause — **HDF5's single-open-per-process constraint is exposed raw by
`medh5`'s public API** — plus a handful of smaller ergonomic gaps. Each
item below is framed as: *what happened, why it matters, suggested fix.*

---

## 2. Findings — behavioural bugs / footguns

### 2.1 `set_review_status` needs exclusive write access but doesn't advertise it

**What we saw.** The widget keeps a shared read-only `MEDH5File` open for
lazy layers. Calling `set_review_status(path, ...)` while that handle is
open raises `OSError: unable to open file (file is already open for
read-only)`. The error surfaces as a raw HDF5 message with no hint that
the conflict is caused by another handle *in the same process*.

**Why it matters.** Any consumer that keeps a read handle open (viewers,
IDE previews, dashboards) must discover this constraint empirically. We
only found it because an integration test tried the full round-trip.

**Suggested fix.**

1. Document the constraint on the `set_review_status` docstring.
2. Detect the conflict (try/except around the `h5py.File(..., "a")`
   open) and raise a `MEDH5FileError("file is already open in this
   process for reading; close other handles before updating review
   status")` with the original as `__cause__`.
3. Optionally, offer a **swap-and-rename** write path (write a new file
   alongside, then atomically `os.replace`), which would let concurrent
   readers keep working off the old inode until they re-open.

### 2.2 `MEDH5File.update` has the same single-open footgun

**What we saw.** Same root cause as 2.1. Any consumer holding a read
handle for lazy slicing must explicitly close it before calling
`MEDH5File.update`. We solved it downstream by adding a reference-counted
handle registry (`napari_medh5._handles.REGISTRY`) plus an explicit
`REGISTRY.drop(path)` call immediately before every `update`.

**Why it matters.** Every downstream consumer that wants to offer lazy
reads *and* in-place edits is forced to reinvent the same registry.

**Suggested fix.** Several options, in increasing ambition:

- Document the constraint on `MEDH5File.update`.
- Provide a cooperative hook: `MEDH5File.update(path, *, on_reopen=None)`
  that calls `on_reopen(path)` after the internal `h5py.File` handle is
  closed, letting consumers re-issue their lazy readers.
- Offer atomic swap-and-rename semantics so readers don't have to
  cooperate at all (see 2.4 — the two items share this remedy).

### 2.3 Post-update signal for lazy-reader consumers

**What we saw.** Even after a consumer correctly drops its read handle
and calls `update`, every lazy `dask.array` already handed to a client
(e.g. a napari layer) still points at a now-closed `h5py.Dataset` — the
next slice read raises. Downstream we built
`rebind_viewer_layers(path, viewer)` to swap `layer.data` to fresh
dask views after every `update`.

**Why it matters.** This is the class of bugs that's invisible in unit
tests and blows up only when the user scrolls after saving.

**Suggested fix.**

- Emit an event/callback (`MEDH5File.update` accepts an
  `on_reopened: Callable[[MEDH5File], None]` argument, invoked after
  the append-mode handle is closed).
- Or provide a low-level helper `medh5.reopen_datasets(path, names) →
  dict[str, h5py.Dataset]` so consumers have a blessed way to rebuild
  their cached views.

### 2.4 `MEDH5File.write` has no atomic-replace semantics

**What we saw.** A partial failure during `write` (disk full, user
interrupt, exception from a validator) leaves the destination in an
inconsistent state — the target file exists but is partially populated.

**Why it matters.** This is the classic "my sample file was corrupted
when I hit Ctrl-C" class of bug. It also compounds 2.1 / 2.2 since a
failing `update` can leave the file unreadable until manual recovery.

**Suggested fix.** Write to `<dest>.tmp` (same directory, so
`os.replace` stays atomic on POSIX), fsync, then `os.replace`. Expose a
public `atomic=True` parameter (defaulting to `True` would be safer,
but may be a breaking change — make it opt-in first, opt-out later).

### 2.5 `MEDH5File.verify` can't distinguish "verified good" from "no checksum stored"

**What we saw.** `verify(path)` returns `True` both when the stored
checksum matches **and** when no checksum was ever written. The widget
currently renders a single green "OK" in both cases, which is
misleading.

**Why it matters.** A consumer can't build a trustworthy audit workflow
— a file with no checksum is indistinguishable from a file that
verified successfully.

**Suggested fix.** Return a tri-state result:

```python
class VerifyResult(Enum):
    OK = "ok"
    MISSING = "missing"
    MISMATCH = "mismatch"
```

or, less invasively, expose `SampleMeta.has_checksum: bool` so
consumers can gate the UI themselves.

### 2.6 Handle leaks on failed writes

**What we saw.** When `set_review_status` / `update` raised mid-way
(seen during test fuzzing with malformed `extra` payloads), the HDF5
handle sometimes remained locked until process exit, requiring a
full pytest-worker restart.

**Why it matters.** Hard to reproduce, but it surfaces as "the file is
locked and I don't know why" — the worst class of bug to debug.

**Suggested fix.** Audit `medh5/core.py` and `medh5/review.py` for
`h5py.File(...)` opens that aren't wrapped in a `with` statement *or*
`try/finally`. Guarantee close on every path, including exceptions
raised inside validators/encoders.

---

## 3. Findings — API / ergonomic improvements

### 3.1 No context-manager idiom for concurrent-safe reads

**Problem.** Consumers that want to coexist with writers currently have
to roll their own registry. Downstream we wrote ~90 lines of locked
reference-counting code (`_handles.py`).

**Suggested fix.** Either:

- Add SWMR support (`MEDH5File(path, mode="r", swmr=True)`), which lets
  multiple readers coexist with one writer at the HDF5 level.
- Or ship a `medh5.open_shared(path)` helper that implements
  reference-counted open/close inside the library.

Either approach obsoletes our downstream registry — every consumer
would benefit.

### 3.2 `read_meta` should report malformed `extra` via warnings, not silent coercion

**Problem.** The napari widget surfaces `meta.extra["nnunetv2"]["labels"]`
as a class-name lookup table. If `labels` comes back as anything other
than `{str: int}`, the plugin has to implement defensive guards (a
type-narrowing `_nnunet_labels` helper). Without upstream validation,
every consumer re-implements this check.

**Suggested fix.** In `read_meta`, validate well-known `extra`
subsystems (`nnunetv2`, `review`, `checksum`, …) and emit a
`UserWarning` with a clear message if the payload is malformed. Keep
the raw dict accessible so consumers can still introspect.

### 3.3 `SpatialMeta.as_affine(ndim)` convenience

**Problem.** The plugin has to compose rotation × spacing + origin into
an `(ndim+1, ndim+1)` homogeneous matrix by hand
(`napari_medh5._layers._spatial_to_affine`, ~30 lines). This is
boilerplate every viewer-style consumer will need.

**Suggested fix.** Add `SpatialMeta.as_affine(ndim: int) ->
np.ndarray | None` that returns `None` when direction is identity/absent
(so consumers can fall back to `scale`+`translate`) and a full affine
otherwise. Trivial to land; high impact.

### 3.4 Bbox shape-validation helper

**Problem.** Writers must clamp bboxes to sample bounds to prevent
off-by-one issues after user edits. Downstream we reconstruct
`sample_shape` from the first image's shape and clamp inside
`_bbox.shapes_to_arrays`. Every producer will need this.

**Suggested fix.** Expose
`medh5.validate_bboxes(bboxes, sample_shape) -> (clamped, report)`
that returns both the clamped array and a list of `(index, axis,
reason)` tuples. Consumers can then decide whether to warn, reject,
or silently clamp.

### 3.5 Schema versioning on `extra` subsystems

**Problem.** `extra["nnunetv2"]`, `extra["review"]`, `extra["checksum"]`
are all consumed by downstream code. A schema change upstream would
silently corrupt plugin output.

**Suggested fix.** Stamp each subsystem:

```python
meta.extra = {
    "review": {"schema_version": 1, "status": "...", ...},
    "nnunetv2": {"schema_version": 1, "labels": {...}, ...},
}
```

Consumers can then fail loudly on unknown versions instead of
silently mis-rendering.

### 3.6 `set_review_status` returns `None`

**Problem.** Consumers need to know the freshly written status (e.g. to
refresh UI); currently they must re-read the file.

**Suggested fix.** Return the written `ReviewStatus` dataclass. Non-
breaking change if consumers aren't checking the return value today.

### 3.7 Review history missing the initial "pending" entry

**What we saw.** `set_review_status` appends to history only on
transitions **after** the first explicit call. The initial
implicit/pending state is never recorded, so the audit trail skips the
sample's entire pre-review life.

**Suggested fix.** Either record the initial state at file-creation time
(in `MEDH5File.write`) or, less invasively, document that history
tracks transitions from the first recorded status onwards.

### 3.8 `ValidationReport` entries need a location field

**Problem.** The widget renders validation findings as a flat
`(severity, code, message)` triple. Adding a `path` / `location` field
(e.g. `"images/CT"`, `"extra.nnunetv2.labels"`) would let downstream
consumers highlight the offending dataset without re-parsing `message`
strings.

**Suggested fix.** Extend the dataclass with an optional `location:
str | None` field. Non-breaking — existing consumers ignore it; new UIs
can opt in.

### 3.9 Redundant metadata-read surface

**Problem.** `MEDH5File.read`, `MEDH5File.read_meta`, and the
`MEDH5File(path, mode="r")` constructor all surface metadata with
subtly different completeness. Downstream code accidentally used the
wrong one more than once during this testing pass.

**Suggested fix.** Either consolidate (one canonical metadata entry
point) or document the distinction prominently in the README and
docstrings ("use `read_meta` for metadata-only; `read` for full
materialised arrays; the constructor returns a lazy handle").

### 3.10 Pydantic V2 `json_encoders` deprecation

**What we saw.** `medh5.meta` triggers `pydantic.PydanticDeprecatedSince20`
warnings during tests due to the removed `Config.json_encoders` pattern.

**Suggested fix.** Migrate to field serializers
(`@field_serializer`) or `model_serializer`. Unblocks Pydantic V3
cleanly.

### 3.11 npe2 plugin manifest template alignment (if upstream hosts one)

**What we saw.** An earlier version of the napari plugin manifest used
`image{1,}` as a writer `layer_types` pattern; npe2 rejects `{min,}`
without an explicit max and silently leaves the plugin unregistered.
We fixed it to `image+`.

**Suggested fix.** If `medh5` ships or documents an example plugin
manifest anywhere, align it with current npe2 syntax and pin
`npe2>=0.8` in the example's dev deps.

---

## 4. Recommended priority

If only a subset of this list gets picked up, the highest-leverage
items — ranked by "how much pain does fixing this remove from every
downstream consumer" — are:

1. **2.1 + 2.2** (single-open footgun docs + error message): small
   change, huge debuggability win.
2. **3.1** (`open_shared` or SWMR): removes the entire registry layer
   downstream and fixes a class of bugs at its source.
3. **2.4** (atomic rewrite): prevents corruption on interrupted writes.
4. **2.5 + 3.8** (tri-state verify + ValidationReport locations):
   unlocks a much more useful review UI.
5. **3.3** (`SpatialMeta.as_affine`): tiny win, zero-risk, high-use.

Items 3.5–3.11 are polish — worth batching into a single "cleanup"
release once the higher-priority items have shipped.

---

## 5. Evidence — what these findings are grounded in

All items above were observed during the headless integration-test pass
for `napari-medh5` (20 new tests in `tests/test_integration_napari.py`,
85 total, 97% coverage). The tests drive a real `napari.Viewer`,
exercising:

- `viewer.open(path, plugin="napari-medh5")` → reader + lazy dask
  layers + affine fallback + nnU-Net renaming + bbox wireframe.
- `viewer.layers.save(path, plugin="napari-medh5")` → in-place and
  full-rewrite writer paths, multi-source rejection, bbox round-trips.
- `MEDH5Widget` against a live viewer → sample picker, nnU-Net label
  rendering, checksum verify, review-status round-trip, layer-removal
  handle lifecycle.
- Corrupt / rotated / deep-bbox fixtures to force the edge cases.

Four plugin bugs found and fixed in-pass (invalid npe2 manifest
syntax, silent-failing review save, post-save lazy-layer crash,
silent bbox clamping); three of those four trace directly back to
items 2.1 / 2.2 / 2.3 above. The remaining findings are ergonomic
gaps that surfaced while reading the `medh5` API closely enough to
write those tests.

---

*End of report.*
