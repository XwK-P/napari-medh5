# napari-medh5

A [napari](https://napari.org) plugin for viewing, annotating, and reviewing
[`.medh5`](https://github.com/XwK-P/medh5) medical-imaging samples.

Built against **MEDH5 format 1.0**.

## Contributions

- **Reader** for `*.medh5` and `*.medh5c`: every image is a lazy `Image` layer,
  every voxel annotation a lazy `Labels` layer decoded window by window out of
  whichever of the five encodings the file uses, and boxes are `Shapes` layers.
  Spacing and origin become napari `scale`/`translate`; an oblique `direction`
  becomes a single affine, because mixing the two double-applies the spacing.
  A multi-visit sample tags each layer with its timepoint.
- **Writer** for `*.medh5`: an amend when the destination is the source and the
  image set matches — only the edited annotations are re-encoded, and unknown
  objects are copied through untouched — or a full write for Save As, carrying
  the source's identity, timepoints, label set and geometry. Both record a
  provenance activity naming napari, so a viewer edit is distinguishable from a
  model prediction a year later.
- **Dock widget**: the validator at a selectable level, per-object digest
  verification, **per-annotation** quality records with their provenance, the
  label set, and the sample document.

Two conventions the plugin is careful about, because getting either wrong is
silent:

- **Boxes sit at voxel edges**; napari draws at voxel centres. The half-voxel
  is applied once on each side, and a test holds the round trip exact.
- **A class examined and not found** is not the same as a class nobody looked
  for (§11.3). Erasing a mask in the viewer keeps the class declared, so the
  file still says it was searched for.

## Install

```bash
pip install napari-medh5
```

For development against the library:

```bash
pip install -e ".[dev]"
pip install -e "../medh5"   # only when developing medh5 alongside
```

## Launch

```bash
napari --plugin napari-medh5 path/to/sample.medh5
```

Open the dock via **Plugins → napari-medh5: Metadata & Review**.

The file format is documented at
[medh5/docs](https://github.com/XwK-P/medh5/blob/main/docs/index.md); the
normative specification is
[docs/spec/medh5-1.0.md](https://github.com/XwK-P/medh5/blob/main/docs/spec/medh5-1.0.md).

## Development

```bash
ruff check . && ruff format --check . && mypy src && pytest -v
```
