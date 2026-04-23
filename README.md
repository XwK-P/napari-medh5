# napari-medh5

A [napari](https://napari.org) plugin for viewing, annotating, and reviewing
[`.medh5`](https://github.com/XwK-P/medh5) medical-imaging samples.

## Contributions

- **Reader** for `*.medh5`: opens each modality as a lazy `Image` layer, each
  segmentation class as a `Labels` layer, and bounding boxes as `Shapes`
  layers, preserving spacing / origin / direction as napari affine transforms.
- **Writer** for `*.medh5`: saves edited samples in place via
  `MEDH5File.update()` when only seg / bbox / metadata changed, or falls back
  to a full `MEDH5File.write()` for Save As.
- **Metadata & Review dock widget**: validation report, checksum verification,
  review-status controls (pending / reviewed / flagged / rejected) with an
  audit trail, editable metadata tree, and nnU-Net v2 class-name surfacing.

## Install

```bash
pip install -e "../medh5"          # the core library, editable
pip install -e ".[dev]"            # this plugin, with dev extras
```

## Launch

```bash
napari --plugin napari-medh5 path/to/sample.medh5
```

Open the dock via **Plugins → napari-medh5: Metadata & Review**.

## Development

```bash
ruff check . && ruff format --check . && mypy src && pytest -v
```
