# ONNX Workbench Design Notes

## Scope

This document captures the implemented design decisions for the ONNX workbench flow in this repository.

## Compare Contract

- Compare operates on `Confirmed Crop Artifact` objects.
- Compare checks **I/O compatibility**, not internal graph isomorphism.
- Two crops can be compared even if:
  - node counts differ
  - operator order differs
  - internal topology differs
- Compare requires:
  - complete input bindings for participating inputs
  - at least one output binding
  - identical dtype
  - identical rank
  - identical concrete shape at compare runtime
- v1 does not perform:
  - automatic dtype conversion
  - broadcasting-based reconciliation
  - internal node-by-node diff
  - intermediate tensor diff

## Host / Webview Split

### Host responsibilities

- manage VS Code multi-panel lifecycle
- own ONNX model parsing and session storage
- build lightweight render snapshots for the webview
- own crop artifact creation and staleness semantics
- export crop to ONNX
- run inference with `onnxruntime-node`
- own global compare state and result cache

### Webview responsibilities

- render graph snapshots through Netron view primitives
- handle edge selection and highlight UX
- present `Model Tools` drawer
- show task status, results, and activity
- submit crop / inference / compare requests back to host

## Data Model

### ModelSession

- source ONNX URI
- original ONNX proto
- analyzed graph index
- render snapshot
- current latest artifact id

### CropArtifact

- immutable confirmed crop definition
- selected start/end keys
- selected node ids
- input/output keys
- fixed I/O signature
- thumbnail data URI
- export cache metadata
- stale marker when draft changes afterward

### CompareCenterState

- slot A summary
- slot B summary
- input bindings
- output bindings
- imported compare input preview/token
- compare task status
- latest compare result

## UI State Machine

### Crop

- `Draft`: selection exists, not yet confirmed
- `Confirmed`: crop artifact exists and matches current selection
- `Stale`: crop artifact exists but current draft changed afterward

### Run

- no artifact => run disabled unless `Use Full Graph` is enabled
- import mode without imported token => run disabled
- busy task => run disabled

### Compare

- current panel can only assign the local confirmed artifact to slot A or B
- actual binding and compare execution occur in the shared bottom `Netron Compare` panel

## Waiting State

### Model load

- host sends staged progress
- webview shows task card + status line
- initial empty panel still uses Netron welcome screen until snapshot render arrives

### Crop / export / run / compare

- graph remains visible
- drawer task card shows current stage and status
- cancel action is best-effort and mainly prevents follow-up UI actions from applying stale results

## Offline Testing

The repository includes:

- `scripts/generate_onnx_test_models.py`
- `scripts/smoke_workbench.js`

Generated models cover:

- large single-file matmul ONNX
- large external-data matmul ONNX
- small branch crop model
- compare pair with matching I/O and different internal structure

## Platform Packaging

- The VSIX bundles `onnxruntime-node` native binaries for `linux/x64` and `win32/x64`.
- `linux/arm64` and `darwin/arm64` binaries are excluded from packaging.
- Packaging was verified by extracting the VSIX and running `lib/onnx-workbench.js` against the bundled Linux x64 runtime.
