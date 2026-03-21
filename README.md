# Netron VSCode Workbench

A VS Code extension workspace that previews ONNX models with Netron rendering and adds host-managed crop, export, inference, and compare workflows.

## Commands

Use command palette (`Ctrl+Shift+P`) and run:

- `Netron: Preview Model`
- `Netron: Focus Compare`
- `Netron: Clear Compare`

You can also right-click a file in explorer and run `Netron: Preview Model`.

## Implemented Features

- Multi-panel model preview in VS Code Webview.
- Host-managed ONNX loading path for large models.
- `Model Tools` drawer integrated into the Netron bottom toolbar.
- Crop workflow:
  - select start / end tensors by clicking edges
  - confirm crop to create a `Confirmed Crop Artifact`
  - draft vs confirmed vs stale state handling
- Export workflow:
  - export confirmed crop directly to `.onnx`
  - save current graph as PNG
- Inference workflow:
  - run on confirmed crop or full graph
  - auto input generation: zeros / ones / random
  - import inputs from `.json` or `.npz`
- Compare workflow:
  - compare two crop artifacts by I/O compatibility
  - no internal graph isomorphism requirement
  - A/B slots managed by a shared bottom `Netron Compare` panel
  - output diff metrics include max abs / mean abs / RMSE / max relative diff / cosine similarity
- Activity workflow:
  - per-panel activity log
  - task status, stage display, and busy state propagation

## Design Notes

- Compare is defined as comparing two subgraphs with compatible input/output contracts.
- Compare does **not** require internal topology or operator structure to match.
- ONNX is managed on the extension host side; the webview receives render snapshots rather than full model binaries.

## Platform Support

- Bundled ONNX Runtime native binaries target `linux x64` and `windows x64`.
- `darwin` and `arm64` runtimes are intentionally excluded from the VSIX to control package size.
- The packaged VSIX was validated by extracting the archive and running the host-side smoke flow against the bundled Linux x64 runtime.

## Offline Test Model Generation

Generate local ONNX fixtures without downloading public models:

```bash
npm run generate:testmodels
```

This writes models into `testdata/generated/`, including:

- `large-matmul-singlefile.onnx`
- `large-matmul-external-data.onnx`
- `branch-crop-small.onnx`
- `dual-io-compare-a.onnx`
- `dual-io-compare-b.onnx`

## Smoke Test

Run a host-side smoke test against a generated model:

```bash
node scripts/smoke_workbench.js testdata/generated/branch-crop-small.onnx
```

## VSIX Packaging

```bash
npm install
npx @vscode/vsce package -o dist/netron-vscode-workbench-0.1.0.vsix
```

## VS Code Install

```bash
code --uninstall-extension local.netron-vscode-preview || true
code --install-extension dist/netron-vscode-workbench-0.1.0.vsix --force
```
