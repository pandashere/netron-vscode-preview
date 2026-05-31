# ONNX Workbench Manual QA Checklist

## Preconditions

- Install the packaged VSIX.
- Open a workspace folder.
- Install sample exporter/analyzer tools when checking text export or AI analysis:

```bash
npm run qa:install-ai-tools
```

- The tool registry roots are global to the VS Code extension host:
  - exporters: `~/.netron/vscode-preview/exporters`
  - analyzers: `~/.netron/vscode-preview/analyzers`
- In Remote SSH, WSL, or container sessions, `~` is resolved in the remote extension-host environment, not on the local desktop.
- Generate offline fixtures when needed:

```bash
npm run generate:testmodels
```

Recommended fixtures:

- `testdata/generated/branch-crop-small.onnx`
- `testdata/generated/dual-io-compare-a.onnx`
- `testdata/generated/dual-io-compare-b.onnx`
- `testdata/generated/large-matmul-singlefile.onnx`
- `testdata/generated/large-matmul-external-data.onnx`

## 1. Multi-Panel Model Open

- Run `Netron: Preview Model` twice.
- Open two different ONNX models in two different tabs.
- Verify both tabs remain open simultaneously.
- Verify actions in one tab do not overwrite the other tab's crop, activity, or task state.

## 2. Crop Draft / Confirmed / Stale

- Open `branch-crop-small.onnx`.
- Click `Model Tools` in the bottom toolbar.
- Enter `Select Start Tensor` mode and pick one valid input edge.
- Enter `Select End Tensor` mode and pick one valid output edge.
- Click `Confirm Crop`.
- Verify crop summary shows `Confirmed`.
- Change either the start or end selection.
- Verify crop summary changes to `Stale`.
- Verify `Export Crop ONNX`, `Run Inference`, and `Set As A/B` are disabled until reconfirmed.

## 3. Crop Export

- Reconfirm the crop.
- Click `Export Crop ONNX`.
- Save the file.
- Reopen the exported ONNX in a fresh Netron tab.
- Verify the exported graph loads and shows only the cropped subgraph.

## 4. Scriptable Text Export

- Reconfirm a crop.
- In the Crop tab, select `Crop JSON Summary` in the Text Export exporter dropdown.
- Verify unavailable entries, if any, are disabled and can expose details from the small disclosure arrow.
- Click `Copy Export Text`.
- Paste the clipboard into a scratch editor.
- Verify the copied text includes model, artifact, graph, input, output, node count, and tensor count lines.
- Temporarily break the selected exporter's manifest, for example by removing `command`, and verify `Copy Export Text` is disabled with a visible unavailable reason.
- Restore the manifest and verify the dropdown returns to a ready state after refresh.

## 5. AI Analysis Panel

- Reconfirm a crop.
- In the AI tab, select `Crop JSON Summary` as Formatter and `Line Count Analysis` as Analyzer.
- Click `Analyze`.
- Verify the global `AI Analysis` panel opens or focuses.
- Verify the panel shows status metadata and the analyzer result as plain text.
- Verify the AI tab itself shows task status only, not the full analyzer result.
- Verify the result can be copied from the AI panel.
- Start a long-running analyzer variant and verify Cancel is available from both the AI tab and AI panel.
- Verify `Analyze` reruns formatter and analyzer each time, rather than reusing old output.

## 6. Single-Model Inference

- On a confirmed crop, use `Run` tab.
- Run with `Auto / zeros`.
- Run with `Auto / ones`.
- Run with `Auto / random`.
- Import a `.json` input and run again.
- Import a `.npz` input and run again.
- Verify result table appears with output name, dtype, shape, and summary stats.

## 7. Compare With Non-Isomorphic Subgraphs

- Open `dual-io-compare-a.onnx` in one tab.
- Open `dual-io-compare-b.onnx` in another tab.
- Confirm a crop in each tab spanning the full graph.
- In one tab, click `Set As A`.
- In the other tab, click `Set As B`.
- Verify the bottom `Netron Compare` panel auto-focuses.
- Verify A/B slot cards show different source models.
- Verify compare can run even though the internal operator ordering differs.
- Run compare with `Auto / ones`.
- Verify the result table is produced from output bindings, not graph topology.

## 8. Compare Binding Behavior

- Use a pair of crops whose port names differ but shapes and dtypes are compatible.
- Verify binding rows appear in the bottom `Netron Compare` panel.
- Manually remap inputs and outputs.
- Run compare.
- Verify compare uses the manual mapping successfully.

## 9. Large ONNX Load Behavior

- Open `large-matmul-singlefile.onnx`.
- Verify the model opens through the host-managed path.
- Verify task stage text updates during load.
- Verify UI remains responsive while loading.
- Repeat with `large-matmul-external-data.onnx`.
- Confirm crop export still works for the external-data source.

## 10. Compare Panel Persistence Semantics

- Put a confirmed crop into slot A.
- Close the source model tab.
- Run `Netron: Focus Compare`.
- Verify the slot entry still exists and is visible.
- Run `Netron: Clear Compare`.
- Verify the slots are emptied.

## 11. Busy-State / Anti-Misclick Checks

- Start a long-running action.
- Verify the related buttons are disabled while busy.
- Verify task status shows stage and message.
- Click `Cancel`.
- Verify the UI reflects cancellation request and avoids applying stale follow-up actions.

## 12. Packaging Sanity

- Verify the packaged VSIX installs successfully.
- Verify Linux x64 runtime works in the local environment.
- Verify the package includes only Linux x64 and Windows x64 ONNX Runtime binaries.
