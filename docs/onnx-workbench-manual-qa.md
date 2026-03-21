# ONNX Workbench Manual QA Checklist

## Preconditions

- Install the packaged VSIX.
- Open a workspace folder.
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

## 4. Single-Model Inference

- On a confirmed crop, use `Run` tab.
- Run with `Auto / zeros`.
- Run with `Auto / ones`.
- Run with `Auto / random`.
- Import a `.json` input and run again.
- Import a `.npz` input and run again.
- Verify result table appears with output name, dtype, shape, and summary stats.

## 5. Compare With Non-Isomorphic Subgraphs

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

## 6. Compare Binding Behavior

- Use a pair of crops whose port names differ but shapes and dtypes are compatible.
- Verify binding rows appear in the bottom `Netron Compare` panel.
- Manually remap inputs and outputs.
- Run compare.
- Verify compare uses the manual mapping successfully.

## 7. Large ONNX Load Behavior

- Open `large-matmul-singlefile.onnx`.
- Verify the model opens through the host-managed path.
- Verify task stage text updates during load.
- Verify UI remains responsive while loading.
- Repeat with `large-matmul-external-data.onnx`.
- Confirm crop export still works for the external-data source.

## 8. Compare Panel Persistence Semantics

- Put a confirmed crop into slot A.
- Close the source model tab.
- Run `Netron: Focus Compare`.
- Verify the slot entry still exists and is visible.
- Run `Netron: Clear Compare`.
- Verify the slots are emptied.

## 9. Busy-State / Anti-Misclick Checks

- Start a long-running action.
- Verify the related buttons are disabled while busy.
- Verify task status shows stage and message.
- Click `Cancel`.
- Verify the UI reflects cancellation request and avoids applying stale follow-up actions.

## 10. Packaging Sanity

- Verify the packaged VSIX installs successfully.
- Verify Linux x64 runtime works in the local environment.
- Verify the package includes only Linux x64 and Windows x64 ONNX Runtime binaries.
