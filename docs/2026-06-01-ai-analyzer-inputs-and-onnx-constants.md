# 2026-06-01 Update: Analyzer Inputs And ONNX Constants

This document records the files changed in this update and the meaningful diff behind each group of changes.

## Summary

This update adds analyzer prompt inputs, analyzer descriptions, a real DeepSeek prompt-template analyzer example, and an ONNX Constant crop fix.

Analyzer behavior:

- Existing analyzers stay compatible. If `userInputs` is absent or empty, stdin is still plain exported text.
- Analyzers can declare `description` and up to 3 `userInputs` in `analyzer.json`.
- When `userInputs` is present, the extension sends a JSON stdin envelope with `kind: "netron-analyzer-input"`, `exportedText`, and `userInputs`.
- The AI panel renders the analyzer description and input boxes, then runs the normal export -> analyze pipeline.
- Editable fields in the workbench stop `Backspace` propagation so Netron global menu shortcuts do not steal text deletion.

ONNX behavior:

- ONNX `Constant` node outputs are normalized as initializer tensors during graph analysis.
- Crop no longer promotes those constants into graph inputs with `unknown` type.
- A smoke test exports a crop containing a Constant and verifies ONNX Runtime can execute it.

## Changed Files

```text
M  docs/private-ir-integration-guide.md
M  docs/showcase.html
M  examples/README.md
M  examples/tools/analyzers/codex-one-shot-analysis/analyzer.json
M  examples/tools/analyzers/codex-one-shot-analysis/codex-one-shot-analysis.js
A  examples/tools/analyzers/deepseek-prompt-template-analysis/analyzer.json
A  examples/tools/analyzers/deepseek-prompt-template-analysis/deepseek-prompt-template-analysis.js
M  extension.js
M  lib/cli-tools.js
M  lib/onnx-workbench.js
M  netron/source/workbench-ui.js
M  package.json
M  scripts/smoke_ai_task_host_flow.js
M  scripts/smoke_all.js
M  scripts/smoke_cli_tools.js
M  scripts/smoke_webview_contract.js
A  scripts/smoke_onnx_constant_crop.js
```

## Diff By Area

### Analyzer Manifest Schema

Files:

- `lib/cli-tools.js`
- `scripts/smoke_cli_tools.js`

Diff:

- Exposed `description` and `userInputs` in public tool entries.
- Added manifest validation for `userInputs`.
- Enforced at most 3 user input fields.
- Validated each input's `id`, `label`, `placeholder`, `description`, `required`, and `multiline`.
- Disabled duplicate input ids as manifest errors.
- Added smoke coverage for ready analyzer metadata and invalid over-limit manifests.

### AI Panel UI

Files:

- `netron/source/workbench-ui.js`
- `scripts/smoke_webview_contract.js`

Diff:

- Added analyzer description and input containers to the AI tab.
- Rendered analyzer-declared user inputs as `textarea` or text `input`.
- Preserved typed values across UI refreshes.
- Sent `analyzerInputs` with the `runAiAnalysis` message.
- Added dark-mode styling for text inputs.
- Stopped `Backspace` propagation from editable workbench controls to avoid Netron global shortcut interception.
- Added webview contract checks for new selectors, helpers, message payload, and Backspace isolation.

### Extension Host Analyzer Pipeline

Files:

- `extension.js`
- `scripts/smoke_ai_task_host_flow.js`

Diff:

- Added host-side normalization for analyzer user inputs.
- Required fields are checked before launching the analyzer process.
- Analyzers without `userInputs` still receive plain text stdin.
- Analyzers with `userInputs` receive this JSON envelope:

```json
{
  "kind": "netron-analyzer-input",
  "schemaVersion": 1,
  "exportedText": "...",
  "userInputs": {
    "focus": "..."
  }
}
```

- Smoke coverage now verifies that the extension sends the JSON envelope and includes user input values.

### Analyzer Examples

Files:

- `examples/tools/analyzers/codex-one-shot-analysis/analyzer.json`
- `examples/tools/analyzers/codex-one-shot-analysis/codex-one-shot-analysis.js`
- `examples/tools/analyzers/deepseek-prompt-template-analysis/analyzer.json`
- `examples/tools/analyzers/deepseek-prompt-template-analysis/deepseek-prompt-template-analysis.js`
- `examples/README.md`

Diff:

- `codex-one-shot-analysis` now declares analyzer description and optional prompt fields.
- The Codex script supports both old plain-text stdin and the new JSON envelope.
- Added `deepseek-prompt-template-analysis`, a minimal real-model analyzer using DeepSeek chat completions.
- The DeepSeek prompt-template analyzer reads `input1` and `input2`, builds a Chinese graph-analysis prompt, and calls `https://api.deepseek.com`.
- Example documentation now lists line-count, DeepSeek graph, DeepSeek prompt-template, and Codex analyzers.

### ONNX Constant Crop Fix

Files:

- `lib/onnx-workbench.js`
- `scripts/smoke_onnx_constant_crop.js`
- `scripts/smoke_all.js`
- `package.json`

Diff:

- Added helpers to normalize Constant node tensors and scalar/vector Constant attributes.
- Recognized ONNX Constant outputs as initializers during graph analysis.
- Supported Constant attributes such as `value`, `value_float`, `value_floats`, `value_int`, `value_ints`, `value_string`, `value_strings`, and `sparse_value`.
- Added `smoke:onnx-constant-crop`.
- Included the Constant crop smoke in `smoke:all`.

### Documentation And Showcase

Files:

- `docs/private-ir-integration-guide.md`
- `docs/showcase.html`
- `docs/2026-06-01-ai-analyzer-inputs-and-onnx-constants.md`

Diff:

- Documented analyzer `description` and `userInputs`.
- Documented old stdin compatibility and the new JSON envelope.
- Added a copyable DeepSeek prompt-template manifest and prompt-building snippet.
- Added install commands for the new analyzer example.
- Added troubleshooting notes for analyzer prompt inputs, editable Backspace behavior, and ONNX Constant tensors.
- Updated the showcase AI section to describe formatter/exporter -> analyzer prompt pipelines with user inputs.

## Verification

Commands used for this update:

```bash
node --check lib/cli-tools.js
node --check extension.js
node --check netron/source/workbench-ui.js
node --check examples/tools/analyzers/codex-one-shot-analysis/codex-one-shot-analysis.js
node --check examples/tools/analyzers/deepseek-prompt-template-analysis/deepseek-prompt-template-analysis.js
npm run smoke:cli-tools
npm run smoke:webview-contract
npm run smoke:ai-task-host-flow
npm run smoke:private-ir-guide-examples
npm run smoke:all
git diff --check
```

The DeepSeek prompt-template analyzer was also manually tested with a tiny graph text and returned a model-generated analysis.
