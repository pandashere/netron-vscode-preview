# Provider, Text Export, and AI Analysis Design

## Status

Draft spec for the next implementation plan.

## Goal

Refactor the current ONNX-centered VS Code Netron workbench toward a format-neutral core, then add a scriptable text export layer and an AI analysis workflow built on top of the same confirmed crop target.

The design keeps three concerns separate:

- Format providers turn model formats into a shared graph/artifact abstraction and execute provider-owned inference.
- Text exporters turn a confirmed crop context into customer-defined text.
- AI analyzers consume exporter text and produce a plain-text analysis result.

The implementation should share registry, process-running, provenance, task-state, and diagnostics infrastructure instead of duplicating those mechanics for exporters and analyzers.

## Current Context

The repository currently has:

- `extension.js` as the VS Code extension host entry point.
- `lib/onnx-workbench.js` as the ONNX-specific host workbench for load, crop, export, inference, compare, and tensor preview.
- `netron/source/workbench-ui.js` as the webview-side Model Tools drawer.
- A Compare WebviewView registered through `package.json`.

The current implementation is strongly ONNX-specific. The new design should introduce extension points without breaking the existing ONNX workflow.

## Architecture Overview

### Format Provider Layer

Use a `FormatProvider + CoreGraph + CoreArtifact + CoreWorkbench` shape.

Provider responsibilities:

- Detect and load supported model files.
- Produce normalized `CoreGraph` data.
- Provide format/runtime-specific inference for full model or crop artifact when supported.
- Provide runnable artifact export when supported.
- Surface capability and health diagnostics.

Core responsibilities:

- Select providers.
- Own crop semantics and create structural crop artifacts from `CoreGraph`.
- Convert `CoreGraph` to Netron render snapshots.
- Orchestrate inference and compare.
- Own compare metrics and dense numeric tensor normalization.
- Own shared target/provenance state used by export and AI workflows.

Provider forms:

- In-process Node providers.
- External CLI/process providers.

Provider registration is code-level for the first private/internal version. Public third-party compatibility is not a v1 goal.

### Shared Target State

Introduce a host-side current target/provenance object for confirmed crops.

It should represent:

- Source model metadata.
- Runtime artifact id.
- Deterministic graph id.
- Crop stale state.
- Availability status and unavailable reason.

This object is internal to the extension host. It is not a public provider API.

Suggested conceptual shape:

```ts
interface WorkbenchTarget {
  model: {
    format: string;
    fileName: string;
    filePath: string;
  };
  artifact: {
    id: string;
    createdAt: string;
    stale: boolean;
  };
  graph: {
    id: string;
    name: string;
  };
  availability: {
    canExportText: boolean;
    canAnalyze: boolean;
    reason?: string;
  };
}
```

### Text Export Layer

Text export remains an important standalone feature, but it is also the formatter stage for AI analysis.

Discovery:

- Scan `~/.netron/vscode-preview/exporters/`.
- In Remote SSH, this path is resolved on the remote server where the extension host runs.
- Watch for hot-plug changes.

Execution:

- Exporters are arbitrary CLI commands.
- Use direct process spawn with `shell: false`.
- `cwd` is the exporter directory.
- Inherit `process.env` and merge manifest string `env`.
- stdin is text export context JSON.
- stdout is final plain text.
- Empty stdout is failure.
- Timeout is manifest-configurable with default and maximum bounds.

Manifest fields:

```json
{
  "id": "markdown-summary",
  "label": "Markdown Summary",
  "command": "python3",
  "args": ["export.py"],
  "timeoutMs": 30000,
  "env": {
    "PYTHONPATH": "/path/to/libs"
  }
}
```

Registry behavior:

- Duplicate ids disable all conflicting entries.
- Invalid entries remain visible as disabled/error entries.
- Deleted current selection falls back to the first available entry.
- Current selection that becomes invalid remains selected and shows its error.
- Active tasks use the manifest snapshot resolved at task start.

### Text Export Context

v1 supports only current confirmed crop.

The context is structural text-export input, not a runnable model export format.

```json
{
  "kind": "text-export-context",
  "schemaVersion": 1,
  "target": "crop",
  "model": {
    "format": "onnx",
    "fileName": "model.onnx",
    "filePath": "/path/to/model.onnx"
  },
  "artifact": {
    "id": "artifact-42",
    "createdAt": "2026-05-31T10:20:30.000Z"
  },
  "graph": {
    "id": "model:crop:a13f09c2",
    "name": "model:crop:a13f09c2",
    "inputs": ["input_0"],
    "outputs": ["output_0"],
    "nodes": [
      {
        "id": "conv-0",
        "name": "Conv_0",
        "type": "Conv",
        "domain": "ai.onnx",
        "inputs": [{ "name": "X", "tensor": "input_0" }],
        "outputs": [{ "name": "Y", "tensor": "conv_out" }],
        "attributes": { "kernel_shape": [3, 3] },
        "omittedAttributes": []
      }
    ],
    "tensors": [
      {
        "name": "input_0",
        "dtype": "float32",
        "rawDtype": "FLOAT",
        "shape": [1, "N", 224, 224],
        "kind": "input"
      }
    ]
  }
}
```

Rules:

- `graph.id` is generated by core using a deterministic rule, such as original graph name plus a hash identifier.
- `graph.name` mirrors `graph.id`.
- `graph.nodes` are topologically ordered.
- No standalone `edges` array.
- `graph.inputs` and `graph.outputs` are simple tensor name arrays.
- `graph.tensors` includes only referenced tensors.
- Tensor data, values, payloads, and size summaries are excluded.
- Initializers, constants, and weights are tensor metadata only.
- `tensor.name` is unique in the crop graph and is the reference key.
- Tensor `shape` uses numbers and strings for symbolic dimensions; unknown rank may be `null`.
- Tensor `dtype` is normalized; `rawDtype` preserves source-format dtype.
- Node `id` should be stable where possible.
- Node `name` is required and falls back to `id`.
- Node `domain` is always present, empty string if unknown.
- Node attributes are lightweight JSON-safe values only.
- `omittedAttributes` records intentionally omitted attributes and reasons.
- Exclude `model.sessionId`, `model.id`, `artifact.summary`, `artifact.ioSignature`, and crop selection start/end tensors.

### AI Analysis Layer

AI analysis consumes exporter text and runs a customer-defined analyzer.

Analyzer discovery:

- Scan `~/.netron/vscode-preview/analyzers/`.
- Registry behavior matches exporters: hot-plug refresh, invalid entries visible/disabled, duplicate id conflicts disable all conflicting entries.

Analyzer execution:

- Analyzer manifest fields match exporter manifest fields.
- Analyzer default timeout is longer than exporter default timeout.
- stdin is the plain text produced by the selected exporter.
- stdout is the analysis result plain text.
- Empty stdout is failure.
- stderr is shown in the AI panel only on failure.
- Result rendering is plain text only in v1.
- Stage detail is shown only in detailed errors, not in normal running UI.

AI analysis pipeline:

```text
confirmed crop target
  -> selected exporter
  -> exported text
  -> selected analyzer
  -> analysis result
```

Analyzer scripts own prompt/message construction and may call custom models or one-shot agent CLIs.

### Shared CLI Runner

Exporter and analyzer execution should share one internal runner/registry substrate.

It should handle:

- Manifest loading and validation.
- Directory watching and debounced refresh.
- Duplicate id diagnostics.
- Disabled/error entries.
- Environment merging.
- Direct spawn with `shell: false`.
- stdin writing and stdout/stderr collection.
- Timeout and cancellation.
- Start-time manifest snapshotting.
- Output Channel diagnostics.
- UI availability reasons.

## UI Design

### Model Tools: Crop Tab

Keep crop selection and crop export in the Crop tab.

Existing controls remain:

- Selection Mode.
- Start Tensors.
- End Tensors.
- Confirmed Crop.
- Confirm Crop.
- Export Crop ONNX.
- Save Crop PNG.

Add Text Export controls alongside crop export actions:

```text
Text Export
Exporter: [Markdown Summary]
[Copy Export Text]
Availability: Ready / unavailable reason
```

Behavior:

- Uses the exporter registry.
- Remembers its own selected exporter independently from the AI tab formatter selection.
- Shows real-time availability and unavailable reasons.
- `Copy Export Text` does not open/focus the AI panel.
- `Copy Export Text` has no Cancel button and relies on timeout.
- `Copy Export Text` records success/failure in Activity.
- Do not record exported text content in Activity.

### Model Tools: AI Tab

Add a new AI tab.

```text
AI Analysis

Target
Confirmed Crop: yes/no
Graph: <graph id>
Status: ready/stale/unavailable

Pipeline
Formatter: [Markdown Summary]
Analyzer: [Agent Review]

Actions
[Analyze] [Cancel]

Status
Ready / Running analysis... / Failed / Cancelled / Succeeded
```

Behavior:

- The formatter selection uses the same exporter registry as Crop Text Export, but remembers a separate selected item.
- The analyzer selection uses analyzer registry.
- The AI tab shows status summary only, not result content.
- Analyze opens/focuses the AI Analysis panel at task start.
- Analyze reruns exporter and analyzer every time.
- Analyze and Copy Export Text share one global task lock.
- Running text is `Running analysis...`.
- Cancel appears in both AI tab and AI Analysis panel while analysis is running.
- Activity records analyze success, failure, and cancellation metadata only.

### AI Analysis Panel

Register a global singleton WebviewView similar to Compare.

The panel is display-only and does not initiate analysis except for Cancel and Copy Result.

Layout:

```text
AI Analysis

Status
Running analysis... / Succeeded / Failed / Cancelled

Source
Model File: model.onnx
Model Path: /path/to/model.onnx
Artifact ID: artifact-42
Graph ID: model:crop:a13f09c2
Exporter ID: markdown-summary
Analyzer ID: agent-review
Time: 2026-05-31 14:20:30

Result [Stale if applicable]
<plain text result>

[Copy Result] [Cancel if running]
```

Behavior:

- Source metadata uses readable labels with internal ids/paths as values.
- Source and result are expanded by default with no extra folding.
- Result rendering is plain text.
- No input preview is shown.
- No Copy Input Text action.
- Copy Result remains enabled for stale previous successful results.
- Running clears the result area and disables Copy Result.
- If a later task fails or is cancelled, restore the previous successful result and mark it stale in both top status and result badge.
- Failure shows error and stderr. Stage detail appears only in detailed errors.

### Availability and Error Details

Controls should show whether functionality is available and why not.

Examples:

- No confirmed crop.
- Current crop is stale. Confirm crop again.
- No exporters found.
- Selected exporter is invalid.
- Exporter naming/configuration conflict.
- No analyzers found.
- Selected analyzer is invalid.
- Another export/analysis task is running.

Dropdowns:

- Normal entries show labels, not ids.
- Invalid/error entries are visible but disabled/greyed out.
- Use a small disclosure control near dropdowns to show current selected/active item error details.
- Details show only the current item error, not all registry errors.
- Full diagnostics go to Output Channel.

## Task Model

There is one global export/analysis task lock.

- `Copy Export Text` and `Analyze` cannot run concurrently.
- `Copy Export Text` status is `Copying export text...`.
- `Analyze` status is `Running analysis...`.
- Analyze supports Cancel.
- Copy Export Text has no Cancel and relies on timeout.
- Active tasks use start-time manifest snapshots.

AI result state:

- Success replaces current result.
- Running clears result area.
- Failure/cancellation restores previous successful result if any and marks it stale.
- No history is persisted.

## Out of Scope for v1

- Public third-party provider API stability.
- Workspace-level exporter/analyzer directories.
- Markdown or HTML rendering for analyzer output.
- Streaming analyzer output.
- Full graph/inference/compare text export targets.
- Analyzer JSON envelope input.
- Persisted AI analysis history.
- Runnable model export through text export schema.
- Manual paste/edit input for AI analysis.

## Implementation Notes

- Keep provider abstraction, text export, and AI analysis logically separate.
- Share implementation substrate for registry and CLI process execution.
- Avoid exposing webview clipboard directly; use extension host clipboard APIs.
- Avoid storing sensitive exporter/analyzer stdout in Activity.
- Keep UI dense and operational, consistent with existing Model Tools and Compare panel style.
