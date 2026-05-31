# Provider, Text Export, and AI Analysis Implementation Plan

## Goal Description

Implement a format-neutral workbench foundation, then add scriptable text export and AI analysis workflows that operate on confirmed crop targets.

The work should preserve the current ONNX workflow while introducing internal abstractions that allow private formats, text exporters, and analyzers to be added without duplicating registry, task, process, diagnostics, and UI state handling.

## Acceptance Criteria

- AC-1: The extension exposes an internal format-neutral provider/workbench foundation without breaking the current ONNX preview, crop, export, inference, and compare flows.
  - Positive Tests (expected to PASS):
    - Open an ONNX model and confirm that graph rendering still works.
    - Confirm a crop and verify the existing ONNX crop artifact path still works.
    - Run existing compare smoke tests after the refactor.
  - Negative Tests (expected to FAIL):
    - Attempt to open an unsupported file and verify the provider selection reports a clear unavailable reason.
    - Register two providers for the same format id and verify the extension rejects or reports ambiguous provider resolution.

- AC-2: Confirmed crop targets have a shared host-side provenance object used consistently by crop export, text export, and AI analysis.
  - Positive Tests (expected to PASS):
    - Confirm a crop and verify the target contains model file metadata, artifact id, graph id, createdAt, and stale state.
    - Modify crop selection after confirming and verify export/analyze actions become unavailable with a stale reason.
  - Negative Tests (expected to FAIL):
    - Request text export without a confirmed crop and verify no CLI process starts.
    - Request analysis for a stale crop and verify the UI disables the action with a reason.

- AC-3: Text exporter registry scans `~/.netron/vscode-preview/exporters/`, validates manifests, handles hot updates, and exposes ready/error entries to the UI.
  - Positive Tests (expected to PASS):
    - Add a valid exporter manifest and verify it appears in the Crop tab.
    - Delete the selected exporter and verify selection falls back to the first available exporter.
    - Make the selected exporter invalid and verify it remains selected with an error reason.
  - Negative Tests (expected to FAIL):
    - Add duplicate exporter ids and verify all conflicting exporters are disabled.
    - Add malformed `exporter.json` and verify it appears as an invalid disabled entry.

- AC-4: Text export context generation produces the v1 crop-only schema without tensor payloads.
  - Positive Tests (expected to PASS):
    - Generate context for a confirmed crop and verify top-level `kind`, `schemaVersion`, `target`, `model`, `artifact`, and `graph`.
    - Verify `graph.nodes` are topologically ordered.
    - Verify `graph.tensors` only includes referenced tensor metadata.
    - Verify dynamic shape dimensions may be strings.
  - Negative Tests (expected to FAIL):
    - Verify tensor values, raw data, byte payloads, and size summaries are absent.
    - Verify `artifact.ioSignature`, crop selection start/end tensors, `model.sessionId`, and `model.id` are absent.

- AC-5: Copy Export Text runs the selected exporter CLI through shared process infrastructure and copies stdout to the VS Code clipboard.
  - Positive Tests (expected to PASS):
    - Use a simple exporter that echoes text and verify clipboard content matches stdout.
    - Verify the Crop tab shows `Copying export text...` while running.
    - Verify success/failure metadata is recorded in Activity without storing stdout content.
  - Negative Tests (expected to FAIL):
    - Exporter exits non-zero and verify clipboard is not updated and an error is shown.
    - Exporter stdout is empty and verify the action fails.
    - Exporter exceeds timeout and verify the action fails.

- AC-6: Analyzer registry scans `~/.netron/vscode-preview/analyzers/` and reuses exporter manifest and registry behavior with analyzer-specific defaults.
  - Positive Tests (expected to PASS):
    - Add a valid analyzer manifest and verify it appears in the AI tab.
    - Hot-edit analyzer manifest and verify the AI tab reflects the change.
    - Verify invalid analyzer entries are visible but disabled.
  - Negative Tests (expected to FAIL):
    - Duplicate analyzer ids disable all conflicting analyzers.
    - Missing analyzer command disables analysis and shows an unavailable reason.

- AC-7: AI analysis runs the selected exporter then selected analyzer as one global task pipeline.
  - Positive Tests (expected to PASS):
    - Run Analyze and verify the exporter receives context JSON on stdin.
    - Verify analyzer receives exporter stdout as plain-text stdin.
    - Verify analyzer stdout appears in the AI Analysis panel as plain text.
    - Verify the AI panel opens/focuses when Analyze starts.
  - Negative Tests (expected to FAIL):
    - Analyzer exits non-zero and verify the panel shows failure and stderr.
    - Analyzer stdout is empty and verify the action fails.
    - Registry changes during a running task do not affect that running task.

- AC-8: Global task lock prevents concurrent export/analyze CLI work.
  - Positive Tests (expected to PASS):
    - While Analyze is running, verify Copy Export Text and Analyze are disabled elsewhere.
    - While Copy Export Text is running, verify Analyze is disabled.
  - Negative Tests (expected to FAIL):
    - Attempt to start a second Analyze task and verify no second process starts.
    - Attempt to start Copy Export Text while Analyze is running and verify no exporter process starts.

- AC-9: AI task cancellation is available from both Model Tools AI tab and AI Analysis panel.
  - Positive Tests (expected to PASS):
    - Start a long-running analyzer and cancel from the AI tab.
    - Start a long-running analyzer and cancel from the AI panel.
    - Verify the child process is terminated and status becomes cancelled.
  - Negative Tests (expected to FAIL):
    - Cancel when no task is running and verify no state corruption occurs.
    - Cancel Copy Export Text and verify no Cancel control is available for that task type.

- AC-10: AI Analysis panel is a global singleton display-only WebviewView.
  - Positive Tests (expected to PASS):
    - Verify the panel displays status, source metadata, result, and actions.
    - Verify results render as plain text.
    - Verify Copy Result copies the latest available result.
  - Negative Tests (expected to FAIL):
    - Verify no manual input editor/paste field exists.
    - Verify no markdown or HTML rendering is applied to analyzer stdout.

- AC-11: Failure, cancellation, running, and stale result states follow the agreed UI behavior.
  - Positive Tests (expected to PASS):
    - Running clears the result area and disables Copy Result.
    - Failure restores the previous successful result and marks it stale when one exists.
    - Cancellation restores the previous successful result and marks it stale when one exists.
    - Stale markers appear in both top status and result title/badge.
  - Negative Tests (expected to FAIL):
    - Verify a stale previous result is not shown as the result of a failed/cancelled task.
    - Verify Copy Result is disabled during running.

- AC-12: UI availability reasons are visible in Crop Text Export and AI tab controls.
  - Positive Tests (expected to PASS):
    - No confirmed crop shows a clear unavailable reason.
    - Stale crop shows a clear unavailable reason.
    - Invalid selected exporter/analyzer shows a clear unavailable reason.
    - A disclosure control shows current item error details.
  - Negative Tests (expected to FAIL):
    - Verify a disabled action never lacks a reason.
    - Verify the disclosure area does not show all registry errors, only the current item error.

## Path Boundaries

### Upper Bound (Maximum Scope)

The maximum acceptable implementation includes:

- Internal provider/workbench abstractions sufficient to support ONNX as the first provider and future private providers.
- Shared host-side target/provenance object.
- Shared CLI registry/runner for exporters and analyzers.
- Exporter registry and Copy Export Text UI.
- Analyzer registry and full AI analysis task pipeline.
- Global AI Analysis WebviewView.
- Updated Model Tools Crop and AI tabs.
- Activity and Output Channel diagnostics.
- Focused unit/smoke tests for schema, registry, runner, state transitions, and existing ONNX workflow.

### Lower Bound (Minimum Scope)

The minimum viable implementation includes:

- Keep ONNX behavior working.
- Add text export context builder for confirmed ONNX crop artifacts.
- Add exporter registry and Copy Export Text.
- Add analyzer registry and Analyze pipeline.
- Add AI tab and AI Analysis panel with global single-task state.
- Implement availability reasons and invalid registry entry handling.

The lower bound may defer broad provider refactoring if needed, but it must avoid hard-coding new export/analyze logic in a way that blocks the planned provider abstraction.

### Allowed Choices

- Can use Node.js extension host modules including `fs`, `path`, `os`, and `child_process`.
- Can use VS Code WebviewView for the AI Analysis panel.
- Can use existing Model Tools drawer patterns and existing Compare panel conventions.
- Can add internal modules under `lib/`.
- Can add focused smoke scripts under `scripts/` if useful.
- Cannot require users to install a separate VS Code extension for exporters/analyzers.
- Cannot rely on webview clipboard writes for primary copy behavior.
- Cannot store exporter/analyzer stdout in Activity.
- Cannot make text export schema a runnable model export format.

## Dependencies and Sequence

### Milestones

1. Milestone 1: Extract shared target and context foundations.
   - Phase A: Introduce internal target/provenance object for confirmed crops.
   - Phase B: Implement deterministic crop graph id generation.
   - Phase C: Build text export context from existing ONNX crop snapshots.
   - Phase D: Add tests for context schema and stale target availability.

2. Milestone 2: Add shared registry and CLI runner substrate.
   - Phase A: Implement manifest loader and validator.
   - Phase B: Implement directory scanner/watcher for global exporter/analyzer roots.
   - Phase C: Implement duplicate-id diagnostics and disabled/error entries.
   - Phase D: Implement shared process runner with stdin, stdout, stderr, timeout, cancellation, environment merging, and start-time manifest snapshots.

3. Milestone 3: Implement Text Export UI and host flow.
   - Phase A: Add Crop tab Text Export controls.
   - Phase B: Wire exporter registry state to model webviews.
   - Phase C: Implement Copy Export Text host command.
   - Phase D: Add Activity logging and clipboard status.

4. Milestone 4: Implement AI analysis state and panel.
   - Phase A: Add package contributions for AI Analysis WebviewView and focus command if needed.
   - Phase B: Implement global AI task state.
   - Phase C: Build AI panel HTML/message handling.
   - Phase D: Implement Copy Result and Cancel actions.

5. Milestone 5: Implement Model Tools AI tab.
   - Phase A: Add AI tab UI.
   - Phase B: Wire formatter/exporter and analyzer selection state.
   - Phase C: Implement Analyze command from model panel.
   - Phase D: Implement status summary and availability reasons.

6. Milestone 6: Integrate task locking, stale/failure behavior, and diagnostics.
   - Phase A: Share global lock between Copy Export Text and Analyze.
   - Phase B: Implement running/failure/cancel/stale result transitions.
   - Phase C: Add disclosure details for current invalid dropdown item.
   - Phase D: Verify Output Channel diagnostics and Activity metadata.

7. Milestone 7: Provider abstraction follow-through.
   - Phase A: Move ONNX-specific behavior behind the provider/workbench boundary in small steps.
   - Phase B: Ensure text export context is built from provider-neutral graph/artifact data where available.
   - Phase C: Keep ONNX compatibility and compare smoke tests passing after each step.

## Feasibility Hints

- Implement the shared CLI registry/runner once and parameterize root directory, default timeout, and labels for exporter/analyzer.
- Keep text export and analysis task state in the extension host. Webviews should render state and send user actions.
- Use existing `enqueuePanelMessage` and compare panel state broadcasting patterns as references.
- Build v1 text export context from existing ONNX crop snapshot first, then migrate source data toward `CoreGraph` as provider abstraction lands.
- Treat runner output as sensitive. Store only status and metadata in Activity.
- Avoid custom dropdowns in v1. Use native selects, disabled/error entries, and a separate disclosure details control.
- Prefer focused tests around pure functions: manifest validation, context building, graph id generation, state transitions, and process runner behavior.

## Implementation Notes

- Code should not contain acceptance-criteria labels or milestone terminology.
- Keep UI wording concise and operational.
- Use `Exporter` in Crop tab and `Formatter` in AI tab where appropriate, while metadata may expose exporter/analyzer ids.
- Do not add a separate Export tab in v1.
- Do not add persisted AI result history in v1.
