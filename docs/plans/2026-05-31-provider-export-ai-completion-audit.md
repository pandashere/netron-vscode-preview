# Provider, Exporter, and AI Plan Completion Audit

Date: 2026-05-31

This audit maps the generated implementation plan acceptance criteria to the current implementation and verification evidence. It intentionally separates code coverage from real VS Code Extension Host/UI verification.

## Summary

Overall status: implemented with broad automated smoke coverage, but not fully complete because real interactive VS Code verification is still blocked in this WSL environment.

Latest verification evidence recorded before this audit:

- `npm run smoke:cli-tools` passed after adding timeout, manifest-snapshot, analyzer duplicate, and watcher hot-update coverage.
- `npm run smoke:webview-contract` passed after adding UI selection fallback, invalid selection preservation, disabled entry, hover reason, disclosure, and availability reason contract checks.
- `npm run smoke:webview-contract` also covers legacy fallback propagation of provider-unavailable reasons after unsupported host-provider selection.
- `npm run smoke:provider-artifact-export` passed after adding dynamic host-flow coverage that opens an unsupported host-provider file and verifies the legacy `loadModel` message carries `providerUnavailableReason`.
- `npm run smoke:all` passed after those coverage additions.
- `git diff --check` passed.
- `code --version` failed in WSL with `UtilBindVsockAnyPort:309: socket failed 1`, so a real VS Code GUI/Extension Host session has not been verified.

## Acceptance Criteria Audit

| AC | Status | Evidence | Remaining Gap |
| --- | --- | --- | --- |
| AC-1 Format-neutral provider foundation | Mostly complete | `lib/format-providers.js` defines provider diagnostics, capability checks, registry resolution, duplicate id rejection, unsupported and ambiguous provider reasons, and ONNX provider adapter. `extension.js` exposes `registerFormatProvider`, `unregisterFormatProvider`, provider diagnostics, and routes host-managed open through the registry. Unsupported host-provider files still fall back to legacy Netron preview for compatibility, but now carry `providerUnavailableReason` into the webview status so provider selection reports a clear reason. Covered by `smoke:provider-registry`, `smoke:extension-provider-api`, `smoke:private-provider-contract`, `smoke:extension-activation`, `smoke:provider-artifact-export`, `smoke:webview-contract`, and compare smokes. | Real ONNX preview/crop/export/inference/compare still needs interactive VS Code verification. |
| AC-2 Shared crop target provenance | Mostly complete | `lib/text-export-context.js` builds a shared crop target with model metadata, artifact id, graph id/name, createdAt, stale state, and availability. `ONNXWorkbench.getCropTarget()` exposes it. Copy Export Text and AI Analyze both call `getTargetAndContextForPanelArtifact()` through the active provider. UI disables export/analyze for `draftDirty` stale crop. Covered by `smoke:text-export-context`, `smoke:copy-export-host-flow`, and `smoke:ai-task-host-flow`. | Stale UI behavior is not verified in a real webview. |
| AC-3 Exporter registry hot-plug/errors | Complete by automated evidence | `ToolRegistry` scans `~/.netron/vscode-preview/exporters`, validates manifests, creates disabled invalid entries, disables duplicate ids, watches root and child dirs, and broadcasts registry snapshots. UI resolves deleted selections to the first ready entry and preserves invalid selected entries by id/key. Covered by `smoke:cli-tools`, including hot-add, hot-edit invalidation, hot-delete removal, duplicate ids, invalid env/command, and by `smoke:webview-contract` for selection/disclosure contracts. | Real webview rendering still needs interactive verification. |
| AC-4 Text export context schema | Complete by automated evidence | `lib/text-export-context.js` implements v1 crop-only context, deterministic graph id/name mirroring, node ports, referenced tensor metadata only, symbolic string dims, normalized/raw dtype, omitted attributes, and excludes payload/session/internal fields. Covered by `smoke:text-export-context` and `smoke:core-text-export-context`. | Real exporter interoperability should still be checked with customer scripts. |
| AC-5 Copy Export Text host flow | Mostly complete | `handleCopyExportText()` runs selected exporter with context JSON on stdin, copies stdout through `vscode.env.clipboard.writeText()`, records metadata-only Activity, treats empty/non-zero/timeout as errors via `runTool()`, and does not store stdout in Activity. Covered by `smoke:copy-export-host-flow`, `smoke:cli-tools`, and `smoke:manual-ai-tools`; `smoke:cli-tools` now explicitly verifies empty stdout, timeout, capped timeout, and start-time entry snapshots. | Visual running state is covered through mocked message/state flow, not real webview interaction. |
| AC-6 Analyzer registry | Complete by automated evidence | Analyzer registry is another `ToolRegistry` rooted at `~/.netron/vscode-preview/analyzers` with analyzer-specific default timeout. It reuses duplicate/invalid/watch behavior and is rendered in the AI tab. Covered by `smoke:manual-ai-tools`, `smoke:ai-task-host-flow`, `smoke:cli-tools` analyzer duplicate coverage, and `smoke:webview-contract` AI formatter/analyzer selection checks. | Real webview rendering still needs interactive verification. |
| AC-7 AI exporter-to-analyzer pipeline | Complete by automated evidence | `handleRunAiAnalysis()` runs exporter with context JSON, passes exporter stdout as analyzer stdin, updates AI panel state with analyzer stdout, opens/focuses the AI view, captures failure stage, and uses manifest snapshots at task start. Covered by `smoke:ai-task-host-flow`, `smoke:ai-panel-contract`, `smoke:manual-ai-tools`, and `smoke:cli-tools` start-time manifest snapshot checks. | No known implementation gap beyond real UI verification. |
| AC-8 Global task lock | Complete by automated evidence | `state.globalTask` gates Copy Export Text and Analyze, broadcasts busy state, and UI disables both actions when busy. Covered by `smoke:ai-task-host-flow` and `smoke:copy-export-host-flow`. | No known implementation gap. |
| AC-9 AI cancellation | Complete by automated evidence | `cancelGlobalTask()` marks cancellation and kills attached child processes; `attachGlobalTaskProcess()` handles the race where cancellation happens before child attachment. Cancel is available from Model Tools AI tab and AI panel; Copy Export Text has no cancel action. Covered by `smoke:ai-task-host-flow` and `smoke:ai-analysis-state`. | No known implementation gap beyond real UI verification. |
| AC-10 AI Analysis panel singleton display-only | Complete by automated evidence | `createAiViewProvider()` contributes one AI Analysis WebviewView. The panel has status/source/result/actions, no manual input field, renders result with `textContent`, and Copy Result sends raw text to host clipboard. Covered by `smoke:ai-panel-contract` and `smoke:package-manifest`. | No known implementation gap beyond real UI verification. |
| AC-11 Failure/cancel/stale result behavior | Complete by automated evidence | `lib/ai-analysis-state.js` implements running clear, success, failure restore previous result as stale, cancellation restore previous result as stale, and no fabricated stale result on first failure. AI panel renders stale badge/note. Covered by `smoke:ai-analysis-state` and `smoke:ai-panel-contract`. | Real panel rendering still needs interactive verification. |
| AC-12 UI availability reasons/details | Mostly complete | `workbench-ui.js` shows provider capability reasons, no confirmed crop/stale reasons, current exporter/formatter/analyzer reason text, disabled invalid dropdown entries with title, and a disclosure button that shows only current item details. Covered by `smoke:webview-contract`, including disabled invalid options, hover title reason, current item disclosure, and reason priority contracts. | Real webview rendering/hover behavior still needs interactive verification. |

## Cross-Design Notes

- Provider abstraction, exporter, and analyzer now share the same crop target/context path: both standalone export and AI analysis use the active panel provider through `getTargetAndContextForPanelArtifact()`.
- Runnable crop artifact export remains provider-specific through `provider.getExportTarget()` and `provider.exportArtifact()`, separate from text export context generation.
- Compare is no longer purely ONNX-hosted at the extension boundary. The host uses compare slots carrying provider provenance and `runCrossProviderCompare()` can orchestrate two providers through `runCompareArtifact()`.
- Exporters and analyzers are intentionally remote-host local under `os.homedir()` because VS Code Remote SSH runs the extension host on the remote server.

## Current Completion Decision

Do not mark the full implementation goal complete yet.

The implementation is strong enough for continued code review and local smoke testing, but the original plan still contains these open items:

- Add interactive VS Code verification: blocked by current WSL `code` failure and not replaced by a real Extension Host session.
