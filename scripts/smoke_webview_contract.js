#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const extension = fs.readFileSync(path.join(root, 'extension.js'), 'utf8');
const workbenchUi = fs.readFileSync(path.join(root, 'netron', 'source', 'workbench-ui.js'), 'utf8');
const vscodeSource = fs.readFileSync(path.join(root, 'netron', 'source', 'vscode.js'), 'utf8');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function includes(source, text, label) {
    assert(source.includes(text), `Missing ${label}: ${text}`);
}

function main() {
    for (const id of [
        'wb-text-exporter',
        'wb-copy-export-text',
        'wb-exporter-details',
        'wb-ai-formatter',
        'wb-ai-analyzer',
        'wb-run-ai',
        'wb-cancel-ai',
        'wb-ai-status',
        'wb-formatter-details',
        'wb-analyzer-details',
        'wb-analyzer-description',
        'wb-analyzer-inputs'
    ]) {
        includes(workbenchUi, `#${id}`, `UI selector ${id}`);
    }

    for (const tab of ['data-tab="crop"', 'data-tab="run"', 'data-tab="compare"', 'data-tab="ai"', 'data-tab="activity"']) {
        includes(workbenchUi, tab, `Model Tools tab ${tab}`);
    }

    const uiToHost = [
        'selectExporter',
        'copyExportText',
        'selectFormatter',
        'selectAnalyzer',
        'runAiAnalysis',
        'cancelAiTask'
    ];
    for (const type of uiToHost) {
        includes(workbenchUi, `type: '${type}'`, `UI post message ${type}`);
        includes(extension, `case '${type}'`, `extension message handler ${type}`);
    }

    const hostToUi = [
        'renderGraphSnapshot',
        'toolStateUpdate',
        'exportTextCopied',
        'exportTextError',
        'aiAnalysisStatus'
    ];
    for (const type of hostToUi) {
        includes(extension, `type: '${type}'`, `extension outbound message ${type}`);
        includes(workbenchUi, `case '${type}'`, `UI host message handler ${type}`);
    }

    for (const stateField of [
        'selectedExporterId',
        'selectedFormatterId',
        'selectedAnalyzerId',
        'providerInfo',
        'exporterDetailsOpen',
        'formatterDetailsOpen',
        'analyzerDetailsOpen'
    ]) {
        includes(workbenchUi, stateField, `UI state field ${stateField}`);
    }

    for (const helper of [
        '_renderToolControls',
        '_resolveToolSelection',
        '_renderToolSelect',
        '_renderDetails',
        '_renderAnalyzerInputControls',
        '_collectAnalyzerInputs',
        '_renderAiStatus',
        '_providerSupports',
        '_capabilityReason'
    ]) {
        includes(workbenchUi, helper, `UI helper ${helper}`);
    }

    includes(workbenchUi, "entries.find((entry) => entry.status === 'ready') || entries[0] || null", 'deleted tool selection fallback');
    includes(workbenchUi, 'entry.id === selectedId || entry.key === selectedId', 'selected invalid tool remains selected by id/key');
    includes(workbenchUi, "option.disabled = entry.status !== 'ready'", 'invalid tool options disabled');
    includes(workbenchUi, "option.title = entry.reason || ''", 'invalid tool hover reason');
    includes(workbenchUi, 'wb-detail-toggle', 'low-emphasis detail disclosure button');
    includes(workbenchUi, "button.textContent = isOpen && hasDetails ? '▾' : 'ⓘ'", 'current item disclosure icon');
    includes(workbenchUi, "button.title = hasDetails", 'current item disclosure tooltip state');
    includes(workbenchUi, "button.classList.toggle('has-error', hasDetails)", 'current item disclosure error emphasis');
    includes(workbenchUi, "const lines = [entry.reason || 'Unavailable'].concat(Array.isArray(entry.details) ? entry.details : [])", 'current item detail lines');
    includes(workbenchUi, "panel.innerHTML = lines.map((line) => `<div>${escapeHtml(line)}</div>`).join('')", 'current item details only');
    includes(workbenchUi, 'wb-spinner', 'Model Tools running spinner');
    includes(workbenchUi, 'data-input-id', 'analyzer user input fields');
    includes(workbenchUi, 'analyzerInputs: this._collectAnalyzerInputs()', 'UI posts analyzer user inputs');
    includes(workbenchUi, 'isEditableTarget', 'editable target helper');
    includes(workbenchUi, "event.key === 'Backspace' && isEditableTarget(event.target)", 'workbench editable Backspace isolation');
    includes(workbenchUi, '_setButtonRunning', 'running action button renderer');
    includes(workbenchUi, '_renderStatusLine', 'running status line renderer');
    includes(workbenchUi, "globalTask && globalTask.kind === 'copy-export'", 'Copy Export running state');
    includes(workbenchUi, "globalTask && globalTask.kind === 'analysis'", 'AI analysis running state');
    includes(extension, '.spinner { width: 12px;', 'AI panel running spinner CSS');
    includes(extension, "status === 'running'", 'AI panel running status rendering');
    includes(extension, "kind: 'netron-analyzer-input'", 'analyzer user-input envelope');
    includes(extension, 'normalizeAnalyzerUserInputs', 'analyzer user-input host normalization');
    includes(workbenchUi, "textExportReason || targetReason || taskReason || this._toolReason(exporter, 'No exporters found.')", 'exporter availability reason priority');
    includes(workbenchUi, "textExportReason || targetReason || taskReason || this._toolReason(analyzer, 'No analyzers found.')", 'analyzer availability reason priority');
    includes(extension, 'providerUnavailableReason: typeof providerUnavailableReason === \'string\' ? providerUnavailableReason : \'\'', 'legacy load carries provider unavailable reason');
    includes(vscodeSource, 'providerUnavailableReason: typeof payload.providerUnavailableReason === \'string\'', 'legacy model-opened forwards provider unavailable reason');
    includes(workbenchUi, 'event.detail.providerUnavailableReason', 'legacy model-opened consumes provider unavailable reason');
    includes(workbenchUi, 'Legacy model loaded. Model Tools actions are unavailable for this file.${suffix}', 'legacy status includes provider unavailable reason');

    includes(extension, 'provider: providerInfo(provider)', 'provider metadata in renderGraphSnapshot');
    includes(extension, 'model: { ...session.snapshot, sessionId: session.id }', 'host injects sessionId into render model snapshot');
    includes(workbenchUi, 'message.provider || snapshot.provider', 'UI provider metadata consumption');
    includes(workbenchUi, "this._providerSupports('exportArtifact')", 'UI export artifact capability gate');
    includes(workbenchUi, "this._providerSupports('inference')", 'UI inference capability gate');
    includes(workbenchUi, "this._providerSupports('compare')", 'UI compare capability gate');
    includes(workbenchUi, "this._providerSupports('textExportContext')", 'UI text export capability gate');

    console.log('webview contract ok');
}

try {
    main();
} catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
}
