const fs = require('fs');
const os = require('os');
const path = require('path');
const vscode = require('vscode');
const { ONNXWorkbench, isOnnxFileName } = require('./lib/onnx-workbench');

const WEBVIEW_READY_TIMEOUT_MS = 10000;
const COMPARE_CENTER_HTML_VERSION = 3;
const COMPARE_VIEW_CONTAINER_ID = 'netronComparePanel';
const COMPARE_VIEW_ID = 'netronCompare.compareView';

const state = {
    context: null,
    output: null,
    panelSeq: 0,
    compareView: null,
    compareViewReady: false,
    pendingCompareState: null,
    panels: new Map(),
    workbench: null
};

function createRequestId() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatDetail(detail) {
    if (detail === undefined || detail === null) {
        return '';
    }
    if (typeof detail === 'string') {
        return detail;
    }
    try {
        return JSON.stringify(detail);
    } catch {
        return String(detail);
    }
}

function appendLog(kind, message, detail) {
    if (!state.output) {
        return;
    }
    const suffix = detail !== undefined ? ` detail=${formatDetail(detail)}` : '';
    state.output.appendLine(`[NetronWorkbench] [${kind}] ${message}${suffix}`);
}

function createPanelState(panel) {
    state.panelSeq += 1;
    const panelId = `panel-${state.panelSeq}`;
    const entry = {
        id: panelId,
        panel,
        ready: false,
        readyTimer: null,
        pendingMessages: [],
        currentModelUri: null,
        currentSessionId: null,
        currentArtifactId: null,
        activity: [],
        currentTask: null,
        cancelRequested: false
    };
    state.panels.set(panelId, entry);
    return entry;
}

function enqueuePanelMessage(panelState, message) {
    if (!panelState || !panelState.panel) {
        return;
    }
    if (!panelState.ready) {
        panelState.pendingMessages.push(message);
        return;
    }
    panelState.panel.webview.postMessage(message).catch((error) => {
        appendLog('warn', 'postMessage failed', { panelId: panelState.id, message: error.message });
    });
}

async function flushPanelMessages(panelState) {
    if (!panelState || !panelState.ready || !panelState.panel) {
        return;
    }
    const queue = panelState.pendingMessages.splice(0);
    for (const message of queue) {
        await panelState.panel.webview.postMessage(message);
    }
}

function addPanelActivity(panelState, level, message, detail) {
    if (!panelState) {
        return;
    }
    const item = {
        id: createRequestId(),
        level,
        message,
        detail,
        createdAt: new Date().toISOString()
    };
    panelState.activity.unshift(item);
    panelState.activity = panelState.activity.slice(0, 60);
    enqueuePanelMessage(panelState, {
        type: 'activityLog',
        entries: panelState.activity
    });
}

function updatePanelTask(panelState, patch) {
    if (!panelState) {
        return;
    }
    panelState.currentTask = {
        ...(panelState.currentTask || {
            status: 'idle',
            stage: '',
            message: '',
            startedAt: null,
            updatedAt: null,
            cancellable: false,
            busy: false
        }),
        ...patch,
        updatedAt: new Date().toISOString()
    };
    enqueuePanelMessage(panelState, {
        type: 'taskState',
        task: panelState.currentTask
    });
}

function clearPanelTask(panelState) {
    updatePanelTask(panelState, {
        status: 'idle',
        stage: '',
        message: '',
        startedAt: null,
        cancellable: false,
        busy: false
    });
    panelState.cancelRequested = false;
}

function disposePanelState(panelState) {
    if (!panelState) {
        return;
    }
    if (panelState.readyTimer) {
        clearTimeout(panelState.readyTimer);
    }
    state.panels.delete(panelState.id);
}

function createCompareViewProvider() {
    return {
        resolveWebviewView(webviewView) {
            const disposables = [];
            state.compareView = webviewView;
            state.compareViewReady = false;
            webviewView.title = 'Compare';
            webviewView.description = 'Shared A/B compare state';
            webviewView.webview.options = {
                enableScripts: true
            };
            webviewView.webview.html = buildCompareCenterHtml(webviewView.webview);
            webviewView.__compareHtmlVersion = COMPARE_CENTER_HTML_VERSION;
            webviewView.onDidDispose(() => {
                if (state.compareView === webviewView) {
                    state.compareView = null;
                    state.compareViewReady = false;
                }
                while (disposables.length > 0) {
                    const disposable = disposables.pop();
                    if (disposable) {
                        disposable.dispose();
                    }
                }
            }, null, disposables);
            webviewView.onDidChangeVisibility(() => {
                if (state.compareView === webviewView && webviewView.visible) {
                    flushCompareState();
                }
            }, null, disposables);
            webviewView.webview.onDidReceiveMessage((message) => {
                handleCompareCenterMessage(message).catch((error) => {
                    appendLog('error', 'compare view message failed', { type: message && message.type, message: error.message });
                    vscode.window.showErrorMessage(error.message);
                });
            }, null, disposables);
            flushCompareState();
        }
    };
}

function canPostCompareState() {
    return !!(state.compareView && state.compareViewReady && state.compareView.visible);
}

function flushCompareState() {
    if (!canPostCompareState() || !state.pendingCompareState) {
        return;
    }
    const compareState = state.pendingCompareState;
    state.compareView.webview.postMessage({ type: 'compareStateUpdate', state: compareState }).then((posted) => {
        if (posted !== false && state.pendingCompareState === compareState) {
            state.pendingCompareState = null;
        }
    }).catch(() => {});
}

async function focusCompareView(preserveFocus = false) {
    if (state.compareView && state.compareView.__compareHtmlVersion !== COMPARE_CENTER_HTML_VERSION) {
        state.compareView.webview.html = buildCompareCenterHtml(state.compareView.webview);
        state.compareView.__compareHtmlVersion = COMPARE_CENTER_HTML_VERSION;
        state.compareViewReady = false;
    }
    if (!state.compareView) {
        await vscode.commands.executeCommand(`workbench.view.extension.${COMPARE_VIEW_CONTAINER_ID}`);
    }
    if (state.compareView) {
        state.compareView.show(preserveFocus);
    }
    pushCompareState();
}

async function activate(context) {
    state.context = context;
    state.output = vscode.window.createOutputChannel('Netron Preview');
    context.subscriptions.push(state.output);
    state.workbench = new ONNXWorkbench(context, (level, message, detail) => appendLog(level, message, detail));
    state.workbench.onChange(() => {
        broadcastCompareState();
    });
    context.subscriptions.push(vscode.window.registerWebviewViewProvider(COMPARE_VIEW_ID, createCompareViewProvider(), {
        webviewOptions: {
            retainContextWhenHidden: true
        }
    }));

    context.subscriptions.push(vscode.commands.registerCommand('netronPreview.openPreview', async (resource) => {
        try {
            const uri = await resolveModelUri(resource);
            if (!uri) {
                return;
            }
            const panelState = createModelPanel(context);
            await openModelInPanel(panelState, uri, resource instanceof vscode.Uri ? 'explorer-context' : 'command-palette');
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            appendLog('error', 'open preview failed', { message });
            vscode.window.showErrorMessage(`Netron preview failed: ${message}`);
        }
    }));

    context.subscriptions.push(vscode.commands.registerCommand('netronPreview.openCompareCenter', async () => {
        await focusCompareView(false);
    }));

    context.subscriptions.push(vscode.commands.registerCommand('netronPreview.clearCompareCenter', async () => {
        state.workbench.clearCompare();
        vscode.window.showInformationMessage('Netron Compare cleared.');
    }));
}

async function resolveModelUri(resource) {
    if (resource instanceof vscode.Uri && resource.scheme === 'file') {
        return resource;
    }
    const picked = await vscode.window.showOpenDialog({
        canSelectMany: false,
        canSelectFiles: true,
        canSelectFolders: false,
        defaultUri: getDefaultFolder(),
        openLabel: 'Open Model',
        title: 'Select model file to preview'
    });
    return Array.isArray(picked) && picked.length > 0 ? picked[0] : null;
}

function createModelPanel(context) {
    const sourceRoot = vscode.Uri.joinPath(context.extensionUri, 'netron', 'source');
    const panel = vscode.window.createWebviewPanel(
        'netronPreview',
        'Netron Preview',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [sourceRoot]
        }
    );
    const panelState = createPanelState(panel);
    panel.webview.html = buildNetronHtml(context, panel.webview, sourceRoot, panelState.id);
    panelState.readyTimer = setTimeout(() => {
        if (!panelState.ready) {
            vscode.window.showWarningMessage('Netron Webview 初始化超时，请尝试 Developer: Reload Window。');
        }
    }, WEBVIEW_READY_TIMEOUT_MS);
    panel.onDidDispose(() => disposePanelState(panelState), null, context.subscriptions);
    panel.webview.onDidReceiveMessage((message) => {
        handleModelPanelMessage(panelState, message).catch((error) => {
            appendLog('error', 'model panel message failed', { panelId: panelState.id, type: message && message.type, message: error.message });
            addPanelActivity(panelState, 'error', error.message, { type: message && message.type });
            clearPanelTask(panelState);
            enqueuePanelMessage(panelState, { type: 'notify', level: 'error', message: error.message });
        });
    }, null, context.subscriptions);
    return panelState;
}

function buildNetronHtml(context, webview, sourceRoot, panelId) {
    const indexFile = path.join(context.extensionPath, 'netron', 'source', 'index.html');
    let html = fs.readFileSync(indexFile, 'utf8');
    const csp = webview.cspSource;
    const baseHref = `${webview.asWebviewUri(sourceRoot).toString()}/`;
    const vscodeScriptUri = webview.asWebviewUri(vscode.Uri.joinPath(sourceRoot, 'vscode.js')).toString();
    const cspMeta = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${csp} data: blob:; style-src ${csp} 'unsafe-inline'; script-src ${csp}; worker-src ${csp} blob:; font-src ${csp}; connect-src ${csp};">`;
    html = html.replace(/<meta http-equiv="Content-Security-Policy"[^>]*>/i, cspMeta);
    html = html.replace('<head>', `<head>\n<base href="${baseHref}">\n<meta name="netron-panel-id" content="${panelId}">`);
    html = html.replace(/<script[^>]*src=["'](?:\.\/)?index\.js["'][^>]*><\/script>/i, `<script id="netron-vscode-entry" type="module" src="${vscodeScriptUri}"></script>`);
    return html;
}

function buildCompareCenterHtml(webview) {
    const nonce = createRequestId().replace(/[^a-z0-9]/gi, '');
    const csp = webview.cspSource;
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${csp} data:; style-src ${csp} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Netron Compare</title>
<style>
html, body { overflow-x: hidden; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 12px 12px 16px 12px; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); }
.app { width: 100%; max-width: 1280px; margin: 0 auto; }
.section { border: 1px solid var(--vscode-panel-border); border-radius: 12px; padding: 14px; margin-bottom: 12px; background: color-mix(in srgb, var(--vscode-editor-background) 90%, white); }
.section-collapse { padding: 0; overflow: hidden; }
.section-collapse > summary { list-style: none; cursor: pointer; padding: 14px; }
.section-collapse > summary::-webkit-details-marker { display: none; }
.section-collapse > .collapse-body { padding: 0 14px 14px 14px; }
.sticky { position: sticky; top: 0; z-index: 2; backdrop-filter: blur(6px); }
.grid { display: grid; gap: 12px; }
.grid-two { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.title-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.title { font-size: 16px; font-weight: 600; }
.subtitle { font-size: 12px; opacity: 0.75; }
.controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.card { min-width: 0; border: 1px solid var(--vscode-panel-border); border-radius: 10px; padding: 12px; background: color-mix(in srgb, var(--vscode-editor-background) 95%, white); }
.slot-entry { display: grid; grid-template-columns: minmax(240px, 320px) minmax(0, 1fr); gap: 14px; align-items: start; }
.slot-preview { min-width: 0; }
.thumb { width: 100%; aspect-ratio: 18 / 10; height: auto; object-fit: contain; object-position: center; display: block; border-radius: 8px; border: 1px solid var(--vscode-panel-border); background: #111; }
.slot-meta { min-width: 0; display: grid; gap: 6px; }
.slot-title { font-size: 14px; font-weight: 600; word-break: break-word; }
.slot-id { word-break: break-all; }
.label { font-size: 12px; opacity: 0.8; }
.muted { opacity: 0.72; }
.mono, code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--vscode-panel-border); background: color-mix(in srgb, var(--vscode-editor-background) 92%, white); }
.chip.ok { border-color: color-mix(in srgb, var(--vscode-testing-iconPassed) 40%, var(--vscode-panel-border)); }
.chip.warn { border-color: color-mix(in srgb, var(--vscode-testing-iconQueued) 45%, var(--vscode-panel-border)); }
.chip.bad { border-color: color-mix(in srgb, var(--vscode-testing-iconFailed) 45%, var(--vscode-panel-border)); }
.status { font-size: 12px; margin-top: 8px; opacity: 0.9; }
.status.running { color: var(--vscode-textLink-foreground); }
.status.failed { color: var(--vscode-testing-iconFailed); }
button, select, textarea { font: inherit; }
button, select { min-height: 32px; }
button { border: 1px solid var(--vscode-button-border, transparent); background: var(--vscode-button-background); color: var(--vscode-button-foreground); border-radius: 6px; padding: 6px 12px; cursor: pointer; }
button.secondary { background: transparent; color: inherit; border-color: var(--vscode-panel-border); }
button[disabled] { opacity: 0.5; cursor: not-allowed; }
textarea { width: 100%; min-height: 84px; border-radius: 6px; border: 1px solid var(--vscode-panel-border); background: var(--vscode-input-background); color: var(--vscode-input-foreground); padding: 8px; box-sizing: border-box; resize: vertical; }
select { max-width: 100%; border-radius: 6px; border: 1px solid var(--vscode-dropdown-border, var(--vscode-panel-border)); background: var(--vscode-dropdown-background, var(--vscode-input-background)); color: var(--vscode-dropdown-foreground, var(--vscode-input-foreground)); padding: 4px 8px; }
.binding-list { display: grid; gap: 10px; }
.binding-card { display: grid; gap: 10px; border: 1px solid var(--vscode-panel-border); border-radius: 10px; padding: 12px; background: color-mix(in srgb, var(--vscode-editor-background) 95%, white); }
.binding-top { display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr) minmax(220px, 1.2fr) auto; align-items: start; }
.binding-port { min-width: 0; }
.binding-name { font-weight: 600; word-break: break-word; }
.binding-meta { margin-top: 4px; font-size: 12px; opacity: 0.76; word-break: break-word; }
.binding-select { min-width: 0; }
.binding-select select { width: 100%; }
.bindings-scroll { overflow-x: auto; overflow-y: hidden; padding-bottom: 2px; }
.bindings-scroll > * { min-width: max(100%, 680px); }
.reason-tag { display: inline-flex; align-items: center; justify-content: center; min-height: 28px; padding: 0 10px; border-radius: 999px; border: 1px solid var(--vscode-panel-border); font-size: 12px; white-space: nowrap; }
.reason-auto { border-color: color-mix(in srgb, var(--vscode-testing-iconPassed) 40%, var(--vscode-panel-border)); }
.reason-manual { border-color: color-mix(in srgb, var(--vscode-textLink-foreground) 45%, var(--vscode-panel-border)); }
.reason-empty { opacity: 0.7; }
.results-scroll { max-height: min(48vh, 520px); overflow: auto; padding-right: 2px; padding-bottom: 2px; }
.results-scroll > * { min-width: max(100%, 760px); }
.results-list { display: grid; gap: 10px; }
.result-card { border: 1px solid var(--vscode-panel-border); border-radius: 10px; background: color-mix(in srgb, var(--vscode-editor-background) 95%, white); overflow: hidden; }
.result-card summary { list-style: none; cursor: pointer; padding: 12px; display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr) auto auto auto; align-items: center; }
.result-card summary::-webkit-details-marker { display: none; }
.result-main { min-width: 0; }
.result-title { font-weight: 600; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.result-pair { word-break: break-word; }
.result-summary { font-size: 12px; opacity: 0.78; margin-top: 4px; }
.result-stat { font-size: 12px; text-align: right; white-space: nowrap; }
.status-tag { display: inline-flex; align-items: center; justify-content: center; min-width: 72px; min-height: 28px; padding: 0 10px; border-radius: 999px; font-size: 12px; border: 1px solid var(--vscode-panel-border); }
.status-ok { border-color: color-mix(in srgb, var(--vscode-testing-iconPassed) 40%, var(--vscode-panel-border)); }
.status-warn { border-color: color-mix(in srgb, var(--vscode-testing-iconQueued) 45%, var(--vscode-panel-border)); }
.status-bad { border-color: color-mix(in srgb, var(--vscode-testing-iconFailed) 45%, var(--vscode-panel-border)); }
.result-details { border-top: 1px solid var(--vscode-panel-border); padding: 12px; display: grid; gap: 8px; }
.kv-grid { display: grid; gap: 8px 12px; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
.kv-item { border: 1px solid var(--vscode-panel-border); border-radius: 8px; padding: 8px; }
.kv-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.02em; opacity: 0.72; margin-bottom: 4px; }
.kv-value { font-size: 12px; word-break: break-word; }
.compare-meta-grid { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 10px; }
.meta-card { border: 1px solid var(--vscode-panel-border); border-radius: 8px; padding: 10px; background: color-mix(in srgb, var(--vscode-editor-background) 95%, white); }
.meta-title { font-size: 12px; text-transform: uppercase; letter-spacing: 0.02em; opacity: 0.72; margin-bottom: 6px; }
.preview-grid { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.tensor-pane { border: 1px solid var(--vscode-panel-border); border-radius: 8px; padding: 10px; background: color-mix(in srgb, var(--vscode-editor-background) 96%, white); }
.preview-actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.preview-code { margin: 0; padding: 8px; border-radius: 6px; border: 1px solid var(--vscode-panel-border); background: var(--vscode-textCodeBlock-background, color-mix(in srgb, var(--vscode-editor-background) 92%, white)); font-size: 12px; white-space: pre-wrap; word-break: break-word; }
.mini-kv-grid { display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); margin: 8px 0; }
@media (max-width: 900px) {
  body { padding: 12px; }
  .section { padding: 10px; margin-bottom: 10px; }
  .section-collapse > summary { padding: 10px; }
  .section-collapse > .collapse-body { padding: 0 10px 10px 10px; }
  .title { font-size: 14px; }
  .subtitle { display: none; }
  .slot-entry { grid-template-columns: 1fr; }
  .controls { gap: 6px; }
  button, select { min-height: 30px; }
  textarea { min-height: 64px; }
  .binding-top { grid-template-columns: 1fr; }
  .result-card summary { grid-template-columns: 1fr; }
  .result-stat { text-align: left; }
  .sticky { position: static; }
}
</style>
</head>
<body>
<div class="app">
<div class="section sticky">
  <div class="title-row">
    <div>
      <div class="title">Netron Compare</div>
      <div class="subtitle">Bottom panel for shared A/B crop comparison, bindings, and exports.</div>
    </div>
    <div class="chips" id="summaryChips"></div>
  </div>
  <div class="controls">
    <select id="inputMode">
      <option value="zeros">Auto / zeros</option>
      <option value="ones">Auto / ones</option>
      <option value="random">Auto / random</option>
      <option value="import">Import JSON / NPZ</option>
    </select>
    <button id="importCompare" class="secondary">Import Input</button>
    <button id="runCompare">Run Compare</button>
    <button id="clearCompare" class="secondary">Clear</button>
    <button id="exportJson" class="secondary">Export JSON</button>
    <button id="exportCsv" class="secondary">Export CSV</button>
  </div>
  <div class="status" id="status"></div>
  <div class="status" id="importPreview"></div>
</div>
<div class="section">
  <div class="title-row">
    <div>
      <div class="title">Slots</div>
      <div class="subtitle">A/B source snapshots and artifact identities.</div>
    </div>
  </div>
  <div class="grid grid-two">
    <div class="card"><div class="label">Slot A</div><div id="slotA"></div></div>
    <div class="card"><div class="label">Slot B</div><div id="slotB"></div></div>
  </div>
</div>
<div class="section">
  <div class="title-row">
    <div>
      <div class="title">Input Bindings</div>
      <div class="subtitle">Map A-side inputs to B-side candidates.</div>
    </div>
  </div>
  <div class="bindings-scroll"><div id="inputBindings"></div></div>
</div>
<div class="section">
  <div class="title-row">
    <div>
      <div class="title">Output Bindings</div>
      <div class="subtitle">Choose outputs to compare side by side.</div>
    </div>
  </div>
  <div class="bindings-scroll"><div id="outputBindings"></div></div>
</div>
<div class="section">
  <div class="title-row">
    <div>
      <div class="title">Results</div>
      <div class="subtitle">Primary metrics stay visible; details expand per row.</div>
    </div>
  </div>
  <div class="results-scroll">
    <div id="results"></div>
  </div>
</div>
<details class="section section-collapse" id="shapeSection">
  <summary class="title-row">
    <div>
      <div class="title">Shape Overrides</div>
      <div class="subtitle">Optional concrete shapes for dynamic inputs.</div>
    </div>
    <div class="chip">Optional</div>
  </summary>
  <div class="collapse-body">
    <textarea id="shapes" placeholder='{"input_name": [1, 3, 224, 224]}'></textarea>
  </div>
</details>
</div>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
let compareState = null;
const el = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');
const formatShape = (shape) => Array.isArray(shape) ? JSON.stringify(shape) : '';
const formatMetric = (value) => {
  if (value === null || value === undefined || value === '') {
    return '—';
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return String(value);
    }
    const abs = Math.abs(value);
    if ((abs >= 1000 || (abs > 0 && abs < 0.001))) {
      return value.toExponential(3);
    }
    return Number(value.toFixed(6)).toString();
  }
  return escapeHtml(value);
};
const reasonLabel = (reason) => reason ? escapeHtml(reason) : 'pending';
const reasonClass = (reason) => {
  if (!reason) {
    return 'reason-empty';
  }
  return /auto/i.test(reason) ? 'reason-auto' : 'reason-manual';
};
const statusClass = (status) => {
  const value = String(status || '').toLowerCase();
  if (value === 'ok' || value === 'matched') {
    return 'status-ok';
  }
  if (value === 'warn' || value === 'warning' || value === 'skipped') {
    return 'status-warn';
  }
  return value ? 'status-bad' : '';
};
const renderSlot = (slot) => {
  if (!slot) return '<div class="label">(empty)</div>';
  return '<div class="slot-entry">' +
    '<div class="slot-preview"><img class="thumb" src="' + escapeHtml(slot.thumbnail) + '" /></div>' +
    '<div class="slot-meta">' +
      '<div class="slot-title">' + escapeHtml(slot.summary.modelName) + '</div>' +
      '<div class="label">' + escapeHtml(slot.summary.graphName) + '</div>' +
      '<div class="label">' + escapeHtml(slot.summary.nodeCount) + ' nodes · ' + escapeHtml(slot.summary.inputCount) + ' in / ' + escapeHtml(slot.summary.outputCount) + ' out</div>' +
      '<div class="label mono slot-id">' + escapeHtml(slot.artifactId) + '</div>' +
    '</div>' +
  '</div>';
};
const renderBindings = (container, bindings, kind) => {
  if (!bindings || bindings.length === 0) {
    container.innerHTML = '<div class="label">(not ready)</div>';
    return;
  }
  const cards = bindings.map((binding) => {
    const options = ['<option value="">(unpaired)</option>']
      .concat((binding.candidates || []).map((candidate) => '<option value="' + escapeHtml(candidate.name) + '" ' + (candidate.name === binding.targetName ? 'selected' : '') + '>' + escapeHtml(candidate.name) + '</option>'))
      .join('');
    const select = '<select data-kind="' + kind + '" data-source="' + escapeHtml(binding.sourceName) + '">' + options + '</select>';
    const sourcePort = binding.sourcePort || {};
    const targetPort = binding.targetPort || null;
    const targetSummary = binding.targetName && targetPort
      ? escapeHtml(binding.targetName) + ' · ' + escapeHtml(targetPort.dtype || '') + ' · ' + escapeHtml(formatShape(targetPort.shape))
      : 'No target selected';
    return '<div class="binding-card">' +
      '<div class="binding-top">' +
        '<div class="binding-port">' +
          '<div class="label">Source</div>' +
          '<div class="binding-name mono">' + escapeHtml(binding.sourceName) + '</div>' +
          '<div class="binding-meta">' + escapeHtml(sourcePort.dtype || '') + ' · ' + escapeHtml(formatShape(sourcePort.shape)) + '</div>' +
        '</div>' +
        '<div class="binding-select">' +
          '<div class="label">Target</div>' +
          select +
          '<div class="binding-meta">' + targetSummary + '</div>' +
        '</div>' +
        '<div class="reason-tag ' + reasonClass(binding.reason) + '">' + reasonLabel(binding.reason) + '</div>' +
      '</div>' +
    '</div>';
  }).join('');
  container.innerHTML = '<div class="binding-list">' + cards + '</div>';
};
const renderSummaryChips = (state) => {
  const chips = [];
  const status = state && state.compareRunStatus ? state.compareRunStatus : { status: 'idle' };
  const rows = state && state.compareResult && Array.isArray(state.compareResult.rows) ? state.compareResult.rows : [];
  const okCount = rows.filter((row) => String(row.status || '').toLowerCase() === 'ok').length;
  const badCount = rows.filter((row) => {
    const value = String(row.status || '').toLowerCase();
    return value && value !== 'ok';
  }).length;
  const statusValue = String(status.status || '').toLowerCase();
  const statusText = statusValue === 'running'
    ? 'Running: ' + escapeHtml(status.stage || 'working')
    : statusValue === 'failed'
      ? 'Failed: ' + escapeHtml(status.message || 'Unknown error')
      : 'Status: Idle';
  const statusTone = statusValue === 'running' ? 'warn' : statusValue === 'failed' ? 'bad' : 'ok';
  chips.push('<div class="chip ' + statusTone + '">' + statusText + '</div>');
  if (state && state.slotA) {
    chips.push('<div class="chip">A ready</div>');
  }
  if (state && state.slotB) {
    chips.push('<div class="chip">B ready</div>');
  }
  if (rows.length > 0) {
    chips.push('<div class="chip ok">Rows: ' + rows.length + '</div>');
    chips.push('<div class="chip ok">OK: ' + okCount + '</div>');
    if (badCount > 0) {
      chips.push('<div class="chip bad">Attention: ' + badCount + '</div>');
    }
  }
  if (state && state.compareResult && state.compareResult.summary) {
    chips.push('<div class="chip warn">Max Abs: ' + formatMetric(state.compareResult.summary.maxAbs) + '</div>');
  }
  el('summaryChips').innerHTML = chips.join('');
};
const renderPreviewText = (preview) => {
  if (!preview || !Array.isArray(preview.sampleValues) || preview.sampleValues.length === 0) {
    return '[]';
  }
  const values = preview.sampleValues.map((item) => item === undefined ? null : item);
  const suffix = preview.truncated ? ', ...' : '';
  return JSON.stringify(values) + suffix;
};
const renderMiniStats = (preview, summary) => {
  const items = [];
  if (preview && preview.elementCount !== undefined) {
    items.push(['Elements', preview.elementCount]);
  }
  if (summary) {
    items.push(['Min', formatMetric(summary.min)]);
    items.push(['Max', formatMetric(summary.max)]);
    items.push(['Mean', formatMetric(summary.mean)]);
    items.push(['Abs Mean', formatMetric(summary.absMean)]);
    items.push(['L2', formatMetric(summary.l2Norm)]);
    items.push(['NonZero', summary.nonZeroCount]);
  }
  if (items.length === 0) {
    return '<div class="label">No tensor stats.</div>';
  }
  return '<div class="mini-kv-grid">' + items.map(([label, value]) =>
    '<div class="kv-item"><div class="kv-label">' + escapeHtml(label) + '</div><div class="kv-value">' + escapeHtml(String(value)) + '</div></div>'
  ).join('') + '</div>';
};
const renderSubgraphCard = (title, meta) => {
  if (!meta) {
    return '';
  }
  return '<div class="meta-card">' +
    '<div class="meta-title">' + escapeHtml(title) + '</div>' +
    '<div><strong>' + escapeHtml(meta.modelName || '') + '</strong></div>' +
    '<div class="label">' + escapeHtml(meta.graphName || '') + '</div>' +
    '<div class="label">' + escapeHtml(meta.nodeCount || 0) + ' nodes · ' + escapeHtml(meta.inputCount || 0) + ' in / ' + escapeHtml(meta.outputCount || 0) + ' out</div>' +
    '<div class="label mono">' + escapeHtml(meta.artifactId || '') + '</div>' +
  '</div>';
};
const renderCompareStatsCard = (stats) => {
  if (!stats) {
    return '';
  }
  return '<div class="meta-card">' +
    '<div class="meta-title">Compare Stats</div>' +
    '<div class="label">Input bindings: ' + escapeHtml(stats.inputBindingCount || 0) + '</div>' +
    '<div class="label">Output bindings: ' + escapeHtml(stats.outputBindingCount || 0) + '</div>' +
    '<div class="label">Rows: ' + escapeHtml(stats.rowCount || 0) + '</div>' +
    '<div class="label">OK: ' + escapeHtml(stats.okCount || 0) + ' · Skipped: ' + escapeHtml(stats.skippedCount || 0) + '</div>' +
  '</div>';
};
const renderTensorPane = (side, row) => {
  const preview = side === 'A' ? row.sourcePreview : row.targetPreview;
  const summary = side === 'A' ? row.sourceStats : row.targetStats;
  const tensorName = side === 'A' ? (row.sourceName || '') : (row.targetName || '');
  const button = tensorName
    ? '<button class="secondary" data-action="export-output-npy" data-side="' + side + '" data-source="' + escapeHtml(row.sourceName || '') + '" data-target="' + escapeHtml(row.targetName || '') + '">Export ' + side + ' NPY</button>'
    : '';
  return '<div class="tensor-pane">' +
    '<div class="meta-title">Output ' + side + '</div>' +
    '<div class="mono">' + escapeHtml(tensorName) + '</div>' +
    '<div class="preview-actions">' + button + '</div>' +
    renderMiniStats(preview, summary) +
    '<pre class="preview-code">' + escapeHtml(renderPreviewText(preview)) + '</pre>' +
  '</div>';
};
const renderResults = (result) => {
  if (!result || !result.rows) {
    return '<div class="label">(no compare result)</div>';
  }
  const summary = result.summary
    ? '<div class="chips" style="margin-bottom:10px;">' +
        '<div class="chip warn">Max diff output: <span class="mono">' + escapeHtml(result.summary.maxDiffOutput) + '</span></div>' +
        '<div class="chip">Max Abs: ' + formatMetric(result.summary.maxAbs) + '</div>' +
      '</div>'
    : '<div class="label">No numeric comparable outputs.</div>';
  const meta = result.subgraphs
    ? '<div class="compare-meta-grid">' + renderSubgraphCard('Subgraph A', result.subgraphs.A) + renderSubgraphCard('Subgraph B', result.subgraphs.B) + renderCompareStatsCard(result.compareStats) + '</div>'
    : '';
  const rows = result.rows.map((row) => {
    const status = escapeHtml(row.status || 'unknown');
    const sourceName = escapeHtml(row.sourceName || '');
    const targetName = escapeHtml(row.targetName || '');
    const dtype = escapeHtml(row.dtype || '');
    const shape = escapeHtml(formatShape(row.shape));
    const reason = escapeHtml(row.reason || '');
    return '<details class="result-card">' +
      '<summary>' +
        '<div class="result-main">' +
          '<div class="result-title"><span class="result-pair mono">' + sourceName + '</span><span>→</span><span class="result-pair mono">' + targetName + '</span></div>' +
          '<div class="result-summary">' + dtype + (shape ? ' · ' + shape : '') + (reason ? ' · ' + reason : '') + '</div>' +
        '</div>' +
        '<div class="status-tag ' + statusClass(row.status) + '">' + status + '</div>' +
        '<div class="result-stat"><div class="label">Max Abs</div><div>' + formatMetric(row.maxAbs) + '</div></div>' +
        '<div class="result-stat"><div class="label">Mean Abs</div><div>' + formatMetric(row.meanAbs) + '</div></div>' +
      '</summary>' +
      '<div class="result-details">' +
        '<div class="kv-grid">' +
          '<div class="kv-item"><div class="kv-label">A</div><div class="kv-value mono">' + sourceName + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">B</div><div class="kv-value mono">' + targetName + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">Status</div><div class="kv-value">' + status + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">Reason</div><div class="kv-value">' + (reason || '—') + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">DType</div><div class="kv-value">' + (dtype || '—') + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">Shape</div><div class="kv-value mono">' + (shape || '—') + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">Max Abs</div><div class="kv-value">' + formatMetric(row.maxAbs) + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">Mean Abs</div><div class="kv-value">' + formatMetric(row.meanAbs) + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">RMSE</div><div class="kv-value">' + formatMetric(row.rmse) + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">Max Rel</div><div class="kv-value">' + formatMetric(row.maxRelativeDiff) + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">Cosine</div><div class="kv-value">' + formatMetric(row.cosineSimilarity) + '</div></div>' +
          '<div class="kv-item"><div class="kv-label">Pearson</div><div class="kv-value">' + formatMetric(row.pearsonCorrelation) + '</div></div>' +
        '</div>' +
        '<div class="preview-grid">' + renderTensorPane('A', row) + renderTensorPane('B', row) + '</div>' +
      '</div>' +
    '</details>';
  }).join('');
  return meta + summary + '<div class="results-list">' + rows + '</div>';
};
const render = () => {
  el('slotA').innerHTML = renderSlot(compareState && compareState.slotA);
  el('slotB').innerHTML = renderSlot(compareState && compareState.slotB);
  renderBindings(el('inputBindings'), compareState && compareState.inputBindings, 'input');
  renderBindings(el('outputBindings'), compareState && compareState.outputBindings, 'output');
  el('results').innerHTML = renderResults(compareState && compareState.compareResult);
  const status = compareState && compareState.compareRunStatus ? compareState.compareRunStatus : { status: 'idle' };
  const statusValue = String(status.status || '').toLowerCase();
  el('status').textContent = statusValue === 'running'
    ? ('Running: ' + (status.stage || 'working'))
    : statusValue === 'failed'
      ? ('Failed: ' + (status.message || 'Unknown error'))
      : 'Idle';
  el('status').className = 'status' + (statusValue === 'running' ? ' running' : statusValue === 'failed' ? ' failed' : '');
  const imported = compareState && compareState.importedInput;
  el('importPreview').innerHTML = imported && imported.preview && imported.preview.length
    ? imported.preview.map((entry) => escapeHtml(entry.name) + ': ' + escapeHtml(entry.dtype || '') + ' ' + escapeHtml(JSON.stringify(entry.shape || []))).join('<br>')
    : '(no imported compare input)';
  const inputMode = el('inputMode').value;
  const readyInputs = compareState && compareState.inputBindings && compareState.inputBindings.length > 0 && compareState.inputBindings.every((item) => !!item.targetName);
  const readyOutputs = compareState && compareState.outputBindings && compareState.outputBindings.some((item) => !!item.targetName);
  el('runCompare').disabled = !(readyInputs && readyOutputs) || (inputMode === 'import' && !(imported && imported.token)) || (status.status === 'running');
  renderSummaryChips(compareState);
};
window.addEventListener('message', (event) => {
  const message = event.data;
  if (message.type === 'compareStateUpdate') {
    compareState = message.state;
    render();
  }
});
document.addEventListener('change', (event) => {
  const target = event.target;
  if (target && target.matches('select[data-kind]')) {
    vscode.postMessage({
      type: 'setCompareBinding',
      kind: target.getAttribute('data-kind'),
      sourceName: target.getAttribute('data-source'),
      targetName: target.value || null
    });
  }
});
document.addEventListener('click', (event) => {
  const button = event.target && event.target.closest ? event.target.closest('button[data-action="export-output-npy"]') : null;
  if (!button) {
    return;
  }
  event.preventDefault();
  vscode.postMessage({
    type: 'exportCompareOutputNpy',
    side: button.getAttribute('data-side') || 'A',
    sourceName: button.getAttribute('data-source') || '',
    targetName: button.getAttribute('data-target') || ''
  });
});
el('importCompare').addEventListener('click', () => vscode.postMessage({ type: 'importCompareInput' }));
el('runCompare').addEventListener('click', () => {
  let inputShapes = {};
  const text = el('shapes').value.trim();
  if (text) {
    try { inputShapes = JSON.parse(text); } catch (error) { el('status').textContent = 'Invalid shapes JSON.'; return; }
  }
  vscode.postMessage({ type: 'runCompare', inputMode: el('inputMode').value, inputShapes });
});
el('clearCompare').addEventListener('click', () => vscode.postMessage({ type: 'clearCompare' }));
el('exportJson').addEventListener('click', () => vscode.postMessage({ type: 'exportCompareJson' }));
el('exportCsv').addEventListener('click', () => vscode.postMessage({ type: 'exportCompareCsv' }));
vscode.postMessage({ type: 'ready' });
</script>
</body>
</html>`;
}

async function handleModelPanelMessage(panelState, message) {
    if (!message || typeof message.type !== 'string') {
        return;
    }
    switch (message.type) {
        case 'ready':
            panelState.ready = true;
            if (panelState.readyTimer) {
                clearTimeout(panelState.readyTimer);
                panelState.readyTimer = null;
            }
            await flushPanelMessages(panelState);
            enqueuePanelMessage(panelState, { type: 'compareStateUpdate', state: state.workbench.getCompareState() });
            enqueuePanelMessage(panelState, { type: 'activityLog', entries: panelState.activity });
            break;
        case 'requestOpenModel': {
            const uri = await resolveModelUri(null);
            if (uri) {
                await openModelInPanel(panelState, uri, 'webview-open');
            }
            break;
        }
        case 'requestOpenCompareCenter':
            await focusCompareView(false);
            break;
        case 'confirmCrop': {
            const artifact = await state.workbench.createCropArtifact({
                sessionId: panelState.currentSessionId,
                startKeys: message.startKeys || [],
                endKeys: message.endKeys || []
            });
            panelState.currentArtifactId = artifact.id;
            addPanelActivity(panelState, 'info', 'Crop confirmed', { artifactId: artifact.id, summary: artifact.summary });
            enqueuePanelMessage(panelState, {
                type: 'cropConfirmed',
                artifact: {
                    id: artifact.id,
                    stale: artifact.stale,
                    summary: artifact.summary,
                    ioSignature: artifact.ioSignature,
                    createdAt: artifact.createdAt,
                    thumbnail: artifact.thumbnail
                },
                graph: artifact.cropGraphSnapshot
            });
            break;
        }
        case 'exportCropOnnx': {
            const artifact = state.workbench.getArtifact(message.artifactId || panelState.currentArtifactId);
            if (!artifact) {
                throw new Error('No confirmed crop artifact available.');
            }
            const session = state.workbench.getSession(artifact.modelSessionId);
            const baseName = `${path.basename(session.filePath, path.extname(session.filePath))}.${artifact.id}.crop.onnx`;
            const saveUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.joinPath(getDefaultFolder(), baseName),
                filters: { ONNX: ['onnx'] },
                title: 'Export Crop ONNX'
            });
            if (!saveUri) {
                break;
            }
            updatePanelTask(panelState, { status: 'running', stage: '重建 ONNX', message: 'Exporting crop ONNX...', startedAt: new Date().toISOString(), cancellable: false, busy: true });
            const useExternal = /external/i.test(message.weightMode || '') || (session.graphInfo.initializers.size > 0 && ensureExternalData(session));
            const result = await state.workbench.exportArtifact(artifact.id, saveUri.fsPath || saveUri.path, { externalData: useExternal, inlineWeights: !useExternal });
            clearPanelTask(panelState);
            addPanelActivity(panelState, 'info', 'Crop ONNX exported', result);
            enqueuePanelMessage(panelState, { type: 'artifactExported', exportInfo: result });
            vscode.window.showInformationMessage(`Crop ONNX exported: ${result.filePath}`);
            break;
        }
        case 'importInputFile': {
            const picked = await vscode.window.showOpenDialog({
                canSelectMany: false,
                canSelectFiles: true,
                canSelectFolders: false,
                filters: { Input: ['json', 'npz'] },
                defaultUri: getDefaultFolder(),
                title: 'Import inference input (.json / .npz)'
            });
            if (!picked || picked.length === 0) {
                break;
            }
            const imported = await state.workbench.importInputFile(picked[0].fsPath || picked[0].path);
            enqueuePanelMessage(panelState, { type: 'inputImported', token: imported.token, preview: imported.preview });
            addPanelActivity(panelState, 'info', 'Input file imported', { preview: imported.preview });
            break;
        }
        case 'runInference': {
            updatePanelTask(panelState, { status: 'running', stage: '执行推理', message: 'Running inference...', startedAt: new Date().toISOString(), cancellable: false, busy: true });
            const result = await state.workbench.runInference({
                artifactId: message.artifactId || panelState.currentArtifactId,
                sessionId: panelState.currentSessionId,
                useFullGraph: !!message.useFullGraph,
                inputMode: message.inputMode || 'zeros',
                inputShapes: message.inputShapes || {},
                importToken: message.importToken || null
            });
            clearPanelTask(panelState);
            addPanelActivity(panelState, 'info', 'Inference completed', { runId: result.runId });
            enqueuePanelMessage(panelState, { type: 'inferenceResult', result });
            break;
        }
        case 'assignCompareSlot': {
            const compareState = state.workbench.assignCompareSlot(message.slot, message.artifactId || panelState.currentArtifactId);
            addPanelActivity(panelState, 'info', `Assigned artifact to compare slot ${message.slot}`, { artifactId: message.artifactId || panelState.currentArtifactId });
            broadcastCompareState(compareState);
            await focusCompareView(false);
            break;
        }
        case 'requestCompareState':
            enqueuePanelMessage(panelState, { type: 'compareStateUpdate', state: state.workbench.getCompareState() });
            break;
        case 'requestTensorPreview': {
            try {
                const preview = await state.workbench.getTensorPreview(message.sessionId || panelState.currentSessionId, message.tensorName, { limit: message.limit || 64 });
                enqueuePanelMessage(panelState, {
                    type: 'tensorPreviewResult',
                    requestId: message.requestId || null,
                    ok: true,
                    preview
                });
            } catch (error) {
                enqueuePanelMessage(panelState, {
                    type: 'tensorPreviewResult',
                    requestId: message.requestId || null,
                    ok: false,
                    error: error && error.message ? error.message : String(error)
                });
            }
            break;
        }
        case 'cancelTask':
            panelState.cancelRequested = true;
            addPanelActivity(panelState, 'warn', 'Cancellation requested', { panelId: panelState.id });
            updatePanelTask(panelState, { message: 'Cancellation requested. Some runtime tasks may complete before stopping.' });
            break;
        case 'saveFile':
            await handleSaveFile(panelState.panel, message);
            break;
        case 'copyText':
            await handleCopyText(panelState.panel, message);
            break;
        case 'readBundledText':
            await handleReadBundledText(panelState.panel, message);
            break;
        case 'openExternal':
            if (message.url) {
                await vscode.env.openExternal(vscode.Uri.parse(message.url));
            }
            break;
        case 'notify':
            handleNotify(message);
            break;
        default:
            break;
    }
}

async function handleCompareCenterMessage(message) {
    if (!message || typeof message.type !== 'string') {
        return;
    }
    switch (message.type) {
        case 'ready':
            state.compareViewReady = true;
            pushCompareState();
            break;
        case 'clearCompare':
            state.workbench.clearCompare();
            vscode.window.showInformationMessage('Netron Compare cleared.');
            break;
        case 'setCompareBinding':
            state.workbench.setCompareBinding(message.kind, message.sourceName, message.targetName);
            break;
        case 'importCompareInput': {
            const picked = await vscode.window.showOpenDialog({
                canSelectMany: false,
                canSelectFiles: true,
                canSelectFolders: false,
                filters: { Input: ['json', 'npz'] },
                defaultUri: getDefaultFolder(),
                title: 'Import compare input (.json / .npz)'
            });
            if (picked && picked.length > 0) {
                const imported = await state.workbench.importInputFile(picked[0].fsPath || picked[0].path);
                state.workbench.setCompareImportedInput(imported);
            }
            break;
        }
        case 'runCompare': {
            const compareState = await state.workbench.runCompare({
                inputMode: message.inputMode || 'zeros',
                inputShapes: message.inputShapes || {},
                importToken: state.workbench.getCompareState().importedInput && state.workbench.getCompareState().importedInput.token
            });
            pushCompareState(compareState);
            break;
        }
        case 'exportCompareJson': {
            const saveUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.joinPath(getDefaultFolder(), 'compare-result.json'),
                filters: { JSON: ['json'] },
                title: 'Export compare result as JSON'
            });
            if (saveUri) {
                await vscode.workspace.fs.writeFile(saveUri, Buffer.from(state.workbench.exportCompareResultAsJson(), 'utf8'));
            }
            break;
        }
        case 'exportCompareCsv': {
            const saveUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.joinPath(getDefaultFolder(), 'compare-result.csv'),
                filters: { CSV: ['csv'] },
                title: 'Export compare result as CSV'
            });
            if (saveUri) {
                await vscode.workspace.fs.writeFile(saveUri, Buffer.from(state.workbench.exportCompareResultAsCsv(), 'utf8'));
            }
            break;
        }
        case 'exportCompareOutputNpy': {
            const exported = state.workbench.exportCompareOutputAsNpy({
                side: message.side,
                sourceName: message.sourceName,
                targetName: message.targetName
            });
            const saveUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.joinPath(getDefaultFolder(), exported.fileName),
                filters: { NPY: ['npy'] },
                title: `Export compare output ${message.side || 'A'} as NPY`
            });
            if (saveUri) {
                await vscode.workspace.fs.writeFile(saveUri, new Uint8Array(exported.bytes));
            }
            break;
        }
        default:
            break;
    }
}

async function openModelInPanel(panelState, modelUri, trigger) {
    panelState.currentModelUri = modelUri;
    panelState.currentArtifactId = null;
    const fileName = path.basename(modelUri.fsPath || modelUri.path);
    panelState.panel.title = `Netron Preview: ${fileName}`;
    addPanelActivity(panelState, 'info', 'Open model requested', { file: fileName, trigger });

    if (!isOnnxFileName(fileName)) {
        return openLegacyModelInPanel(panelState, modelUri, trigger);
    }

    updatePanelTask(panelState, {
        status: 'running',
        stage: '读取文件',
        message: `Opening ${fileName}`,
        startedAt: new Date().toISOString(),
        cancellable: true,
        busy: true
    });

    const session = await state.workbench.loadModel(modelUri, {
        onStage: (stage, detail) => {
            updatePanelTask(panelState, { status: 'running', stage, message: fileName, cancellable: true, busy: true });
            addPanelActivity(panelState, 'info', stage, detail);
        }
    });

    panelState.currentSessionId = session.id;
    clearPanelTask(panelState);
    enqueuePanelMessage(panelState, {
        type: 'renderGraphSnapshot',
        sessionId: session.id,
        model: session.snapshot,
        fileName,
        filePath: modelUri.fsPath || modelUri.path
    });
    addPanelActivity(panelState, 'info', 'Host-managed ONNX render ready', { sessionId: session.id });
}

async function openLegacyModelInPanel(panelState, modelUri, trigger) {
    const bytes = await vscode.workspace.fs.readFile(modelUri);
    const fileName = path.basename(modelUri.fsPath || modelUri.path);
    addPanelActivity(panelState, 'info', 'Legacy model load', { file: fileName, trigger, sizeBytes: bytes.byteLength });
    enqueuePanelMessage(panelState, {
        type: 'loadModel',
        requestId: createRequestId(),
        name: fileName,
        base64: Buffer.from(bytes).toString('base64'),
        sizeBytes: bytes.byteLength,
        sentAt: Date.now()
    });
}

function pushCompareState(snapshot) {
    state.pendingCompareState = snapshot || state.workbench.getCompareState();
    flushCompareState();
}

function broadcastCompareState(snapshot) {
    const compareState = snapshot || state.workbench.getCompareState();
    pushCompareState(compareState);
    for (const panelState of state.panels.values()) {
        enqueuePanelMessage(panelState, { type: 'compareStateUpdate', state: compareState });
    }
}

async function handleSaveFile(panel, message) {
    const fileName = typeof message.fileName === 'string' && message.fileName.length > 0 ? message.fileName : 'output.bin';
    const saveUri = await vscode.window.showSaveDialog({
        defaultUri: vscode.Uri.joinPath(getDefaultFolder(), fileName),
        title: 'Save generated file',
        filters: normalizeFilters(message.filters)
    });
    if (!saveUri) {
        return;
    }
    const content = typeof message.base64 === 'string'
        ? Buffer.from(message.base64, 'base64')
        : typeof message.text === 'string'
            ? Buffer.from(message.text, 'utf8')
            : null;
    if (!content) {
        throw new Error('Invalid save payload.');
    }
    await vscode.workspace.fs.writeFile(saveUri, new Uint8Array(content));
    await panel.webview.postMessage({ type: 'fileSaved', path: saveUri.fsPath || saveUri.path });
}

async function handleCopyText(panel, message) {
    const text = typeof message.text === 'string' ? message.text : '';
    const label = typeof message.label === 'string' && message.label.length > 0 ? message.label : 'Text';
    if (!text) {
        await panel.webview.postMessage({ type: 'clipboardError', label, message: 'No text content to copy.' });
        return;
    }
    try {
        await vscode.env.clipboard.writeText(text);
        await panel.webview.postMessage({ type: 'clipboardCopied', label });
    } catch (error) {
        await panel.webview.postMessage({ type: 'clipboardError', label, message: error.message });
    }
}

async function handleReadBundledText(panel, message) {
    const requestId = typeof message.requestId === 'string' ? message.requestId : null;
    const file = typeof message.file === 'string' ? message.file : '';
    if (!requestId || !/^[a-z0-9._-]+-metadata\.json$/i.test(file)) {
        return;
    }
    try {
        const sourceRoot = vscode.Uri.joinPath(state.context.extensionUri, 'netron', 'source');
        const target = vscode.Uri.joinPath(sourceRoot, file);
        const bytes = await vscode.workspace.fs.readFile(target);
        await panel.webview.postMessage({ type: 'readBundledTextResult', requestId, ok: true, text: Buffer.from(bytes).toString('utf8') });
    } catch (error) {
        await panel.webview.postMessage({ type: 'readBundledTextResult', requestId, ok: false, error: error.message });
    }
}

function handleNotify(message) {
    const text = typeof message.message === 'string' ? message.message : '';
    if (!text) {
        return;
    }
    if (message.level === 'error') {
        vscode.window.showErrorMessage(text);
    } else if (message.level === 'warn') {
        vscode.window.showWarningMessage(text);
    } else {
        vscode.window.showInformationMessage(text);
    }
}

function ensureExternalData(session) {
    return Array.from(session.graphInfo.initializers.values()).some((tensor) => tensor.dataLocation === 1 && Array.isArray(tensor.externalData) && tensor.externalData.length > 0);
}

function getDefaultFolder() {
    return vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0
        ? vscode.workspace.workspaceFolders[0].uri
        : vscode.Uri.file(os.homedir());
}

function normalizeFilters(value) {
    if (!value || typeof value !== 'object') {
        return undefined;
    }
    const result = {};
    for (const [key, extensions] of Object.entries(value)) {
        const items = Array.isArray(extensions) ? extensions.map((item) => String(item).replace(/^\./, '')) : [];
        if (items.length > 0) {
            result[key] = items;
        }
    }
    return Object.keys(result).length > 0 ? result : undefined;
}

function deactivate() {
    for (const panelState of state.panels.values()) {
        if (panelState.readyTimer) {
            clearTimeout(panelState.readyTimer);
        }
    }
}

module.exports = {
    activate,
    deactivate
};
