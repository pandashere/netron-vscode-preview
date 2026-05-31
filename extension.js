const fs = require('fs');
const os = require('os');
const path = require('path');
const vscode = require('vscode');
const { ONNXWorkbench, isOnnxFileName } = require('./lib/onnx-workbench');
const { ToolRegistry, runTool } = require('./lib/cli-tools');
const { FormatProviderRegistry, createOnnxProvider, providerDiagnostics } = require('./lib/format-providers');
const { createDevIrProvider } = require('./lib/dev-ir-provider');
const {
    assignCompareSlot,
    cloneCompareState,
    createEmptyCompareState,
    setCompareBinding,
    setCompareRunStatus,
    setImportedInput
} = require('./lib/host-compare-state');
const {
    exportCompareOutputAsNpy,
    exportCompareResultAsCsv,
    exportCompareResultAsJson,
    runCrossProviderCompare
} = require('./lib/compare-engine');
const {
    analysisCancelling,
    analysisFailed,
    analysisStarted,
    analysisSucceeded,
    createInitialAiAnalysisState
} = require('./lib/ai-analysis-state');

const WEBVIEW_READY_TIMEOUT_MS = 10000;
const COMPARE_CENTER_HTML_VERSION = 3;
const AI_ANALYSIS_HTML_VERSION = 1;
const COMPARE_VIEW_CONTAINER_ID = 'netronComparePanel';
const COMPARE_VIEW_ID = 'netronCompare.compareView';
const AI_VIEW_ID = 'netronAI.analysisView';

const state = {
    context: null,
    output: null,
    panelSeq: 0,
    compareView: null,
    compareViewReady: false,
    pendingCompareState: null,
    compareState: createEmptyCompareState(),
    compareRawOutputs: new Map(),
    compareProviderId: null,
    aiView: null,
    aiViewReady: false,
    pendingAiState: null,
    panels: new Map(),
    workbench: null,
    exporterRegistry: null,
    analyzerRegistry: null,
    providerRegistry: null,
    globalTask: null,
    aiState: createInitialAiAnalysisState()
};
const pendingFormatProviders = [];

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
        currentProviderId: null,
        selectedExporterId: null,
        selectedFormatterId: null,
        selectedAnalyzerId: null,
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

function getToolRoots() {
    const base = path.join(os.homedir(), '.netron', 'vscode-preview');
    return {
        exporters: path.join(base, 'exporters'),
        analyzers: path.join(base, 'analyzers')
    };
}

function getToolState() {
    return {
        exporters: state.exporterRegistry ? state.exporterRegistry.getSnapshot() : { kind: 'exporter', entries: [] },
        analyzers: state.analyzerRegistry ? state.analyzerRegistry.getSnapshot() : { kind: 'analyzer', entries: [] },
        task: state.globalTask ? {
            id: state.globalTask.id,
            kind: state.globalTask.kind,
            status: state.globalTask.status,
            message: state.globalTask.message,
            startedAt: state.globalTask.startedAt,
            sourcePanelId: state.globalTask.sourcePanelId
        } : null
    };
}

function broadcastToolState() {
    const toolState = getToolState();
    for (const panelState of state.panels.values()) {
        enqueuePanelMessage(panelState, { type: 'toolStateUpdate', state: toolState });
    }
}

function setGlobalTask(task) {
    state.globalTask = task;
    broadcastToolState();
}

function clearGlobalTask(taskId) {
    if (!taskId || (state.globalTask && state.globalTask.id === taskId)) {
        state.globalTask = null;
        broadcastToolState();
    }
}

function attachGlobalTaskProcess(taskId, child) {
    if (!state.globalTask || state.globalTask.id !== taskId) {
        return;
    }
    state.globalTask.process = child;
    if (state.globalTask.cancelRequested && child && typeof child.kill === 'function') {
        try {
            child.kill('SIGTERM');
        } catch (error) {
            appendLog('warn', 'failed to kill cancelled task process', { message: error.message });
        }
    }
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

function createAiViewProvider() {
    return {
        resolveWebviewView(webviewView) {
            const disposables = [];
            state.aiView = webviewView;
            state.aiViewReady = false;
            webviewView.title = 'AI Analysis';
            webviewView.description = 'Latest crop analysis result';
            webviewView.webview.options = {
                enableScripts: true
            };
            webviewView.webview.html = buildAiAnalysisHtml(webviewView.webview);
            webviewView.__aiHtmlVersion = AI_ANALYSIS_HTML_VERSION;
            webviewView.onDidDispose(() => {
                if (state.aiView === webviewView) {
                    state.aiView = null;
                    state.aiViewReady = false;
                }
                while (disposables.length > 0) {
                    const disposable = disposables.pop();
                    if (disposable) {
                        disposable.dispose();
                    }
                }
            }, null, disposables);
            webviewView.onDidChangeVisibility(() => {
                if (state.aiView === webviewView && webviewView.visible) {
                    flushAiState();
                }
            }, null, disposables);
            webviewView.webview.onDidReceiveMessage((message) => {
                handleAiPanelMessage(message).catch((error) => {
                    appendLog('error', 'ai view message failed', { type: message && message.type, message: error.message });
                    vscode.window.showErrorMessage(error.message);
                });
            }, null, disposables);
            flushAiState();
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

function canPostAiState() {
    return !!(state.aiView && state.aiViewReady && state.aiView.visible);
}

function pushAiState(snapshot) {
    state.pendingAiState = snapshot || state.aiState;
    flushAiState();
}

function flushAiState() {
    if (!canPostAiState() || !state.pendingAiState) {
        return;
    }
    const aiState = state.pendingAiState;
    state.aiView.webview.postMessage({ type: 'aiStateUpdate', state: aiState }).then((posted) => {
        if (posted !== false && state.pendingAiState === aiState) {
            state.pendingAiState = null;
        }
    }).catch(() => {});
}

async function focusAiView(preserveFocus = false) {
    if (state.aiView && state.aiView.__aiHtmlVersion !== AI_ANALYSIS_HTML_VERSION) {
        state.aiView.webview.html = buildAiAnalysisHtml(state.aiView.webview);
        state.aiView.__aiHtmlVersion = AI_ANALYSIS_HTML_VERSION;
        state.aiViewReady = false;
    }
    if (!state.aiView) {
        await vscode.commands.executeCommand(`workbench.view.extension.${COMPARE_VIEW_CONTAINER_ID}`);
    }
    if (state.aiView) {
        state.aiView.show(preserveFocus);
    }
    pushAiState();
}

function updateAiState(patch) {
    state.aiState = {
        ...state.aiState,
        ...patch,
        updatedAt: new Date().toISOString()
    };
    pushAiState(state.aiState);
}

function registerFormatProvider(provider) {
    if (state.providerRegistry) {
        state.providerRegistry.register(provider);
        appendLog('info', 'format provider registered', { providerId: provider.id });
    } else {
        const diagnostics = providerDiagnostics(provider);
        if (diagnostics.errors.length > 0) {
            throw new Error(diagnostics.errors.join(' '));
        }
        if (pendingFormatProviders.some((item) => item.id === provider.id)) {
            throw new Error(`Duplicate provider id: ${provider.id}`);
        }
        pendingFormatProviders.push(provider);
    }
    return {
        dispose() {
            unregisterFormatProvider(provider.id);
        }
    };
}

function unregisterFormatProvider(providerId) {
    const pendingIndex = pendingFormatProviders.findIndex((provider) => provider.id === providerId);
    if (pendingIndex >= 0) {
        pendingFormatProviders.splice(pendingIndex, 1);
        return true;
    }
    if (state.providerRegistry) {
        const removed = state.providerRegistry.unregister(providerId);
        if (removed) {
            appendLog('info', 'format provider unregistered', { providerId });
        }
        return removed;
    }
    return false;
}

function getFormatProviders() {
    if (state.providerRegistry) {
        return state.providerRegistry.list().map((provider) => ({
            id: provider.id,
            label: provider.label || provider.id,
            capabilities: { ...(provider.capabilities || {}) }
        }));
    }
    return pendingFormatProviders.map((provider) => ({
        id: provider.id,
        label: provider.label || provider.id,
        capabilities: { ...(provider.capabilities || {}) }
    }));
}

function getFormatProviderDiagnostics(providerId) {
    if (state.providerRegistry) {
        return state.providerRegistry.getDiagnostics(providerId);
    }
    if (providerId) {
        const provider = pendingFormatProviders.find((item) => item.id === providerId);
        return provider ? providerDiagnostics(provider) : { errors: [], warnings: [`Provider '${providerId}' is not registered.`] };
    }
    return pendingFormatProviders.map((provider) => ({
        id: provider.id,
        ...providerDiagnostics(provider)
    }));
}

function createExtensionApi() {
    return {
        registerFormatProvider,
        unregisterFormatProvider,
        getFormatProviders,
        getFormatProviderDiagnostics
    };
}

function providerInfo(provider) {
    if (!provider) {
        return null;
    }
    return {
        id: provider.id,
        label: provider.label || provider.id,
        capabilities: { ...(provider.capabilities || {}) }
    };
}

async function activate(context) {
    state.context = context;
    state.output = vscode.window.createOutputChannel('Netron Preview');
    context.subscriptions.push(state.output);
    state.workbench = new ONNXWorkbench(context, (level, message, detail) => appendLog(level, message, detail));
    state.providerRegistry = new FormatProviderRegistry();
    state.providerRegistry.register(createOnnxProvider(state.workbench, isOnnxFileName));
    state.providerRegistry.register(createDevIrProvider({ id: 'dev-ir-a', label: 'Dev IR A' }));
    state.providerRegistry.register(createDevIrProvider({ id: 'dev-ir-b', label: 'Dev IR B' }));
    while (pendingFormatProviders.length > 0) {
        state.providerRegistry.register(pendingFormatProviders.shift());
    }
    state.workbench.onChange(() => {
        broadcastCompareState();
    });
    context.subscriptions.push(vscode.window.registerWebviewViewProvider(COMPARE_VIEW_ID, createCompareViewProvider(), {
        webviewOptions: {
            retainContextWhenHidden: true
        }
    }));
    const toolRoots = getToolRoots();
    state.exporterRegistry = new ToolRegistry({
        kind: 'exporter',
        rootDir: toolRoots.exporters,
        defaultTimeoutMs: 30000,
        maxTimeoutMs: 300000,
        logger: (level, message, detail) => appendLog(level, message, detail)
    });
    state.analyzerRegistry = new ToolRegistry({
        kind: 'analyzer',
        rootDir: toolRoots.analyzers,
        defaultTimeoutMs: 120000,
        maxTimeoutMs: 300000,
        logger: (level, message, detail) => appendLog(level, message, detail)
    });
    state.exporterRegistry.onChange(() => broadcastToolState());
    state.analyzerRegistry.onChange(() => broadcastToolState());
    state.exporterRegistry.refresh();
    state.analyzerRegistry.refresh();
    state.exporterRegistry.startWatching();
    state.analyzerRegistry.startWatching();
    context.subscriptions.push({ dispose: () => state.exporterRegistry && state.exporterRegistry.stopWatching() });
    context.subscriptions.push({ dispose: () => state.analyzerRegistry && state.analyzerRegistry.stopWatching() });
    context.subscriptions.push(vscode.window.registerWebviewViewProvider(AI_VIEW_ID, createAiViewProvider(), {
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
        clearActiveCompare();
        vscode.window.showInformationMessage('Netron Compare cleared.');
    }));

    context.subscriptions.push(vscode.commands.registerCommand('netronPreview.openAiAnalysis', async () => {
        await focusAiView(false);
    }));
    return createExtensionApi();
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

function buildAiAnalysisHtml(webview) {
    const nonce = createRequestId().replace(/[^a-z0-9]/gi, '');
    const csp = webview.cspSource;
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${csp} data:; style-src ${csp} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AI Analysis</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 12px; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); }
.app { width: 100%; max-width: 1200px; margin: 0 auto; display: grid; gap: 12px; }
.section { border: 1px solid var(--vscode-panel-border); border-radius: 10px; padding: 12px; background: color-mix(in srgb, var(--vscode-editor-background) 92%, white); }
.title { font-size: 15px; font-weight: 600; }
.subtitle { font-size: 12px; opacity: 0.75; }
.meta { display: grid; gap: 6px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 8px; }
.meta-item { border: 1px solid var(--vscode-panel-border); border-radius: 8px; padding: 8px; }
.meta-label { font-size: 11px; text-transform: uppercase; opacity: 0.72; margin-bottom: 4px; }
.meta-value { font-size: 12px; word-break: break-word; }
.result { white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; min-height: 120px; }
.status { font-size: 12px; padding: 6px 0; }
.status.running { color: var(--vscode-textLink-foreground); }
.status.failed { color: var(--vscode-testing-iconFailed); }
.status.warn { color: var(--vscode-testing-iconQueued); }
.status-line { display: inline-flex; align-items: center; gap: 6px; }
.spinner { width: 12px; height: 12px; border: 2px solid currentColor; border-right-color: transparent; border-radius: 50%; display: inline-block; animation: spin 0.8s linear infinite; flex: 0 0 auto; }
@keyframes spin { to { transform: rotate(360deg); } }
button { border: 1px solid var(--vscode-button-border, transparent); background: var(--vscode-button-background); color: var(--vscode-button-foreground); border-radius: 6px; padding: 6px 12px; min-height: 32px; cursor: pointer; }
button.secondary { background: transparent; color: inherit; border-color: var(--vscode-panel-border); }
button[disabled] { opacity: 0.5; cursor: not-allowed; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; }
.reason { font-size: 12px; opacity: 0.82; }
.badge { display: inline-flex; align-items: center; min-height: 24px; border-radius: 999px; padding: 0 8px; border: 1px solid var(--vscode-panel-border); margin-left: 8px; font-size: 12px; }
.stale { border-color: color-mix(in srgb, var(--vscode-testing-iconQueued) 45%, var(--vscode-panel-border)); }
.stale-note { margin-top: 8px; font-size: 12px; opacity: 0.8; }
</style>
</head>
<body>
<div class="app">
  <div class="section">
    <div class="title">AI Analysis</div>
    <div id="status" class="status">Loading...</div>
    <div id="reason" class="reason"></div>
    <div class="meta" id="meta"></div>
  </div>
  <div class="section">
    <div class="title">Result <span id="staleBadge" class="badge stale" style="display:none;">Stale</span></div>
    <div id="result" class="result">(no result)</div>
    <div id="staleNote" class="stale-note" style="display:none;">This result is from a previous successful analysis.</div>
  </div>
  <div class="section">
    <div class="actions">
      <button id="copyResult" class="secondary" disabled>Copy Result</button>
      <button id="cancelTask" class="secondary" disabled>Cancel</button>
    </div>
  </div>
</div>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
const el = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');
let aiState = null;
const renderMeta = (state) => {
  if (!state || !state.source) {
    el('meta').innerHTML = '<div class="meta-item"><div class="meta-label">Source</div><div class="meta-value">(none)</div></div>';
    return;
  }
  const source = state.source;
  const items = [
    ['Model File', source.modelFile || ''],
    ['Model Path', source.modelPath || ''],
    ['Artifact ID', source.artifactId || ''],
    ['Graph ID', source.graphId || ''],
    ['Exporter ID', source.exporterId || ''],
    ['Analyzer ID', source.analyzerId || ''],
    ['Time', source.time || '']
  ];
  el('meta').innerHTML = items.map(([label, value]) => '<div class="meta-item"><div class="meta-label">' + escapeHtml(label) + '</div><div class="meta-value">' + escapeHtml(value || '(none)') + '</div></div>').join('');
};
const render = (state) => {
  aiState = state || {};
  const status = String(aiState.status || 'idle');
  const message = aiState.message || 'Ready.';
  el('status').innerHTML = status === 'running'
    ? '<span class="status-line"><span class="spinner" aria-hidden="true"></span><span>' + escapeHtml(message) + '</span></span>'
    : escapeHtml(message);
  el('status').className = 'status ' + (status === 'running' ? 'running' : status === 'failed' ? 'failed' : status === 'stale' ? 'warn' : '');
  el('reason').textContent = aiState.error && aiState.error.message ? aiState.error.message : '';
  renderMeta(aiState);
  const result = aiState.result;
  if (result && typeof result.text === 'string' && result.text.length > 0) {
    el('result').textContent = result.text;
    el('result').classList.remove('empty');
    el('copyResult').disabled = false;
    if (aiState.resultStale) {
      el('staleBadge').style.display = 'inline-flex';
      el('staleNote').style.display = 'block';
    } else {
      el('staleBadge').style.display = 'none';
      el('staleNote').style.display = 'none';
    }
  } else {
    el('result').textContent = status === 'running' ? '(running)' : '(no result)';
    el('copyResult').disabled = true;
    el('staleBadge').style.display = 'none';
    el('staleNote').style.display = 'none';
  }
  el('cancelTask').disabled = status !== 'running';
};
window.addEventListener('message', (event) => {
  const message = event.data;
  if (message && message.type === 'aiStateUpdate') {
    render(message.state);
  } else if (message && message.type === 'clipboardCopied') {
    el('reason').textContent = (message.label || 'Text') + ' copied to clipboard.';
  } else if (message && message.type === 'clipboardError') {
    el('reason').textContent = message.message || 'Clipboard operation failed.';
  }
});
el('copyResult').addEventListener('click', () => {
  if (aiState && aiState.result && typeof aiState.result.text === 'string') {
    vscode.postMessage({ type: 'copyText', text: aiState.result.text, label: 'AI Result' });
  }
});
el('cancelTask').addEventListener('click', () => vscode.postMessage({ type: 'cancelTask' }));
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
            enqueuePanelMessage(panelState, { type: 'compareStateUpdate', state: getCompareState() });
            enqueuePanelMessage(panelState, { type: 'activityLog', entries: panelState.activity });
            enqueuePanelMessage(panelState, { type: 'toolStateUpdate', state: getToolState() });
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
            const provider = getPanelProvider(panelState);
            if (typeof provider.createCropArtifact !== 'function') {
                throw new Error(`Provider '${provider.id}' does not support crop artifacts.`);
            }
            const artifact = await provider.createCropArtifact({
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
            const provider = getPanelProvider(panelState);
            const artifactId = message.artifactId || panelState.currentArtifactId;
            if (!artifactId) {
                throw new Error('No confirmed crop artifact available.');
            }
            if (typeof provider.exportArtifact !== 'function') {
                throw new Error(`Provider '${provider.id}' does not support artifact export.`);
            }
            const exportTarget = typeof provider.getExportTarget === 'function'
                ? provider.getExportTarget(artifactId, { weightMode: message.weightMode || '' })
                : {
                    artifactId,
                    defaultFileName: `${artifactId}.crop.bin`,
                    filters: { Model: ['bin'] },
                    title: 'Export Crop Artifact',
                    stage: 'Export artifact',
                    message: 'Exporting crop artifact...',
                    options: {}
                };
            const saveUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.joinPath(getDefaultFolder(), exportTarget.defaultFileName || `${artifactId}.crop.bin`),
                filters: exportTarget.filters || { Model: ['bin'] },
                title: exportTarget.title || 'Export Crop Artifact'
            });
            if (!saveUri) {
                break;
            }
            updatePanelTask(panelState, {
                status: 'running',
                stage: exportTarget.stage || 'Export artifact',
                message: exportTarget.message || 'Exporting crop artifact...',
                startedAt: new Date().toISOString(),
                cancellable: false,
                busy: true
            });
            const result = await provider.exportArtifact(exportTarget.artifactId || artifactId, saveUri.fsPath || saveUri.path, exportTarget.options || {});
            clearPanelTask(panelState);
            addPanelActivity(panelState, 'info', 'Crop artifact exported', result);
            enqueuePanelMessage(panelState, { type: 'artifactExported', exportInfo: result });
            vscode.window.showInformationMessage(`Crop artifact exported: ${result.filePath || (saveUri.fsPath || saveUri.path)}`);
            break;
        }
        case 'importInputFile': {
            const provider = getPanelProvider(panelState);
            if (typeof provider.importInputFile !== 'function') {
                throw new Error(`Provider '${provider.id}' does not support input import.`);
            }
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
            const imported = await provider.importInputFile(picked[0].fsPath || picked[0].path);
            enqueuePanelMessage(panelState, { type: 'inputImported', token: imported.token, preview: imported.preview });
            addPanelActivity(panelState, 'info', 'Input file imported', { preview: imported.preview });
            break;
        }
        case 'runInference': {
            const provider = getPanelProvider(panelState);
            if (typeof provider.runInference !== 'function') {
                throw new Error(`Provider '${provider.id}' does not support inference.`);
            }
            updatePanelTask(panelState, { status: 'running', stage: '执行推理', message: 'Running inference...', startedAt: new Date().toISOString(), cancellable: false, busy: true });
            const result = await provider.runInference({
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
            const provider = getPanelProvider(panelState);
            setCompareProvider(provider);
            if (typeof provider.getCompareSlot !== 'function') {
                throw new Error(`Provider '${provider.id}' does not support compare slots.`);
            }
            const slot = provider.getCompareSlot(message.artifactId || panelState.currentArtifactId);
            const compareState = assignHostCompareSlot(message.slot, slot);
            addPanelActivity(panelState, 'info', `Assigned artifact to compare slot ${message.slot}`, { artifactId: message.artifactId || panelState.currentArtifactId });
            broadcastCompareState(compareState);
            await focusCompareView(false);
            break;
        }
        case 'requestCompareState':
            enqueuePanelMessage(panelState, { type: 'compareStateUpdate', state: getCompareState() });
            break;
        case 'selectExporter':
            panelState.selectedExporterId = typeof message.id === 'string' ? message.id : null;
            enqueuePanelMessage(panelState, { type: 'toolStateUpdate', state: getToolState() });
            break;
        case 'selectFormatter':
            panelState.selectedFormatterId = typeof message.id === 'string' ? message.id : null;
            enqueuePanelMessage(panelState, { type: 'toolStateUpdate', state: getToolState() });
            break;
        case 'selectAnalyzer':
            panelState.selectedAnalyzerId = typeof message.id === 'string' ? message.id : null;
            enqueuePanelMessage(panelState, { type: 'toolStateUpdate', state: getToolState() });
            break;
        case 'copyExportText':
            await handleCopyExportText(panelState, message);
            break;
        case 'runAiAnalysis':
            await handleRunAiAnalysis(panelState, message);
            break;
        case 'cancelAiTask':
            cancelGlobalTask();
            break;
        case 'requestTensorPreview': {
            try {
                const provider = getPanelProvider(panelState);
                if (typeof provider.getTensorPreview !== 'function') {
                    throw new Error(`Provider '${provider.id}' does not support tensor preview.`);
                }
                const preview = await provider.getTensorPreview(message.sessionId || panelState.currentSessionId, message.tensorName, { limit: message.limit || 64 });
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
            clearActiveCompare();
            vscode.window.showInformationMessage('Netron Compare cleared.');
            break;
        case 'setCompareBinding':
            setHostCompareBinding(message.kind, message.sourceName, message.targetName);
            broadcastCompareState();
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
                const provider = getCompareProvider();
                const imported = await provider.importInputFile(picked[0].fsPath || picked[0].path);
                setImportedInput(state.compareState, imported, provider.id);
                broadcastCompareState();
            }
            break;
        }
        case 'runCompare': {
            const compareState = await runHostCompare({
                inputMode: message.inputMode || 'zeros',
                inputShapes: message.inputShapes || {}
            });
            broadcastCompareState(compareState);
            break;
        }
        case 'exportCompareJson': {
            const saveUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.joinPath(getDefaultFolder(), 'compare-result.json'),
                filters: { JSON: ['json'] },
                title: 'Export compare result as JSON'
            });
            if (saveUri) {
                await vscode.workspace.fs.writeFile(saveUri, Buffer.from(exportCompareResultAsJson(state.compareState), 'utf8'));
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
                await vscode.workspace.fs.writeFile(saveUri, Buffer.from(exportCompareResultAsCsv(state.compareState), 'utf8'));
            }
            break;
        }
        case 'exportCompareOutputNpy': {
            const exported = exportCompareOutputAsNpy(state.compareState, state.compareRawOutputs, {
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

async function handleAiPanelMessage(message) {
    if (!message || typeof message.type !== 'string') {
        return;
    }
    switch (message.type) {
        case 'ready':
            state.aiViewReady = true;
            pushAiState();
            break;
        case 'copyText':
            await handleCopyText(state.aiView, message);
            break;
        case 'cancelTask':
            cancelGlobalTask();
            break;
        default:
            break;
    }
}

function cancelGlobalTask() {
    if (!state.globalTask) {
        return;
    }
    state.globalTask.cancelRequested = true;
    state.globalTask.message = 'Cancelling...';
    if (state.globalTask.process && typeof state.globalTask.process.kill === 'function') {
        try {
            state.globalTask.process.kill('SIGTERM');
        } catch (error) {
            appendLog('warn', 'failed to kill global task process', { message: error.message });
        }
    }
    broadcastToolState();
    if (state.globalTask.kind === 'analysis') {
        updateAiState(analysisCancelling());
    }
}

function resolveReadyEntry(registry, id, label) {
    const entry = id ? registry.getEntry(id) : registry.getFirstReady();
    if (!entry) {
        throw new Error(`No ${label} available.`);
    }
    if (entry.status !== 'ready') {
        throw new Error(entry.reason || `Selected ${label} is not available.`);
    }
    return entry;
}

function getPanelProvider(panelState) {
    if (!panelState || !panelState.currentProviderId || !state.providerRegistry) {
        throw new Error('No host provider is active for this model.');
    }
    const provider = state.providerRegistry.get(panelState.currentProviderId);
    if (!provider) {
        throw new Error(`Active provider '${panelState.currentProviderId}' is not registered.`);
    }
    return provider;
}

function getCompareProvider() {
    if (!state.providerRegistry) {
        throw new Error('No provider registry is available.');
    }
    const providerId = state.compareProviderId || 'onnx';
    const provider = state.providerRegistry.get(providerId);
    if (!provider) {
        throw new Error(`Compare provider '${providerId}' is not registered.`);
    }
    return provider;
}

function setCompareProvider(provider) {
    if (!provider || !provider.id) {
        throw new Error('Compare provider is required.');
    }
    state.compareProviderId = provider.id;
}

function clearActiveCompare() {
    if (state.providerRegistry) {
        for (const provider of state.providerRegistry.list()) {
            if (provider && typeof provider.clearCompare === 'function') {
                provider.clearCompare();
            }
        }
    }
    state.compareProviderId = null;
    state.compareState = createEmptyCompareState();
    state.compareRawOutputs.clear();
    broadcastCompareState();
}

function getCompareState() {
    return cloneCompareState(state.compareState);
}

function assignHostCompareSlot(slotName, slot) {
    assignCompareSlot(state.compareState, slotName, slot);
    state.compareRawOutputs.clear();
    return getCompareState();
}

function setHostCompareBinding(kind, sourceName, targetName) {
    setCompareBinding(state.compareState, kind, sourceName, targetName);
    return getCompareState();
}

function resolveHostImportedInput() {
    const imported = state.compareState.importedInput;
    if (!imported || !imported.token) {
        return null;
    }
    const provider = state.providerRegistry && state.providerRegistry.get(imported.providerId || state.compareProviderId || 'onnx');
    if (!provider || typeof provider.resolveImportedInput !== 'function') {
        throw new Error('Imported compare input is not available.');
    }
    return provider.resolveImportedInput(imported.token);
}

async function runHostCompare(options = {}) {
    setCompareRunStatus(state.compareState, 'running', '校验/生成共享输入', '');
    broadcastCompareState();
    try {
        const result = await runCrossProviderCompare(state.compareState, state.providerRegistry, {
            inputMode: options.inputMode || 'zeros',
            inputShapes: options.inputShapes || {},
            importedInput: resolveHostImportedInput(),
            createRunId: () => createRequestId(),
            onStage: (stage) => {
                const label = stage === 'B' ? '执行 B' : '执行 A';
                setCompareRunStatus(state.compareState, 'running', label, '');
                broadcastCompareState();
            }
        });
        state.compareState = result.compareState;
        state.compareRawOutputs.set(state.compareState.compareResult.rawOutputRef, result.rawOutputs);
        return getCompareState();
    } catch (error) {
        setCompareRunStatus(state.compareState, 'failed', '', error && error.message ? error.message : String(error));
        broadcastCompareState();
        throw error;
    }
}

function buildSourceFromTarget(target, exporterEntry, analyzerEntry) {
    return {
        modelFile: target.model.fileName,
        modelPath: target.model.filePath,
        artifactId: target.artifact.id,
        graphId: target.graph.id,
        exporterId: exporterEntry && exporterEntry.id ? exporterEntry.id : '',
        analyzerId: analyzerEntry && analyzerEntry.id ? analyzerEntry.id : '',
        time: new Date().toISOString()
    };
}

function getTargetAndContextForPanelArtifact(panelState) {
    const artifactId = panelState.currentArtifactId;
    if (!artifactId) {
        throw new Error('No confirmed crop artifact available.');
    }
    const provider = getPanelProvider(panelState);
    if (typeof provider.getCropTarget !== 'function' || typeof provider.buildTextExportContext !== 'function') {
        throw new Error(`Provider '${provider.id}' does not support text export context.`);
    }
    const target = provider.getCropTarget(artifactId);
    const context = provider.buildTextExportContext(artifactId);
    return { target, context };
}

async function handleCopyExportText(panelState, message) {
    if (state.globalTask) {
        throw new Error('Another export/analysis task is running.');
    }
    const taskId = createRequestId();
    const exporterId = message.exporterId || panelState.selectedExporterId;
    const exporter = resolveReadyEntry(state.exporterRegistry, exporterId, 'exporter');
    const { target, context } = getTargetAndContextForPanelArtifact(panelState);
    setGlobalTask({
        id: taskId,
        kind: 'copy-export',
        status: 'running',
        message: 'Copying export text...',
        startedAt: new Date().toISOString(),
        sourcePanelId: panelState.id,
        process: null,
        cancelRequested: false
    });
    updatePanelTask(panelState, {
        status: 'running',
        stage: 'Copying export text',
        message: 'Copying export text...',
        startedAt: new Date().toISOString(),
        cancellable: false,
        busy: true
    });
    try {
        const result = await runTool(exporter, JSON.stringify(context, null, 2), {
            kind: 'exporter',
            label: 'Exporter',
            onProcess: (child) => {
                attachGlobalTaskProcess(taskId, child);
            }
        });
        await vscode.env.clipboard.writeText(result.stdout);
        addPanelActivity(panelState, 'info', 'Export text copied', {
            exporterId: exporter.id,
            artifactId: target.artifact.id,
            graphId: target.graph.id
        });
        enqueuePanelMessage(panelState, { type: 'exportTextCopied', exporterId: exporter.id });
        appendLog('info', 'export text copied', { exporterId: exporter.id, artifactId: target.artifact.id, stderr: result.stderr ? result.stderr.slice(0, 500) : '' });
    } catch (error) {
        addPanelActivity(panelState, 'error', 'Export text failed', {
            exporterId: exporter.id,
            artifactId: target.artifact.id,
            graphId: target.graph.id,
            message: error.message
        });
        enqueuePanelMessage(panelState, { type: 'exportTextError', message: error.message });
        appendLog('error', 'export text failed', { exporterId: exporter.id, message: error.message, stderr: error.stderr });
        throw error;
    } finally {
        clearPanelTask(panelState);
        clearGlobalTask(taskId);
    }
}

async function handleRunAiAnalysis(panelState, message) {
    if (state.globalTask) {
        throw new Error('Another export/analysis task is running.');
    }
    const taskId = createRequestId();
    const exporterId = message.exporterId || panelState.selectedFormatterId;
    const analyzerId = message.analyzerId || panelState.selectedAnalyzerId;
    const exporter = resolveReadyEntry(state.exporterRegistry, exporterId, 'formatter');
    const analyzer = resolveReadyEntry(state.analyzerRegistry, analyzerId, 'analyzer');
    const { target, context } = getTargetAndContextForPanelArtifact(panelState);
    const source = buildSourceFromTarget(target, exporter, analyzer);
    setGlobalTask({
        id: taskId,
        kind: 'analysis',
        status: 'running',
        message: 'Running analysis...',
        startedAt: new Date().toISOString(),
        sourcePanelId: panelState.id,
        process: null,
        cancelRequested: false
    });
    updateAiState(analysisStarted(source));
    updatePanelTask(panelState, {
        status: 'running',
        stage: 'AI Analysis',
        message: 'Running analysis...',
        startedAt: new Date().toISOString(),
        cancellable: true,
        busy: true
    });
    await focusAiView(false);
    let exportedText = '';
    let failedStage = 'exporter';
    try {
        const exportResult = await runTool(exporter, JSON.stringify(context, null, 2), {
            kind: 'exporter',
            label: 'Exporter',
            onProcess: (child) => {
                attachGlobalTaskProcess(taskId, child);
            }
        });
        exportedText = exportResult.stdout;
        failedStage = 'analyzer';
        const analysisResult = await runTool(analyzer, exportedText, {
            kind: 'analyzer',
            label: 'Analyzer',
            onProcess: (child) => {
                attachGlobalTaskProcess(taskId, child);
            }
        });
        updateAiState(analysisSucceeded(source, analysisResult.stdout, new Date().toISOString()));
        addPanelActivity(panelState, 'info', 'AI analysis completed', {
            exporterId: exporter.id,
            analyzerId: analyzer.id,
            artifactId: target.artifact.id,
            graphId: target.graph.id
        });
        enqueuePanelMessage(panelState, { type: 'aiAnalysisStatus', status: 'succeeded', message: 'Analysis completed.' });
        appendLog('info', 'ai analysis completed', { exporterId: exporter.id, analyzerId: analyzer.id, artifactId: target.artifact.id, stderr: analysisResult.stderr ? analysisResult.stderr.slice(0, 500) : '' });
    } catch (error) {
        const wasCancelled = state.globalTask && state.globalTask.id === taskId && state.globalTask.cancelRequested;
        updateAiState(analysisFailed(state.aiState, {
            cancelled: wasCancelled,
            source,
            message: error.message,
            stage: failedStage,
            stderr: error.stderr || ''
        }));
        addPanelActivity(panelState, wasCancelled ? 'warn' : 'error', wasCancelled ? 'AI analysis cancelled' : 'AI analysis failed', {
            exporterId: exporter.id,
            analyzerId: analyzer.id,
            artifactId: target.artifact.id,
            graphId: target.graph.id,
            stage: failedStage,
            message: error.message
        });
        enqueuePanelMessage(panelState, { type: 'aiAnalysisStatus', status: wasCancelled ? 'cancelled' : 'failed', message: error.message });
        appendLog(wasCancelled ? 'warn' : 'error', wasCancelled ? 'ai analysis cancelled' : 'ai analysis failed', { exporterId: exporter.id, analyzerId: analyzer.id, stage: failedStage, message: error.message, stderr: error.stderr });
        if (!wasCancelled) {
            throw error;
        }
    } finally {
        clearPanelTask(panelState);
        clearGlobalTask(taskId);
    }
}

async function openModelInPanel(panelState, modelUri, trigger) {
    panelState.currentModelUri = modelUri;
    panelState.currentArtifactId = null;
    panelState.currentProviderId = null;
    const fileName = path.basename(modelUri.fsPath || modelUri.path);
    panelState.panel.title = `Netron Preview: ${fileName}`;
    addPanelActivity(panelState, 'info', 'Open model requested', { file: fileName, trigger });

    const providerResult = state.providerRegistry ? state.providerRegistry.resolve(modelUri) : { ok: false };
    if (!providerResult.ok) {
        appendLog('info', 'no host provider for model, using legacy Netron load', { fileName, reason: providerResult.reason });
        return openLegacyModelInPanel(panelState, modelUri, trigger, providerResult.reason);
    }
    const provider = providerResult.provider;
    panelState.currentProviderId = provider.id;

    updatePanelTask(panelState, {
        status: 'running',
        stage: '读取文件',
        message: `Opening ${fileName}`,
        startedAt: new Date().toISOString(),
        cancellable: true,
        busy: true
    });

    const session = await provider.loadModel(modelUri, {
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
        model: { ...session.snapshot, sessionId: session.id },
        provider: providerInfo(provider),
        fileName,
        filePath: modelUri.fsPath || modelUri.path
    });
    addPanelActivity(panelState, 'info', 'Host-managed model render ready', { sessionId: session.id, providerId: provider.id });
}

async function openLegacyModelInPanel(panelState, modelUri, trigger, providerUnavailableReason = '') {
    const bytes = await vscode.workspace.fs.readFile(modelUri);
    const fileName = path.basename(modelUri.fsPath || modelUri.path);
    addPanelActivity(panelState, 'info', 'Legacy model load', { file: fileName, trigger, sizeBytes: bytes.byteLength });
    enqueuePanelMessage(panelState, {
        type: 'loadModel',
        requestId: createRequestId(),
        name: fileName,
        base64: Buffer.from(bytes).toString('base64'),
        sizeBytes: bytes.byteLength,
        sentAt: Date.now(),
        providerUnavailableReason: typeof providerUnavailableReason === 'string' ? providerUnavailableReason : ''
    });
}

function pushCompareState(snapshot) {
    state.pendingCompareState = snapshot || getCompareState();
    flushCompareState();
}

function broadcastCompareState(snapshot) {
    const compareState = snapshot || getCompareState();
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
    if (state.exporterRegistry) {
        state.exporterRegistry.stopWatching();
    }
    if (state.analyzerRegistry) {
        state.analyzerRegistry.stopWatching();
    }
    cancelGlobalTask();
}

module.exports = {
    activate,
    deactivate,
    registerFormatProvider,
    unregisterFormatProvider,
    getFormatProviders,
    getFormatProviderDiagnostics
};
