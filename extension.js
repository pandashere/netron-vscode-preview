const fs = require('fs');
const os = require('os');
const path = require('path');
const vscode = require('vscode');

const LOAD_WATCHDOG_MS = 20000;
const WEBVIEW_READY_TIMEOUT_MS = 8000;

const panelState = {
    panel: null,
    ready: false,
    pendingModel: null,
    context: null,
    output: null,
    inflight: new Map(),
    readyTimer: null,
    readyFailed: false,
    panelSeq: 0,
    panelId: null
};

function createRequestId() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function toNumber(value, fallback) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
    }
    return fallback;
}

function formatDetail(detail) {
    if (detail === undefined) {
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

function appendLog(kind, requestId, message, detail) {
    if (!panelState.output) {
        return;
    }
    const requestPart = `[requestId=${requestId || 'n/a'}]`;
    const detailText = formatDetail(detail);
    const suffix = detailText ? ` detail=${detailText}` : '';
    panelState.output.appendLine(`[NetronPreview]${requestPart} [${kind}] ${message}${suffix}`);
}

function clearInflightRequest(requestId) {
    if (!requestId) {
        return null;
    }
    const entry = panelState.inflight.get(requestId);
    if (!entry) {
        return null;
    }
    clearTimeout(entry.timer);
    panelState.inflight.delete(requestId);
    return entry;
}

function clearAllInflightRequests() {
    for (const [requestId, entry] of panelState.inflight.entries()) {
        clearTimeout(entry.timer);
        panelState.inflight.delete(requestId);
    }
}

function clearReadyWatchdog() {
    if (panelState.readyTimer) {
        clearTimeout(panelState.readyTimer);
        panelState.readyTimer = null;
    }
}

function scheduleReadyWatchdog(panel) {
    clearReadyWatchdog();
    panelState.readyTimer = setTimeout(() => {
        if (panelState.ready) {
            return;
        }
        panelState.readyFailed = true;
        appendLog('fail', null, `webview ready timeout after ${WEBVIEW_READY_TIMEOUT_MS}ms`, {
            status: 'webview-not-ready-timeout',
            panelId: panelState.panelId,
            pendingRequestId: panelState.pendingModel ? panelState.pendingModel.requestId || null : null,
            readyFailed: panelState.readyFailed
        });
        if (panelState.output) {
            panelState.output.show(true);
        }
        vscode.window.showWarningMessage('Netron Webview 初始化超时。请执行 Developer: Reload Window 后重试。');
        if (panel && panelState.panel === panel) {
            panel.webview.postMessage({ type: 'showWelcome', reason: 'webview-not-ready-timeout' }).then((ok) => {
                appendLog('stage', null, `showWelcome after ready-timeout posted=${ok}`);
            }).catch((error) => {
                appendLog('fail', null, 'showWelcome after ready-timeout failed', {
                    message: error instanceof Error ? error.message : String(error)
                });
            });
        }
    }, WEBVIEW_READY_TIMEOUT_MS);
}

function activate(context) {
    panelState.context = context;
    panelState.output = vscode.window.createOutputChannel('Netron Preview');
    context.subscriptions.push(panelState.output);
    appendLog('info', null, 'extension activated');

    const disposable = vscode.commands.registerCommand('netronPreview.openPreview', async (resource) => {
        try {
            const modelUri = await resolveModelUri(resource);
            if (!modelUri) {
                return;
            }
            const trigger = resource instanceof vscode.Uri ? 'explorer-context' : 'command-palette';
            const panel = ensurePanel(context, { forceRecreate: true });
            panel.reveal(vscode.ViewColumn.Beside, true);
            await openModel(panel, modelUri, trigger);
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            appendLog('fail', null, `command failed: ${message}`);
            vscode.window.showErrorMessage(`Netron preview failed: ${message}`);
        }
    });

    context.subscriptions.push(disposable);
}

async function resolveModelUri(resource) {
    // Always require explicit user selection instead of auto-opening
    // the current editor or provided resource URI.
    let defaultUri = getDefaultSaveFolder();
    if (resource instanceof vscode.Uri && resource.scheme === 'file') {
        defaultUri = resource;
    }

    const picked = await vscode.window.showOpenDialog({
        canSelectMany: false,
        canSelectFiles: true,
        canSelectFolders: false,
        defaultUri,
        openLabel: 'Preview Model',
        title: 'Select model file to preview'
    });

    return Array.isArray(picked) && picked.length > 0 ? picked[0] : null;
}

function ensurePanel(context, options) {
    const forceRecreate = !!(options && options.forceRecreate);
    if (panelState.panel) {
        if (forceRecreate) {
            appendLog('info', null, 'disposing existing webview panel for fresh reload');
            panelState.panel.dispose();
        } else if (panelState.readyFailed) {
            appendLog('info', null, 'disposing stale webview panel after ready-timeout');
            panelState.panel.dispose();
        } else {
            return panelState.panel;
        }
    }

    if (panelState.panel) {
        return panelState.panel;
    }

    const sourceRoot = vscode.Uri.joinPath(context.extensionUri, 'netron', 'source');
    const panel = vscode.window.createWebviewPanel(
        'netronPreview',
        'Netron Preview',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: false,
            localResourceRoots: [sourceRoot]
        }
    );
    panelState.panelSeq += 1;
    panelState.panelId = `panel-${panelState.panelSeq}`;

    panel.webview.html = buildWebviewHtml(context, panel.webview, sourceRoot);
    appendLog('stage', null, 'webview html injected', {
        hasEntryScript: panel.webview.html.includes('netron-vscode-entry'),
        panelId: panelState.panelId
    });
    panelState.readyFailed = false;
    scheduleReadyWatchdog(panel);

    panel.onDidDispose(() => {
        clearAllInflightRequests();
        clearReadyWatchdog();
        panelState.panel = null;
        panelState.ready = false;
        panelState.pendingModel = null;
        panelState.readyFailed = false;
        panelState.panelId = null;
    }, null, context.subscriptions);

    panel.webview.onDidReceiveMessage(async (message) => {
        await handleWebviewMessage(panel, message);
    }, null, context.subscriptions);

    panelState.panel = panel;
    return panel;
}

function buildWebviewHtml(context, webview, sourceRoot) {
    const indexFile = path.join(context.extensionPath, 'netron', 'source', 'index.html');
    let html = fs.readFileSync(indexFile, 'utf8');
    const csp = webview.cspSource;
    const baseHref = `${webview.asWebviewUri(sourceRoot).toString()}/`;
    const vscodeScriptUri = webview.asWebviewUri(vscode.Uri.joinPath(sourceRoot, 'vscode.js')).toString();

    const cspMeta = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${csp} data: blob:; style-src ${csp} 'unsafe-inline'; script-src ${csp}; worker-src ${csp} blob:; font-src ${csp}; connect-src ${csp};">`;

    html = html.replace(
        /<meta http-equiv="Content-Security-Policy"[^>]*>/i,
        cspMeta
    );

    const indexScriptPattern = /<script[^>]*src=["'](?:\.\/)?index\.js["'][^>]*><\/script>/i;
    if (indexScriptPattern.test(html)) {
        html = html.replace(
            indexScriptPattern,
            `<script id="netron-vscode-entry" type="text/javascript" src="${vscodeScriptUri}"></script>`
        );
    } else if (!/src=["'][^"']*vscode\.js["']/i.test(html)) {
        html = html.replace(
            '</body>',
            `    <script id="netron-vscode-entry" type="text/javascript" src="${vscodeScriptUri}"></script>\n</body>`
        );
    }

    if (html.includes('<head>')) {
        html = html.replace('<head>', `<head>\n<base href="${baseHref}">`);
    }

    return html;
}

async function openModel(panel, modelUri, trigger) {
    const bytes = await vscode.workspace.fs.readFile(modelUri);
    const fileName = path.basename(modelUri.fsPath || modelUri.path);
    panel.title = `Netron Preview: ${fileName}`;
    const requestId = createRequestId();
    const startedAt = Date.now();
    const sizeBytes = bytes.byteLength;

    const payload = {
        type: 'loadModel',
        requestId,
        name: fileName,
        base64: Buffer.from(bytes).toString('base64'),
        sizeBytes,
        sentAt: startedAt
    };

    const timer = setTimeout(() => {
        const active = clearInflightRequest(requestId);
        if (!active) {
            return;
        }
        const elapsedMs = Date.now() - active.startedAt;
        appendLog('fail', requestId, `watchdog timeout after ${elapsedMs}ms`, {
            file: active.fileName,
            status: 'timeout',
            elapsedMs
        });
        if (panelState.output) {
            panelState.output.show(true);
        }
        vscode.window.showWarningMessage(`Netron 预览加载超时（${active.fileName}）。请重试或查看 Output -> Netron Preview。`);
        if (panelState.panel === panel && panelState.ready) {
            panel.webview.postMessage({
                type: 'showWelcome',
                reason: 'load-watchdog-timeout',
                requestId
            }).then((ok) => {
                appendLog('stage', requestId, `showWelcome after load-timeout posted=${ok}`);
            }).catch((error) => {
                appendLog('fail', requestId, 'showWelcome after load-timeout failed', {
                    message: error instanceof Error ? error.message : String(error)
                });
            });
        }
    }, LOAD_WATCHDOG_MS);

    panelState.inflight.set(requestId, {
        timer,
        startedAt,
        fileName,
        trigger: trigger || 'unknown'
    });
    appendLog('stage', requestId, 'enqueue loadModel', {
        file: fileName,
        trigger: trigger || 'unknown',
        uri: modelUri.toString(),
        sizeKB: Number((sizeBytes / 1024).toFixed(2))
    });

    if (panelState.ready) {
        const posted = await panel.webview.postMessage(payload);
        appendLog('stage', requestId, `loadModel posted=${posted}`, {
            route: 'direct'
        });
        if (!posted) {
            clearInflightRequest(requestId);
            appendLog('fail', requestId, 'postMessage failed while webview is ready', {
                status: 'post-message-failed'
            });
            vscode.window.showWarningMessage('Netron preview: message dispatch failed, please reload window and retry.');
        }
    } else {
        if (panelState.pendingModel && panelState.pendingModel.requestId) {
            const previous = clearInflightRequest(panelState.pendingModel.requestId);
            if (previous) {
                appendLog('fail', panelState.pendingModel.requestId, 'replaced by newer pending request before webview ready', {
                    status: 'replaced'
                });
            }
        }
        panelState.pendingModel = payload;
    }
}

async function handleWebviewMessage(panel, message) {
    if (!message || typeof message.type !== 'string') {
        return;
    }

    switch (message.type) {
        case 'ready': {
            panelState.ready = true;
            panelState.readyFailed = false;
            clearReadyWatchdog();
            appendLog('info', null, 'webview ready received');
            if (panelState.pendingModel) {
                const posted = await panel.webview.postMessage(panelState.pendingModel);
                appendLog('stage', panelState.pendingModel.requestId || null, `pending loadModel posted=${posted}`, {
                    route: 'pending'
                });
                if (!posted && panelState.pendingModel.requestId) {
                    clearInflightRequest(panelState.pendingModel.requestId);
                    appendLog('fail', panelState.pendingModel.requestId, 'pending postMessage failed after ready', {
                        status: 'post-message-failed'
                    });
                }
                panelState.pendingModel = null;
            }
            break;
        }
        case 'pong': {
            appendLog('stage', null, 'webview pong received', message);
            break;
        }
        case 'requestOpenModel': {
            const uri = await resolveModelUri(null);
            if (uri) {
                await openModel(panel, uri, 'webview-open');
            }
            break;
        }
        case 'modelOpenStage': {
            const requestId = typeof message.requestId === 'string' ? message.requestId : null;
            const stage = typeof message.stage === 'string' ? message.stage : 'unknown-stage';
            const elapsedMs = toNumber(message.elapsedMs, -1);
            const text = elapsedMs >= 0 ? `${stage} (${elapsedMs}ms)` : stage;
            appendLog('stage', requestId, text, {
                file: message.name,
                detail: message.detail
            });
            break;
        }
        case 'modelOpened': {
            const requestId = typeof message.requestId === 'string' ? message.requestId : null;
            const entry = clearInflightRequest(requestId);
            const elapsedMs = entry
                ? Date.now() - entry.startedAt
                : toNumber(message.elapsedMs, -1);
            appendLog('success', requestId, `model opened${elapsedMs >= 0 ? ` (${elapsedMs}ms)` : ''}`, {
                file: message.name,
                status: message.status || ''
            });
            break;
        }
        case 'modelOpenFailed': {
            const requestId = typeof message.requestId === 'string' ? message.requestId : null;
            const entry = clearInflightRequest(requestId);
            const elapsedMs = entry
                ? Date.now() - entry.startedAt
                : toNumber(message.elapsedMs, -1);
            const status = typeof message.status === 'string' ? message.status : 'unknown';
            const text = typeof message.message === 'string' && message.message.length > 0
                ? message.message
                : 'Model open failed.';
            appendLog('fail', requestId, `model open failed${elapsedMs >= 0 ? ` (${elapsedMs}ms)` : ''}`, {
                file: message.name,
                status,
                message: text,
                detail: message.detail
            });
            if (panelState.output) {
                panelState.output.show(true);
            }
            if (status === 'context-open-failed' || status === 'unsupported-file' || status.includes('timeout')) {
                vscode.window.showWarningMessage(`Netron preview: ${text}`);
            } else {
                vscode.window.showErrorMessage(`Netron preview: ${text}`);
            }
            break;
        }
        case 'modelOpenLog': {
            const requestId = typeof message.requestId === 'string' ? message.requestId : null;
            const level = typeof message.level === 'string' ? message.level : 'info';
            const text = typeof message.message === 'string' ? message.message : '';
            if (text) {
                appendLog(level, requestId, text, message.detail);
            }
            const isFatalBeforeReady = !panelState.ready &&
                level === 'error' &&
                (text.includes('bootstrap failed') || text.includes('terminate before ready'));
            if (isFatalBeforeReady) {
                clearReadyWatchdog();
                panelState.readyFailed = true;
                const pending = panelState.pendingModel;
                if (pending && pending.requestId) {
                    const entry = clearInflightRequest(pending.requestId);
                    const elapsedMs = entry ? Date.now() - entry.startedAt : -1;
                    appendLog('fail', pending.requestId, 'terminated before ready due to bootstrap failure', {
                        status: 'webview-bootstrap-failed',
                        panelId: panelState.panelId,
                        elapsedMs: elapsedMs >= 0 ? elapsedMs : undefined,
                        reason: text,
                        detail: message.detail
                    });
                }
                clearAllInflightRequests();
                panelState.pendingModel = null;
                if (panelState.output) {
                    panelState.output.show(true);
                }
                const reason = (message.detail && typeof message.detail.message === 'string' && message.detail.message.length > 0)
                    ? message.detail.message
                    : text;
                vscode.window.showErrorMessage(`Netron preview bootstrap failed: ${reason}`);
            }
            break;
        }
        case 'saveFile': {
            await handleSaveFile(panel, message);
            break;
        }
        case 'copyText': {
            await handleCopyText(panel, message);
            break;
        }
        case 'readBundledText': {
            await handleReadBundledText(panel, message);
            break;
        }
        case 'openExternal': {
            if (typeof message.url === 'string' && message.url.length > 0) {
                try {
                    await vscode.env.openExternal(vscode.Uri.parse(message.url));
                } catch {
                    vscode.window.showWarningMessage(`Unable to open URL: ${message.url}`);
                }
            }
            break;
        }
        case 'notify': {
            const text = typeof message.message === 'string' ? message.message : '';
            if (!text) {
                break;
            }
            if (message.level === 'error' || message.level === 'warn') {
                appendLog('notify', null, `${message.level}: ${text}`);
            }
            if (message.level === 'error') {
                vscode.window.showErrorMessage(text);
            } else if (message.level === 'warn') {
                vscode.window.showWarningMessage(text);
            } else {
                vscode.window.showInformationMessage(text);
            }
            break;
        }
        default:
            break;
    }
}

async function handleSaveFile(panel, message) {
    const fileName = typeof message.fileName === 'string' && message.fileName.length > 0
        ? message.fileName
        : 'output.bin';

    const defaultFolder = getDefaultSaveFolder();
    const defaultUri = vscode.Uri.joinPath(defaultFolder, fileName);

    const saveUri = await vscode.window.showSaveDialog({
        defaultUri,
        title: 'Save generated file',
        filters: normalizeFilters(message.filters)
    });

    if (!saveUri) {
        return;
    }

    let content;
    if (typeof message.base64 === 'string') {
        content = Buffer.from(message.base64, 'base64');
    } else if (typeof message.text === 'string') {
        content = Buffer.from(message.text, 'utf8');
    } else {
        throw new Error('Invalid save payload.');
    }

    await vscode.workspace.fs.writeFile(saveUri, new Uint8Array(content));
    await panel.webview.postMessage({
        type: 'fileSaved',
        path: saveUri.fsPath || saveUri.path
    });
}

async function handleCopyText(panel, message) {
    const text = typeof message.text === 'string' ? message.text : '';
    const label = typeof message.label === 'string' && message.label.length > 0 ? message.label : 'Text';
    if (!text) {
        await panel.webview.postMessage({
            type: 'clipboardError',
            label,
            message: 'No text content to copy.'
        });
        return;
    }
    try {
        await vscode.env.clipboard.writeText(text);
        await panel.webview.postMessage({
            type: 'clipboardCopied',
            label
        });
    } catch (error) {
        await panel.webview.postMessage({
            type: 'clipboardError',
            label,
            message: error instanceof Error ? error.message : String(error)
        });
    }
}

function isBundledMetadataFileName(file) {
    return typeof file === 'string' && /^[a-z0-9._-]+-metadata\.json$/i.test(file);
}

async function handleReadBundledText(panel, message) {
    const requestId = typeof message.requestId === 'string' && message.requestId.length > 0
        ? message.requestId
        : null;
    const file = typeof message.file === 'string' ? message.file : '';

    if (!requestId) {
        return;
    }
    if (!isBundledMetadataFileName(file)) {
        appendLog('warn', requestId, 'invalid bundled metadata request', { file });
        await panel.webview.postMessage({
            type: 'readBundledTextResult',
            requestId,
            ok: false,
            error: 'Invalid metadata file name.'
        });
        return;
    }

    try {
        if (!panelState.context || !panelState.context.extensionUri) {
            throw new Error('Extension context is unavailable.');
        }
        const sourceRoot = vscode.Uri.joinPath(panelState.context.extensionUri, 'netron', 'source');
        const target = vscode.Uri.joinPath(sourceRoot, file);
        const bytes = await vscode.workspace.fs.readFile(target);
        const text = Buffer.from(bytes).toString('utf8');
        appendLog('stage', requestId, 'bundled metadata fallback loaded', { file });
        await panel.webview.postMessage({
            type: 'readBundledTextResult',
            requestId,
            ok: true,
            text
        });
    } catch (error) {
        const errorText = error instanceof Error ? error.message : String(error);
        appendLog('warn', requestId, 'bundled metadata fallback failed', {
            file,
            error: errorText
        });
        await panel.webview.postMessage({
            type: 'readBundledTextResult',
            requestId,
            ok: false,
            error: errorText
        });
    }
}

function getDefaultSaveFolder() {
    const folder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0
        ? vscode.workspace.workspaceFolders[0].uri
        : vscode.Uri.file(os.homedir());
    return folder;
}

function normalizeFilters(value) {
    if (!value || typeof value !== 'object') {
        return undefined;
    }
    const result = {};
    for (const [key, extensions] of Object.entries(value)) {
        if (!Array.isArray(extensions)) {
            continue;
        }
        const items = extensions
            .filter((item) => typeof item === 'string' && item.length > 0)
            .map((item) => item.replace(/^\./, ''));
        if (items.length > 0) {
            result[key] = items;
        }
    }
    return Object.keys(result).length > 0 ? result : undefined;
}

function deactivate() {
    clearAllInflightRequests();
    clearReadyWatchdog();
}

module.exports = {
    activate,
    deactivate
};
