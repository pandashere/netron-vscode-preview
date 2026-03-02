/* eslint-disable no-var */

window.exports = {};

window.__netron_vscode_base__ = (function() {
    try {
        var document = window.document;
        var entry = document.currentScript || document.getElementById('netron-vscode-entry');
        if (entry && typeof entry.getAttribute === 'function') {
            var src = entry.getAttribute('src');
            if (src && src.length > 0) {
                var absolute = new window.URL(src, window.location.href);
                return new window.URL('.', absolute).toString();
            }
        }
    } catch {
        // continue regardless of error
    }
    return '';
})();

window.exports.require = function(id, callback) {
    if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
        throw new Error("Invalid module '" + id + "'.");
    }
    var document = window.document;
    var base = window.__netron_vscode_base__ || document.baseURI || window.location.href || '';
    base = base.split('?')[0].split('#')[0];
    var index = base.lastIndexOf('/');
    base = index > 0 ? base.substring(0, index + 1) : base;
    base = base.lastIndexOf('/') === base.length - 1 ? base : base + '/';
    var url = base + id + '.js';
    var scripts = document.head.getElementsByTagName('script');
    for (var i = 0; i < scripts.length; i++) {
        if (url === scripts[i].getAttribute('src')) {
            throw new Error("Duplicate import of '" + url + "'.");
        }
    }
    var script = document.createElement('script');
    script.setAttribute('id', id);
    script.setAttribute('type', 'module');
    var loadHandler = function() {
        script.removeEventListener('load', loadHandler);
        script.removeEventListener('error', errorHandler);
        callback();
    };
    var errorHandler = function(e) {
        script.removeEventListener('load', loadHandler);
        script.removeEventListener('error', errorHandler);
        callback(null, new Error("The script '" + e.target.src + "' failed to load."));
    };
    script.addEventListener('load', loadHandler, false);
    script.addEventListener('error', errorHandler, false);
    script.setAttribute('src', url);
    document.head.appendChild(script);
};

window.exports.preload = function(callback) {
    var modules = [
        ['view'],
        ['json', 'xml', 'protobuf', 'hdf5', 'grapher', 'browser'],
        ['base', 'text', 'flatbuffers', 'flexbuffers', 'zip', 'tar', 'python']
    ];
    var next = function() {
        if (modules.length === 0) {
            callback();
        } else {
            var ids = modules.pop();
            var resolved = ids.length;
            for (var i = 0; i < ids.length; i++) {
                window.exports.require(ids[i], function(module, error) {
                    if (error) {
                        callback(null, error);
                    } else {
                        resolved--;
                        if (resolved === 0) {
                            next();
                        }
                    }
                });
            }
        }
    };
    next();
};

window.exports.terminate = function(message) {
    if (window.__netron_vscode_api__) {
        try {
            window.__netron_vscode_api__.postMessage({
                type: 'modelOpenLog',
                level: 'error',
                message: 'terminate before ready',
                detail: {
                    message
                }
            });
        } catch {
            // continue regardless of error
        }
    }
    var document = window.document;
    var text = document.getElementById('message-text');
    if (text) {
        text.innerText = message;
    }
    var button = document.getElementById('message-button');
    if (button) {
        button.style.display = 'none';
        button.onclick = null;
    }
    document.body.setAttribute('class', 'welcome message');
    if (window.__view__) {
        try {
            window.__view__.show('welcome message');
        } catch {
            // continue regardless of error
        }
    }
};

window.addEventListener('error', function(event) {
    var error = event instanceof window.ErrorEvent && event.error && event.error instanceof Error ? event.error : new Error(event && event.message ? event.message : JSON.stringify(event));
    window.exports.terminate(error.message);
});

window.addEventListener('unhandledrejection', function(event) {
    var reason = event && event.reason ? event.reason : 'Unhandled promise rejection';
    var message = reason instanceof Error ? reason.message : String(reason);
    window.exports.terminate(message);
});

(function() {
    const isVSCode = typeof window.acquireVsCodeApi === 'function';
    const vscodeApi = isVSCode ? window.acquireVsCodeApi() : null;
    window.__netron_vscode_api__ = vscodeApi;

    const decodeBase64 = (base64) => {
        const binary = window.atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
    };

    const encodeBase64 = (bytes) => {
        let binary = '';
        const chunk = 0x8000;
        for (let i = 0; i < bytes.length; i += chunk) {
            const part = bytes.subarray(i, Math.min(i + chunk, bytes.length));
            binary += String.fromCharCode.apply(null, part);
        }
        return window.btoa(binary);
    };

    const blobToBase64 = async (blob) => {
        const buffer = await blob.arrayBuffer();
        return encodeBase64(new Uint8Array(buffer));
    };

    const sanitizeFileName = (name, fallback) => {
        const base = (name || fallback || 'model').replace(/[\\/:*?"<>|]+/g, '_').trim();
        return base.length > 0 ? base : (fallback || 'model');
    };

    const safeNumber = (value) => {
        if (typeof value === 'number') {
            if (Number.isNaN(value)) {
                return 'NaN';
            }
            if (!Number.isFinite(value)) {
                return value > 0 ? 'Infinity' : '-Infinity';
            }
        }
        return value;
    };

    const toSerializable = (value, seen, depth) => {
        if (value === null || value === undefined) {
            return value;
        }
        if (depth > 20) {
            return '[DepthLimit]';
        }
        if (typeof value === 'bigint') {
            return value.toString();
        }
        if (typeof value === 'number') {
            return safeNumber(value);
        }
        if (typeof value === 'string' || typeof value === 'boolean') {
            return value;
        }
        if (Array.isArray(value)) {
            return value.map((item) => toSerializable(item, seen, depth + 1));
        }
        if (value instanceof Uint8Array) {
            return {
                type: 'Uint8Array',
                base64: encodeBase64(value)
            };
        }
        if (value instanceof Int8Array ||
            value instanceof Uint16Array ||
            value instanceof Int16Array ||
            value instanceof Uint32Array ||
            value instanceof Int32Array ||
            value instanceof Float32Array ||
            value instanceof Float64Array) {
            return {
                type: value.constructor.name,
                data: Array.from(value)
            };
        }
        if (typeof value === 'object') {
            if (seen.has(value)) {
                return '[Circular]';
            }
            seen.add(value);
            const result = {};
            for (const key of Object.keys(value)) {
                const item = value[key];
                if (typeof item === 'function') {
                    continue;
                }
                result[key] = toSerializable(item, seen, depth + 1);
            }
            seen.delete(value);
            return result;
        }
        return String(value);
    };

    const CONTEXT_OPEN_TIMEOUT_MS = 8000;
    const OPEN_CONTEXT_TIMEOUT_MS = 15000;

    const postModelLog = (level, message, detail) => {
        if (!vscodeApi) {
            return;
        }
        try {
            vscodeApi.postMessage({
                type: 'modelOpenLog',
                level,
                message,
                detail
            });
        } catch {
            // continue regardless of error
        }
    };

    const createVSCodeHostClass = () => {
        const BrowserHost = window.exports.browser && window.exports.browser.Host;
        if (!BrowserHost) {
            throw new Error('Browser host module is not initialized.');
        }
        return class VSCodeHost extends BrowserHost {

        constructor() {
            super();
            this._vscode = vscodeApi;
            this._pendingModel = null;
            this._captureToMemory = false;
            this._memory = {
                screenshot: null
            };
            this._environment.name = 'Netron VSCode Preview';
            this._environment.type = 'VSCode';
            this._environment.menu = true;
            this._environment.serial = true;
            this._environment.packaged = false;
            this._environment.repository = 'https://github.com/lutzroeder/netron';
            this._window.addEventListener('message', (event) => this._handleMessage(event.data));
        }

        async view(view) {
            this._view = view;
        }

        async start() {
            const document = this.document;
            document.addEventListener('dragover', (e) => {
                e.preventDefault();
            });
            document.addEventListener('drop', (e) => {
                e.preventDefault();
            });
            const openFileButton = this._element('open-file-button');
            if (openFileButton) {
                openFileButton.addEventListener('click', () => {
                    this.execute('open');
                });
            }
            this._view.show('welcome');
            this._post({ type: 'ready' });
            this._post({
                type: 'modelOpenLog',
                level: 'info',
                message: 'webview host started',
                detail: {
                    phase: 'runtime',
                    serial: true
                }
            });
            this._post({
                type: 'modelOpenLog',
                level: 'info',
                message: 'worker disabled in VSCode webview; using serial layout',
                detail: {
                    phase: 'bootstrap',
                    serial: true
                }
            });
            if (this._pendingModel) {
                const payload = this._pendingModel;
                this._pendingModel = null;
                await this._openModelPayload(payload);
            }
        }

        openURL(url) {
            this._post({ type: 'openExternal', url });
        }

        async execute(name) {
            switch (name) {
                case 'open':
                    this._post({ type: 'requestOpenModel' });
                    break;
                case 'report-issue':
                    this.openURL(`${this.environment('repository')}/issues/new`);
                    break;
                case 'about':
                    await this.message('Netron VSCode Preview', false, 'OK');
                    break;
                default:
                    await super.execute(name);
                    break;
            }
        }

        async export(file, blob) {
            if (this._captureToMemory) {
                this._memory.screenshot = { file, blob };
                this._window.dispatchEvent(new this._window.CustomEvent('nnjs:screenshot-cached'));
                return;
            }
            await this.saveBlobFile(file, blob, { Image: ['png', 'svg'] });
        }

        async saveBlobFile(fileName, blob, filters) {
            const base64 = await blobToBase64(blob);
            this._post({
                type: 'saveFile',
                fileName,
                base64,
                mimeType: blob.type || 'application/octet-stream',
                filters
            });
        }

        async saveTextFile(fileName, text, filters) {
            this._post({
                type: 'saveFile',
                fileName,
                text,
                filters
            });
        }

        async copyTextToClipboard(text, label) {
            this._post({
                type: 'copyText',
                text: typeof text === 'string' ? text : String(text),
                label: label || 'JSON'
            });
        }

        async captureScreenshotToMemory(fileName) {
            this._captureToMemory = true;
            try {
                await this._view.export(fileName || 'crop-preview.png');
            } finally {
                this._captureToMemory = false;
            }
        }

        async saveCachedScreenshot(fileName) {
            if (!this._memory.screenshot || !this._memory.screenshot.blob) {
                throw new Error('No cached screenshot available.');
            }
            const target = sanitizeFileName(fileName || this._memory.screenshot.file || 'crop-preview.png', 'crop-preview.png');
            await this.saveBlobFile(target, this._memory.screenshot.blob, { Image: ['png'] });
        }

        async copyCachedScreenshot() {
            if (!this._memory.screenshot || !this._memory.screenshot.blob) {
                throw new Error('No cached screenshot available.');
            }
            if (!this._window.ClipboardItem || !this._window.navigator.clipboard || !this._window.navigator.clipboard.write) {
                throw new Error('Clipboard image API is not available in this VS Code Webview runtime.');
            }
            const blob = this._memory.screenshot.blob;
            const type = blob.type || 'image/png';
            await this._window.navigator.clipboard.write([
                new this._window.ClipboardItem({
                    [type]: blob
                })
            ]);
        }

        hasCachedScreenshot() {
            return !!(this._memory.screenshot && this._memory.screenshot.blob);
        }

        notify(level, message) {
            this._post({ type: 'notify', level, message });
        }

        _emitStage(requestId, name, stage, t0, detail) {
            const elapsedMs = Math.max(0, Date.now() - t0);
            this._post({
                type: 'modelOpenStage',
                requestId,
                name,
                stage,
                elapsedMs,
                detail
            });
        }

        _emitFail(requestId, name, status, message, elapsedMs, detail) {
            this._post({
                type: 'modelOpenFailed',
                requestId,
                name,
                status,
                message,
                elapsedMs,
                detail
            });
        }

        async _withTimeout(promise, timeoutMs, status, message) {
            let timeoutId = null;
            try {
                return await Promise.race([
                    promise,
                    new Promise((_, reject) => {
                        timeoutId = this._window.setTimeout(() => {
                            const error = new Error(message);
                            error.status = status;
                            reject(error);
                        }, timeoutMs);
                    })
                ]);
            } finally {
                if (timeoutId !== null) {
                    this._window.clearTimeout(timeoutId);
                }
            }
        }

        async _openModelPayload(payload) {
            if (!payload || typeof payload.base64 !== 'string' || typeof payload.name !== 'string') {
                return;
            }
            const requestId = typeof payload.requestId === 'string' && payload.requestId.length > 0
                ? payload.requestId
                : `unknown-${Date.now()}`;
            const startAt = typeof payload.sentAt === 'number' && Number.isFinite(payload.sentAt)
                ? payload.sentAt
                : Date.now();

            this._emitStage(requestId, payload.name, 'received', startAt, {
                sizeBytes: typeof payload.sizeBytes === 'number' ? payload.sizeBytes : undefined
            });
            this._view.show('welcome spinner');
            try {
                this._emitStage(requestId, payload.name, 'decode_base64_start', startAt);
                const bytes = decodeBase64(payload.base64);
                this._emitStage(requestId, payload.name, 'decode_base64_end', startAt, {
                    sizeBytes: bytes.length
                });
                const file = new this._window.File([bytes], payload.name, { type: 'application/octet-stream' });
                const context = new window.exports.browser.BrowserFileContext(this, file, [file]);

                this._emitStage(requestId, payload.name, 'context_open_start', startAt);
                await this._withTimeout(
                    context.open(),
                    CONTEXT_OPEN_TIMEOUT_MS,
                    'context-open-timeout',
                    'Timed out while preparing model context.'
                );
                this._emitStage(requestId, payload.name, 'context_open_end', startAt);

                this._emitStage(requestId, payload.name, 'open_context_start', startAt);
                const status = await this._withTimeout(
                    this._openContext(context),
                    OPEN_CONTEXT_TIMEOUT_MS,
                    'open-context-timeout',
                    'Timed out while opening model context.'
                );
                this._emitStage(requestId, payload.name, 'open_context_end', startAt, {
                    status
                });

                if (status === '' || status === 'context-open-attachment') {
                    this._window.dispatchEvent(new this._window.CustomEvent('nnjs:model-opened'));
                    this._post({
                        type: 'modelOpened',
                        requestId,
                        name: payload.name,
                        status,
                        elapsedMs: Math.max(0, Date.now() - startAt)
                    });
                    return;
                }

                this._view.show('welcome');
                const statusText = typeof status === 'string' && status.length > 0 ? status : 'unknown-status';
                const message = statusText === 'context-open-failed'
                    ? 'Unsupported model file or failed to open model context.'
                    : statusText === 'context-open-error'
                        ? 'Model open failed with parser or render error.'
                        : `Model open did not complete successfully (${statusText}).`;
                this._emitFail(
                    requestId,
                    payload.name,
                    statusText,
                    message,
                    Math.max(0, Date.now() - startAt),
                    { status: statusText }
                );
            } catch (error) {
                const status = error && typeof error.status === 'string' ? error.status : 'context-open-error';
                const message = error instanceof Error ? error.message : String(error);
                this._emitFail(
                    requestId,
                    payload.name,
                    status,
                    message,
                    Math.max(0, Date.now() - startAt)
                );
                try {
                    const displayError = error instanceof Error ? error : new Error(message);
                    await this._view.error(displayError, 'Model open failed.');
                } catch {
                    // continue regardless of error
                }
                this._view.show('welcome');
            }
        }

        _handleMessage(message) {
            if (!message || typeof message.type !== 'string') {
                return;
            }
            if (message.type === 'showWelcome') {
                if (this._view) {
                    this._view.show('welcome');
                }
                return;
            }
            if (message.type === 'ping') {
                this._post({ type: 'pong', ts: Date.now() });
                return;
            }
            if (message.type === 'loadModel') {
                if (this._view) {
                    this._openModelPayload(message).catch((error) => {
                        this.exception(error, false);
                    });
                } else {
                    this._pendingModel = message;
                }
            }
        }

        _post(message) {
            if (this._vscode) {
                this._vscode.postMessage(message);
            }
        }
        };
    };

    class NNJSToolbar {

        constructor(host, view) {
            this.host = host;
            this.view = view;
            this.mode = null;
            this.startKeys = new Set();
            this.endKeys = new Set();
            this.graphIndex = null;
            this.nnjsCache = null;
            this.croppedCache = null;
            this._elements = {};
        }

        attach() {
            this._injectStyle();
            this._createPanel();
            this._bindEvents();
            this._setStatus('Load a model to enable NNJS and crop tools.');
        }

        _injectStyle() {
            const document = this.host.document;
            const style = document.createElement('style');
            style.textContent = `
            #nnjs-overlay { position: fixed; top: 72px; left: 8px; z-index: 10; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; pointer-events: none; }
            #nnjs-overlay .nnjs-rail { pointer-events: auto; width: 32px; border: 1px solid #cfcfcf; border-radius: 10px; background: rgba(245, 245, 245, 0.96); box-shadow: 0 6px 14px rgba(0,0,0,0.12); display: flex; flex-direction: column; align-items: center; padding: 6px 3px; gap: 6px; }
            #nnjs-overlay .nnjs-rail button { width: 26px; height: 26px; border: 1px solid #777; background: #fff; color: #222; border-radius: 6px; font-size: 11px; line-height: 1; cursor: pointer; padding: 0; }
            #nnjs-overlay .nnjs-rail button:hover { background: #efefef; }
            #nnjs-overlay .nnjs-drawer { pointer-events: auto; position: absolute; top: 0; left: 42px; width: 272px; height: min(74vh, 540px); max-height: calc(100vh - 88px); background: rgba(245, 245, 245, 0.97); color: #111; border: 1px solid #cfcfcf; border-radius: 10px; box-shadow: 0 8px 18px rgba(0,0,0,0.12); overflow: hidden; display: none; flex-direction: column; }
            #nnjs-overlay.expanded .nnjs-drawer { display: flex; }
            #nnjs-overlay .nnjs-header { display: flex; align-items: center; justify-content: space-between; padding: 8px 10px; background: #1d1d1d; color: #fff; font-size: 12px; font-weight: 600; letter-spacing: 0.3px; }
            #nnjs-overlay .nnjs-header button { border: 1px solid #888; background: #2a2a2a; color: #fff; border-radius: 6px; padding: 2px 8px; font-size: 11px; cursor: pointer; }
            #nnjs-overlay .nnjs-body { padding: 10px 10px 20px 10px; display: flex; flex-direction: column; gap: 8px; flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden; scrollbar-gutter: stable; }
            #nnjs-overlay .nnjs-row { display: flex; gap: 6px; flex-wrap: wrap; }
            #nnjs-overlay .nnjs-body button { border: 1px solid #777; background: #fff; color: #222; border-radius: 6px; padding: 4px 8px; font-size: 12px; cursor: pointer; }
            #nnjs-overlay .nnjs-body button:hover { background: #efefef; }
            #nnjs-overlay .nnjs-body button[disabled] { opacity: 0.45; cursor: not-allowed; }
            #nnjs-overlay .nnjs-body button.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
            #nnjs-overlay .nnjs-chip-wrap { display: flex; flex-wrap: wrap; gap: 4px; min-height: 24px; }
            #nnjs-overlay .nnjs-chip { background: #e8eefc; border: 1px solid #b8cbf7; border-radius: 10px; padding: 2px 8px; font-size: 11px; line-height: 16px; }
            #nnjs-overlay .nnjs-chip.end { background: #fde8df; border-color: #efbca7; }
            #nnjs-overlay .nnjs-section-title { font-size: 11px; font-weight: 600; color: #333; }
            #nnjs-overlay .nnjs-status { font-size: 11px; color: #333; min-height: 16px; line-height: 16px; }
            #nnjs-overlay .nnjs-muted { color: #666; }
            .edge-path.nnjs-edge-start { stroke: #2f80ed !important; stroke-width: 1.8px !important; }
            .edge-path.nnjs-edge-end { stroke: #eb6f2d !important; stroke-width: 1.8px !important; }
            @media (prefers-color-scheme: dark) {
                #nnjs-overlay .nnjs-rail { background: rgba(39, 39, 39, 0.96); border-color: #444; }
                #nnjs-overlay .nnjs-rail button { background: #222; color: #ddd; border-color: #5b5b5b; }
                #nnjs-overlay .nnjs-rail button:hover { background: #313131; }
                #nnjs-overlay .nnjs-drawer { background: rgba(39, 39, 39, 0.96); color: #eaeaea; border-color: #444; }
                #nnjs-overlay .nnjs-header button { background: #242424; color: #ddd; border-color: #5b5b5b; }
                #nnjs-overlay .nnjs-body button { background: #222; color: #ddd; border-color: #5b5b5b; }
                #nnjs-overlay .nnjs-body button:hover { background: #313131; }
                #nnjs-overlay .nnjs-chip { background: #1d2c49; border-color: #32518a; color: #cddcff; }
                #nnjs-overlay .nnjs-chip.end { background: #493023; border-color: #8d5a3f; color: #ffd9bf; }
                #nnjs-overlay .nnjs-section-title { color: #dbdbdb; }
                #nnjs-overlay .nnjs-status { color: #d1d1d1; }
                #nnjs-overlay .nnjs-muted { color: #aaaaaa; }
            }
            @media (max-width: 900px) {
                #nnjs-overlay { top: 60px; left: 6px; }
                #nnjs-overlay .nnjs-drawer { width: min(272px, calc(100vw - 54px)); height: min(72vh, 500px); max-height: calc(100vh - 72px); }
            }
            `;
            document.head.appendChild(style);
        }

        _createPanel() {
            const document = this.host.document;
            const overlay = document.createElement('div');
            overlay.id = 'nnjs-overlay';
            overlay.innerHTML = `
                <div class="nnjs-rail">
                    <button id="nnjs-collapse" title="Show NNJS tools">NN</button>
                </div>
                <div class="nnjs-drawer">
                    <div class="nnjs-header">
                        <span>NNJS Tools</span>
                        <button id="nnjs-close">Close</button>
                    </div>
                    <div class="nnjs-body">
                        <div class="nnjs-row">
                            <button id="nnjs-convert">Convert To NNJS</button>
                            <button id="nnjs-save" disabled>Save NNJS</button>
                            <button id="nnjs-copy" disabled>Copy NNJS JSON</button>
                        </div>
                        <div class="nnjs-row">
                            <label class="nnjs-muted"><input id="nnjs-include-weights" type="checkbox"> Include weights (default off)</label>
                        </div>
                        <div class="nnjs-row">
                            <button id="nnjs-crop-toggle">Crop</button>
                        </div>
                        <div id="nnjs-crop-body" style="display:none; flex-direction: column; gap: 8px;">
                            <div class="nnjs-row">
                                <button id="nnjs-mode-start">Select Start Tensor</button>
                                <button id="nnjs-mode-end">Select End Tensor</button>
                            </div>
                            <div>
                                <div class="nnjs-section-title">Start Tensors</div>
                                <div id="nnjs-start-list" class="nnjs-chip-wrap"></div>
                                <div class="nnjs-row"><button id="nnjs-clear-start">Clear Start</button></div>
                            </div>
                            <div>
                                <div class="nnjs-section-title">End Tensors</div>
                                <div id="nnjs-end-list" class="nnjs-chip-wrap"></div>
                                <div class="nnjs-row"><button id="nnjs-clear-end">Clear End</button></div>
                            </div>
                            <div class="nnjs-row">
                                <button id="nnjs-confirm-crop">Confirm Crop</button>
                                <button id="nnjs-save-cropped" disabled>Save Cropped NNJS</button>
                                <button id="nnjs-copy-cropped" disabled>Copy Cropped JSON</button>
                            </div>
                            <div class="nnjs-row">
                                <button id="nnjs-save-shot" disabled>Save Crop Screenshot</button>
                                <button id="nnjs-copy-shot" disabled>Copy Crop Screenshot</button>
                            </div>
                        </div>
                        <div id="nnjs-status" class="nnjs-status"></div>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);

            this._elements.overlay = overlay;
            this._elements.collapse = document.getElementById('nnjs-collapse');
            this._elements.close = document.getElementById('nnjs-close');
            this._elements.convert = document.getElementById('nnjs-convert');
            this._elements.save = document.getElementById('nnjs-save');
            this._elements.copy = document.getElementById('nnjs-copy');
            this._elements.includeWeights = document.getElementById('nnjs-include-weights');
            this._elements.cropToggle = document.getElementById('nnjs-crop-toggle');
            this._elements.cropBody = document.getElementById('nnjs-crop-body');
            this._elements.modeStart = document.getElementById('nnjs-mode-start');
            this._elements.modeEnd = document.getElementById('nnjs-mode-end');
            this._elements.startList = document.getElementById('nnjs-start-list');
            this._elements.endList = document.getElementById('nnjs-end-list');
            this._elements.clearStart = document.getElementById('nnjs-clear-start');
            this._elements.clearEnd = document.getElementById('nnjs-clear-end');
            this._elements.confirmCrop = document.getElementById('nnjs-confirm-crop');
            this._elements.saveCropped = document.getElementById('nnjs-save-cropped');
            this._elements.copyCropped = document.getElementById('nnjs-copy-cropped');
            this._elements.saveShot = document.getElementById('nnjs-save-shot');
            this._elements.copyShot = document.getElementById('nnjs-copy-shot');
            this._elements.status = document.getElementById('nnjs-status');
            this._setToolbarExpanded(false);
        }

        _setToolbarExpanded(expanded) {
            this._elements.overlay.classList.toggle('expanded', expanded);
            this._elements.collapse.textContent = expanded ? '×' : 'NN';
            this._elements.collapse.title = expanded ? 'Hide NNJS tools' : 'Show NNJS tools';
        }

        _bindEvents() {
            this._elements.collapse.addEventListener('click', () => {
                const expanded = !this._elements.overlay.classList.contains('expanded');
                this._setToolbarExpanded(expanded);
            });
            this._elements.close.addEventListener('click', () => {
                this._setToolbarExpanded(false);
            });

            this._elements.convert.addEventListener('click', async () => {
                await this._handleConvert();
            });

            this._elements.save.addEventListener('click', async () => {
                await this._saveNNJSCache();
            });
            this._elements.copy.addEventListener('click', async () => {
                await this._copyNNJSCache();
            });

            this._elements.cropToggle.addEventListener('click', () => {
                const visible = this._elements.cropBody.style.display !== 'none';
                this._elements.cropBody.style.display = visible ? 'none' : 'flex';
            });

            this._elements.modeStart.addEventListener('click', () => {
                this._setMode(this.mode === 'start' ? null : 'start');
            });

            this._elements.modeEnd.addEventListener('click', () => {
                this._setMode(this.mode === 'end' ? null : 'end');
            });

            this._elements.clearStart.addEventListener('click', () => {
                this.startKeys.clear();
                this._renderSelectionLists();
                this._applyEdgeHighlights();
            });

            this._elements.clearEnd.addEventListener('click', () => {
                this.endKeys.clear();
                this._renderSelectionLists();
                this._applyEdgeHighlights();
            });

            this._elements.confirmCrop.addEventListener('click', async () => {
                await this._confirmCrop();
            });

            this._elements.saveCropped.addEventListener('click', async () => {
                await this._saveCroppedNNJS();
            });
            this._elements.copyCropped.addEventListener('click', async () => {
                await this._copyCroppedNNJS();
            });

            this._elements.saveShot.addEventListener('click', async () => {
                try {
                    await this.host.saveCachedScreenshot('crop-preview.png');
                    this._setStatus('Cached screenshot save dialog opened.');
                } catch (error) {
                    this._setStatus(error.message || String(error), true);
                }
            });

            this._elements.copyShot.addEventListener('click', async () => {
                try {
                    await this.host.copyCachedScreenshot();
                    this._setStatus('Crop screenshot copied to clipboard.');
                } catch (error) {
                    this._setStatus(error.message || String(error), true);
                }
            });

            this.host.window.addEventListener('nnjs:model-opened', () => {
                this.mode = null;
                this.startKeys.clear();
                this.endKeys.clear();
                this.nnjsCache = null;
                this.croppedCache = null;
                this._refreshGraphIndex();
                this._renderSelectionLists();
                this._applyEdgeHighlights();
                this._updateButtons();
                this._setStatus('Model loaded. You can now convert to NNJS or start crop selection.');
            });

            this.host.window.addEventListener('nnjs:screenshot-cached', () => {
                this._updateButtons();
                this._setStatus('Crop screenshot cached in memory. Use Save or Copy when needed.');
            });

            this.host.window.addEventListener('message', (event) => {
                const data = event.data;
                if (data && data.type === 'fileSaved' && typeof data.path === 'string') {
                    this._setStatus(`Saved: ${data.path}`);
                } else if (data && data.type === 'clipboardCopied') {
                    const label = typeof data.label === 'string' && data.label.length > 0 ? data.label : 'Text';
                    this._setStatus(`${label} copied to clipboard.`);
                } else if (data && data.type === 'clipboardError') {
                    const message = typeof data.message === 'string' && data.message.length > 0
                        ? data.message
                        : 'Clipboard operation failed.';
                    this._setStatus(message, true);
                }
            });

            this.host.document.addEventListener('click', (event) => {
                this._handleGraphClick(event);
            }, true);
        }

        _setMode(mode) {
            this.mode = mode;
            this._elements.modeStart.classList.toggle('active', mode === 'start');
            this._elements.modeEnd.classList.toggle('active', mode === 'end');
            if (mode === 'start') {
                this._setStatus('Start tensor select mode is ON. Click tensor edges to toggle selection.');
            } else if (mode === 'end') {
                this._setStatus('End tensor select mode is ON. Click tensor edges to toggle selection.');
            } else {
                this._setStatus('Tensor select mode is OFF.');
            }
        }

        _handleGraphClick(event) {
            if (!this.mode) {
                return;
            }
            if (this._elements.overlay.contains(event.target)) {
                return;
            }
            const path = event.target && event.target.closest ? event.target.closest('#edge-paths-hit-test path') : null;
            if (!path) {
                return;
            }
            if (!this._refreshGraphIndex()) {
                return;
            }
            const target = this.view && this.view._target;
            const edge = target && target._focusable ? target._focusable.get(path) : null;
            const value = edge && edge.value ? edge.value.value : null;
            const key = this._valueToKey(value);
            if (!key) {
                return;
            }
            const selectedSet = this.mode === 'start' ? this.startKeys : this.endKeys;
            if (selectedSet.has(key)) {
                selectedSet.delete(key);
            } else {
                selectedSet.add(key);
            }
            this._renderSelectionLists();
            this._applyEdgeHighlights();
            event.preventDefault();
            event.stopPropagation();
        }

        _refreshGraphIndex() {
            const graph = this.view ? this.view.activeTarget : null;
            if (!graph || !Array.isArray(graph.nodes)) {
                this.graphIndex = null;
                return false;
            }
            this.graphIndex = this._buildGraphIndex(graph);
            for (const key of Array.from(this.startKeys)) {
                if (!this.graphIndex.tensorByKey.has(key)) {
                    this.startKeys.delete(key);
                }
            }
            for (const key of Array.from(this.endKeys)) {
                if (!this.graphIndex.tensorByKey.has(key)) {
                    this.endKeys.delete(key);
                }
            }
            return true;
        }

        _buildGraphIndex(graph) {
            const autoTensorIds = new WeakMap();
            let autoTensorCount = 0;
            const keyToValue = new Map();
            const tensorByKey = new Map();
            const nodes = [];

            const getTensorKey = (value) => {
                if (!value) {
                    return null;
                }
                if (typeof value.name === 'string' && value.name.length > 0) {
                    return value.name;
                }
                if (!autoTensorIds.has(value)) {
                    autoTensorIds.set(value, `__tensor_${autoTensorCount++}`);
                }
                return autoTensorIds.get(value);
            };

            const ensureTensor = (value) => {
                const key = getTensorKey(value);
                if (!key) {
                    return null;
                }
                const hasEntry = tensorByKey.has(key);
                if (!hasEntry) {
                    keyToValue.set(key, value);
                    tensorByKey.set(key, {
                        key,
                        value,
                        type: value.type || null,
                        initializer: value.initializer || null,
                        producer: null,
                        consumers: new Set()
                    });
                } else {
                    const tensor = tensorByKey.get(key);
                    const previous = keyToValue.get(key);
                    if (value && value.initializer && (!tensor.initializer || !(previous && previous.initializer))) {
                        tensor.initializer = value.initializer;
                        tensor.value = value;
                        keyToValue.set(key, value);
                    }
                    if (!tensor.type && value && value.type) {
                        tensor.type = value.type;
                    }
                }
                return key;
            };

            const nodeTypeName = (node) => {
                if (!node || !node.type) {
                    return '';
                }
                if (typeof node.type === 'string') {
                    return node.type;
                }
                if (typeof node.type.name === 'string') {
                    return node.type.name;
                }
                return '';
            };

            graph.nodes.forEach((node, index) => {
                const id = `n${index}`;
                const entry = {
                    id,
                    node,
                    name: node.name || id,
                    type: nodeTypeName(node),
                    attributes: Array.isArray(node.attributes) ? node.attributes : [],
                    inputs: [],
                    outputs: []
                };

                for (const argument of node.inputs || []) {
                    const keys = [];
                    for (const value of argument.value || []) {
                        const key = ensureTensor(value);
                        if (!key) {
                            continue;
                        }
                        keys.push(key);
                        tensorByKey.get(key).consumers.add(id);
                    }
                    entry.inputs.push({
                        name: argument.name || '',
                        keys
                    });
                }

                let outputs = node.outputs || [];
                if (node.chain && node.chain.length > 0) {
                    const chainOutputs = node.chain[node.chain.length - 1].outputs;
                    if (Array.isArray(chainOutputs) && chainOutputs.length > 0) {
                        outputs = chainOutputs;
                    }
                }

                for (const argument of outputs) {
                    const keys = [];
                    for (const value of argument.value || []) {
                        const key = ensureTensor(value);
                        if (!key) {
                            continue;
                        }
                        keys.push(key);
                        const tensor = tensorByKey.get(key);
                        if (!tensor.producer) {
                            tensor.producer = id;
                        }
                        if (value.initializer && !tensor.initializer) {
                            tensor.initializer = value.initializer;
                        }
                    }
                    entry.outputs.push({
                        name: argument.name || '',
                        keys
                    });
                }

                nodes.push(entry);
            });

            const graphInputs = new Set();
            for (const argument of graph.inputs || []) {
                for (const value of argument.value || []) {
                    const key = ensureTensor(value);
                    if (key) {
                        graphInputs.add(key);
                    }
                }
            }

            const graphOutputs = new Set();
            for (const argument of graph.outputs || []) {
                for (const value of argument.value || []) {
                    const key = ensureTensor(value);
                    if (key) {
                        graphOutputs.add(key);
                    }
                }
            }

            return {
                graph,
                nodes,
                keyToValue,
                tensorByKey,
                graphInputs,
                graphOutputs
            };
        }

        _valueToKey(value) {
            if (!value || !this.graphIndex) {
                return null;
            }
            if (typeof value.name === 'string' && value.name.length > 0) {
                return value.name;
            }
            for (const [key, candidate] of this.graphIndex.keyToValue.entries()) {
                if (candidate === value) {
                    return key;
                }
            }
            return null;
        }

        _renderSelectionLists() {
            const render = (container, keys, isEnd) => {
                while (container.firstChild) {
                    container.removeChild(container.firstChild);
                }
                const list = Array.from(keys.values());
                if (list.length === 0) {
                    const empty = this.host.document.createElement('span');
                    empty.className = 'nnjs-muted';
                    empty.textContent = '(none)';
                    container.appendChild(empty);
                    return;
                }
                for (const key of list) {
                    const chip = this.host.document.createElement('span');
                    chip.className = isEnd ? 'nnjs-chip end' : 'nnjs-chip';
                    chip.textContent = this._tensorDisplayName(key);
                    container.appendChild(chip);
                }
            };
            render(this._elements.startList, this.startKeys, false);
            render(this._elements.endList, this.endKeys, true);
            this._updateButtons();
        }

        _tensorDisplayName(key) {
            if (!key.startsWith('__tensor_')) {
                return key;
            }
            return `[unnamed] ${key}`;
        }

        _applyEdgeHighlights() {
            const document = this.host.document;
            for (const element of Array.from(document.querySelectorAll('.nnjs-edge-start'))) {
                element.classList.remove('nnjs-edge-start');
            }
            for (const element of Array.from(document.querySelectorAll('.nnjs-edge-end'))) {
                element.classList.remove('nnjs-edge-end');
            }
            const target = this.view && this.view._target;
            if (!target || !target._table || !this.graphIndex) {
                return;
            }

            const apply = (keys, className) => {
                for (const key of keys) {
                    const value = this.graphIndex.keyToValue.get(key);
                    if (!value) {
                        continue;
                    }
                    const viewValue = target._table.get(value);
                    if (!viewValue || !Array.isArray(viewValue._edges)) {
                        continue;
                    }
                    for (const edge of viewValue._edges) {
                        if (edge && edge.element) {
                            edge.element.classList.add(className);
                        }
                    }
                }
            };

            apply(this.startKeys, 'nnjs-edge-start');
            apply(this.endKeys, 'nnjs-edge-end');
        }

        async _handleConvert() {
            try {
                if (!this._refreshGraphIndex()) {
                    this._setStatus('No active graph to convert.', true);
                    return;
                }
                const includeWeights = !!this._elements.includeWeights.checked;
                this.nnjsCache = await this._buildNNJSFromGraph(this.graphIndex, {
                    includeWeights,
                    graphNameSuffix: 'full'
                });
                this._setStatus(`NNJS cached in memory (${includeWeights ? 'with' : 'without'} weights).`);
                this._updateButtons();
            } catch (error) {
                this._setStatus(error.message || String(error), true);
            }
        }

        async _buildNNJSFromGraph(index, options) {
            const includeWeights = !!options.includeWeights;
            const allNodeIds = new Set(index.nodes.map((node) => node.id));
            const selectedNodeIds = options.selectedNodeIds || allNodeIds;
            const graphInputKeys = options.graphInputKeys || index.graphInputs;
            const graphOutputKeys = options.graphOutputKeys || index.graphOutputs;

            const usedTensorKeys = new Set();
            const selectedNodes = index.nodes.filter((node) => selectedNodeIds.has(node.id));
            for (const entry of selectedNodes) {
                for (const argument of entry.inputs) {
                    for (const key of argument.keys) {
                        usedTensorKeys.add(key);
                    }
                }
                for (const argument of entry.outputs) {
                    for (const key of argument.keys) {
                        usedTensorKeys.add(key);
                    }
                }
            }
            for (const key of graphInputKeys) {
                usedTensorKeys.add(key);
            }
            for (const key of graphOutputKeys) {
                usedTensorKeys.add(key);
            }

            const tensors = [];
            const weights = {
                signature: 'nnjs-weights',
                version: 1,
                tensors: {}
            };

            for (const key of usedTensorKeys) {
                const tensor = index.tensorByKey.get(key);
                if (!tensor) {
                    continue;
                }
                const type = tensor.type || {};
                const shape = type.shape && Array.isArray(type.shape.dimensions) ? type.shape.dimensions.map((item) => toSerializable(item, new Set(), 0)) : [];
                const item = {
                    name: key,
                    dataType: type.dataType || '',
                    shape,
                    producer: tensor.producer && selectedNodeIds.has(tensor.producer) ? tensor.producer : null,
                    consumers: Array.from(tensor.consumers).filter((id) => selectedNodeIds.has(id))
                };

                if (includeWeights && tensor.initializer) {
                    const value = await this._readInitializer(tensor.initializer);
                    item.weight = {
                        buffer: 'weights',
                        key
                    };
                    weights.tensors[key] = {
                        dataType: item.dataType,
                        shape,
                        value
                    };
                }
                tensors.push(item);
            }

            const nnjs = {
                signature: 'nnjs',
                version: 1,
                generatedAt: new Date().toISOString(),
                source: {
                    identifier: this.view.model && this.view.model.identifier ? this.view.model.identifier : '',
                    format: this.view.model && this.view.model.format ? this.view.model.format : '',
                    producer: this.view.model && this.view.model.producer ? this.view.model.producer : ''
                },
                graph: {
                    name: index.graph && index.graph.name ? index.graph.name : 'graph',
                    tag: options.graphNameSuffix || 'full',
                    inputs: Array.from(graphInputKeys),
                    outputs: Array.from(graphOutputKeys),
                    nodes: selectedNodes.map((entry) => ({
                        id: entry.id,
                        name: entry.name,
                        type: entry.type,
                        inputs: entry.inputs.map((argument) => ({
                            name: argument.name,
                            tensors: argument.keys
                        })),
                        outputs: entry.outputs.map((argument) => ({
                            name: argument.name,
                            tensors: argument.keys
                        })),
                        attributes: entry.attributes.map((attribute) => ({
                            name: attribute.name || '',
                            type: attribute.type || '',
                            value: toSerializable(attribute.value, new Set(), 0)
                        }))
                    })),
                    tensors
                },
                buffers: includeWeights ? [{
                    id: 'weights',
                    format: 'json',
                    file: '<save-time>.weights.json'
                }] : []
            };

            return {
                includeWeights,
                document: nnjs,
                weights: includeWeights ? weights : null
            };
        }

        async _readInitializer(initializer) {
            try {
                if (initializer && typeof initializer.peek === 'function' && !initializer.peek() && typeof initializer.read === 'function') {
                    await initializer.read();
                }
            } catch {
                // continue regardless of error
            }
            try {
                if (initializer && Object.prototype.hasOwnProperty.call(initializer, 'value')) {
                    return toSerializable(initializer.value, new Set(), 0);
                }
            } catch {
                // continue regardless of error
            }
            try {
                if (initializer && Object.prototype.hasOwnProperty.call(initializer, 'values')) {
                    return toSerializable(initializer.values, new Set(), 0);
                }
            } catch {
                // continue regardless of error
            }
            return null;
        }

        async _saveNNJSCache() {
            if (!this.nnjsCache) {
                this._setStatus('No NNJS cache in memory. Click Convert To NNJS first.', true);
                return;
            }
            const title = sanitizeFileName(this.host.document.title || 'model', 'model');
            const fileName = `${title}.nnjs.json`;
            const content = JSON.stringify(this.nnjsCache.document, null, 2);
            await this.host.saveTextFile(fileName, content, {
                JSON: ['json']
            });

            if (this.nnjsCache.includeWeights && this.nnjsCache.weights) {
                const weightName = `${title}.weights.json`;
                const weightContent = JSON.stringify(this.nnjsCache.weights, null, 2);
                await this.host.saveTextFile(weightName, weightContent, {
                    JSON: ['json']
                });
            }
            this._setStatus('NNJS save dialog opened.');
        }

        async _copyNNJSCache() {
            if (!this.nnjsCache) {
                this._setStatus('No NNJS cache in memory. Click Convert To NNJS first.', true);
                return;
            }
            const content = JSON.stringify(this.nnjsCache.document, null, 2);
            await this.host.copyTextToClipboard(content, 'NNJS JSON');
        }

        _cropGraph(index, startKeys, endKeys) {
            const nodeMap = new Map(index.nodes.map((node) => [node.id, node]));

            const startNodes = new Set();
            for (const key of startKeys) {
                const tensor = index.tensorByKey.get(key);
                if (!tensor) {
                    continue;
                }
                for (const consumer of tensor.consumers) {
                    startNodes.add(consumer);
                }
            }

            const endNodes = new Set();
            for (const key of endKeys) {
                const tensor = index.tensorByKey.get(key);
                if (!tensor || !tensor.producer) {
                    continue;
                }
                endNodes.add(tensor.producer);
            }

            if (startNodes.size === 0) {
                throw new Error('No valid start tensor consumer nodes found.');
            }
            if (endNodes.size === 0) {
                throw new Error('No valid end tensor producer nodes found.');
            }

            const walkForward = new Set();
            const forwardQueue = Array.from(startNodes);
            while (forwardQueue.length > 0) {
                const nodeId = forwardQueue.shift();
                if (walkForward.has(nodeId)) {
                    continue;
                }
                walkForward.add(nodeId);
                const node = nodeMap.get(nodeId);
                if (!node) {
                    continue;
                }
                for (const argument of node.outputs) {
                    for (const key of argument.keys) {
                        const tensor = index.tensorByKey.get(key);
                        if (!tensor) {
                            continue;
                        }
                        for (const consumer of tensor.consumers) {
                            if (!walkForward.has(consumer)) {
                                forwardQueue.push(consumer);
                            }
                        }
                    }
                }
            }

            const walkBackward = new Set();
            const backwardQueue = Array.from(endNodes);
            while (backwardQueue.length > 0) {
                const nodeId = backwardQueue.shift();
                if (walkBackward.has(nodeId)) {
                    continue;
                }
                walkBackward.add(nodeId);
                const node = nodeMap.get(nodeId);
                if (!node) {
                    continue;
                }
                for (const argument of node.inputs) {
                    for (const key of argument.keys) {
                        const tensor = index.tensorByKey.get(key);
                        if (tensor && tensor.producer && !walkBackward.has(tensor.producer)) {
                            backwardQueue.push(tensor.producer);
                        }
                    }
                }
            }

            const selectedNodeIds = new Set(Array.from(walkForward).filter((id) => walkBackward.has(id)));
            if (selectedNodeIds.size === 0) {
                throw new Error('No intersected nodes between start and end tensor paths.');
            }

            const inputKeys = new Set();
            const outputKeys = new Set();
            const tensorKeys = new Set();

            for (const nodeId of selectedNodeIds) {
                const node = nodeMap.get(nodeId);
                if (!node) {
                    continue;
                }

                for (const argument of node.inputs) {
                    for (const key of argument.keys) {
                        tensorKeys.add(key);
                        const tensor = index.tensorByKey.get(key);
                        const producer = tensor ? tensor.producer : null;
                        const isInitializer = !!(tensor && tensor.initializer);
                        if ((!producer || !selectedNodeIds.has(producer)) && !isInitializer) {
                            inputKeys.add(key);
                        }
                    }
                }

                for (const argument of node.outputs) {
                    for (const key of argument.keys) {
                        tensorKeys.add(key);
                        const tensor = index.tensorByKey.get(key);
                        if (!tensor) {
                            continue;
                        }
                        let hasInside = false;
                        let hasOutside = false;
                        for (const consumer of tensor.consumers) {
                            if (selectedNodeIds.has(consumer)) {
                                hasInside = true;
                            } else {
                                hasOutside = true;
                            }
                        }
                        if (!hasInside || hasOutside || endKeys.has(key) || index.graphOutputs.has(key)) {
                            outputKeys.add(key);
                        }
                    }
                }
            }

            for (const key of index.graphInputs) {
                const tensor = index.tensorByKey.get(key);
                if (!tensor) {
                    continue;
                }
                if (tensor.initializer) {
                    continue;
                }
                const usedByCrop = Array.from(tensor.consumers).some((id) => selectedNodeIds.has(id));
                if (usedByCrop) {
                    inputKeys.add(key);
                }
            }

            for (const key of index.graphOutputs) {
                const tensor = index.tensorByKey.get(key);
                if (!tensor) {
                    continue;
                }
                if (tensor.producer && selectedNodeIds.has(tensor.producer)) {
                    outputKeys.add(key);
                }
            }

            for (const key of inputKeys) {
                tensorKeys.add(key);
            }
            for (const key of outputKeys) {
                tensorKeys.add(key);
            }

            return {
                selectedNodeIds,
                inputKeys,
                outputKeys,
                tensorKeys
            };
        }

        _pruneConstantInitializerNodes(index, selectedNodeIds) {
            const filtered = new Set();
            for (const entry of index.nodes) {
                if (!selectedNodeIds.has(entry.id)) {
                    continue;
                }
                const type = typeof entry.type === 'string' ? entry.type.toLowerCase() : '';
                const isConstantOp = type === 'constant';
                if (!isConstantOp) {
                    filtered.add(entry.id);
                    continue;
                }
                const hasInputs = entry.inputs.some((argument) => Array.isArray(argument.keys) && argument.keys.length > 0);
                if (hasInputs) {
                    filtered.add(entry.id);
                    continue;
                }
                const outputKeys = entry.outputs.flatMap((argument) => argument.keys || []);
                if (outputKeys.length === 0) {
                    filtered.add(entry.id);
                    continue;
                }
                const allInitializerOutputs = outputKeys.every((key) => {
                    const tensor = index.tensorByKey.get(key);
                    return !!(tensor && tensor.initializer);
                });
                if (!allInitializerOutputs) {
                    filtered.add(entry.id);
                }
            }
            return filtered.size > 0 ? filtered : selectedNodeIds;
        }

        _buildCroppedGraphObject(index, cropResult, selectedNodeIds) {
            const effectiveNodeIds = selectedNodeIds || cropResult.selectedNodeIds;
            const selectedNodes = index.nodes
                .filter((entry) => effectiveNodeIds.has(entry.id))
                .map((entry) => entry.node);

            const makeArguments = (keys, prefix, skipInitializer) => {
                const result = [];
                let i = 0;
                for (const key of keys) {
                    const value = index.keyToValue.get(key);
                    if (!value) {
                        continue;
                    }
                    if (skipInitializer) {
                        const tensor = index.tensorByKey.get(key);
                        if (tensor && tensor.initializer) {
                            continue;
                        }
                    }
                    const name = value.name && value.name.length > 0 ? value.name : `${prefix}_${++i}`;
                    result.push({
                        name,
                        value: [value],
                        visible: true
                    });
                }
                return result;
            };

            return {
                name: `${index.graph && index.graph.name ? index.graph.name : 'graph'}::crop`,
                nodes: selectedNodes,
                inputs: makeArguments(cropResult.inputKeys, 'crop_input', true),
                outputs: makeArguments(cropResult.outputKeys, 'crop_output', false),
                groups: index.graph && index.graph.groups ? index.graph.groups : false
            };
        }

        async _confirmCrop() {
            try {
                if (this.startKeys.size === 0 || this.endKeys.size === 0) {
                    this._setStatus('Please select at least one start tensor and one end tensor.', true);
                    return;
                }
                if (!this._refreshGraphIndex()) {
                    this._setStatus('No active graph available for crop.', true);
                    return;
                }
                const cropResult = this._cropGraph(this.graphIndex, this.startKeys, this.endKeys);
                const effectiveNodeIds = this._pruneConstantInitializerNodes(this.graphIndex, cropResult.selectedNodeIds);
                const cropGraph = this._buildCroppedGraphObject(this.graphIndex, cropResult, effectiveNodeIds);
                await this.view.pushTarget(cropGraph, null);

                this.croppedCache = await this._buildNNJSFromGraph(this.graphIndex, {
                    includeWeights: !!this._elements.includeWeights.checked,
                    selectedNodeIds: effectiveNodeIds,
                    graphInputKeys: cropResult.inputKeys,
                    graphOutputKeys: cropResult.outputKeys,
                    graphNameSuffix: 'crop'
                });

                await this.host.captureScreenshotToMemory('crop-preview.png');
                this._updateButtons();
                this._applyEdgeHighlights();
                this._setStatus('Crop confirm completed. Cropped graph rendered, screenshot cached in memory.');
            } catch (error) {
                this._setStatus(error.message || String(error), true);
            }
        }

        async _saveCroppedNNJS() {
            if (!this.croppedCache) {
                this._setStatus('No cropped NNJS cache available. Run Confirm Crop first.', true);
                return;
            }
            const title = sanitizeFileName(this.host.document.title || 'model', 'model');
            const fileName = `${title}.cropped.nnjs.json`;
            const content = JSON.stringify(this.croppedCache.document, null, 2);
            await this.host.saveTextFile(fileName, content, {
                JSON: ['json']
            });

            if (this.croppedCache.includeWeights && this.croppedCache.weights) {
                const weightName = `${title}.cropped.weights.json`;
                const weightContent = JSON.stringify(this.croppedCache.weights, null, 2);
                await this.host.saveTextFile(weightName, weightContent, {
                    JSON: ['json']
                });
            }
            this._setStatus('Cropped NNJS save dialog opened.');
        }

        async _copyCroppedNNJS() {
            if (!this.croppedCache) {
                this._setStatus('No cropped NNJS cache available. Run Confirm Crop first.', true);
                return;
            }
            const content = JSON.stringify(this.croppedCache.document, null, 2);
            await this.host.copyTextToClipboard(content, 'Cropped NNJS JSON');
        }

        _updateButtons() {
            this._elements.save.disabled = !this.nnjsCache;
            this._elements.copy.disabled = !this.nnjsCache;
            this._elements.saveCropped.disabled = !this.croppedCache;
            this._elements.copyCropped.disabled = !this.croppedCache;
            const hasCropSelection = this.startKeys.size > 0 && this.endKeys.size > 0;
            this._elements.confirmCrop.disabled = !hasCropSelection;
            const hasScreenshot = this.host.hasCachedScreenshot();
            this._elements.saveShot.disabled = !hasScreenshot;
            this._elements.copyShot.disabled = !hasScreenshot;
        }

        _setStatus(message, isError) {
            this._elements.status.textContent = message;
            if (isError) {
                this._elements.status.style.color = '#c62828';
                this.host.notify('warn', message);
            } else {
                this._elements.status.style.color = '';
            }
        }
    }

    const bootstrap = () => {
        if (!isVSCode) {
            const message = 'This entry point is for VS Code Webview only.';
            window.exports.terminate(message);
            return;
        }
        if (!window.exports.browser || !window.exports.browser.Host) {
            window.exports.terminate('Browser host module is not initialized.');
            return;
        }
        if (!window.exports.view || !window.exports.view.View) {
            window.exports.terminate('View module is not initialized.');
            return;
        }
        try {
            const VSCodeHost = createVSCodeHostClass();
            const host = new VSCodeHost();
            window.__view__ = new window.exports.view.View(host);
            window.__nnjsToolbar = new NNJSToolbar(host, window.__view__);
            window.__nnjsToolbar.attach();
            window.__view__.start();
            postModelLog('info', 'bootstrap completed', {
                phase: 'bootstrap',
                serial: true
            });
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            postModelLog('error', 'bootstrap failed', {
                phase: 'bootstrap',
                serial: true,
                message
            });
            window.exports.terminate(message);
        }
    };

    window.addEventListener('load', function() {
        if (typeof Symbol !== 'function' || typeof Symbol.asyncIterator !== 'symbol' ||
            typeof BigInt !== 'function' || typeof BigInt.asIntN !== 'function' || typeof BigInt.asUintN !== 'function' || typeof DataView.prototype.getBigInt64 !== 'function') {
            throw new Error('Please update your browser to use this application.');
        }
        var ua = window.navigator.userAgent;
        var chrome = ua.match(/Chrom(e|ium)\/([0-9]+)\./);
        var safari = ua.match(/Version\/(\d+)\.(\d+).*Safari/);
        var firefox = ua.match(/Firefox\/([0-9]+)\./);
        if ((Array.isArray(chrome) && parseInt(chrome[2], 10) < 86) ||
            (Array.isArray(safari) && (parseInt(safari[1], 10) < 16 || (parseInt(safari[1], 10) === 16 && parseInt(safari[2], 10) < 4))) ||
            (Array.isArray(firefox) && parseInt(firefox[1], 10) < 114)) {
            throw new Error('Please update your browser to use this application.');
        }
        window.exports.preload(function(value, error) {
            if (error) {
                const message = error instanceof Error ? error.message : String(error);
                postModelLog('error', 'preload failed', {
                    phase: 'preload',
                    serial: true,
                    message
                });
                window.exports.terminate(message);
            } else {
                postModelLog('info', 'preload completed', {
                    phase: 'preload',
                    serial: true
                });
                bootstrap();
            }
        });
    });
})();
