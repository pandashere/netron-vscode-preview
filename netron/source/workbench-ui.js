const toText = (value) => {
    if (value === null || value === undefined) {
        return '';
    }
    if (typeof value === 'string') {
        return value;
    }
    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
};

class TensorShape {
    constructor(dimensions) {
        this.dimensions = Array.isArray(dimensions) ? dimensions.slice() : [];
    }

    toString() {
        if (!Array.isArray(this.dimensions) || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.map((dimension) => dimension === null || dimension === undefined ? '?' : dimension).join(',')}]`;
    }
}

class TensorType {
    constructor(dataType, shape, denotation = null) {
        this.dataType = dataType || 'unknown';
        this.shape = shape instanceof TensorShape ? shape : new TensorShape(shape || []);
        this.denotation = denotation;
        this.layout = null;
    }

    toString() {
        return `${this.dataType}${this.shape ? this.shape.toString() : ''}`;
    }
}

class Tensor {
    constructor(data, previewLoader = null) {
        this.name = data.name || '';
        this.category = data.category || 'Initializer';
        this.type = data.type instanceof TensorType ? data.type : new TensorType(data.type && data.type.dataType, data.type && data.type.shape);
        this.location = data.location || null;
        this.encoding = '<';
        this.values = data.values || null;
        this.preview = data.preview || null;
        this._previewLoader = typeof previewLoader === 'function' ? previewLoader : null;
        this._previewPromise = null;
    }

    peek() {
        return !!this.preview || !this._previewLoader;
    }

    async read() {
        if (this.preview || !this._previewLoader) {
            return this.preview;
        }
        if (!this._previewPromise) {
            this._previewPromise = this._previewLoader().then((preview) => {
                this.preview = preview || { error: 'Tensor preview is unavailable.' };
                return this.preview;
            }).catch((error) => {
                this.preview = { error: error && error.message ? error.message : String(error) };
                return this.preview;
            });
        }
        return this._previewPromise;
    }
}

class Value {
    constructor(name, type = null, initializer = null, description = '') {
        this.name = name || '';
        this.type = type;
        this.initializer = initializer;
        this.description = description || '';
        this.metadata = [];
    }
}

class Argument {
    constructor(name, value, type = null, description = null, visible = true) {
        this.name = name || '';
        this.value = value || [];
        this.type = type;
        this.description = description;
        this.visible = visible;
    }
}

class Node {
    constructor(data, valueOf) {
        this.name = data.name || '';
        this.identifier = data.id || this.name;
        this.type = data.type || { name: 'Unknown', module: '', identifier: 'Unknown' };
        this.inputs = (data.inputs || []).map((argument) => new Argument(argument.name || '', (argument.values || []).map((name) => valueOf(name))));
        this.outputs = (data.outputs || []).map((argument) => new Argument(argument.name || '', (argument.values || []).map((name) => valueOf(name))));
        this.attributes = (data.attributes || []).map((attribute) => ({
            name: attribute.name || '',
            value: attribute.value,
            type: attribute.type || 'attribute',
            visible: attribute.visible !== false
        }));
        this.metadata = [];
        this.chain = [];
    }
}

class Graph {
    constructor(data, options = {}) {
        this.name = data.name || 'graph';
        this.description = data.description || '';
        this.groups = !!data.groups;
        this._valueMap = new Map();
        const valuesData = data.values || {};
        const valueOf = (name) => {
            if (!this._valueMap.has(name)) {
                const source = valuesData[name] || { name, type: { dataType: 'unknown', shape: [] }, initializer: null };
                const type = new TensorType(source.type && source.type.dataType, source.type && source.type.shape, source.type && source.type.denotation);
                const initializer = source.initializer
                    ? new Tensor(source.initializer, typeof options.previewLoader === 'function' ? () => options.previewLoader(name) : null)
                    : null;
                this._valueMap.set(name, new Value(name, type, initializer, source.description || ''));
            }
            return this._valueMap.get(name);
        };
        this.inputs = (data.inputs || []).map((argument) => new Argument(argument.name || '', (argument.values || []).map((name) => valueOf(name))));
        this.outputs = (data.outputs || []).map((argument) => new Argument(argument.name || '', (argument.values || []).map((name) => valueOf(name))));
        this.nodes = (data.nodes || []).map((node) => new Node(node, valueOf));
    }
}

class Model {
    constructor(snapshot, options = {}) {
        this.format = snapshot.format || 'ONNX';
        this.producer = snapshot.producer || '';
        this.metadata = (snapshot.metadata || []).map((entry) => ({ name: entry.name, value: entry.value }));
        this.modules = [new Graph(snapshot.graph, options)];
        this.functions = [];
        const emptyList = () => [];
        this.attachment = {
            metadata: {
                model: emptyList,
                graph: emptyList,
                node: emptyList,
                value: emptyList,
                tensor: emptyList
            },
            metrics: {
                model: emptyList,
                graph: emptyList,
                node: emptyList,
                value: emptyList,
                tensor: emptyList
            }
        };
    }
}

function parseShapeOverrides(text) {
    const value = (text || '').trim();
    if (!value) {
        return {};
    }
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('Shape overrides must be a JSON object.');
    }
    return parsed;
}

function sameSet(a, b) {
    if (a.size !== b.size) {
        return false;
    }
    for (const item of a) {
        if (!b.has(item)) {
            return false;
        }
    }
    return true;
}

export class ONNXWorkbenchUI {
    constructor(host, view) {
        this.host = host;
        this.view = view;
        this.mode = null;
        this.startKeys = new Set();
        this.endKeys = new Set();
        this.graphIndex = null;
        this.currentSessionId = null;
        this.fullModelSnapshot = null;
        this.confirmedArtifact = null;
        this.importedInput = null;
        this.compareState = null;
        this.taskState = null;
        this.activity = [];
        this.draftDirty = false;
        this._elements = {};
        this._renderInProgress = false;
        this._suspendRefit = false;
        this._lastRenderRetryCount = 0;
        this._resizeTimer = null;
        this._resizeObserver = null;
        this._tensorPreviewRequests = new Map();
    }

    attach() {
        this._injectStyle();
        this._injectToolbarButton();
        this._createSelectionBar();
        this._createDrawer();
        this._bindEvents();
        this._syncWorkbenchLayout();
        this._setStatus('Load an ONNX model to use Model Tools.');
    }

    _injectStyle() {
        const style = this.host.document.createElement('style');
        style.textContent = `
        html { --workbench-bottom-inset: 0px; }
        #target { box-sizing: border-box; padding-bottom: var(--workbench-bottom-inset); scroll-padding-bottom: var(--workbench-bottom-inset); }
        #toolbar { bottom: calc(10px + var(--workbench-bottom-inset)); transition: bottom 0.12s ease; }
        #workbench-toolbar-button.toolbar-button { position: relative; }
        #workbench-toolbar-button .toolbar-icon text { font-size: 34px; font-weight: 700; }
        #onnx-workbench-bar { position: fixed; left: 10px; bottom: 56px; z-index: 4; display: none; align-items: center; gap: 10px; flex-wrap: wrap; max-width: min(calc(100vw - 20px), 900px); padding: 10px 12px; border: 1px solid rgba(127,127,127,0.25); border-radius: 12px; background: rgba(248,248,248,0.98); color: #1f2328; box-shadow: 0 12px 24px rgba(0,0,0,0.16); backdrop-filter: blur(8px); }
        #onnx-workbench-bar.visible { display: flex; }
        #onnx-workbench-bar .wb-bar-group { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
        #onnx-workbench-bar .wb-bar-summary { font-size: 12px; opacity: 0.82; min-width: 180px; }
        #onnx-workbench-bar .wb-bar-pill { display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(127,127,127,0.2); background: rgba(31,111,235,0.08); font-size: 11px; }
        #onnx-workbench-bar button { border: 1px solid rgba(127,127,127,0.35); background: #fff; color: inherit; border-radius: 8px; padding: 6px 10px; cursor: pointer; }
        #onnx-workbench-bar button.primary { background: #1f6feb; color: #fff; border-color: #1f6feb; }
        #onnx-workbench-bar button[disabled] { opacity: 0.45; cursor: not-allowed; }
        #onnx-workbench-bar button.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
        #onnx-workbench { position: fixed; left: 10px; bottom: 56px; z-index: 4; width: 420px; max-width: min(420px, calc(100vw - 20px)); border: 1px solid rgba(127,127,127,0.25); border-radius: 12px; background: rgba(248,248,248,0.98); color: #1f2328; box-shadow: 0 12px 24px rgba(0,0,0,0.18); display: none; overflow: hidden; backdrop-filter: blur(8px); }
        #onnx-workbench.with-bar { bottom: 118px; }
        #onnx-workbench.visible { display: block; }
        #onnx-workbench .wb-header { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 10px 12px; border-bottom: 1px solid rgba(127,127,127,0.2); background: rgba(255,255,255,0.7); }
        #onnx-workbench .wb-title { font-size: 13px; font-weight: 700; letter-spacing: 0.2px; }
        #onnx-workbench .wb-close { border: 1px solid rgba(127,127,127,0.35); background: transparent; color: inherit; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
        #onnx-workbench .wb-tabs { display: flex; gap: 4px; padding: 8px 10px 0 10px; }
        #onnx-workbench .wb-tab { border: 1px solid rgba(127,127,127,0.25); background: transparent; color: inherit; border-radius: 8px 8px 0 0; padding: 6px 10px; cursor: pointer; font-size: 12px; }
        #onnx-workbench .wb-tab.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
        #onnx-workbench .wb-panel { display: none; padding: 12px; max-height: min(52vh, 460px); overflow: auto; }
        #onnx-workbench .wb-panel.active { display: block; }
        #onnx-workbench .wb-section { margin-bottom: 12px; }
        #onnx-workbench .wb-section-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.2px; font-weight: 700; opacity: 0.75; margin-bottom: 6px; }
        #onnx-workbench .wb-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
        #onnx-workbench button, #onnx-workbench select, #onnx-workbench textarea { font: inherit; }
        #onnx-workbench button { border: 1px solid rgba(127,127,127,0.35); background: #fff; color: inherit; border-radius: 8px; padding: 6px 10px; cursor: pointer; }
        #onnx-workbench button.primary { background: #1f6feb; color: #fff; border-color: #1f6feb; }
        #onnx-workbench button[disabled] { opacity: 0.45; cursor: not-allowed; }
        #onnx-workbench button.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
        #onnx-workbench .wb-chip-wrap { display: flex; flex-wrap: wrap; gap: 4px; min-height: 22px; }
        #onnx-workbench .wb-chip { background: rgba(31,111,235,0.12); border: 1px solid rgba(31,111,235,0.35); border-radius: 999px; padding: 2px 8px; font-size: 11px; }
        #onnx-workbench .wb-chip.end { background: rgba(235,111,45,0.12); border-color: rgba(235,111,45,0.35); }
        #onnx-workbench .wb-muted { opacity: 0.65; font-size: 12px; }
        #onnx-workbench .wb-status { padding: 10px 12px; border-top: 1px solid rgba(127,127,127,0.2); background: rgba(255,255,255,0.7); font-size: 12px; }
        #onnx-workbench .wb-status.error { color: #c62828; }
        #onnx-workbench .wb-task { border: 1px solid rgba(127,127,127,0.18); background: rgba(31,111,235,0.06); border-radius: 10px; padding: 8px; margin-bottom: 12px; }
        #onnx-workbench .wb-kv { display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; font-size: 12px; }
        #onnx-workbench textarea { width: 100%; min-height: 66px; border-radius: 8px; border: 1px solid rgba(127,127,127,0.25); background: rgba(255,255,255,0.9); color: inherit; padding: 8px; box-sizing: border-box; }
        #onnx-workbench .wb-result-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        #onnx-workbench .wb-result-table th, #onnx-workbench .wb-result-table td { border-top: 1px solid rgba(127,127,127,0.18); padding: 6px; text-align: left; vertical-align: top; }
        .edge-path.workbench-edge-start { stroke: #2f80ed !important; stroke-width: 1.8px !important; }
        .edge-path.workbench-edge-end { stroke: #eb6f2d !important; stroke-width: 1.8px !important; }
        @media (prefers-color-scheme: dark) {
            #onnx-workbench-bar { background: rgba(36,39,46,0.98); color: #e6edf3; border-color: rgba(255,255,255,0.08); }
            #onnx-workbench-bar button { background: #22262e; color: #e6edf3; border-color: rgba(255,255,255,0.12); }
            #onnx-workbench-bar .wb-bar-pill { background: rgba(31,111,235,0.12); border-color: rgba(255,255,255,0.08); }
            #onnx-workbench { background: rgba(36,39,46,0.98); color: #e6edf3; border-color: rgba(255,255,255,0.08); }
            #onnx-workbench .wb-header, #onnx-workbench .wb-status { background: rgba(45,49,57,0.92); }
            #onnx-workbench .wb-tab, #onnx-workbench button, #onnx-workbench textarea, #onnx-workbench select { background: #22262e; color: #e6edf3; border-color: rgba(255,255,255,0.12); }
            #onnx-workbench .wb-task { background: rgba(31,111,235,0.10); }
        }
        @media (max-width: 900px) {
            #onnx-workbench-bar { left: 8px; right: 8px; bottom: 54px; max-width: none; }
            #onnx-workbench { left: 8px; right: 8px; width: auto; max-width: none; bottom: 54px; }
            #onnx-workbench.with-bar { bottom: 116px; }
        }`;
        this.host.document.head.appendChild(style);
    }

    _createSelectionBar() {
        const bar = this.host.document.createElement('div');
        bar.id = 'onnx-workbench-bar';
        bar.innerHTML = `
            <div class="wb-bar-group">
                <button id="wb-bar-mode-start">Start</button>
                <button id="wb-bar-mode-end">End</button>
                <button id="wb-bar-clear">Clear</button>
                <button id="wb-bar-confirm" class="primary">Confirm</button>
            </div>
            <div class="wb-bar-summary" id="wb-bar-summary">Idle · Start 0 · End 0</div>
            <div class="wb-bar-group">
                <span class="wb-bar-pill" id="wb-bar-draft">Draft clean</span>
                <button id="wb-open-drawer">Advanced</button>
                <button id="wb-bar-hide">Hide</button>
            </div>
        `;
        this.host.document.body.appendChild(bar);
        this._elements.selectionBar = bar;
        this._elements.barModeStart = bar.querySelector('#wb-bar-mode-start');
        this._elements.barModeEnd = bar.querySelector('#wb-bar-mode-end');
        this._elements.barClear = bar.querySelector('#wb-bar-clear');
        this._elements.barConfirm = bar.querySelector('#wb-bar-confirm');
        this._elements.barSummary = bar.querySelector('#wb-bar-summary');
        this._elements.barDraft = bar.querySelector('#wb-bar-draft');
        this._elements.openDrawer = bar.querySelector('#wb-open-drawer');
        this._elements.barHide = bar.querySelector('#wb-bar-hide');
    }

    _injectToolbarButton() {
        const toolbar = this.host.document.getElementById('toolbar');
        if (!toolbar) {
            return;
        }
        const button = this.host.document.createElement('button');
        button.id = 'workbench-toolbar-button';
        button.className = 'toolbar-button';
        button.title = 'Model Tools';
        button.innerHTML = `
            <svg class="toolbar-icon" viewBox="0 0 100 100">
                <rect class="border" x="5" y="5" width="90" height="90" rx="18"></rect>
                <text class="fill" x="50" y="63" text-anchor="middle">M</text>
            </svg>`;
        toolbar.appendChild(button);
        this._elements.toolbarButton = button;
    }

    _createDrawer() {
        const root = this.host.document.createElement('div');
        root.id = 'onnx-workbench';
        root.innerHTML = `
            <div class="wb-header">
                <div class="wb-title">Model Tools</div>
                <button class="wb-close" id="wb-close">Close</button>
            </div>
            <div class="wb-tabs">
                <button class="wb-tab active" data-tab="crop">Crop</button>
                <button class="wb-tab" data-tab="run">Run</button>
                <button class="wb-tab" data-tab="compare">Compare</button>
                <button class="wb-tab" data-tab="activity">Activity</button>
            </div>
            <div class="wb-panel active" data-panel="crop">
                <div id="wb-task-crop"></div>
                <div class="wb-section">
                    <div class="wb-section-title">Selection Mode</div>
                    <div class="wb-row">
                        <button id="wb-mode-start">Select Start Tensor</button>
                        <button id="wb-mode-end">Select End Tensor</button>
                    </div>
                </div>
                <div class="wb-section">
                    <div class="wb-section-title">Start Tensors</div>
                    <div class="wb-chip-wrap" id="wb-start-list"></div>
                    <div class="wb-row"><button id="wb-clear-start">Clear Start</button></div>
                </div>
                <div class="wb-section">
                    <div class="wb-section-title">End Tensors</div>
                    <div class="wb-chip-wrap" id="wb-end-list"></div>
                    <div class="wb-row"><button id="wb-clear-end">Clear End</button></div>
                </div>
                <div class="wb-section">
                    <div class="wb-section-title">Confirmed Crop</div>
                    <div id="wb-artifact-summary" class="wb-muted">(none)</div>
                </div>
                <div class="wb-row">
                    <button id="wb-confirm-crop" class="primary">Confirm Crop</button>
                    <button id="wb-export-crop" disabled>Export Crop ONNX</button>
                    <button id="wb-save-crop-png">Save Crop PNG</button>
                </div>
            </div>
            <div class="wb-panel" data-panel="run">
                <div id="wb-task-run"></div>
                <div class="wb-section">
                    <div class="wb-section-title">Run Target</div>
                    <label><input type="checkbox" id="wb-use-full-graph"> Use Full Graph</label>
                </div>
                <div class="wb-section">
                    <div class="wb-section-title">Input Source</div>
                    <div class="wb-row">
                        <select id="wb-input-mode">
                            <option value="zeros">Auto / zeros</option>
                            <option value="ones">Auto / ones</option>
                            <option value="random">Auto / random</option>
                            <option value="import">Import JSON / NPZ</option>
                        </select>
                        <button id="wb-import-input">Import Input</button>
                    </div>
                    <div id="wb-import-preview" class="wb-muted">(no imported input)</div>
                </div>
                <div class="wb-section">
                    <div class="wb-section-title">Shape Overrides (JSON)</div>
                    <textarea id="wb-shape-overrides" placeholder='{"input_name": [1, 3, 224, 224]}'></textarea>
                </div>
                <div class="wb-row">
                    <button id="wb-run-inference" class="primary" disabled>Run Inference</button>
                </div>
                <div class="wb-section">
                    <div class="wb-section-title">Inference Result</div>
                    <div id="wb-run-result" class="wb-muted">(no result)</div>
                </div>
            </div>
            <div class="wb-panel" data-panel="compare">
                <div id="wb-task-compare"></div>
                <div class="wb-section">
                    <div class="wb-section-title">Compare Slots</div>
                    <div id="wb-compare-summary" class="wb-muted">(empty)</div>
                </div>
                <div class="wb-row">
                    <button id="wb-assign-a" disabled>Set As A</button>
                    <button id="wb-assign-b" disabled>Set As B</button>
                    <button id="wb-open-compare">Focus Compare</button>
                </div>
            </div>
            <div class="wb-panel" data-panel="activity">
                <div class="wb-section">
                    <div class="wb-section-title">Recent Activity</div>
                    <div id="wb-activity-list" class="wb-muted">(no activity)</div>
                </div>
            </div>
            <div class="wb-status" id="wb-status">Ready.</div>
        `;
        this.host.document.body.appendChild(root);
        this._elements.root = root;
        this._elements.header = root.querySelector('.wb-header');
        this._elements.tabsBar = root.querySelector('.wb-tabs');
        this._elements.close = root.querySelector('#wb-close');
        this._elements.tabs = Array.from(root.querySelectorAll('.wb-tab'));
        this._elements.panels = Array.from(root.querySelectorAll('.wb-panel'));
        this._elements.modeStart = root.querySelector('#wb-mode-start');
        this._elements.modeEnd = root.querySelector('#wb-mode-end');
        this._elements.startList = root.querySelector('#wb-start-list');
        this._elements.endList = root.querySelector('#wb-end-list');
        this._elements.clearStart = root.querySelector('#wb-clear-start');
        this._elements.clearEnd = root.querySelector('#wb-clear-end');
        this._elements.confirmCrop = root.querySelector('#wb-confirm-crop');
        this._elements.exportCrop = root.querySelector('#wb-export-crop');
        this._elements.saveCropPng = root.querySelector('#wb-save-crop-png');
        this._elements.artifactSummary = root.querySelector('#wb-artifact-summary');
        this._elements.useFullGraph = root.querySelector('#wb-use-full-graph');
        this._elements.inputMode = root.querySelector('#wb-input-mode');
        this._elements.importInput = root.querySelector('#wb-import-input');
        this._elements.importPreview = root.querySelector('#wb-import-preview');
        this._elements.shapeOverrides = root.querySelector('#wb-shape-overrides');
        this._elements.runInference = root.querySelector('#wb-run-inference');
        this._elements.runResult = root.querySelector('#wb-run-result');
        this._elements.assignA = root.querySelector('#wb-assign-a');
        this._elements.assignB = root.querySelector('#wb-assign-b');
        this._elements.compareSummary = root.querySelector('#wb-compare-summary');
        this._elements.openCompare = root.querySelector('#wb-open-compare');
        this._elements.activityList = root.querySelector('#wb-activity-list');
        this._elements.status = root.querySelector('#wb-status');
        this._elements.taskCrop = root.querySelector('#wb-task-crop');
        this._elements.taskRun = root.querySelector('#wb-task-run');
        this._elements.taskCompare = root.querySelector('#wb-task-compare');
    }

    _scheduleRefit() {
        if (this._renderInProgress || this._suspendRefit) {
            this._clearPendingRefit();
            return;
        }
        this._clearPendingRefit();
        this._resizeTimer = this.host.window.setTimeout(() => {
            this._resizeTimer = null;
            this._fitActiveGraph();
        }, 80);
    }

    _clearPendingRefit() {
        if (this._resizeTimer) {
            this.host.window.clearTimeout(this._resizeTimer);
            this._resizeTimer = null;
        }
    }

    _fitActiveGraph() {
        if (this._renderInProgress || this._suspendRefit) {
            return;
        }
        const target = this.view && this.view._target;
        const container = this.host.document.getElementById('target');
        if (!target || !container) {
            return;
        }
        try {
            container.scrollLeft = 0;
            container.scrollTop = 0;
            target.zoom = 0.1;
        } catch {
            // ignore fit errors
        }
    }

    async _waitForFontsReady() {
        const fonts = this.host.document && this.host.document.fonts ? this.host.document.fonts : null;
        if (fonts && fonts.ready) {
            try {
                await fonts.ready;
            } catch {
                // continue regardless of error
            }
        }
    }

    async _waitForAnimationFrames(count = 1) {
        const window = this.host.window;
        await new Promise((resolve) => {
            let remaining = Math.max(0, count);
            const tick = () => {
                if (remaining <= 0) {
                    resolve();
                    return;
                }
                remaining -= 1;
                if (typeof window.requestAnimationFrame === 'function') {
                    window.requestAnimationFrame(tick);
                } else {
                    window.setTimeout(tick, 16);
                }
            };
            tick();
        });
    }

    async _sleep(delay) {
        await new Promise((resolve) => this.host.window.setTimeout(resolve, delay));
    }

    _isTextOutsideBounds(textElement, boundsElement, tolerance = 1) {
        if (!textElement || !boundsElement) {
            return false;
        }
        const textRect = textElement.getBoundingClientRect();
        const boundsRect = boundsElement.getBoundingClientRect();
        if (textRect.width === 0 || textRect.height === 0 || boundsRect.width === 0 || boundsRect.height === 0) {
            return false;
        }
        return textRect.top < (boundsRect.top - tolerance) || textRect.bottom > (boundsRect.bottom + tolerance);
    }

    detectUnstableGraphText() {
        const canvas = this.host.document.getElementById('canvas');
        if (!canvas) {
            return false;
        }
        for (const element of canvas.querySelectorAll('.node-item')) {
            const text = element.querySelector('text');
            const path = element.querySelector('path');
            if (this._isTextOutsideBounds(text, path)) {
                return true;
            }
        }
        for (const element of canvas.querySelectorAll('.node-argument')) {
            const text = element.querySelector('text');
            const rect = element.querySelector('rect');
            if (this._isTextOutsideBounds(text, rect)) {
                return true;
            }
        }
        return false;
    }

    async renderWithStability(renderFn) {
        this._renderInProgress = true;
        this._suspendRefit = true;
        this._lastRenderRetryCount = 0;
        this._clearPendingRefit();
        this._updateButtons();
        try {
            await this._waitForFontsReady();
            await this._waitForAnimationFrames(3);
            await renderFn();
            await this._waitForAnimationFrames(2);
            if (this.detectUnstableGraphText()) {
                this._lastRenderRetryCount = 1;
                this.host.window.console.info('[ONNXWorkbenchUI] stability retry triggered', { attempts: 1 });
                await this._sleep(80);
                await this._waitForAnimationFrames(2);
                await renderFn();
                await this._waitForAnimationFrames(2);
                if (this.detectUnstableGraphText()) {
                    this.host.window.console.warn('[ONNXWorkbenchUI] graph text remained unstable after retry');
                }
            }
        } finally {
            this._renderInProgress = false;
            this._suspendRefit = false;
            this._updateButtons();
        }
    }

    _bindEvents() {
        this._elements.toolbarButton.addEventListener('click', () => this.toggleSelectionBar());
        this._elements.close.addEventListener('click', () => this.toggleDrawer(false));
        this._elements.tabs.forEach((tab) => tab.addEventListener('click', () => this._selectTab(tab.getAttribute('data-tab'))));
        const toggleStartMode = () => this._setMode(this.mode === 'start' ? null : 'start');
        const toggleEndMode = () => this._setMode(this.mode === 'end' ? null : 'end');
        this._elements.modeStart.addEventListener('click', toggleStartMode);
        this._elements.modeEnd.addEventListener('click', toggleEndMode);
        this._elements.barModeStart.addEventListener('click', toggleStartMode);
        this._elements.barModeEnd.addEventListener('click', toggleEndMode);
        this._elements.clearStart.addEventListener('click', () => this._clearSelection('start'));
        this._elements.clearEnd.addEventListener('click', () => this._clearSelection('end'));
        this._elements.barClear.addEventListener('click', () => this._clearSelection(this.mode || 'all'));
        this._elements.confirmCrop.addEventListener('click', () => this._confirmCrop());
        this._elements.barConfirm.addEventListener('click', () => this._confirmCrop());
        this._elements.openDrawer.addEventListener('click', () => this.toggleDrawer(true));
        this._elements.barHide.addEventListener('click', () => this.toggleSelectionBar(false));
        this._elements.exportCrop.addEventListener('click', () => {
            if (!this.confirmedArtifact || this.draftDirty) {
                return;
            }
            this.host._post({
                type: 'exportCropOnnx',
                artifactId: this.confirmedArtifact.id
            });
        });
        this._elements.saveCropPng.addEventListener('click', async () => {
            try {
                await this.view.export('crop-preview.png');
                this._setStatus('Save dialog opened for current graph PNG.');
            } catch (error) {
                this._setStatus(error.message || String(error), true);
            }
        });
        this._elements.importInput.addEventListener('click', () => {
            this.host._post({ type: 'importInputFile' });
        });
        this._elements.runInference.addEventListener('click', () => {
            try {
                const inputShapes = parseShapeOverrides(this._elements.shapeOverrides.value);
                this.host._post({
                    type: 'runInference',
                    artifactId: this.confirmedArtifact ? this.confirmedArtifact.id : null,
                    sessionId: this.currentSessionId,
                    useFullGraph: !!this._elements.useFullGraph.checked,
                    inputMode: this._elements.inputMode.value,
                    inputShapes,
                    importToken: this.importedInput ? this.importedInput.token : null
                });
            } catch (error) {
                this._setStatus(error.message || String(error), true);
            }
        });
        this._elements.assignA.addEventListener('click', () => {
            if (this.confirmedArtifact && !this.draftDirty) {
                this.host._post({ type: 'assignCompareSlot', slot: 'A', artifactId: this.confirmedArtifact.id });
            }
        });
        this._elements.assignB.addEventListener('click', () => {
            if (this.confirmedArtifact && !this.draftDirty) {
                this.host._post({ type: 'assignCompareSlot', slot: 'B', artifactId: this.confirmedArtifact.id });
            }
        });
        this._elements.openCompare.addEventListener('click', () => {
            this.host._post({ type: 'requestOpenCompareCenter' });
        });
        this._elements.selectionBar.addEventListener('click', (event) => event.stopPropagation());
        this._elements.root.addEventListener('click', (event) => event.stopPropagation());
        this.host.document.addEventListener('click', (event) => this._handleGraphClick(event), true);
        this.host.document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && this.mode) {
                this._setMode(null);
            }
        });
        this.host.window.addEventListener('message', (event) => this.handleHostMessage(event.data));
        this.host.window.addEventListener('resize', () => {
            this._scheduleRefit();
            this._syncWorkbenchLayout();
        });
        const targetContainer = this.host.document.getElementById('target');
        if (targetContainer && this.host.window.ResizeObserver) {
            this._resizeObserver = new this.host.window.ResizeObserver(() => this._scheduleRefit());
            this._resizeObserver.observe(targetContainer);
        }
        this.host.window.addEventListener('nnjs:model-opened', () => {
            this.currentSessionId = null;
            this.fullModelSnapshot = null;
            this.confirmedArtifact = null;
            this.importedInput = null;
            this.startKeys.clear();
            this.endKeys.clear();
            this.graphIndex = null;
            this.draftDirty = false;
            this._setMode(null);
            this._renderSelectionLists();
            this._renderArtifactSummary();
            this._updateButtons();
            this._setStatus('Legacy model loaded. ONNX workbench actions are unavailable for this file.', false);
        });
        this.host._post({ type: 'requestCompareState' });
    }

    toggleSelectionBar(force) {
        const visible = force === undefined ? !this._elements.selectionBar.classList.contains('visible') : !!force;
        this._elements.selectionBar.classList.toggle('visible', visible);
        this._elements.root.classList.toggle('with-bar', visible && this._elements.root.classList.contains('visible'));
        if (visible) {
            this._updateSelectionBar();
        }
        this._syncWorkbenchLayout();
    }

    toggleDrawer(force) {
        const visible = force === undefined ? !this._elements.root.classList.contains('visible') : !!force;
        this._elements.root.classList.toggle('visible', visible);
        this._elements.root.classList.toggle('with-bar', visible && this._elements.selectionBar.classList.contains('visible'));
        this._syncWorkbenchLayout();
    }

    _syncWorkbenchLayout() {
        const narrow = this.host.window.matchMedia && this.host.window.matchMedia('(max-width: 900px)').matches;
        const baseBottom = narrow ? 54 : 56;
        const topMargin = narrow ? 8 : 12;
        const viewportHeight = this.host.window.innerHeight || 800;
        this._elements.selectionBar.style.bottom = `${baseBottom}px`;
        const barVisible = this._elements.selectionBar.classList.contains('visible');
        const drawerVisible = this._elements.root.classList.contains('visible');
        const barHeight = barVisible ? this._elements.selectionBar.offsetHeight : 0;
        const drawerBottom = drawerVisible && barVisible ? baseBottom + barHeight + 12 : baseBottom;
        this._elements.root.style.bottom = `${drawerBottom}px`;
        const graphInset = barVisible ? barHeight + 20 : 0;
        this.host.document.documentElement.style.setProperty('--workbench-bottom-inset', `${graphInset}px`);
        const availableHeight = Math.max(220, viewportHeight - drawerBottom - topMargin);
        this._elements.root.style.maxHeight = `${availableHeight}px`;
        const headerHeight = this._elements.header ? this._elements.header.offsetHeight : 44;
        const tabsHeight = this._elements.tabsBar ? this._elements.tabsBar.offsetHeight : 40;
        const statusHeight = this._elements.status ? this._elements.status.offsetHeight : 40;
        const panelMaxHeight = Math.max(120, availableHeight - headerHeight - tabsHeight - statusHeight - 12);
        for (const panel of this._elements.panels) {
            panel.style.maxHeight = `${panelMaxHeight}px`;
        }
    }

    _clearSelection(kind) {
        if (kind === 'start') {
            this.startKeys.clear();
        } else if (kind === 'end') {
            this.endKeys.clear();
        } else {
            this.startKeys.clear();
            this.endKeys.clear();
        }
        this._markDraftDirty();
        this._renderSelectionLists();
        this._applyEdgeHighlights();
    }

    _confirmCrop() {
        if (!this.currentSessionId) {
            this._setStatus('Load an ONNX model before confirming crop.', true);
            return;
        }
        if (this.startKeys.size === 0 || this.endKeys.size === 0) {
            this._setStatus('Please select at least one start tensor and one end tensor.', true);
            return;
        }
        this.host._post({
            type: 'confirmCrop',
            sessionId: this.currentSessionId,
            startKeys: Array.from(this.startKeys),
            endKeys: Array.from(this.endKeys)
        });
    }

    _updateSelectionBar() {
        const modeLabel = this.mode ? this.mode.toUpperCase() : 'IDLE';
        const dirtyLabel = this.draftDirty ? 'Draft dirty' : 'Draft clean';
        this._elements.barSummary.textContent = `${modeLabel} · Start ${this.startKeys.size} · End ${this.endKeys.size}`;
        this._elements.barDraft.textContent = dirtyLabel;
        this._elements.barModeStart.classList.toggle('active', this.mode === 'start');
        this._elements.barModeEnd.classList.toggle('active', this.mode === 'end');
        this._syncWorkbenchLayout();
    }

    _selectTab(tabName) {
        this._elements.tabs.forEach((tab) => tab.classList.toggle('active', tab.getAttribute('data-tab') === tabName));
        this._elements.panels.forEach((panel) => panel.classList.toggle('active', panel.getAttribute('data-panel') === tabName));
    }

    _setMode(mode) {
        this.mode = mode;
        this._elements.modeStart.classList.toggle('active', mode === 'start');
        this._elements.modeEnd.classList.toggle('active', mode === 'end');
        this._updateSelectionBar();
        if (mode === 'start') {
            this._setStatus('Start tensor select mode is ON. Click tensor edges to toggle selection.');
        } else if (mode === 'end') {
            this._setStatus('End tensor select mode is ON. Click tensor edges to toggle selection.');
        } else {
            this._setStatus('Tensor select mode is OFF.');
        }
    }

    async handleHostMessage(message) {
        if (!message || typeof message.type !== 'string') {
            return;
        }
        switch (message.type) {
            case 'renderGraphSnapshot':
                await this._renderFullModel(message.model);
                break;
            case 'cropConfirmed':
                await this._handleCropConfirmed(message);
                break;
            case 'artifactExported':
                this._setStatus(`Exported: ${message.exportInfo.filePath}`);
                break;
            case 'inputImported':
                this.importedInput = { token: message.token, preview: message.preview || [] };
                this._renderImportedInputPreview();
                this._updateButtons();
                this._setStatus('Input file imported and ready to use.');
                break;
            case 'inferenceResult':
                this._renderRunResult(message.result);
                this._setStatus('Inference completed.');
                break;
            case 'compareStateUpdate':
                this.compareState = message.state;
                this._renderCompareSummary();
                break;
            case 'tensorPreviewResult': {
                const pending = message.requestId ? this._tensorPreviewRequests.get(message.requestId) : null;
                if (pending) {
                    this._tensorPreviewRequests.delete(message.requestId);
                    this.host.window.clearTimeout(pending.timer);
                    if (message.ok) {
                        pending.resolve(message.preview || null);
                    } else {
                        pending.reject(new Error(message.error || 'Tensor preview failed.'));
                    }
                }
                break;
            }
            case 'taskState':
                this.taskState = message.task;
                this._renderTaskState();
                break;
            case 'activityLog':
                this.activity = message.entries || [];
                this._renderActivity();
                break;
            case 'fileSaved':
                this._setStatus(`Saved: ${message.path}`);
                break;
            case 'clipboardCopied':
                this._setStatus(`${message.label || 'Text'} copied to clipboard.`);
                break;
            case 'clipboardError':
                this._setStatus(message.message || 'Clipboard operation failed.', true);
                break;
            default:
                break;
        }
    }

    async _renderFullModel(snapshot) {
        this.fullModelSnapshot = snapshot;
        this.currentSessionId = snapshot.sessionId;
        this.confirmedArtifact = null;
        this.importedInput = null;
        this.draftDirty = false;
        this.startKeys.clear();
        this.endKeys.clear();
        await this.renderWithStability(async () => {
            const model = new Model(snapshot, { previewLoader: (tensorName) => this._requestTensorPreview(snapshot.sessionId, tensorName) });
            await this.view._updateTarget(model, [{ target: model.modules[0], signature: null }]);
        });
        this._refreshGraphIndex();
        this._setMode(null);
        this._renderSelectionLists();
        this._renderArtifactSummary();
        this._renderImportedInputPreview();
        this._renderRunResult(null);
        this._updateButtons();
        this._fitActiveGraph();
        this._setStatus('ONNX model loaded. Select crop boundaries and confirm when ready.');
    }

    async _handleCropConfirmed(message) {
        this.confirmedArtifact = message.artifact;
        this.draftDirty = false;
        await this.renderWithStability(async () => {
            const graph = new Graph(message.graph, { previewLoader: (tensorName) => this._requestTensorPreview(this.currentSessionId, tensorName) });
            await this.view.pushTarget(graph, null);
        });
        this._refreshGraphIndex();
        this._setMode(null);
        this._renderArtifactSummary();
        this._updateButtons();
        this._applyEdgeHighlights();
        this._fitActiveGraph();
        this._setStatus('Crop confirmed. Export, run inference, or assign to compare.');
        this._selectTab('crop');
        if (!this._elements.selectionBar.classList.contains('visible')) {
            this.toggleDrawer(true);
        }
    }

    _markDraftDirty() {
        if (!this.confirmedArtifact) {
            return;
        }
        this.draftDirty = true;
        this._renderArtifactSummary();
        this._updateButtons();
    }

    _requestTensorPreview(sessionId, tensorName) {
        const resolvedSessionId = sessionId || this.currentSessionId;
        if (!resolvedSessionId) {
            return Promise.reject(new Error('Model session is not ready for tensor preview.'));
        }
        const requestId = `tensor-preview-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        return new Promise((resolve, reject) => {
            const timer = this.host.window.setTimeout(() => {
                this._tensorPreviewRequests.delete(requestId);
                reject(new Error(`Tensor preview timed out for '${tensorName}'.`));
            }, 15000);
            this._tensorPreviewRequests.set(requestId, { resolve, reject, timer });
            this.host._post({
                type: 'requestTensorPreview',
                requestId,
                sessionId: resolvedSessionId,
                tensorName
            });
        });
    }

    _handleGraphClick(event) {
        if (this._renderInProgress) {
            return;
        }
        if (!this.mode) {
            return;
        }
        if (this._elements.selectionBar.contains(event.target)) {
            return;
        }
        if (this._elements.root.contains(event.target)) {
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
        this._markDraftDirty();
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
            if (!tensorByKey.has(key)) {
                keyToValue.set(key, value);
                tensorByKey.set(key, {
                    key,
                    value,
                    type: value.type || null,
                    initializer: value.initializer || null,
                    producer: null,
                    consumers: new Set()
                });
            }
            return key;
        };
        graph.nodes.forEach((node, index) => {
            const entry = {
                id: `n${index}`,
                node,
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
                    tensorByKey.get(key).consumers.add(entry.id);
                }
                entry.inputs.push({ name: argument.name || '', keys });
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
                        tensor.producer = entry.id;
                    }
                }
                entry.outputs.push({ name: argument.name || '', keys });
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
        return { graph, nodes, keyToValue, tensorByKey, graphInputs, graphOutputs };
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
            const items = Array.from(keys.values());
            if (items.length === 0) {
                const empty = this.host.document.createElement('span');
                empty.className = 'wb-muted';
                empty.textContent = '(none)';
                container.appendChild(empty);
                return;
            }
            for (const key of items) {
                const chip = this.host.document.createElement('span');
                chip.className = isEnd ? 'wb-chip end' : 'wb-chip';
                chip.textContent = key.startsWith('__tensor_') ? `[unnamed] ${key}` : key;
                container.appendChild(chip);
            }
        };
        render(this._elements.startList, this.startKeys, false);
        render(this._elements.endList, this.endKeys, true);
        this._updateSelectionBar();
        this._updateButtons();
    }

    _applyEdgeHighlights() {
        const document = this.host.document;
        Array.from(document.querySelectorAll('.workbench-edge-start')).forEach((element) => element.classList.remove('workbench-edge-start'));
        Array.from(document.querySelectorAll('.workbench-edge-end')).forEach((element) => element.classList.remove('workbench-edge-end'));
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
        apply(this.startKeys, 'workbench-edge-start');
        apply(this.endKeys, 'workbench-edge-end');
    }

    _renderArtifactSummary() {
        if (!this.confirmedArtifact) {
            this._elements.artifactSummary.textContent = '(none)';
            return;
        }
        const suffix = this.draftDirty ? ' · Stale' : ' · Confirmed';
        this._elements.artifactSummary.textContent = `${this.confirmedArtifact.summary.modelName} · ${this.confirmedArtifact.summary.nodeCount} nodes · ${this.confirmedArtifact.summary.inputCount} in / ${this.confirmedArtifact.summary.outputCount} out${suffix}`;
    }

    _renderImportedInputPreview() {
        if (!this.importedInput || !Array.isArray(this.importedInput.preview) || this.importedInput.preview.length === 0) {
            this._elements.importPreview.textContent = '(no imported input)';
            return;
        }
        this._elements.importPreview.innerHTML = this.importedInput.preview.map((entry) => `${entry.name}: ${entry.dtype || ''} ${Array.isArray(entry.shape) ? JSON.stringify(entry.shape) : ''}`).join('<br>');
    }

    _renderRunResult(result) {
        if (!result) {
            this._elements.runResult.textContent = '(no result)';
            return;
        }
        const rows = (result.outputsSummary || []).map((output) => `<tr><td><code>${output.name}</code></td><td>${output.dtype}</td><td>${JSON.stringify(output.shape || [])}</td><td>${output.summary ? `min=${output.summary.min}, max=${output.summary.max}, mean=${output.summary.mean}` : ''}</td></tr>`).join('');
        this._elements.runResult.innerHTML = `
            <div class="wb-kv">
                <div>Run ID</div><div><code>${result.runId}</code></div>
                <div>Input Mode</div><div>${result.inputMode}</div>
                <div>Outputs</div><div>${(result.outputsSummary || []).length}</div>
                <div>Cache</div><div><code>${result.cacheKey}</code></div>
            </div>
            <table class="wb-result-table"><thead><tr><th>Name</th><th>DType</th><th>Shape</th><th>Summary</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    _renderCompareSummary() {
        const state = this.compareState;
        if (!state || (!state.slotA && !state.slotB)) {
            this._elements.compareSummary.textContent = '(empty)';
            return;
        }
        const slotLine = (label, slot) => slot
            ? `${label}: ${slot.summary.modelName} · ${slot.summary.nodeCount} nodes · ${slot.summary.inputCount} in / ${slot.summary.outputCount} out`
            : `${label}: (empty)`;
        const lines = [slotLine('A', state.slotA), slotLine('B', state.slotB)];
        if (state.compareResult && state.compareResult.summary) {
            lines.push(`Latest compare max diff: ${state.compareResult.summary.maxDiffOutput} (maxAbs=${state.compareResult.summary.maxAbs})`);
        }
        this._elements.compareSummary.innerHTML = lines.map((line) => `<div>${line}</div>`).join('');
    }

    _renderTaskState() {
        const task = this.taskState;
        const renderTarget = (container, show) => {
            container.innerHTML = show && task && task.busy
                ? `<div class="wb-task"><div><strong>${task.stage || 'Working'}</strong></div><div class="wb-muted">${task.message || ''}</div><div class="wb-row" style="margin-top:8px;"><button id="wb-cancel-task">Cancel</button></div></div>`
                : '';
        };
        renderTarget(this._elements.taskCrop, true);
        renderTarget(this._elements.taskRun, true);
        renderTarget(this._elements.taskCompare, true);
        const cancelButton = this.host.document.getElementById('wb-cancel-task');
        if (cancelButton) {
            cancelButton.addEventListener('click', () => this.host._post({ type: 'cancelTask' }), { once: true });
        }
        this._updateButtons();
    }

    _renderActivity() {
        if (!Array.isArray(this.activity) || this.activity.length === 0) {
            this._elements.activityList.textContent = '(no activity)';
            return;
        }
        this._elements.activityList.innerHTML = this.activity.map((entry) => `<div style="margin-bottom:8px;"><strong>${entry.message}</strong><div class="wb-muted">${entry.createdAt}</div><div class="wb-muted">${toText(entry.detail)}</div></div>`).join('');
    }

    _updateButtons() {
        const hasSelection = this.startKeys.size > 0 && this.endKeys.size > 0;
        const hasConfirmed = !!this.confirmedArtifact;
        const canUseConfirmed = hasConfirmed && !this.draftDirty;
        const isBusy = !!(this.taskState && this.taskState.busy) || this._renderInProgress;
        this._elements.confirmCrop.disabled = !hasSelection || !this.currentSessionId || isBusy;
        this._elements.barConfirm.disabled = this._elements.confirmCrop.disabled;
        this._elements.exportCrop.disabled = !canUseConfirmed || isBusy;
        this._elements.runInference.disabled = (!canUseConfirmed && !this._elements.useFullGraph.checked) || !this.currentSessionId || isBusy || (this._elements.inputMode.value === 'import' && !this.importedInput);
        this._elements.assignA.disabled = !canUseConfirmed || isBusy;
        this._elements.assignB.disabled = !canUseConfirmed || isBusy;
        this._elements.importInput.disabled = isBusy;
        this._elements.barModeStart.disabled = !this.currentSessionId || isBusy;
        this._elements.barModeEnd.disabled = !this.currentSessionId || isBusy;
        this._elements.barClear.disabled = (this.startKeys.size === 0 && this.endKeys.size === 0) || isBusy;
        this._elements.openDrawer.disabled = isBusy;
        this._updateSelectionBar();
    }

    _setStatus(message, isError = false) {
        this._elements.status.textContent = message;
        this._elements.status.classList.toggle('error', !!isError);
        this._syncWorkbenchLayout();
    }
}
