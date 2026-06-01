#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { createMockPrivateProvider } = require('./fixtures/mock_private_provider');

const root = path.resolve(__dirname, '..');
const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-ai-task-host-'));
process.env.HOME = tempHome;

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

class Uri {
    constructor(fsPath, scheme = 'file') {
        this.fsPath = fsPath;
        this.path = fsPath;
        this.scheme = scheme;
    }

    toString() {
        return this.scheme === 'file' ? this.fsPath : `${this.scheme}:${this.path}`;
    }

    static file(filePath) {
        return new Uri(filePath, 'file');
    }

    static joinPath(base, ...segments) {
        return Uri.file(path.join(base.fsPath || base.path || String(base), ...segments));
    }

    static parse(value) {
        const match = /^([a-z][a-z0-9+.-]*):(.*)$/i.exec(String(value));
        return new Uri(match ? match[2] : String(value), match ? match[1] : 'file');
    }
}

function createWebview() {
    const posted = [];
    return {
        posted,
        _handler: null,
        html: '',
        options: {},
        asWebviewUri(uri) {
            return { toString: () => `vscode-resource:${uri.fsPath || uri.path}` };
        },
        postMessage(message) {
            posted.push(message);
            return Promise.resolve(true);
        },
        onDidReceiveMessage(handler) {
            this._handler = handler;
            return { dispose() {} };
        }
    };
}

function createPanel() {
    const webview = createWebview();
    return {
        webview,
        title: '',
        onDidDispose() {
            return { dispose() {} };
        }
    };
}

function createView() {
    const webview = createWebview();
    return {
        webview,
        title: '',
        description: '',
        visible: true,
        show() {
            this.visible = true;
        },
        onDidDispose() {
            return { dispose() {} };
        },
        onDidChangeVisibility() {
            return { dispose() {} };
        }
    };
}

function disposable() {
    return { dispose() {} };
}

function delay(ms = 0) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function writeTool(kind, name, script, timeoutMs = 5000, manifestOverrides = {}) {
    const dir = path.join(tempHome, '.netron', 'vscode-preview', `${kind}s`, name);
    fs.mkdirSync(dir, { recursive: true });
    const scriptPath = path.join(dir, `${name}.js`);
    fs.writeFileSync(scriptPath, script);
    const manifest = {
        id: name,
        label: name,
        command: process.execPath,
        args: [scriptPath],
        timeoutMs,
        ...manifestOverrides
    };
    fs.writeFileSync(path.join(dir, `${kind}.json`), `${JSON.stringify(manifest, null, 2)}\n`);
}

function installTools() {
    const promptCapturePath = path.join(tempHome, 'prompt-analyzer-input.json');
    writeTool('exporter', 'echo-context', `
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  const context = JSON.parse(input || '{}');
  process.stdout.write('artifact=' + context.artifact.id + '\\n');
});
`);
    writeTool('analyzer', 'slow-analyzer', `
let input = '';
process.stdin.resume();
process.stdin.on('data', () => {});
process.stdin.on('end', () => {
  setTimeout(() => process.stdout.write('analysis complete\\n'), 2000);
});
`, 10000);
    writeTool('analyzer', 'prompt-analyzer', `
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  require('fs').writeFileSync(process.env.PROMPT_CAPTURE_PATH, input);
  const payload = JSON.parse(input);
  process.stdout.write([
    payload.kind,
    payload.exportedText.trim(),
    payload.userInputs.focus
  ].join('\\n'));
});
`, 5000, {
        description: 'Prompt analyzer description.',
        env: {
            PROMPT_CAPTURE_PATH: promptCapturePath
        },
        userInputs: [
            {
                id: 'focus',
                label: 'Focus',
                placeholder: 'Focus area',
                required: true,
                multiline: true
            }
        ]
    });
    return { promptCapturePath };
}

async function waitFor(predicate, message, timeoutMs = 3000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        const value = predicate();
        if (value) {
            return value;
        }
        await delay(25);
    }
    throw new Error(message);
}

async function runAndCancelAnalysis({ panel, cropMessage, clipboardWrites, cancel, cancelledPredicate }) {
    const startIndex = panel.webview.posted.length;
    panel.webview._handler({
        type: 'runAiAnalysis',
        artifactId: cropMessage.artifact.id,
        exporterId: 'echo-context',
        analyzerId: 'slow-analyzer'
    });
    await waitFor(() => panel.webview.posted.slice(startIndex).find((message) => message.type === 'toolStateUpdate' && message.state && message.state.task && message.state.task.kind === 'analysis'), 'Analysis task did not start.');

    panel.webview._handler({
        type: 'copyExportText',
        artifactId: cropMessage.artifact.id,
        exporterId: 'echo-context'
    });
    await waitFor(() => panel.webview.posted.slice(startIndex).find((message) => message.type === 'notify' && /Another export\/analysis task is running/.test(message.message || '')), 'Copy Export Text should be rejected while analysis is running.');
    assert(clipboardWrites.length === 0, 'Clipboard should not be written by rejected concurrent copy.');

    await cancel();
    const cancelled = await waitFor(cancelledPredicate, 'Analysis did not report cancelled.');
    assert(/exited|timed out|signal|SIGTERM/i.test(cancelled.message), 'Cancelled task should include process termination reason.');
    const taskCleared = panel.webview.posted.filter((message) => message.type === 'toolStateUpdate').pop();
    assert(taskCleared && taskCleared.state && taskCleared.state.task === null, 'Global task should be cleared after cancellation.');
}

async function main() {
    const toolPaths = installTools();
    const createdPanels = [];
    const registered = [];
    const registeredViews = new Map();
    const clipboardWrites = [];
    const errors = [];
    const vscodeMock = {
        Uri,
        ViewColumn: { Beside: 2 },
        window: {
            createOutputChannel() {
                return { appendLine() {}, dispose() {} };
            },
            registerWebviewViewProvider(id, provider) {
                registeredViews.set(id, provider);
                return disposable();
            },
            createWebviewPanel() {
                const panel = createPanel();
                createdPanels.push(panel);
                return panel;
            },
            showErrorMessage(message) {
                errors.push(message);
            },
            showWarningMessage() {},
            showInformationMessage() {},
            showOpenDialog() {
                return Promise.resolve([]);
            },
            showSaveDialog() {
                return Promise.resolve(null);
            }
        },
        commands: {
            registerCommand(id, callback) {
                registered.push({ id, callback });
                return disposable();
            },
            executeCommand(id) {
                if (id === 'workbench.view.extension.netronComparePanel') {
                    const provider = registeredViews.get('netronAI.analysisView');
                    if (provider && !vscodeMock._aiView) {
                        vscodeMock._aiView = createView();
                        provider.resolveWebviewView(vscodeMock._aiView);
                    }
                }
                return Promise.resolve();
            }
        },
        workspace: {
            workspaceFolders: [{ uri: Uri.file(root) }],
            fs: {
                readFile(uri) {
                    return fs.promises.readFile(uri.fsPath || uri.path);
                },
                writeFile(uri, bytes) {
                    return fs.promises.writeFile(uri.fsPath || uri.path, Buffer.from(bytes));
                }
            }
        },
        env: {
            clipboard: {
                writeText(text) {
                    clipboardWrites.push(text);
                    return Promise.resolve();
                }
            },
            openExternal() {
                return Promise.resolve();
            }
        }
    };

    const originalLoad = Module._load;
    Module._load = function patchedLoad(request, parent, isMain) {
        if (request === 'vscode') {
            return vscodeMock;
        }
        return originalLoad.call(this, request, parent, isMain);
    };

    try {
        const extension = require(path.join(root, 'extension.js'));
        const api = await extension.activate({
            extensionUri: Uri.file(root),
            extensionPath: root,
            subscriptions: []
        });
        api.registerFormatProvider(createMockPrivateProvider({
            id: 'private-ai-task',
            extension: '.pai'
        }));

        const open = registered.find((item) => item.id === 'netronPreview.openPreview');
        assert(open, 'Open preview command should be registered.');
        await open.callback(Uri.file('/tmp/model.pai'));
        const panel = createdPanels[createdPanels.length - 1];
        await panel.webview._handler({ type: 'ready' });
        await waitFor(() => panel.webview.posted.find((message) => message.type === 'renderGraphSnapshot'), 'Model did not render.');
        await panel.webview._handler({ type: 'confirmCrop', startKeys: ['x'], endKeys: ['y'] });
        const cropMessage = await waitFor(() => panel.webview.posted.find((message) => message.type === 'cropConfirmed'), 'Crop did not confirm.');

        await panel.webview._handler({
            type: 'runAiAnalysis',
            artifactId: cropMessage.artifact.id,
            exporterId: 'echo-context',
            analyzerId: 'prompt-analyzer',
            analyzerInputs: { focus: 'check constants' }
        });
        const promptSucceeded = await waitFor(() => panel.webview.posted.find((message) => message.type === 'aiAnalysisStatus' && message.status === 'succeeded'), 'Prompt analysis did not complete.');
        assert(promptSucceeded, 'Prompt analysis should report success.');
        const promptPayload = JSON.parse(fs.readFileSync(toolPaths.promptCapturePath, 'utf8'));
        assert(promptPayload.kind === 'netron-analyzer-input', 'Prompt analyzer should receive JSON envelope.');
        assert(/artifact=/.test(promptPayload.exportedText), 'Prompt analyzer envelope should include exported text.');
        assert(promptPayload.userInputs.focus === 'check constants', 'Prompt analyzer envelope should include user input.');

        await runAndCancelAnalysis({
            panel,
            cropMessage,
            clipboardWrites,
            cancel: () => panel.webview._handler({ type: 'cancelAiTask' }),
            cancelledPredicate: () => panel.webview.posted.find((message) => message.type === 'aiAnalysisStatus' && message.status === 'cancelled')
        });

        const aiView = vscodeMock._aiView;
        assert(aiView && typeof aiView.webview._handler === 'function', 'AI Analysis view should be created and have a message handler.');
        await aiView.webview._handler({ type: 'ready' });
        await delay(25);
        const aiStartIndex = aiView.webview.posted.length;
        await runAndCancelAnalysis({
            panel,
            cropMessage,
            clipboardWrites,
            cancel: () => aiView.webview._handler({ type: 'cancelTask' }),
            cancelledPredicate: () => {
                const stateMessage = aiView.webview.posted.slice(aiStartIndex).find((message) => message.type === 'aiStateUpdate' && message.state && message.state.status === 'cancelled');
                return stateMessage ? { message: stateMessage.state.error && stateMessage.state.error.message ? stateMessage.state.error.message : stateMessage.state.message } : null;
            }
        });
        assert(errors.length === 0, `No VS Code error toast expected during cancellation: ${errors.join('; ')}`);

        extension.deactivate();
        console.log('ai task host flow ok');
    } finally {
        Module._load = originalLoad;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
