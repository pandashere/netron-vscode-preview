#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { createMockPrivateProvider } = require('./fixtures/mock_private_provider');

const root = path.resolve(__dirname, '..');
const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-copy-export-host-'));
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

function disposable() {
    return { dispose() {} };
}

function delay(ms = 0) {
    return new Promise((resolve) => setTimeout(resolve, ms));
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

function writeExporter(name, script) {
    const dir = path.join(tempHome, '.netron', 'vscode-preview', 'exporters', name);
    fs.mkdirSync(dir, { recursive: true });
    const scriptPath = path.join(dir, `${name}.js`);
    fs.writeFileSync(scriptPath, script);
    fs.writeFileSync(path.join(dir, 'exporter.json'), `${JSON.stringify({
        id: name,
        label: name,
        command: process.execPath,
        args: [scriptPath],
        timeoutMs: 5000
    }, null, 2)}\n`);
}

function installExporters() {
    writeExporter('copy-success', `
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  const context = JSON.parse(input || '{}');
  process.stdout.write('SECRET_EXPORT_TEXT artifact=' + context.artifact.id + '\\n');
});
`);
    writeExporter('copy-fail', `
process.stdin.resume();
process.stderr.write('copy failed intentionally\\n');
process.exit(7);
`);
}

async function main() {
    installExporters();
    const createdPanels = [];
    const registered = [];
    const clipboardWrites = [];
    const errors = [];
    const vscodeMock = {
        Uri,
        ViewColumn: { Beside: 2 },
        window: {
            createOutputChannel() {
                return { appendLine() {}, dispose() {} };
            },
            registerWebviewViewProvider() {
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
            executeCommand() {
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
            id: 'private-copy',
            extension: '.pcopy'
        }));

        const open = registered.find((item) => item.id === 'netronPreview.openPreview');
        assert(open, 'Open preview command should be registered.');
        await open.callback(Uri.file('/tmp/model.pcopy'));
        const panel = createdPanels[createdPanels.length - 1];
        await panel.webview._handler({ type: 'ready' });
        await waitFor(() => panel.webview.posted.find((message) => message.type === 'renderGraphSnapshot'), 'Model did not render.');
        await panel.webview._handler({ type: 'confirmCrop', startKeys: ['x'], endKeys: ['y'] });
        const cropMessage = await waitFor(() => panel.webview.posted.find((message) => message.type === 'cropConfirmed'), 'Crop did not confirm.');

        panel.webview._handler({
            type: 'copyExportText',
            artifactId: cropMessage.artifact.id,
            exporterId: 'copy-success'
        });
        await waitFor(() => panel.webview.posted.find((message) => message.type === 'exportTextCopied'), 'Copy Export Text did not report success.');
        assert(clipboardWrites.length === 1 && /SECRET_EXPORT_TEXT/.test(clipboardWrites[0]), 'Successful export text should be copied to clipboard.');
        const successActivity = panel.webview.posted.filter((message) => message.type === 'activityLog').pop();
        const successSerialized = JSON.stringify(successActivity);
        assert(successSerialized.includes('Export text copied'), 'Success activity should be recorded.');
        assert(!successSerialized.includes('SECRET_EXPORT_TEXT'), 'Success activity must not store exported text.');

        panel.webview._handler({
            type: 'copyExportText',
            artifactId: cropMessage.artifact.id,
            exporterId: 'copy-fail'
        });
        await waitFor(() => panel.webview.posted.find((message) => message.type === 'exportTextError'), 'Copy Export Text did not report failure.');
        assert(clipboardWrites.length === 1, 'Failed export should not update clipboard.');
        const failureActivity = panel.webview.posted.filter((message) => message.type === 'activityLog').pop();
        const failureSerialized = JSON.stringify(failureActivity);
        assert(failureSerialized.includes('Export text failed'), 'Failure activity should be recorded.');
        assert(failureSerialized.includes('copy-fail'), 'Failure activity should include exporter id.');
        assert(!failureSerialized.includes('SECRET_EXPORT_TEXT'), 'Failure activity must not leak prior exported text.');
        const notifyError = panel.webview.posted.find((message) => message.type === 'notify' && message.level === 'error' && /Exporter exited with code 7/.test(message.message || ''));
        assert(notifyError || errors.some((message) => /Exporter exited with code 7/.test(message)), 'Failed copy should surface an error notification.');

        extension.deactivate();
        console.log('copy export host flow ok');
    } finally {
        Module._load = originalLoad;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
