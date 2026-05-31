#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const root = path.resolve(__dirname, '..');
const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-vscode-activation-'));
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

function disposable(label, collection) {
    return {
        dispose() {
            collection.push(label);
        }
    };
}

async function main() {
    const registeredCommands = [];
    const registeredViews = [];
    const disposed = [];
    const vscodeMock = {
        Uri,
        ViewColumn: { Beside: 2 },
        window: {
            createOutputChannel(name) {
                return {
                    name,
                    appendLine() {},
                    dispose() {
                        disposed.push(`output:${name}`);
                    }
                };
            },
            registerWebviewViewProvider(id, provider, options) {
                registeredViews.push({ id, provider, options });
                return disposable(`view:${id}`, disposed);
            },
            registerCommand() {
                throw new Error('registerCommand should be accessed through vscode.commands.');
            },
            createWebviewPanel() {
                throw new Error('createWebviewPanel should not be called during activation smoke.');
            },
            showErrorMessage() {},
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
                registeredCommands.push({ id, callback });
                return disposable(`command:${id}`, disposed);
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
                writeText() {
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
        const context = {
            extensionUri: Uri.file(root),
            subscriptions: []
        };
        await extension.activate(context);
        for (const id of [
            'netronPreview.openPreview',
            'netronPreview.openCompareCenter',
            'netronPreview.clearCompareCenter',
            'netronPreview.openAiAnalysis'
        ]) {
            assert(registeredCommands.some((item) => item.id === id), `Missing registered command: ${id}`);
        }
        for (const id of ['netronCompare.compareView', 'netronAI.analysisView']) {
            assert(registeredViews.some((item) => item.id === id), `Missing registered WebviewView provider: ${id}`);
        }
        assert(fs.existsSync(path.join(tempHome, '.netron', 'vscode-preview', 'exporters')), 'Exporter registry directory was not initialized.');
        assert(fs.existsSync(path.join(tempHome, '.netron', 'vscode-preview', 'analyzers')), 'Analyzer registry directory was not initialized.');
        extension.deactivate();
        for (const item of context.subscriptions) {
            if (item && typeof item.dispose === 'function') {
                item.dispose();
            }
        }
        console.log('extension activation ok', { commands: registeredCommands.length, views: registeredViews.length });
    } finally {
        Module._load = originalLoad;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
