#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const root = path.resolve(__dirname, '..');
const extensionSource = fs.readFileSync(path.join(root, 'extension.js'), 'utf8');
const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-ai-panel-'));
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

function createView() {
    const webview = {
        cspSource: 'vscode-webview:',
        html: '',
        options: {},
        posted: [],
        _handler: null,
        postMessage(message) {
            this.posted.push(message);
            return Promise.resolve(true);
        },
        onDidReceiveMessage(handler) {
            this._handler = handler;
            return { dispose() {} };
        }
    };
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

async function main() {
    assert(extensionSource.includes('<button id="copyResult"'), 'AI panel should expose Copy Result button.');
    assert(extensionSource.includes('<button id="cancelTask"'), 'AI panel should expose Cancel button.');
    assert(!/<textarea[^>]*id=["'](?:ai|prompt|input|message)/i.test(extensionSource), 'AI panel should not include manual text input.');
    assert(!/markdown|marked|innerHTML\s*=\s*result\.text|innerHTML\s*=\s*aiState\.result/i.test(extensionSource), 'AI panel should not render analyzer output as markdown/HTML.');
    assert(extensionSource.includes("el('result').textContent = result.text"), 'AI panel should render analyzer output through textContent.');
    assert(extensionSource.includes("vscode.postMessage({ type: 'copyText', text: aiState.result.text, label: 'AI Result' })"), 'Copy Result should post text to extension clipboard handler.');

    const registeredViews = new Map();
    const clipboardWrites = [];
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
                throw new Error('Model panel should not be created during AI panel smoke.');
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
            registerCommand() {
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
        await extension.activate({
            extensionUri: Uri.file(root),
            extensionPath: root,
            subscriptions: []
        });
        const aiProvider = registeredViews.get('netronAI.analysisView');
        assert(aiProvider, 'AI Analysis WebviewView provider should be registered.');
        const aiView = createView();
        aiProvider.resolveWebviewView(aiView);
        assert(aiView.webview.html.includes('AI Analysis'), 'AI panel HTML should be initialized.');
        assert(!/<textarea/i.test(aiView.webview.html), 'AI panel HTML should not include textarea input.');
        assert(aiView.webview.html.includes("el('result').textContent = result.text"), 'AI panel HTML should use textContent for result output.');
        assert(typeof aiView.webview._handler === 'function', 'AI panel message handler should be registered.');
        await aiView.webview._handler({ type: 'ready' });
        assert(aiView.webview.posted.some((message) => message.type === 'aiStateUpdate'), 'AI panel should receive initial AI state on ready.');
        await aiView.webview._handler({ type: 'copyText', text: 'plain <b>not html</b>', label: 'AI Result' });
        assert(clipboardWrites.length === 1 && clipboardWrites[0] === 'plain <b>not html</b>', 'AI panel Copy Result should use VS Code clipboard with raw plain text.');
        assert(aiView.webview.posted.some((message) => message.type === 'clipboardCopied' && message.label === 'AI Result'), 'AI panel should acknowledge copied result.');
        extension.deactivate();
        console.log('ai panel contract ok');
    } finally {
        Module._load = originalLoad;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
