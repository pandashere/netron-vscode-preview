#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const root = path.resolve(__dirname, '..');
const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-provider-api-'));
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

function createProvider(id, extension) {
    return {
        id,
        label: id.toUpperCase(),
        capabilities: {},
        canOpen(uri) {
            const fileName = uri && (uri.fsPath || uri.path || String(uri));
            return typeof fileName === 'string' && fileName.endsWith(extension);
        },
        async loadModel(uri) {
            return { id: `${id}-session`, format: id, filePath: uri.fsPath || uri.path || String(uri) };
        }
    };
}

function disposable() {
    return { dispose() {} };
}

async function main() {
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
                throw new Error('createWebviewPanel should not be called during provider API smoke.');
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
            clipboard: { writeText() { return Promise.resolve(); } },
            openExternal() { return Promise.resolve(); }
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
        const preActivationDisposable = extension.registerFormatProvider(createProvider('private-before', '.before'));
        assert(extension.getFormatProviders().some((provider) => provider.id === 'private-before'), 'Pre-activation provider should be listed.');

        let duplicateFailed = false;
        try {
            extension.registerFormatProvider(createProvider('private-before', '.dup'));
        } catch (error) {
            duplicateFailed = /Duplicate provider id/.test(error.message);
        }
        assert(duplicateFailed, 'Duplicate pre-activation provider should fail.');

        const api = await extension.activate({
            extensionUri: Uri.file(root),
            subscriptions: []
        });
        assert(api && typeof api.registerFormatProvider === 'function', 'activate() should return provider API.');
        assert(api.getFormatProviders().some((provider) => provider.id === 'onnx'), 'ONNX provider should be registered after activation.');
        assert(api.getFormatProviders().some((provider) => provider.id === 'dev-ir-a'), 'Dev IR A provider should be registered after activation.');
        assert(api.getFormatProviders().some((provider) => provider.id === 'dev-ir-b'), 'Dev IR B provider should be registered after activation.');
        assert(api.getFormatProviders().some((provider) => provider.id === 'private-before'), 'Queued provider should register during activation.');

        const postActivationDisposable = api.registerFormatProvider(createProvider('private-after', '.after'));
        assert(api.getFormatProviders().some((provider) => provider.id === 'private-after'), 'Post-activation provider should be registered.');
        assert(api.getFormatProviderDiagnostics('private-after').errors.length === 0, 'Post-activation provider diagnostics should be clean.');

        postActivationDisposable.dispose();
        assert(!api.getFormatProviders().some((provider) => provider.id === 'private-after'), 'Disposing provider registration should unregister it.');

        preActivationDisposable.dispose();
        assert(!api.getFormatProviders().some((provider) => provider.id === 'private-before'), 'Disposing queued provider registration should unregister it after activation.');

        extension.deactivate();
        console.log('extension provider api ok');
    } finally {
        Module._load = originalLoad;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
