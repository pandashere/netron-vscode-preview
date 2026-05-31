#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { createMockPrivateProvider } = require('./fixtures/mock_private_provider');

const root = path.resolve(__dirname, '..');
const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-provider-export-'));
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
            return {
                toString() {
                    return `vscode-resource:${uri.fsPath || uri.path}`;
                }
            };
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

function delay() {
    return new Promise((resolve) => setImmediate(resolve));
}

async function main() {
    const createdPanels = [];
    const saveDialogs = [];
    const infoMessages = [];
    const registered = [];
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
            showErrorMessage() {},
            showWarningMessage() {},
            showInformationMessage(message) {
                infoMessages.push(message);
            },
            showOpenDialog() {
                return Promise.resolve([]);
            },
            showSaveDialog(options) {
                saveDialogs.push(options);
                return Promise.resolve(Uri.file(path.join(tempHome, 'exported.private')));
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
        const provider = createMockPrivateProvider({
            id: 'private-export',
            extension: '.pex'
        });
        const api = await extension.activate({
            extensionUri: Uri.file(root),
            extensionPath: root,
            subscriptions: []
        });
        api.registerFormatProvider(provider);

        const modelUri = Uri.file('/tmp/model.pex');
        const open = registered.find((item) => item.id === 'netronPreview.openPreview');
        assert(open, 'Open preview command should be registered.');

        const devIrUri = Uri.file(path.join(root, 'testdata', 'dev-ir', 'model-a.netronir.json'));
        await open.callback(devIrUri);
        const devIrPanel = createdPanels[createdPanels.length - 1];
        assert(devIrPanel && typeof devIrPanel.webview._handler === 'function', 'Dev IR model panel message handler should be registered.');
        await devIrPanel.webview._handler({ type: 'ready' });
        await delay();
        const devIrRender = devIrPanel.webview.posted.find((message) => message.type === 'renderGraphSnapshot');
        assert(devIrRender && devIrRender.provider && devIrRender.provider.id === 'dev-ir-a', 'Built-in Dev IR A provider should render through host snapshot.');
        assert(devIrRender.model && devIrRender.model.sessionId === devIrRender.sessionId, 'Host render model should carry sessionId for webview crop gating.');
        assert(devIrRender.model.graph.nodes[0].type.category === 'Layer', 'Dev IR render snapshot should carry node category for Netron colors.');
        assert(devIrRender.model.graph.values.a_weight.initializer && devIrRender.model.graph.values.a_weight.initializer.category === 'Initializer', 'Dev IR render snapshot should carry initializer object.');
        await devIrPanel.webview._handler({ type: 'confirmCrop', startKeys: ['image_a'], endKeys: ['a_hidden'] });
        await delay();
        const devIrPartialCrop = devIrPanel.webview.posted.filter((message) => message.type === 'cropConfirmed').pop();
        assert(devIrPartialCrop && devIrPartialCrop.graph && devIrPartialCrop.graph.nodes.length === 1, 'Dev IR host crop should trim graph nodes.');
        assert(devIrPartialCrop.graph.outputs[0].values[0] === 'a_hidden', 'Dev IR host crop should expose selected internal end tensor.');
        await devIrPanel.webview._handler({ type: 'confirmCrop', startKeys: ['image_a'], endKeys: ['logits_a'] });
        await delay();
        const devIrCrop = devIrPanel.webview.posted.filter((message) => message.type === 'cropConfirmed').pop();
        assert(devIrCrop && devIrCrop.artifact && devIrCrop.artifact.id, 'Dev IR crop should confirm.');
        await devIrPanel.webview._handler({
            type: 'runInference',
            artifactId: devIrCrop.artifact.id,
            sessionId: devIrRender.sessionId,
            inputMode: 'zeros'
        });
        await delay();
        const devIrInference = devIrPanel.webview.posted.find((message) => message.type === 'inferenceResult');
        assert(devIrInference && devIrInference.result && devIrInference.result.outputsSummary[0].name === 'logits_a', 'Dev IR inference should return stubbed output.');

        await open.callback(modelUri);
        const panel = createdPanels[createdPanels.length - 1];
        assert(panel && typeof panel.webview._handler === 'function', 'Model panel message handler should be registered.');
        await panel.webview._handler({ type: 'ready' });
        await delay();
        const renderMessage = panel.webview.posted.find((message) => message.type === 'renderGraphSnapshot');
        assert(renderMessage && renderMessage.model && renderMessage.model.format === 'private-export', 'Private provider model should render through host snapshot.');
        await panel.webview._handler({ type: 'confirmCrop', startKeys: ['x'], endKeys: ['y'] });
        await delay();
        const cropMessage = panel.webview.posted.find((message) => message.type === 'cropConfirmed');
        assert(cropMessage && cropMessage.artifact && cropMessage.artifact.id, 'Private provider crop should confirm.');
        await panel.webview._handler({ type: 'exportCropOnnx', artifactId: cropMessage.artifact.id });
        await delay();
        const exportMessage = panel.webview.posted.filter((message) => message.type === 'artifactExported').pop();
        assert(exportMessage && exportMessage.exportInfo.providerId === 'private-export', 'Host export should call private provider exportArtifact().');
        assert(saveDialogs[0] && saveDialogs[0].title === 'Export Private Crop', 'Host export should use provider export target metadata.');
        assert(saveDialogs[0].filters && saveDialogs[0].filters.Private, 'Host export should use provider filters.');
        assert(infoMessages.some((message) => /Crop artifact exported/.test(message)), 'Host export should report generic crop artifact export success.');

        const unsupportedPath = path.join(tempHome, 'unsupported.bin');
        fs.writeFileSync(unsupportedPath, Buffer.from('not-a-host-provider-model'));
        await open.callback(Uri.file(unsupportedPath));
        const legacyPanel = createdPanels[createdPanels.length - 1];
        assert(legacyPanel && typeof legacyPanel.webview._handler === 'function', 'Legacy model panel message handler should be registered.');
        await legacyPanel.webview._handler({ type: 'ready' });
        await delay();
        const legacyLoadMessage = legacyPanel.webview.posted.find((message) => message.type === 'loadModel');
        assert(legacyLoadMessage, 'Unsupported host-provider file should fall back to legacy loadModel.');
        assert(
            /No registered provider can open this file/.test(legacyLoadMessage.providerUnavailableReason || ''),
            'Legacy loadModel should include provider unavailable reason.'
        );

        extension.deactivate();
        console.log('provider artifact export ok');
    } finally {
        Module._load = originalLoad;
        fs.rmSync(tempHome, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
