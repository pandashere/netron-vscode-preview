#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function hasCommand(pkg, command) {
    return Array.isArray(pkg.contributes && pkg.contributes.commands)
        && pkg.contributes.commands.some((item) => item.command === command);
}

function hasActivation(pkg, activation) {
    return Array.isArray(pkg.activationEvents) && pkg.activationEvents.includes(activation);
}

function hasView(pkg, containerId, viewId) {
    const views = pkg.contributes && pkg.contributes.views && pkg.contributes.views[containerId];
    return Array.isArray(views) && views.some((item) => item.id === viewId && item.type === 'webview');
}

function main() {
    const packagePath = path.join(root, 'package.json');
    const extensionPath = path.join(root, 'extension.js');
    const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
    const extension = fs.readFileSync(extensionPath, 'utf8');

    for (const command of [
        'netronPreview.openPreview',
        'netronPreview.openCompareCenter',
        'netronPreview.clearCompareCenter',
        'netronPreview.openAiAnalysis'
    ]) {
        assert(hasCommand(pkg, command), `Missing command contribution: ${command}`);
        assert(hasActivation(pkg, `onCommand:${command}`), `Missing command activation event: ${command}`);
        assert(extension.includes(`registerCommand('${command}'`), `Extension does not register command: ${command}`);
    }

    assert(hasActivation(pkg, 'onView:netronCompare.compareView'), 'Missing Compare view activation.');
    assert(hasActivation(pkg, 'onView:netronAI.analysisView'), 'Missing AI view activation.');
    assert(hasView(pkg, 'netronComparePanel', 'netronCompare.compareView'), 'Missing Compare webview contribution.');
    assert(hasView(pkg, 'netronComparePanel', 'netronAI.analysisView'), 'Missing AI Analysis webview contribution.');
    assert(extension.includes("registerWebviewViewProvider(COMPARE_VIEW_ID"), 'Extension does not register Compare WebviewView provider.');
    assert(extension.includes("registerWebviewViewProvider(AI_VIEW_ID"), 'Extension does not register AI WebviewView provider.');

    const panelContainer = pkg.contributes && pkg.contributes.viewsContainers && pkg.contributes.viewsContainers.panel;
    const comparePanel = Array.isArray(panelContainer) ? panelContainer.find((item) => item.id === 'netronComparePanel') : null;
    assert(comparePanel, 'Missing netronComparePanel container.');
    assert(comparePanel.icon && fs.existsSync(path.join(root, comparePanel.icon)), 'Compare panel icon is missing.');

    for (const [name, command] of Object.entries(pkg.scripts || {})) {
        if (!name.startsWith('smoke:')) {
            continue;
        }
        const match = /^node\s+(.+\.js)$/.exec(command);
        assert(match, `Smoke script '${name}' should run a Node JS file.`);
        assert(fs.existsSync(path.join(root, match[1])), `Smoke script '${name}' points to a missing file: ${match[1]}`);
    }

    console.log('package manifest ok');
}

try {
    main();
} catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
}
