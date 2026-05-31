#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { createMyIrProvider } = require('../examples/private-ir/my-ir-provider');
const { providerDiagnostics } = require('../lib/format-providers');
const { ToolRegistry } = require('../lib/cli-tools');

const root = path.resolve(__dirname, '..');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function copyDirectory(source, target) {
    fs.mkdirSync(target, { recursive: true });
    for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
        const sourcePath = path.join(source, entry.name);
        const targetPath = path.join(target, entry.name);
        if (entry.isDirectory()) {
            copyDirectory(sourcePath, targetPath);
        } else {
            fs.copyFileSync(sourcePath, targetPath);
        }
    }
}

function shellQuote(value) {
    return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function runNodeEntryWithFiles(entry, inputText, options = {}) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-guide-tool-run-'));
    const inputPath = path.join(dir, 'stdin.txt');
    const outputPath = path.join(dir, 'stdout.txt');
    const errorPath = path.join(dir, 'stderr.txt');
    fs.writeFileSync(inputPath, inputText);
    try {
        const scriptPath = path.resolve(entry.dir, entry.args[0]);
        const command = [
            shellQuote(process.execPath),
            shellQuote(scriptPath),
            '<',
            shellQuote(inputPath),
            '>',
            shellQuote(outputPath),
            '2>',
            shellQuote(errorPath)
        ].join(' ');
        const result = spawnSync('/bin/sh', ['-c', command], {
            env: {
                ...process.env,
                ...(options.env || {})
            }
        });
        return {
            status: result.status,
            stdout: fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8') : '',
            stderr: fs.existsSync(errorPath) ? fs.readFileSync(errorPath, 'utf8') : ''
        };
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

async function main() {
    const provider = createMyIrProvider();
    const diagnostics = providerDiagnostics(provider);
    assert(diagnostics.errors.length === 0, `Example provider diagnostics failed: ${diagnostics.errors.join(' ')}`);

    const samplePath = path.join(root, 'examples', 'private-ir', 'sample.myir.json');
    assert(provider.canOpen({ fsPath: samplePath }), 'Example provider should open sample.myir.json.');
    const session = await provider.loadModel({ fsPath: samplePath });
    assert(session.snapshot.sessionId === session.id, 'Example snapshot should carry sessionId.');
    assert(session.snapshot.graph.nodes[0].type.category === 'Layer', 'Example snapshot should carry node category.');
    assert(session.snapshot.graph.values.weight.initializer, 'Example snapshot should carry initializer object.');

    const artifact = await provider.createCropArtifact({
        sessionId: session.id,
        startKeys: ['input'],
        endKeys: ['hidden']
    });
    assert(artifact.summary.nodeCount === 1, 'Example crop should trim to selected path.');
    assert(artifact.outputKeys[0] === 'hidden', 'Example crop should expose selected internal tensor.');
    const context = provider.buildTextExportContext(artifact.id);
    assert(context.kind === 'text-export-context', 'Example provider should build text export context.');
    assert(context.graph.nodes.length === 1, 'Example text export should use cropped graph.');

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-private-ir-guide-'));
    try {
        const exporterRoot = path.join(tempRoot, 'exporters');
        const analyzerRoot = path.join(tempRoot, 'analyzers');
        copyDirectory(path.join(root, 'examples', 'tools', 'exporters'), exporterRoot);
        copyDirectory(path.join(root, 'examples', 'tools', 'analyzers'), analyzerRoot);

        const exporters = new ToolRegistry({ kind: 'exporter', rootDir: exporterRoot, defaultTimeoutMs: 30000 });
        const analyzers = new ToolRegistry({ kind: 'analyzer', rootDir: analyzerRoot, defaultTimeoutMs: 30000 });
        const exporterSnapshot = exporters.refresh();
        const analyzerSnapshot = analyzers.refresh();
        assert(exporterSnapshot.entries.some((entry) => entry.id === 'crop-json-summary' && entry.status === 'ready'), 'Crop summary exporter should be ready.');
        assert(exporterSnapshot.entries.some((entry) => entry.id === 'graph-edge-list' && entry.status === 'ready'), 'Graph edge-list exporter should be ready.');
        assert(analyzerSnapshot.entries.some((entry) => entry.id === 'line-count-analysis' && entry.status === 'ready'), 'Line count analyzer should be ready.');
        assert(analyzerSnapshot.entries.some((entry) => entry.id === 'deepseek-graph-analysis' && entry.status === 'ready'), 'DeepSeek analyzer manifest should be ready.');

        const contextText = JSON.stringify(context, null, 2);
        const summary = runNodeEntryWithFiles(exporters.getEntry('crop-json-summary'), contextText);
        assert(summary.status === 0, `Crop summary exporter failed: ${summary.stderr}`);
        assert(summary.stdout.includes('Model: sample.myir.json'), 'Crop summary exporter should include model file.');

        const edgeList = runNodeEntryWithFiles(exporters.getEntry('graph-edge-list'), contextText);
        assert(edgeList.status === 0, `Graph edge-list exporter failed: ${edgeList.stderr}`);
        assert(edgeList.stdout.includes('[edges]'), 'Graph edge-list exporter should include edges section.');
        assert(edgeList.stdout.includes('GRAPH_INPUT[input]'), 'Graph edge-list exporter should include graph input boundary.');

        const lineCount = runNodeEntryWithFiles(analyzers.getEntry('line-count-analysis'), edgeList.stdout);
        assert(lineCount.status === 0, `Line count analyzer failed: ${lineCount.stderr}`);
        assert(lineCount.stdout.includes('Analysis Result'), 'Line count analyzer should produce a result.');

        const missingKey = runNodeEntryWithFiles(analyzers.getEntry('deepseek-graph-analysis'), edgeList.stdout, {
            env: {
                DEEPSEEK_API_KEY: '',
                DEEPSEEK_API_KEY_FILE: path.join(tempRoot, 'missing-key'),
                DEEPSEEK_BASE_URL: 'https://api.deepseek.com',
                DEEPSEEK_MODEL: 'deepseek-v4-flash'
            }
        });
        assert(missingKey.status !== 0, 'DeepSeek example should fail without a key.');
        assert(/DEEPSEEK_API_KEY/.test(missingKey.stderr), 'DeepSeek example should fail clearly when no key is configured.');
    } finally {
        fs.rmSync(tempRoot, { recursive: true, force: true });
    }

    const guide = fs.readFileSync(path.join(root, 'docs', 'private-ir-integration-guide.md'), 'utf8');
    assert(guide.includes('Private IR Integration Guide'), 'Guide title missing.');
    assert(guide.includes('createCropArtifact'), 'Guide should document crop provider method.');
    assert(!/sk-[A-Za-z0-9]/.test(guide), 'Guide must not contain API keys.');

    console.log('private ir guide examples ok');
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
