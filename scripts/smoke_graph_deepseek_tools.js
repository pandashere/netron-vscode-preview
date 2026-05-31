#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { installGraphDeepseekTools } = require('./install_graph_deepseek_tools');
const { ToolRegistry } = require('../lib/cli-tools');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function shellQuote(value) {
    return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function runNodeToolWithFiles(scriptPath, inputText, options = {}) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-tool-run-'));
    const inputPath = path.join(dir, 'input.txt');
    const outputPath = path.join(dir, 'stdout.txt');
    const errorPath = path.join(dir, 'stderr.txt');
    fs.writeFileSync(inputPath, inputText);
    try {
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
            signal: result.signal,
            stdout: fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8') : '',
            stderr: fs.existsSync(errorPath) ? fs.readFileSync(errorPath, 'utf8') : ''
        };
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

async function main() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-graph-deepseek-'));
    try {
        installGraphDeepseekTools(root);
        const exporterRegistry = new ToolRegistry({
            kind: 'exporter',
            rootDir: path.join(root, 'exporters'),
            defaultTimeoutMs: 30000
        });
        const analyzerRegistry = new ToolRegistry({
            kind: 'analyzer',
            rootDir: path.join(root, 'analyzers'),
            defaultTimeoutMs: 30000
        });
        exporterRegistry.refresh();
        analyzerRegistry.refresh();
        const exporter = exporterRegistry.getEntry('graph-edge-list');
        const analyzer = analyzerRegistry.getEntry('deepseek-graph-analysis');
        assert(exporter && exporter.status === 'ready', 'graph-edge-list exporter should be ready.');
        assert(analyzer && analyzer.status === 'ready', 'deepseek-graph-analysis analyzer should be ready.');

        const context = {
            kind: 'text-export-context',
            schemaVersion: 1,
            target: 'crop',
            model: {
                format: 'mock',
                fileName: 'sample.mock',
                filePath: '/tmp/sample.mock'
            },
            artifact: {
                id: 'artifact-1',
                createdAt: '2026-05-31T00:00:00.000Z'
            },
            graph: {
                id: 'sample:crop:abc',
                name: 'sample:crop:abc',
                inputs: ['x'],
                outputs: ['z'],
                nodes: [
                    {
                        id: 'conv-1',
                        name: 'Conv_1',
                        type: 'Conv',
                        domain: 'ai.onnx',
                        inputs: [{ name: 'X', tensor: 'x' }, { name: 'W', tensor: 'w' }],
                        outputs: [{ name: 'Y', tensor: 'y' }],
                        attributes: {},
                        omittedAttributes: []
                    },
                    {
                        id: 'relu-1',
                        name: 'Relu_1',
                        type: 'Relu',
                        domain: 'ai.onnx',
                        inputs: [{ name: 'X', tensor: 'y' }],
                        outputs: [{ name: 'Y', tensor: 'z' }],
                        attributes: {},
                        omittedAttributes: []
                    }
                ],
                tensors: [
                    { name: 'x', dtype: 'float32', rawDtype: 'FLOAT', shape: [1, 'N'], kind: 'input' },
                    { name: 'w', dtype: 'float32', rawDtype: 'FLOAT', shape: [3, 3], kind: 'initializer' },
                    { name: 'y', dtype: 'float32', rawDtype: 'FLOAT', shape: [1, 'N'], kind: 'intermediate' },
                    { name: 'z', dtype: 'float32', rawDtype: 'FLOAT', shape: [1, 'N'], kind: 'output' }
                ]
            }
        };

        const exportResult = runNodeToolWithFiles(exporter.args[0], JSON.stringify(context));
        assert(exportResult.status === 0, `Exporter failed: ${exportResult.stderr}`);
        assert(exportResult.stdout.includes('# Netron Crop Graph Edge List'), 'Exporter header missing.');
        assert(exportResult.stdout.includes('GRAPH_INPUT[x] -> conv-1.X'), 'Graph input edge missing.');
        assert(exportResult.stdout.includes('CONST[w] -> conv-1.W'), 'Initializer edge missing.');
        assert(exportResult.stdout.includes('conv-1.Y -> relu-1.X'), 'Node-to-node edge missing.');
        assert(exportResult.stdout.includes('relu-1.Y -> GRAPH_OUTPUT[z]'), 'Graph output edge missing.');

        const analysisResult = runNodeToolWithFiles(analyzer.args[0], exportResult.stdout, {
            env: {
                DEEPSEEK_API_KEY: '',
                DEEPSEEK_API_KEY_FILE: path.join(root, 'missing-key')
            }
        });
        assert(analysisResult.status !== 0, 'Analyzer should fail without a DeepSeek API key.');
        assert(/DEEPSEEK_API_KEY/.test(analysisResult.stderr), 'Analyzer should explain missing DeepSeek API key.');

        console.log('graph deepseek tools ok');
    } finally {
        fs.rmSync(root, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
