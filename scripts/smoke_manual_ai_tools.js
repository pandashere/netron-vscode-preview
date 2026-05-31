#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const { ToolRegistry, runTool } = require('../lib/cli-tools');
const { installManualAiTools } = require('./install_manual_ai_tools');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

async function main() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-manual-ai-tools-'));
    try {
        installManualAiTools(root);

        const exporterRegistry = new ToolRegistry({
            kind: 'exporter',
            rootDir: path.join(root, 'exporters'),
            defaultTimeoutMs: 1000
        });
        const analyzerRegistry = new ToolRegistry({
            kind: 'analyzer',
            rootDir: path.join(root, 'analyzers'),
            defaultTimeoutMs: 1000
        });
        const exporters = exporterRegistry.refresh();
        const analyzers = analyzerRegistry.refresh();

        assert(exporters.entries.some((entry) => entry.id === 'crop-json-summary' && entry.status === 'ready'), 'Sample exporter was not registered.');
        assert(analyzers.entries.some((entry) => entry.id === 'line-count-analysis' && entry.status === 'ready'), 'Sample analyzer was not registered.');

        const context = {
            kind: 'text-export-context',
            schemaVersion: 1,
            target: 'crop',
            model: {
                fileName: 'sample.onnx'
            },
            artifact: {
                id: 'crop-sample'
            },
            graph: {
                id: 'sample-graph',
                name: 'sample-graph',
                inputs: ['input'],
                outputs: ['output'],
                nodes: [
                    { id: 'node-1', name: 'MatMul_0', type: 'MatMul' },
                    { id: 'node-2', name: 'Relu_0', type: 'Relu' }
                ],
                tensors: [
                    { name: 'input', dtype: 'float32', shape: [1, 4] },
                    { name: 'output', dtype: 'float32', shape: [1, 4] }
                ]
            }
        };

        const exporterResult = await runTool(
            exporterRegistry.getEntry('crop-json-summary'),
            JSON.stringify(context),
            { kind: 'exporter', label: 'Exporter' }
        );
        assert(exporterResult.stdout.includes('Model: sample.onnx'), 'Exporter did not summarize the model.');
        assert(exporterResult.stdout.includes('Nodes: 2'), 'Exporter did not summarize nodes.');

        const analyzerResult = await runTool(
            analyzerRegistry.getEntry('line-count-analysis'),
            exporterResult.stdout,
            { kind: 'analyzer', label: 'Analyzer' }
        );
        assert(analyzerResult.stdout.includes('Analysis Result'), 'Analyzer result header missing.');
        assert(analyzerResult.stdout.includes('First line: Model: sample.onnx'), 'Analyzer did not receive exporter stdout.');

        console.log('manual ai tools ok', {
            exporter: 'crop-json-summary',
            analyzer: 'line-count-analysis'
        });
    } finally {
        fs.rmSync(root, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
