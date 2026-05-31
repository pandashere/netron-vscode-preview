#!/usr/bin/env node
const path = require('path');
const { ONNXWorkbench } = require('../lib/onnx-workbench');
const { buildTextExportContext } = require('../lib/text-export-context');

const vscode = {
    Uri: {
        file(filePath) {
            return { fsPath: filePath, path: filePath };
        }
    }
};

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

async function main() {
    const modelPath = path.resolve(process.argv[2] || 'testdata/generated/dual-io-compare-a.onnx');
    const workbench = new ONNXWorkbench({}, () => {});
    const session = await workbench.loadModel(vscode.Uri.file(modelPath));
    const artifact = await workbench.createCropArtifact({
        sessionId: session.id,
        startKeys: ['tmp'],
        endKeys: ['shared_out']
    });
    const target = workbench.getCropTarget(artifact.id);
    const context = workbench.buildTextExportContext(artifact.id);

    assert(target.kind === 'crop-target', 'Unexpected target kind.');
    assert(target.schemaVersion === 1, 'Unexpected target schema version.');
    assert(target.providerId === 'onnx', 'Unexpected target provider.');
    assert(target.target === 'crop', 'Unexpected target scope.');
    assert(target.model.filePath === modelPath, 'Target model file path mismatch.');
    assert(target.artifact.id === artifact.id, 'Target artifact id mismatch.');
    assert(target.artifact.createdAt === artifact.createdAt, 'Target artifact timestamp mismatch.');
    assert(target.artifact.stale === false, 'Target should not be stale.');
    assert(target.graph.id && target.graph.name === target.graph.id, 'Target graph id/name mismatch.');

    assert(context.kind === 'text-export-context', 'Unexpected context kind.');
    assert(context.schemaVersion === 1, 'Unexpected schema version.');
    assert(context.target === 'crop', 'Unexpected target.');
    assert(context.model.format === 'onnx', `Unexpected model format: ${context.model.format}`);
    assert(context.model.fileName.endsWith('.onnx'), 'Missing model file name.');
    assert(context.model.filePath === modelPath, 'Missing model file path.');
    assert(!Object.prototype.hasOwnProperty.call(context.model, 'sessionId'), 'model.sessionId should be excluded.');
    assert(!Object.prototype.hasOwnProperty.call(context.model, 'id'), 'model.id should be excluded.');
    assert(context.artifact.id === artifact.id, 'Artifact id mismatch.');
    assert(context.artifact.createdAt === artifact.createdAt, 'Artifact timestamp mismatch.');
    assert(!Object.prototype.hasOwnProperty.call(context.artifact, 'ioSignature'), 'artifact.ioSignature should be excluded.');
    assert(!Object.prototype.hasOwnProperty.call(context.artifact, 'summary'), 'artifact.summary should be excluded.');
    assert(context.graph.id && context.graph.name === context.graph.id, 'Graph id/name mismatch.');
    assert(context.graph.id === target.graph.id, 'Context should use the shared target graph id.');
    assert(Array.isArray(context.graph.inputs) && context.graph.inputs.includes('tmp'), 'Missing graph input.');
    assert(Array.isArray(context.graph.outputs) && context.graph.outputs.includes('shared_out'), 'Missing graph output.');
    assert(Array.isArray(context.graph.nodes) && context.graph.nodes.length > 0, 'Missing nodes.');
    assert(Array.isArray(context.graph.tensors) && context.graph.tensors.length > 0, 'Missing tensors.');

    const tensorNames = new Set(context.graph.tensors.map((tensor) => tensor.name));
    for (const node of context.graph.nodes) {
        assert(typeof node.id === 'string' && node.id.length > 0, 'Node id is required.');
        assert(typeof node.name === 'string' && node.name.length > 0, 'Node name is required.');
        assert(typeof node.domain === 'string', 'Node domain should be present.');
        assert(Array.isArray(node.inputs), 'Node inputs should be an array.');
        assert(Array.isArray(node.outputs), 'Node outputs should be an array.');
        for (const port of [...node.inputs, ...node.outputs]) {
            assert(tensorNames.has(port.tensor), `Missing tensor metadata for ${port.tensor}.`);
        }
    }
    for (const name of [...context.graph.inputs, ...context.graph.outputs]) {
        assert(tensorNames.has(name), `Missing boundary tensor metadata for ${name}.`);
    }
    const serialized = JSON.stringify(context);
    assert(!serialized.includes('sampleValues'), 'Tensor sample values should be excluded.');
    assert(!serialized.includes('rawData'), 'Tensor raw data should be excluded.');

    artifact.stale = true;
    let staleFailed = false;
    try {
        workbench.getCropTarget(artifact.id);
    } catch (error) {
        staleFailed = /stale/i.test(error.message);
    }
    assert(staleFailed, 'Stale artifact should not build crop target.');

    staleFailed = false;
    try {
        workbench.buildTextExportContext(artifact.id);
    } catch (error) {
        staleFailed = /stale/i.test(error.message);
    }
    assert(staleFailed, 'Stale artifact should not build export context.');

    const symbolic = buildTextExportContext({
        format: 'onnx',
        filePath: '/tmp/symbolic.onnx',
        graphInfo: { name: 'symbolic_graph' }
    }, {
        id: 'artifact-symbolic',
        createdAt: '2026-05-31T00:00:00.000Z',
        stale: false,
        inputKeys: ['X'],
        outputKeys: ['Y'],
        cropGraphSnapshot: {
            inputs: [{ name: 'input', values: ['X'] }],
            outputs: [{ name: 'output', values: ['Y'] }],
            values: {
                X: { type: { dataType: 'float32', shape: [1, 'N', 4] } },
                Y: { type: { dataType: 'float32', shape: [1, 'N', 4] } }
            },
            nodes: [{
                id: 'identity-0',
                name: 'Identity_0',
                type: { name: 'Identity', module: '' },
                inputs: [{ name: 'input', values: ['X'] }],
                outputs: [{ name: 'output', values: ['Y'] }],
                attributes: []
            }]
        }
    });
    const symbolicTensor = symbolic.graph.tensors.find((tensor) => tensor.name === 'X');
    assert(symbolicTensor && symbolicTensor.shape[1] === 'N', 'Symbolic dimensions should be preserved as strings.');

    console.log('text export context ok', {
        graphId: context.graph.id,
        nodes: context.graph.nodes.length,
        tensors: context.graph.tensors.length
    });
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
