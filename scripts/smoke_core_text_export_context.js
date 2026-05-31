#!/usr/bin/env node
const {
    buildCropTargetFromCoreGraph,
    buildTextExportContextFromCoreGraph,
    normalizeCoreGraph
} = require('../lib/text-export-context');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function baseInput() {
    return {
        providerId: 'private-format',
        model: {
            format: 'private-format',
            fileName: 'model.pf',
            filePath: '/tmp/model.pf'
        },
        artifact: {
            id: 'artifact-1',
            createdAt: '2026-05-31T00:00:00.000Z',
            stale: false,
            summary: { nodeCount: 1 },
            ioSignature: { inputs: [], outputs: [] }
        },
        graph: {
            id: 'private-format:crop:artifact-1',
            name: 'private-format:crop:artifact-1',
            inputs: ['x'],
            outputs: ['y'],
            nodes: [{
                id: 'n0',
                name: 'PrivateOp_0',
                type: 'PrivateOp',
                domain: 'private.domain',
                inputs: [{ name: 'input', tensor: 'x' }],
                outputs: [{ name: 'output', tensor: 'y' }],
                attributes: { alpha: 1 },
                omittedAttributes: [{ name: 'weights', reason: 'tensor-data' }]
            }],
            tensors: [
                { name: 'x', dtype: 'float', rawDtype: 'FLOAT', shape: [1, 'N'], kind: 'input', values: [1, 2] },
                { name: 'y', dtype: 'float32', rawDtype: 'FLOAT', shape: [1, 'N'], kind: 'output', rawData: Buffer.from([1, 2]) }
            ]
        }
    };
}

function assertThrows(pattern, callback, message) {
    let failed = false;
    try {
        callback();
    } catch (error) {
        failed = pattern.test(error.message);
    }
    assert(failed, message);
}

function main() {
    const input = baseInput();
    const target = buildCropTargetFromCoreGraph(input);
    const context = buildTextExportContextFromCoreGraph(input);

    assert(target.providerId === 'private-format', 'Target provider mismatch.');
    assert(target.graph.id === input.graph.id && target.graph.name === input.graph.name, 'Target graph identity mismatch.');
    assert(!Object.prototype.hasOwnProperty.call(context.artifact, 'summary'), 'Context artifact should not expose summary.');
    assert(!Object.prototype.hasOwnProperty.call(context.artifact, 'ioSignature'), 'Context artifact should not expose ioSignature.');
    assert(context.graph.nodes[0].attributes.alpha === 1, 'Node attributes should be preserved.');
    assert(context.graph.nodes[0].omittedAttributes[0].reason === 'tensor-data', 'Omitted attributes should be preserved.');
    assert(context.graph.tensors[0].dtype === 'float32', 'Tensor dtype should be normalized.');
    assert(context.graph.tensors[0].shape[1] === 'N', 'Symbolic dimensions should be preserved.');
    assert(!JSON.stringify(context).includes('rawData'), 'Tensor raw data should be stripped.');
    assert(!JSON.stringify(context).includes('values'), 'Tensor values should be stripped.');

    const normalized = normalizeCoreGraph(input.graph);
    assert(normalized.nodes[0].inputs[0].tensor === 'x', 'Core graph ports should normalize.');

    const missingBoundaryTensor = baseInput();
    missingBoundaryTensor.graph.tensors = missingBoundaryTensor.graph.tensors.filter((tensor) => tensor.name !== 'x');
    assertThrows(/Boundary tensor 'x'/, () => buildTextExportContextFromCoreGraph(missingBoundaryTensor), 'Missing boundary tensor should fail.');

    const missingNodeTensor = baseInput();
    missingNodeTensor.graph.nodes[0].outputs[0].tensor = 'missing';
    assertThrows(/Node tensor 'missing'/, () => buildTextExportContextFromCoreGraph(missingNodeTensor), 'Missing node tensor should fail.');

    const stale = baseInput();
    stale.artifact.stale = true;
    assertThrows(/stale/i, () => buildTextExportContextFromCoreGraph(stale), 'Stale artifact should fail.');

    console.log('core text export context ok');
}

try {
    main();
} catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
}
