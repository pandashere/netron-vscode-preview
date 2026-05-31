#!/usr/bin/env node
const { FormatProviderRegistry, providerDiagnostics } = require('../lib/format-providers');
const { assignCompareSlot, createEmptyCompareState } = require('../lib/host-compare-state');
const { runCrossProviderCompare } = require('../lib/compare-engine');
const { createMockPrivateProvider } = require('./fixtures/mock_private_provider');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

async function main() {
    const providerA = createMockPrivateProvider({
        id: 'private-a',
        extension: '.pa',
        inputName: 'x',
        outputName: 'y',
        outputValues: [1, 2, 3]
    });
    const providerB = createMockPrivateProvider({
        id: 'private-b',
        extension: '.pb',
        inputName: 'x_b',
        outputName: 'y_b',
        outputValues: [1, 4, 2]
    });
    assert(providerDiagnostics(providerA).errors.length === 0, 'Provider A should satisfy its declared contract.');
    assert(providerDiagnostics(providerB).errors.length === 0, 'Provider B should satisfy its declared contract.');

    const registry = new FormatProviderRegistry();
    registry.register(providerA);
    registry.register(providerB);
    assert(registry.resolve({ fsPath: '/tmp/model.pa' }).provider.id === 'private-a', 'Expected provider A resolution.');
    assert(registry.resolve({ fsPath: '/tmp/model.pb' }).provider.id === 'private-b', 'Expected provider B resolution.');

    const sessionA = await providerA.loadModel({ fsPath: '/tmp/model.pa' });
    const sessionB = await providerB.loadModel({ fsPath: '/tmp/model.pb' });
    const artifactA = await providerA.createCropArtifact({ sessionId: sessionA.id });
    const artifactB = await providerB.createCropArtifact({ sessionId: sessionB.id });
    const target = providerA.getCropTarget(artifactA.id);
    const context = providerA.buildTextExportContext(artifactA.id);
    assert(target.providerId === 'private-a', 'Expected crop target provider id.');
    assert(context.kind === 'text-export-context', 'Expected text export context.');
    assert(context.graph.tensors.every((tensor) => !Object.prototype.hasOwnProperty.call(tensor, 'values')), 'Tensor values should not be exported.');

    const compareState = createEmptyCompareState();
    assignCompareSlot(compareState, 'A', providerA.getCompareSlot(artifactA.id));
    assignCompareSlot(compareState, 'B', providerB.getCompareSlot(artifactB.id));
    assert(compareState.inputBindings[0].targetName === 'x_b', 'Expected cross-provider input binding.');
    assert(compareState.outputBindings[0].targetName === 'y_b', 'Expected cross-provider output binding.');

    const imported = await providerA.importInputFile('/tmp/input.json');
    const result = await runCrossProviderCompare(compareState, registry, {
        inputMode: 'import',
        importedInput: providerA.resolveImportedInput(imported.token),
        createRunId() {
            return 'private-compare-run';
        },
        now() {
            return '2026-05-31T00:00:00.000Z';
        }
    });
    const compareResult = result.compareState.compareResult;
    assert(compareResult.subgraphs.A.providerId === 'private-a', 'Expected side A provider provenance.');
    assert(compareResult.subgraphs.B.providerId === 'private-b', 'Expected side B provider provenance.');
    assert(compareResult.rows[0].status === 'ok', 'Expected comparable output row.');
    assert(compareResult.rows[0].maxAbs === 2, 'Unexpected compare maxAbs.');
    console.log('private provider contract ok', { providers: registry.list().length, rows: compareResult.rows.length });
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
