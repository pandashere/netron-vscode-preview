#!/usr/bin/env node
const path = require('path');
const { createDevIrProvider, loadDevIrDocument } = require('../lib/dev-ir-provider');
const { FormatProviderRegistry, providerDiagnostics } = require('../lib/format-providers');
const { assignCompareSlot, createEmptyCompareState } = require('../lib/host-compare-state');
const { runCrossProviderCompare } = require('../lib/compare-engine');

const root = path.resolve(__dirname, '..');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

async function main() {
    const fileA = path.join(root, 'testdata', 'dev-ir', 'model-a.netronir.json');
    const fileB = path.join(root, 'testdata', 'dev-ir', 'model-b.netronir.json');
    const docA = loadDevIrDocument(fileA, 'dev-ir-a');
    const docB = loadDevIrDocument(fileB, 'dev-ir-b');
    assert(docA.graph.inputs[0] === 'image_a', 'Unexpected Dev IR A input.');
    assert(docB.graph.outputs[0] === 'scores_b', 'Unexpected Dev IR B output.');

    const providerA = createDevIrProvider({ id: 'dev-ir-a', label: 'Dev IR A' });
    const providerB = createDevIrProvider({ id: 'dev-ir-b', label: 'Dev IR B' });
    assert(providerDiagnostics(providerA).errors.length === 0, 'Dev IR A diagnostics should be clean.');
    assert(providerDiagnostics(providerB).errors.length === 0, 'Dev IR B diagnostics should be clean.');
    assert(providerA.canOpen({ fsPath: fileA }), 'Dev IR A should open model A.');
    assert(!providerA.canOpen({ fsPath: fileB }), 'Dev IR A should not open model B.');
    assert(providerB.canOpen({ fsPath: fileB }), 'Dev IR B should open model B.');

    const registry = new FormatProviderRegistry();
    registry.register(providerA);
    registry.register(providerB);
    assert(registry.resolve({ fsPath: fileA }).provider.id === 'dev-ir-a', 'Registry should resolve Dev IR A.');
    assert(registry.resolve({ fsPath: fileB }).provider.id === 'dev-ir-b', 'Registry should resolve Dev IR B.');

    const sessionA = await providerA.loadModel({ fsPath: fileA });
    const sessionB = await providerB.loadModel({ fsPath: fileB });
    assert(sessionA.snapshot.sessionId === sessionA.id, 'Dev IR snapshot should carry sessionId for webview crop gating.');
    assert(sessionA.snapshot.graph.nodes.length === 2, 'Dev IR A snapshot should render nodes.');
    assert(sessionA.snapshot.graph.nodes[0].type.category === 'Layer', 'Dev IR Conv node should carry Netron color category.');
    assert(sessionB.snapshot.graph.nodes[0].type.category === 'Transform', 'Dev IR PatchEmbed node should carry Netron color category.');
    assert(sessionA.snapshot.graph.values.a_weight.initializer && sessionA.snapshot.graph.values.a_weight.initializer.category === 'Initializer', 'Dev IR initializer tensor should use Netron initializer object shape.');
    assert(sessionB.snapshot.graph.values.scores_b.type.shape[1] === 4, 'Dev IR B output shape missing.');

    const convOnlyArtifact = await providerA.createCropArtifact({
        sessionId: sessionA.id,
        startKeys: ['image_a'],
        endKeys: ['a_hidden']
    });
    assert(convOnlyArtifact.summary.nodeCount === 1, 'Dev IR crop should trim to selected path nodes.');
    assert(convOnlyArtifact.outputKeys.length === 1 && convOnlyArtifact.outputKeys[0] === 'a_hidden', 'Dev IR crop should expose selected end tensor as output.');
    assert(convOnlyArtifact.cropGraphSnapshot.nodes.length === 1, 'Dev IR crop graph snapshot should contain only cropped nodes.');
    assert(convOnlyArtifact.cropGraphSnapshot.nodes[0].id === 'a-conv', 'Dev IR crop should keep the producer path node.');
    assert(!convOnlyArtifact.cropGraphSnapshot.values.logits_a, 'Dev IR crop should remove tensors outside the selected path.');
    assert(convOnlyArtifact.cropGraphSnapshot.values.a_weight.initializer, 'Dev IR crop should retain initializer tensors used by selected nodes.');

    const artifactA = await providerA.createCropArtifact({
        sessionId: sessionA.id,
        startKeys: ['image_a'],
        endKeys: ['logits_a']
    });
    const artifactB = await providerB.createCropArtifact({
        sessionId: sessionB.id,
        startKeys: ['image_b'],
        endKeys: ['scores_b']
    });
    const contextA = providerA.buildTextExportContext(artifactA.id);
    assert(contextA.kind === 'text-export-context', 'Dev IR text export context kind mismatch.');
    assert(contextA.model.format === 'dev-ir-a', 'Dev IR text export model format mismatch.');
    assert(contextA.graph.nodes[0].id === 'a-conv', 'Dev IR text export graph node mismatch.');

    const inference = await providerA.runInference({ sessionId: sessionA.id, artifactId: artifactA.id });
    assert(inference.outputsSummary[0].name === 'logits_a', 'Dev IR inference output missing.');
    assert(inference.outputsSummary[0].values.length === 4, 'Dev IR inference values should match output shape.');

    const compareState = createEmptyCompareState();
    assignCompareSlot(compareState, 'A', providerA.getCompareSlot(artifactA.id));
    assignCompareSlot(compareState, 'B', providerB.getCompareSlot(artifactB.id));
    assert(compareState.inputBindings[0].targetName === 'image_b', 'Dev IR cross-format input should auto-bind by shape.');
    assert(compareState.outputBindings[0].targetName === 'scores_b', 'Dev IR cross-format output should auto-bind by shape.');

    const compare = await runCrossProviderCompare(compareState, registry, {
        inputMode: 'zeros',
        createRunId() {
            return 'dev-ir-compare';
        },
        now() {
            return '2026-05-31T00:00:00.000Z';
        }
    });
    const result = compare.compareState.compareResult;
    assert(result.subgraphs.A.providerId === 'dev-ir-a', 'Compare result should preserve provider A provenance.');
    assert(result.subgraphs.B.providerId === 'dev-ir-b', 'Compare result should preserve provider B provenance.');
    assert(result.rows[0].status === 'ok', 'Dev IR compare row should be comparable.');
    assert(Number.isFinite(result.rows[0].maxAbs), 'Dev IR compare should compute numeric diff.');

    console.log('dev ir provider ok', {
        providers: registry.list().map((provider) => provider.id).join(','),
        maxAbs: Number(result.rows[0].maxAbs.toFixed(6))
    });
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
