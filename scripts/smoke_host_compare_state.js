#!/usr/bin/env node
const {
    assignCompareSlot,
    cloneCompareState,
    createEmptyCompareState,
    setCompareBinding,
    setCompareRunStatus,
    setImportedInput
} = require('../lib/host-compare-state');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function slot(providerId, artifactId, inputs, outputs) {
    return {
        providerId,
        artifactId,
        modelSessionId: `${artifactId}-session`,
        ioSignature: { inputs, outputs },
        summary: { modelName: `${artifactId}.model` }
    };
}

function main() {
    const state = createEmptyCompareState();
    assert(state.compareRunStatus.status === 'idle', 'Expected idle state.');
    assignCompareSlot(state, 'A', slot('a', 'artifact-a', [
        { name: 'x', dtype: 'float32', rank: 1, shape: [4] }
    ], [
        { name: 'y', dtype: 'float32', rank: 1, shape: [4] }
    ]));
    assert(state.inputBindings.length === 0, 'Bindings should wait for both slots.');
    assignCompareSlot(state, 'B', slot('b', 'artifact-b', [
        { name: 'x_b', dtype: 'float32', rank: 1, shape: [4] }
    ], [
        { name: 'y_b', dtype: 'float32', rank: 1, shape: [4] }
    ]));
    assert(state.inputBindings.length === 1, 'Expected input binding after both slots.');
    assert(state.outputBindings.length === 1, 'Expected output binding after both slots.');
    assert(state.inputBindings[0].targetName === 'x_b', 'Expected static-compatible input target.');

    setCompareBinding(state, 'output', 'y', 'y_b');
    assert(state.outputBindings[0].confirmed, 'Expected manual output binding.');
    assert(state.outputBindings[0].targetPort.name === 'y_b', 'Expected target port metadata.');

    setImportedInput(state, { token: 'input-1', preview: [{ name: 'x' }] }, 'a');
    assert(state.importedInput.providerId === 'a', 'Expected imported input provider id.');
    assert(state.importedInput.preview.length === 1, 'Expected imported input preview.');

    setCompareRunStatus(state, 'running', 'A', '');
    assert(state.compareRunStatus.status === 'running' && state.compareRunStatus.stage === 'A', 'Expected running status.');

    const clone = cloneCompareState(state);
    clone.slotA.artifactId = 'changed';
    assert(state.slotA.artifactId === 'artifact-a', 'Clone should not mutate original state.');

    let failed = false;
    try {
        setCompareBinding(state, 'input', 'missing', 'x_b');
    } catch (error) {
        failed = /Binding source/.test(error.message);
    }
    assert(failed, 'Missing binding source should fail.');
    console.log('host compare state ok', { inputs: state.inputBindings.length, outputs: state.outputBindings.length });
}

try {
    main();
} catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
}
