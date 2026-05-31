const { buildCompareBindings } = require('./compare-core');

function nowIso() {
    return new Date().toISOString();
}

function createEmptyCompareState() {
    return {
        slotA: null,
        slotB: null,
        inputBindings: [],
        outputBindings: [],
        compareRunStatus: { status: 'idle', stage: '', message: '', updatedAt: nowIso() },
        compareResult: null,
        importedInput: null
    };
}

function cloneCompareState(compareState) {
    return JSON.parse(JSON.stringify(compareState || createEmptyCompareState()));
}

function recomputeCompareBindings(compareState) {
    const slotA = compareState.slotA;
    const slotB = compareState.slotB;
    if (!slotA || !slotB) {
        compareState.inputBindings = [];
        compareState.outputBindings = [];
        return compareState;
    }
    compareState.inputBindings = buildCompareBindings(slotA.ioSignature.inputs, slotB.ioSignature.inputs);
    compareState.outputBindings = buildCompareBindings(slotA.ioSignature.outputs, slotB.ioSignature.outputs);
    return compareState;
}

function assignCompareSlot(compareState, slotName, slot) {
    if (!slot) {
        throw new Error(`Compare slot ${slotName} is not available.`);
    }
    if (slotName === 'A') {
        compareState.slotA = slot;
    } else if (slotName === 'B') {
        compareState.slotB = slot;
    } else {
        throw new Error(`Unsupported compare slot '${slotName}'.`);
    }
    compareState.compareResult = null;
    compareState.compareRunStatus = { status: 'idle', stage: '', message: '', updatedAt: nowIso() };
    recomputeCompareBindings(compareState);
    return compareState;
}

function setCompareBinding(compareState, kind, sourceName, targetName) {
    const listName = kind === 'output' ? 'outputBindings' : 'inputBindings';
    const binding = compareState[listName].find((item) => item.sourceName === sourceName);
    if (!binding) {
        throw new Error('Binding source not found.');
    }
    binding.targetName = targetName || null;
    binding.confirmed = !!targetName;
    binding.reason = binding.confirmed ? 'manual' : binding.reason;
    const slotB = compareState.slotB;
    const ports = kind === 'output'
        ? slotB && slotB.ioSignature && slotB.ioSignature.outputs
        : slotB && slotB.ioSignature && slotB.ioSignature.inputs;
    binding.targetPort = Array.isArray(ports) ? ports.find((item) => item.name === binding.targetName) || null : null;
    compareState.compareResult = null;
    return compareState;
}

function setImportedInput(compareState, imported, providerId) {
    compareState.importedInput = imported ? {
        token: imported.token,
        preview: imported.preview || [],
        providerId
    } : null;
    return compareState;
}

function setCompareRunStatus(compareState, status, stage = '', message = '') {
    compareState.compareRunStatus = { status, stage, message, updatedAt: nowIso() };
    return compareState;
}

module.exports = {
    assignCompareSlot,
    cloneCompareState,
    createEmptyCompareState,
    recomputeCompareBindings,
    setCompareBinding,
    setCompareRunStatus,
    setImportedInput
};
