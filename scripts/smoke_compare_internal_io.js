#!/usr/bin/env node
const path = require('path');
const { ONNXWorkbench } = require('../lib/onnx-workbench');

const vscode = {
    Uri: {
        file(filePath) {
            return { fsPath: filePath, path: filePath };
        }
    }
};

function assertTensorPort(port, expectedName) {
    if (!port) {
        throw new Error(`Missing port '${expectedName}'.`);
    }
    if (port.name !== expectedName) {
        throw new Error(`Unexpected port name '${port.name}', expected '${expectedName}'.`);
    }
    if (port.dtype !== 'float32') {
        throw new Error(`Expected '${expectedName}' dtype float32, got '${port.dtype}'.`);
    }
    if (JSON.stringify(port.shape || []) !== JSON.stringify([1, 4])) {
        throw new Error(`Expected '${expectedName}' shape [1,4], got ${JSON.stringify(port.shape)}.`);
    }
}

async function main() {
    const baseDir = path.resolve(process.argv[2] || 'testdata/generated');
    const modelAPath = path.join(baseDir, 'dual-io-compare-a.onnx');
    const modelBPath = path.join(baseDir, 'dual-io-compare-b.onnx');

    const workbench = new ONNXWorkbench({}, () => {});
    const sessionA = await workbench.loadModel(vscode.Uri.file(modelAPath));
    const sessionB = await workbench.loadModel(vscode.Uri.file(modelBPath));

    const artifactA = await workbench.createCropArtifact({
        sessionId: sessionA.id,
        startKeys: ['tmp'],
        endKeys: ['shared_out']
    });
    const artifactB = await workbench.createCropArtifact({
        sessionId: sessionB.id,
        startKeys: ['tmp0'],
        endKeys: ['shared_out']
    });

    assertTensorPort(artifactA.ioSignature.inputs[0], 'tmp');
    assertTensorPort(artifactB.ioSignature.inputs[0], 'tmp0');
    assertTensorPort(artifactA.ioSignature.outputs[0], 'shared_out');
    assertTensorPort(artifactB.ioSignature.outputs[0], 'shared_out');

    workbench.assignCompareSlot('A', artifactA.id);
    workbench.assignCompareSlot('B', artifactB.id);
    const state = workbench.getCompareState();
    if (!state.inputBindings[0] || state.inputBindings[0].targetName !== 'tmp0') {
        throw new Error(`Expected compare input binding tmp -> tmp0, got ${JSON.stringify(state.inputBindings)}.`);
    }

    const result = await workbench.runCompare({ inputMode: 'ones', inputShapes: {} });
    console.log('compare ok', {
        input: artifactA.ioSignature.inputs[0],
        summary: result.compareResult.summary
    });
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
