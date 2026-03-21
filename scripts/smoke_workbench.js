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

async function run() {
    const target = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve('testdata/generated/branch-crop-small.onnx');
    const workbench = new ONNXWorkbench({}, () => {});
    const session = await workbench.loadModel(vscode.Uri.file(target));
    console.log('session', session.id, session.graphInfo.name, session.graphInfo.nodes.length);
    const artifact = await workbench.createCropArtifact({
        sessionId: session.id,
        startKeys: [session.graphInfo.graphInputNames[0]],
        endKeys: [session.graphInfo.graphOutputNames[0]]
    });
    console.log('artifact', artifact.id, artifact.summary);
    const inference = await workbench.runInference({ artifactId: artifact.id, inputMode: 'ones', inputShapes: {} });
    console.log('inference outputs', inference.outputsSummary.map((entry) => ({ name: entry.name, shape: entry.shape, dtype: entry.dtype })));
    workbench.assignCompareSlot('A', artifact.id);
    workbench.assignCompareSlot('B', artifact.id);
    const compare = await workbench.runCompare({ inputMode: 'ones', inputShapes: {} });
    console.log('compare summary', compare.compareResult.summary);
}

run().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
