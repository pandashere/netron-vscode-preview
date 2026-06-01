#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const ort = require('onnxruntime-node');
const { onnx } = require('onnx-proto');
const { ONNXWorkbench } = require('../lib/onnx-workbench');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function tensorValueInfo(name, elemType, shape) {
    return onnx.ValueInfoProto.create({
        name,
        type: onnx.TypeProto.create({
            tensorType: onnx.TypeProto.Tensor.create({
                elemType,
                shape: onnx.TensorShapeProto.create({
                    dim: shape.map((dimValue) => onnx.TensorShapeProto.Dimension.create({ dimValue }))
                })
            })
        })
    });
}

function makeConstantAddModel(filePath) {
    const dataType = onnx.TensorProto.DataType.FLOAT;
    const constant = onnx.TensorProto.create({
        name: 'constant_attr',
        dataType,
        dims: [1, 4],
        floatData: [1, 2, 3, 4]
    });
    const graph = onnx.GraphProto.create({
        name: 'constant_crop_graph',
        input: [tensorValueInfo('X', dataType, [1, 4])],
        output: [tensorValueInfo('Y', dataType, [1, 4])],
        initializer: [],
        node: [
            onnx.NodeProto.create({
                name: 'make_c',
                opType: 'Constant',
                output: ['C'],
                attribute: [
                    onnx.AttributeProto.create({
                        name: 'value',
                        type: onnx.AttributeProto.AttributeType.TENSOR,
                        t: constant
                    })
                ]
            }),
            onnx.NodeProto.create({
                name: 'add_c',
                opType: 'Add',
                input: ['X', 'C'],
                output: ['Y']
            })
        ]
    });
    const model = onnx.ModelProto.create({
        irVersion: 8,
        producerName: 'onnx-constant-crop-smoke',
        opsetImport: [onnx.OperatorSetIdProto.create({ domain: '', version: 13 })],
        graph
    });
    fs.writeFileSync(filePath, Buffer.from(onnx.ModelProto.encode(model).finish()));
}

async function main() {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-onnx-constant-crop-'));
    try {
        const modelPath = path.join(tempDir, 'constant-add.onnx');
        const exportPath = path.join(tempDir, 'constant-add-crop.onnx');
        makeConstantAddModel(modelPath);

        const workbench = new ONNXWorkbench({}, () => {});
        const session = await workbench.loadModel({ fsPath: modelPath, path: modelPath });
        const constantValue = session.graphInfo.values.get('C');
        assert(constantValue && constantValue.initializer, 'Constant node output should be treated as an initializer.');
        assert(constantValue.type && constantValue.type.dataType === 'float32', 'Constant node output should carry tensor type.');
        assert(!session.graphInfo.graphInputNames.includes('C'), 'Constant node output must not become a graph input.');

        const artifact = await workbench.createCropArtifact({
            sessionId: session.id,
            startKeys: ['X'],
            endKeys: ['Y']
        });
        assert(artifact.inputKeys.length === 1 && artifact.inputKeys[0] === 'X', 'Crop input should only contain real graph input X.');
        assert(!artifact.inputKeys.includes('C'), 'Crop input must not include constant tensor C.');
        assert(artifact.cropGraphSnapshot.values.C.initializer, 'Crop snapshot should expose constant tensor C as initializer.');
        assert(artifact.cropGraphSnapshot.values.C.type.dataType === 'float32', 'Crop snapshot constant tensor should keep dtype.');

        await workbench.exportArtifact(artifact.id, exportPath, { inlineWeights: true, externalData: false });
        const exported = onnx.ModelProto.decode(fs.readFileSync(exportPath));
        assert(exported.graph.input.length === 1 && exported.graph.input[0].name === 'X', 'Exported crop should not turn C into a graph input.');
        assert(exported.graph.initializer.some((tensor) => tensor.name === 'C'), 'Exported crop should include C as graph initializer.');

        const ortSession = await ort.InferenceSession.create(exportPath);
        assert(Object.keys(ortSession.inputNames || {}).length !== 0 || ortSession.inputNames, 'ORT session should expose inputs.');
        const result = await ortSession.run({
            X: new ort.Tensor('float32', Float32Array.from([10, 10, 10, 10]), [1, 4])
        });
        const values = Array.from(result.Y.data);
        assert(values.join(',') === '11,12,13,14', `Unexpected ORT result: ${values.join(',')}`);
    } finally {
        fs.rmSync(tempDir, { recursive: true, force: true });
    }
    console.log('onnx constant crop ok');
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
