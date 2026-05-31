const fs = require('fs');
const path = require('path');
const {
    buildCropTargetFromCoreGraph,
    buildTextExportContextFromCoreGraph,
    normalizeCoreGraph,
    stableHash
} = require('../../lib/text-export-context');

const FORMAT_ID = 'my-ir';
const FILE_EXTENSION = '.myir.json';
const CATEGORY_FALLBACK = new Map([
    ['Conv', 'Layer'],
    ['Linear', 'Layer'],
    ['Gemm', 'Layer'],
    ['MatMul', 'Layer'],
    ['Relu', 'Activation'],
    ['Sigmoid', 'Activation'],
    ['Tanh', 'Activation'],
    ['Softmax', 'Activation'],
    ['MaxPool', 'Pool'],
    ['AveragePool', 'Pool'],
    ['BatchNormalization', 'Normalization'],
    ['LayerNormalization', 'Normalization'],
    ['Reshape', 'Shape'],
    ['Transpose', 'Transform'],
    ['Concat', 'Tensor'],
    ['Slice', 'Tensor'],
    ['Gather', 'Tensor'],
    ['Constant', 'Constant'],
    ['Identity', 'Control']
]);

function filePathOf(uri) {
    return uri && (uri.fsPath || uri.path || String(uri));
}

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function normalizeTensor(tensor, index) {
    const name = tensor && typeof tensor.name === 'string' ? tensor.name : '';
    if (!name) {
        throw new Error(`Tensor ${index} is missing name.`);
    }
    return {
        name,
        dtype: String(tensor.dtype || 'float32').toLowerCase(),
        rawDtype: tensor.rawDtype || tensor.dtype || 'float32',
        shape: Array.isArray(tensor.shape) ? tensor.shape.slice() : [],
        kind: tensor.kind || 'activation'
    };
}

function normalizePort(port) {
    if (!port || typeof port.tensor !== 'string' || !port.tensor) {
        return null;
    }
    return {
        name: typeof port.name === 'string' ? port.name : '',
        tensor: port.tensor
    };
}

function normalizeNode(node, index) {
    const id = node && typeof node.id === 'string' && node.id ? node.id : `node-${index}`;
    return {
        id,
        name: node && typeof node.name === 'string' && node.name ? node.name : id,
        type: node && typeof node.type === 'string' && node.type ? node.type : 'PrivateOp',
        category: node && typeof node.category === 'string' && node.category ? node.category : 'Custom',
        domain: node && typeof node.domain === 'string' ? node.domain : FORMAT_ID,
        inputs: Array.isArray(node && node.inputs) ? node.inputs.map(normalizePort).filter(Boolean) : [],
        outputs: Array.isArray(node && node.outputs) ? node.outputs.map(normalizePort).filter(Boolean) : [],
        attributes: node && node.attributes && typeof node.attributes === 'object' && !Array.isArray(node.attributes) ? { ...node.attributes } : {},
        omittedAttributes: []
    };
}

function normalizePrivateGraph(raw, fileName) {
    const graph = raw && raw.graph && typeof raw.graph === 'object' ? raw.graph : {};
    const tensors = Array.isArray(graph.tensors) ? graph.tensors.map(normalizeTensor) : [];
    const nodes = Array.isArray(graph.nodes) ? graph.nodes.map(normalizeNode) : [];
    const inputs = Array.isArray(graph.inputs) ? graph.inputs.filter((name) => typeof name === 'string' && name) : [];
    const outputs = Array.isArray(graph.outputs) ? graph.outputs.filter((name) => typeof name === 'string' && name) : [];
    const id = typeof graph.id === 'string' && graph.id
        ? graph.id
        : `${FORMAT_ID}:${fileName}:${stableHash({ inputs, outputs, nodes })}`;
    return normalizeCoreGraph({
        id,
        name: typeof graph.name === 'string' && graph.name ? graph.name : id,
        inputs,
        outputs,
        nodes,
        tensors
    });
}

function tensorType(tensor) {
    return {
        dataType: tensor.dtype,
        shape: tensor.shape
    };
}

function isInitializer(tensor) {
    return /initializer|constant|weight/i.test(tensor.kind || '');
}

function graphToSnapshot(graph) {
    const values = {};
    for (const tensor of graph.tensors) {
        values[tensor.name] = {
            type: tensorType(tensor),
            initializer: isInitializer(tensor)
                ? {
                    name: tensor.name,
                    category: /constant/i.test(tensor.kind) ? 'Constant' : 'Initializer',
                    type: tensorType(tensor),
                    location: 'inline',
                    preview: null
                }
                : null
        };
    }
    return {
        name: graph.name,
        inputs: graph.inputs.map((name) => ({ name, values: [name] })),
        outputs: graph.outputs.map((name) => ({ name, values: [name] })),
        nodes: graph.nodes.map((node) => ({
            id: node.id,
            name: node.name,
            type: {
                name: node.type,
                module: node.domain || FORMAT_ID,
                identifier: `${node.domain || FORMAT_ID}.${node.type}`,
                category: node.category || CATEGORY_FALLBACK.get(node.type) || 'Custom'
            },
            inputs: node.inputs.map((port) => ({ name: port.name || 'input', values: [port.tensor] })),
            outputs: node.outputs.map((port) => ({ name: port.name || 'output', values: [port.tensor] })),
            attributes: Object.entries(node.attributes || {}).map(([name, value]) => ({ name, value }))
        })),
        values
    };
}

function createGraphIndex(graph) {
    const values = new Map();
    const ensureValue = (name) => {
        if (!values.has(name)) {
            const tensor = graph.tensors.find((item) => item.name === name) || { name, kind: 'activation' };
            values.set(name, {
                name,
                initializer: isInitializer(tensor),
                producer: null,
                consumers: new Set()
            });
        }
        return values.get(name);
    };
    const nodes = graph.nodes.map((node) => {
        const inputs = node.inputs.map((port) => port.tensor);
        const outputs = node.outputs.map((port) => port.tensor);
        for (const name of inputs) {
            ensureValue(name).consumers.add(node.id);
        }
        for (const name of outputs) {
            const value = ensureValue(name);
            value.producer = value.producer || node.id;
        }
        return { id: node.id, node, inputs, outputs };
    });
    for (const name of [...graph.inputs, ...graph.outputs]) {
        ensureValue(name);
    }
    return {
        nodeMap: new Map(nodes.map((node) => [node.id, node])),
        nodes,
        values,
        graphInputs: graph.inputs.slice(),
        graphOutputs: graph.outputs.slice()
    };
}

function cropGraph(graph, startKeys, endKeys) {
    const index = createGraphIndex(graph);
    const starts = new Set(Array.isArray(startKeys) ? startKeys : []);
    const ends = new Set(Array.isArray(endKeys) ? endKeys : []);
    if (starts.size === 0 && ends.size === 0) {
        return graph;
    }

    const startNodes = new Set();
    for (const key of starts) {
        const value = index.values.get(key);
        if (value) {
            for (const consumer of value.consumers) {
                startNodes.add(consumer);
            }
        }
    }
    const endNodes = new Set();
    for (const key of ends) {
        const value = index.values.get(key);
        if (value && value.producer) {
            endNodes.add(value.producer);
        }
    }
    if (startNodes.size === 0) {
        throw new Error('No valid start tensor consumer nodes found.');
    }
    if (endNodes.size === 0) {
        throw new Error('No valid end tensor producer nodes found.');
    }

    const forward = new Set();
    const forwardQueue = Array.from(startNodes);
    while (forwardQueue.length > 0) {
        const nodeId = forwardQueue.shift();
        if (forward.has(nodeId)) {
            continue;
        }
        forward.add(nodeId);
        const node = index.nodeMap.get(nodeId);
        for (const output of node ? node.outputs : []) {
            const value = index.values.get(output);
            for (const consumer of value ? value.consumers : []) {
                forwardQueue.push(consumer);
            }
        }
    }

    const backward = new Set();
    const backwardQueue = Array.from(endNodes);
    while (backwardQueue.length > 0) {
        const nodeId = backwardQueue.shift();
        if (backward.has(nodeId)) {
            continue;
        }
        backward.add(nodeId);
        const node = index.nodeMap.get(nodeId);
        for (const input of node ? node.inputs : []) {
            const value = index.values.get(input);
            if (value && value.producer) {
                backwardQueue.push(value.producer);
            }
        }
    }

    const selected = new Set(Array.from(forward).filter((nodeId) => backward.has(nodeId)));
    if (selected.size === 0) {
        throw new Error('No intersected nodes between start and end tensor paths.');
    }

    const inputKeys = new Set();
    const outputKeys = new Set();
    const tensorNames = new Set();
    const nodes = graph.nodes.filter((node) => selected.has(node.id));
    for (const node of nodes) {
        for (const port of node.inputs) {
            tensorNames.add(port.tensor);
            const value = index.values.get(port.tensor);
            if (value && !value.initializer && (!value.producer || !selected.has(value.producer))) {
                inputKeys.add(port.tensor);
            }
        }
        for (const port of node.outputs) {
            tensorNames.add(port.tensor);
            const value = index.values.get(port.tensor);
            const consumers = value ? Array.from(value.consumers) : [];
            const hasInside = consumers.some((consumer) => selected.has(consumer));
            const hasOutside = consumers.some((consumer) => !selected.has(consumer));
            if (!hasInside || hasOutside || ends.has(port.tensor) || graph.outputs.includes(port.tensor)) {
                outputKeys.add(port.tensor);
            }
        }
    }
    for (const name of [...inputKeys, ...outputKeys]) {
        tensorNames.add(name);
    }
    const id = `${graph.id}:crop:${stableHash({
        inputs: Array.from(inputKeys),
        outputs: Array.from(outputKeys),
        nodes: nodes.map((node) => node.id)
    })}`;
    return normalizeCoreGraph({
        id,
        name: id,
        inputs: Array.from(inputKeys),
        outputs: Array.from(outputKeys),
        nodes,
        tensors: graph.tensors.filter((tensor) => tensorNames.has(tensor.name))
    });
}

function tensorSpec(graph, name) {
    const tensor = graph.tensors.find((item) => item.name === name);
    return {
        name,
        dtype: tensor ? tensor.dtype : 'unknown',
        rank: tensor && Array.isArray(tensor.shape) ? tensor.shape.length : null,
        shape: tensor && Array.isArray(tensor.shape) ? tensor.shape.slice() : [],
        optional: false
    };
}

function fakeOutputValues(graph, artifact, name) {
    const tensor = graph.tensors.find((item) => item.name === name);
    const shape = tensor && Array.isArray(tensor.shape) ? tensor.shape : [1];
    const size = shape.reduce((total, dimension) => total * Math.max(1, Number(dimension) || 1), 1);
    const seed = parseInt(stableHash(`${artifact.id}:${name}`).slice(0, 8), 16);
    const values = [];
    for (let index = 0; index < size; index++) {
        values.push(Number((((seed + index * 1103515245) >>> 0) / 0xffffffff).toFixed(6)));
    }
    return {
        name,
        dtype: tensor ? tensor.dtype : 'float32',
        shape,
        values,
        preview: {
            elementCount: values.length,
            sampleCount: Math.min(values.length, 8),
            sampleValues: values.slice(0, 8),
            truncated: values.length > 8
        },
        summary: values.length > 0
            ? {
                min: Math.min(...values),
                max: Math.max(...values),
                mean: values.reduce((sum, value) => sum + value, 0) / values.length
            }
            : null
    };
}

function createMyIrProvider() {
    const sessions = new Map();
    const artifacts = new Map();
    let sessionSeq = 0;
    let artifactSeq = 0;

    function getSession(sessionId) {
        const session = sessions.get(sessionId);
        if (!session) {
            throw new Error(`Session '${sessionId}' not found.`);
        }
        return session;
    }

    function getArtifact(artifactId) {
        const artifact = artifacts.get(artifactId);
        if (!artifact) {
            throw new Error(`Artifact '${artifactId}' not found.`);
        }
        return artifact;
    }

    return {
        id: FORMAT_ID,
        label: 'My Private IR',
        capabilities: {
            crop: true,
            exportArtifact: true,
            inference: true,
            compare: true,
            textExportContext: true
        },
        canOpen(uri) {
            const filePath = filePathOf(uri);
            return typeof filePath === 'string' && filePath.toLowerCase().endsWith(FILE_EXTENSION);
        },
        async loadModel(uri) {
            const filePath = filePathOf(uri);
            const raw = readJson(filePath);
            const graph = normalizePrivateGraph(raw, path.basename(filePath));
            sessionSeq += 1;
            const session = {
                id: `${FORMAT_ID}-session-${sessionSeq}`,
                format: FORMAT_ID,
                filePath,
                graph,
                snapshot: {
                    format: FORMAT_ID,
                    fileName: path.basename(filePath),
                    filePath,
                    graph: graphToSnapshot(graph)
                }
            };
            session.snapshot.sessionId = session.id;
            sessions.set(session.id, session);
            return session;
        },
        async createCropArtifact({ sessionId, startKeys, endKeys }) {
            const session = getSession(sessionId);
            const graph = cropGraph(session.graph, startKeys, endKeys);
            artifactSeq += 1;
            const artifact = {
                id: `${FORMAT_ID}-artifact-${artifactSeq}`,
                modelSessionId: session.id,
                createdAt: new Date().toISOString(),
                stale: false,
                graph,
                inputKeys: graph.inputs.slice(),
                outputKeys: graph.outputs.slice(),
                ioSignature: {
                    inputs: graph.inputs.map((name) => tensorSpec(graph, name)),
                    outputs: graph.outputs.map((name) => tensorSpec(graph, name))
                },
                summary: {
                    modelName: path.basename(session.filePath),
                    graphName: graph.name,
                    nodeCount: graph.nodes.length,
                    inputCount: graph.inputs.length,
                    outputCount: graph.outputs.length
                },
                cropGraphSnapshot: graphToSnapshot(graph)
            };
            artifacts.set(artifact.id, artifact);
            return artifact;
        },
        getCropTarget(artifactId) {
            const artifact = getArtifact(artifactId);
            const session = getSession(artifact.modelSessionId);
            return buildCropTargetFromCoreGraph({
                providerId: FORMAT_ID,
                model: { format: FORMAT_ID, fileName: path.basename(session.filePath), filePath: session.filePath },
                artifact,
                graph: artifact.graph
            });
        },
        buildTextExportContext(artifactId) {
            const artifact = getArtifact(artifactId);
            const session = getSession(artifact.modelSessionId);
            return buildTextExportContextFromCoreGraph({
                providerId: FORMAT_ID,
                model: { format: FORMAT_ID, fileName: path.basename(session.filePath), filePath: session.filePath },
                artifact,
                graph: artifact.graph
            });
        },
        getExportTarget(artifactId) {
            return {
                artifactId,
                defaultFileName: `${artifactId}.myir.json`,
                filters: { 'My IR': ['json'] },
                title: 'Export My IR Crop',
                stage: 'Export My IR crop',
                message: 'Exporting My IR crop...',
                options: {}
            };
        },
        async exportArtifact(artifactId, targetPath) {
            const artifact = getArtifact(artifactId);
            fs.writeFileSync(targetPath, `${JSON.stringify({
                kind: FORMAT_ID,
                graph: artifact.graph,
                exportedArtifact: { id: artifact.id, createdAt: artifact.createdAt }
            }, null, 2)}\n`);
            return { filePath: targetPath, artifactId, providerId: FORMAT_ID };
        },
        getCompareSlot(artifactId) {
            const artifact = getArtifact(artifactId);
            return {
                providerId: FORMAT_ID,
                artifactId,
                modelSessionId: artifact.modelSessionId,
                ioSignature: artifact.ioSignature,
                summary: artifact.summary,
                createdAt: artifact.createdAt
            };
        },
        async runCompareArtifact({ artifactId }) {
            const artifact = getArtifact(artifactId);
            return {
                outputsSummary: artifact.graph.outputs.map((name) => fakeOutputValues(artifact.graph, artifact, name))
            };
        },
        async runInference(options = {}) {
            let artifact = options.artifactId ? artifacts.get(options.artifactId) : null;
            if (!artifact) {
                artifact = await this.createCropArtifact({ sessionId: options.sessionId });
            }
            return {
                runId: `my-ir-run-${Date.now()}`,
                ...(await this.runCompareArtifact({ artifactId: artifact.id }))
            };
        }
    };
}

module.exports = {
    createMyIrProvider
};
