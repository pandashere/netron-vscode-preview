const fs = require('fs');
const path = require('path');
const {
    buildCropTargetFromCoreGraph,
    buildTextExportContextFromCoreGraph,
    normalizeCoreGraph,
    stableHash
} = require('./text-export-context');

const DEV_IR_KIND = 'netron-dev-ir';
const DEV_IR_EXTENSION = '.netronir.json';
const CATEGORY_FALLBACK = new Map([
    ['Conv', 'Layer'],
    ['ConvTranspose', 'Layer'],
    ['Gemm', 'Layer'],
    ['Linear', 'Layer'],
    ['MatMul', 'Layer'],
    ['PatchEmbed', 'Transform'],
    ['Relu', 'Activation'],
    ['Sigmoid', 'Activation'],
    ['Tanh', 'Activation'],
    ['Softmax', 'Activation'],
    ['BatchNormalization', 'Normalization'],
    ['LayerNormalization', 'Normalization'],
    ['MaxPool', 'Pool'],
    ['AveragePool', 'Pool'],
    ['Reshape', 'Shape'],
    ['Transpose', 'Transform'],
    ['Concat', 'Tensor'],
    ['Slice', 'Tensor'],
    ['Gather', 'Tensor'],
    ['QuantizeLinear', 'Quantization'],
    ['DequantizeLinear', 'Quantization'],
    ['Attention', 'Attention'],
    ['Constant', 'Constant'],
    ['Identity', 'Control']
]);

function filePathOf(uri) {
    return uri && (uri.fsPath || uri.path || String(uri));
}

function readJsonFile(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function normalizeFormatId(value) {
    return String(value || '').trim().toLowerCase();
}

function isDevIrFileName(filePath) {
    return typeof filePath === 'string' && filePath.toLowerCase().endsWith(DEV_IR_EXTENSION);
}

function readFormatId(filePath) {
    if (!isDevIrFileName(filePath) || !fs.existsSync(filePath)) {
        return '';
    }
    try {
        const json = readJsonFile(filePath);
        return normalizeFormatId(json.formatId || json.providerId || json.format);
    } catch {
        return '';
    }
}

function normalizePortList(values, fallbackName) {
    if (!Array.isArray(values)) {
        return [];
    }
    return values.map((item, index) => {
        if (typeof item === 'string') {
            return item;
        }
        if (item && typeof item.name === 'string') {
            return item.name;
        }
        return `${fallbackName || 'tensor'}_${index}`;
    });
}

function normalizeTensor(tensor, index) {
    if (!tensor || typeof tensor !== 'object') {
        throw new Error(`Tensor at index ${index} must be an object.`);
    }
    const name = typeof tensor.name === 'string' && tensor.name ? tensor.name : '';
    if (!name) {
        throw new Error(`Tensor at index ${index} is missing name.`);
    }
    const dtype = String(tensor.dtype || tensor.rawDtype || 'float32').toLowerCase();
    const shape = Array.isArray(tensor.shape) ? tensor.shape.slice() : [];
    return {
        name,
        dtype,
        rawDtype: tensor.rawDtype || dtype,
        shape,
        kind: typeof tensor.kind === 'string' && tensor.kind ? tensor.kind : 'activation'
    };
}

function normalizeNode(node, index) {
    if (!node || typeof node !== 'object') {
        throw new Error(`Node at index ${index} must be an object.`);
    }
    const id = typeof node.id === 'string' && node.id ? node.id : `node-${index}`;
    return {
        id,
        name: typeof node.name === 'string' && node.name ? node.name : id,
        type: typeof node.type === 'string' && node.type ? node.type : 'DevIROp',
        category: typeof node.category === 'string' && node.category ? node.category : '',
        domain: typeof node.domain === 'string' ? node.domain : 'dev.ir',
        inputs: Array.isArray(node.inputs) ? node.inputs.map((port) => ({
            name: port && typeof port.name === 'string' ? port.name : '',
            tensor: port && typeof port.tensor === 'string' ? port.tensor : ''
        })).filter((port) => port.tensor) : [],
        outputs: Array.isArray(node.outputs) ? node.outputs.map((port) => ({
            name: port && typeof port.name === 'string' ? port.name : '',
            tensor: port && typeof port.tensor === 'string' ? port.tensor : ''
        })).filter((port) => port.tensor) : [],
        attributes: node.attributes && typeof node.attributes === 'object' && !Array.isArray(node.attributes) ? { ...node.attributes } : {},
        omittedAttributes: Array.isArray(node.omittedAttributes) ? node.omittedAttributes.map((item) => ({ ...item })) : []
    };
}

function isInitializerTensor(tensor) {
    return /initializer|constant|weight/i.test(tensor.kind);
}

function tensorTypeSnapshot(tensor) {
    return {
        dataType: tensor.dtype,
        shape: tensor.shape
    };
}

function tensorInitializerSnapshot(tensor) {
    if (!isInitializerTensor(tensor)) {
        return null;
    }
    return {
        name: tensor.name,
        category: /constant/i.test(tensor.kind) ? 'Constant' : 'Initializer',
        type: tensorTypeSnapshot(tensor),
        location: 'inline',
        preview: null
    };
}

function nodeCategory(node) {
    return node.category || CATEGORY_FALLBACK.get(node.type) || 'Custom';
}

function createGraphInfo(graph) {
    const values = new Map();
    const ensureValue = (name) => {
        if (!values.has(name)) {
            const tensor = graph.tensors.find((item) => item.name === name) || {
                name,
                dtype: 'unknown',
                rawDtype: 'unknown',
                shape: [],
                kind: 'activation'
            };
            values.set(name, {
                name,
                tensor,
                initializer: isInitializerTensor(tensor),
                producer: null,
                consumers: new Set()
            });
        }
        return values.get(name);
    };
    const nodes = graph.nodes.map((node) => {
        const inputKeys = node.inputs.map((port) => port.tensor).filter(Boolean);
        const outputKeys = node.outputs.map((port) => port.tensor).filter(Boolean);
        for (const key of inputKeys) {
            ensureValue(key).consumers.add(node.id);
        }
        for (const key of outputKeys) {
            const value = ensureValue(key);
            if (!value.producer) {
                value.producer = node.id;
            }
        }
        return {
            id: node.id,
            node,
            inputs: inputKeys,
            outputs: outputKeys
        };
    });
    for (const key of [...graph.inputs, ...graph.outputs]) {
        ensureValue(key);
    }
    return {
        nodeMap: new Map(nodes.map((node) => [node.id, node])),
        nodes,
        values,
        graphInputNames: graph.inputs.slice(),
        graphOutputNames: graph.outputs.slice()
    };
}

function cropGraphInfo(graphInfo, startKeys, endKeys) {
    const startSet = new Set(Array.isArray(startKeys) ? startKeys.filter((key) => typeof key === 'string' && key) : []);
    const endSet = new Set(Array.isArray(endKeys) ? endKeys.filter((key) => typeof key === 'string' && key) : []);
    if (startSet.size === 0 && endSet.size === 0) {
        return {
            selectedNodeIds: new Set(graphInfo.nodes.map((node) => node.id)),
            inputKeys: new Set(graphInfo.graphInputNames),
            outputKeys: new Set(graphInfo.graphOutputNames)
        };
    }
    const startNodes = new Set();
    for (const key of startSet) {
        const tensor = graphInfo.values.get(key);
        if (!tensor) {
            continue;
        }
        for (const consumer of tensor.consumers) {
            startNodes.add(consumer);
        }
    }
    const endNodes = new Set();
    for (const key of endSet) {
        const tensor = graphInfo.values.get(key);
        if (tensor && tensor.producer) {
            endNodes.add(tensor.producer);
        }
    }
    if (startNodes.size === 0) {
        throw new Error('No valid start tensor consumer nodes found.');
    }
    if (endNodes.size === 0) {
        throw new Error('No valid end tensor producer nodes found.');
    }

    const walkForward = new Set();
    const forwardQueue = Array.from(startNodes);
    while (forwardQueue.length > 0) {
        const nodeId = forwardQueue.shift();
        if (walkForward.has(nodeId)) {
            continue;
        }
        walkForward.add(nodeId);
        const node = graphInfo.nodeMap.get(nodeId);
        if (!node) {
            continue;
        }
        for (const key of node.outputs) {
            const tensor = graphInfo.values.get(key);
            if (!tensor) {
                continue;
            }
            for (const consumer of tensor.consumers) {
                if (!walkForward.has(consumer)) {
                    forwardQueue.push(consumer);
                }
            }
        }
    }

    const walkBackward = new Set();
    const backwardQueue = Array.from(endNodes);
    while (backwardQueue.length > 0) {
        const nodeId = backwardQueue.shift();
        if (walkBackward.has(nodeId)) {
            continue;
        }
        walkBackward.add(nodeId);
        const node = graphInfo.nodeMap.get(nodeId);
        if (!node) {
            continue;
        }
        for (const key of node.inputs) {
            const tensor = graphInfo.values.get(key);
            if (tensor && tensor.producer && !walkBackward.has(tensor.producer)) {
                backwardQueue.push(tensor.producer);
            }
        }
    }

    const selectedNodeIds = new Set(Array.from(walkForward).filter((id) => walkBackward.has(id)));
    if (selectedNodeIds.size === 0) {
        throw new Error('No intersected nodes between start and end tensor paths.');
    }

    const inputKeys = new Set();
    const outputKeys = new Set();
    for (const nodeId of selectedNodeIds) {
        const node = graphInfo.nodeMap.get(nodeId);
        if (!node) {
            continue;
        }
        for (const key of node.inputs) {
            const tensor = graphInfo.values.get(key);
            if (!tensor) {
                continue;
            }
            if ((!tensor.producer || !selectedNodeIds.has(tensor.producer)) && !tensor.initializer) {
                inputKeys.add(key);
            }
        }
        for (const key of node.outputs) {
            const tensor = graphInfo.values.get(key);
            if (!tensor) {
                continue;
            }
            let hasInside = false;
            let hasOutside = false;
            for (const consumer of tensor.consumers) {
                if (selectedNodeIds.has(consumer)) {
                    hasInside = true;
                } else {
                    hasOutside = true;
                }
            }
            if (!hasInside || hasOutside || endSet.has(key) || graphInfo.graphOutputNames.includes(key)) {
                outputKeys.add(key);
            }
        }
    }
    for (const key of graphInfo.graphInputNames) {
        const tensor = graphInfo.values.get(key);
        if (tensor && Array.from(tensor.consumers).some((id) => selectedNodeIds.has(id))) {
            inputKeys.add(key);
        }
    }
    for (const key of graphInfo.graphOutputNames) {
        const tensor = graphInfo.values.get(key);
        if (tensor && tensor.producer && selectedNodeIds.has(tensor.producer)) {
            outputKeys.add(key);
        }
    }
    return { selectedNodeIds, inputKeys, outputKeys };
}

function buildCroppedGraph(graph, cropResult) {
    const selectedNodeIds = cropResult.selectedNodeIds;
    const nodes = graph.nodes.filter((node) => selectedNodeIds.has(node.id));
    const tensorNames = new Set([...cropResult.inputKeys, ...cropResult.outputKeys]);
    for (const node of nodes) {
        for (const port of [...node.inputs, ...node.outputs]) {
            if (port.tensor) {
                tensorNames.add(port.tensor);
            }
        }
    }
    const tensors = graph.tensors.filter((tensor) => tensorNames.has(tensor.name));
    const id = `${graph.id}:crop:${stableHash({
        inputs: Array.from(cropResult.inputKeys),
        outputs: Array.from(cropResult.outputKeys),
        nodes: nodes.map((node) => node.id)
    })}`;
    return normalizeCoreGraph({
        id,
        name: id,
        inputs: Array.from(cropResult.inputKeys),
        outputs: Array.from(cropResult.outputKeys),
        nodes,
        tensors
    });
}

function normalizeGraph(raw, formatId, fileBaseName) {
    const graph = raw && raw.graph && typeof raw.graph === 'object' ? raw.graph : {};
    const tensors = Array.isArray(graph.tensors) ? graph.tensors.map(normalizeTensor) : [];
    const inputs = normalizePortList(graph.inputs, 'input');
    const outputs = normalizePortList(graph.outputs, 'output');
    const nodes = Array.isArray(graph.nodes) ? graph.nodes.map(normalizeNode) : [];
    const id = typeof graph.id === 'string' && graph.id
        ? graph.id
        : `${formatId}:${fileBaseName.replace(/\.[^.]+$/, '')}:${stableHash({ inputs, outputs, nodes })}`;
    return normalizeCoreGraph({
        id,
        name: typeof graph.name === 'string' && graph.name ? graph.name : id,
        inputs,
        outputs,
        nodes,
        tensors
    });
}

function loadDevIrDocument(filePath, expectedFormatId) {
    const raw = readJsonFile(filePath);
    const kind = typeof raw.kind === 'string' ? raw.kind : DEV_IR_KIND;
    if (kind !== DEV_IR_KIND) {
        throw new Error(`Dev IR kind must be '${DEV_IR_KIND}'.`);
    }
    const formatId = normalizeFormatId(raw.formatId || raw.providerId || raw.format);
    if (!formatId) {
        throw new Error('Dev IR formatId is required.');
    }
    if (expectedFormatId && formatId !== expectedFormatId) {
        throw new Error(`Dev IR file formatId '${formatId}' does not match provider '${expectedFormatId}'.`);
    }
    const fileName = path.basename(filePath);
    const graph = normalizeGraph(raw, formatId, fileName);
    return {
        raw,
        formatId,
        label: typeof raw.label === 'string' && raw.label ? raw.label : formatId,
        filePath,
        fileName,
        graph,
        runtime: raw.runtime && typeof raw.runtime === 'object' ? raw.runtime : {}
    };
}

function graphToSnapshot(graph, formatId) {
    const values = {};
    for (const tensor of graph.tensors) {
        values[tensor.name] = {
            type: tensorTypeSnapshot(tensor),
            initializer: tensorInitializerSnapshot(tensor)
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
                module: node.domain || formatId,
                identifier: node.domain ? `${node.domain}.${node.type}` : node.type,
                category: nodeCategory(node)
            },
            inputs: node.inputs.map((port) => ({ name: port.name || 'input', values: [port.tensor] })),
            outputs: node.outputs.map((port) => ({ name: port.name || 'output', values: [port.tensor] })),
            attributes: Object.entries(node.attributes || {}).map(([name, value]) => ({ name, value }))
        })),
        values
    };
}

function tensorSpec(graph, name) {
    const tensor = graph.tensors.find((item) => item.name === name);
    if (!tensor) {
        throw new Error(`Tensor '${name}' is not defined.`);
    }
    return {
        name: tensor.name,
        dtype: tensor.dtype,
        rank: Array.isArray(tensor.shape) ? tensor.shape.length : null,
        shape: Array.isArray(tensor.shape) ? tensor.shape.slice() : [],
        optional: false
    };
}

function summarizeValues(values) {
    const numeric = values.map(Number).filter((value) => Number.isFinite(value));
    if (numeric.length === 0) {
        return null;
    }
    const total = numeric.reduce((sum, value) => sum + value, 0);
    return {
        min: Math.min(...numeric),
        max: Math.max(...numeric),
        mean: total / numeric.length
    };
}

function product(shape) {
    return Array.isArray(shape) && shape.length > 0
        ? shape.reduce((total, dimension) => total * (Number.isFinite(Number(dimension)) ? Number(dimension) : 1), 1)
        : 1;
}

function hashSeed(text) {
    const hash = stableHash(text);
    return parseInt(hash.slice(0, 8), 16) >>> 0;
}

function mulberry32(seed) {
    let state = seed >>> 0;
    return () => {
        state += 0x6D2B79F5;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}

function runtimeOutputConfig(document, outputName) {
    const outputs = document.runtime && document.runtime.outputs && typeof document.runtime.outputs === 'object'
        ? document.runtime.outputs
        : {};
    return outputs[outputName] && typeof outputs[outputName] === 'object' ? outputs[outputName] : {};
}

function generateOutputValues(document, artifact, tensor) {
    const config = runtimeOutputConfig(document, tensor.name);
    if (Array.isArray(config.values)) {
        return config.values.map(Number);
    }
    const size = product(tensor.shape);
    const seedText = `${document.formatId}:${document.graph.id}:${artifact.id}:${tensor.name}:${document.runtime.seed || 0}:${config.seed || 0}`;
    const random = mulberry32(hashSeed(seedText));
    const scale = Number.isFinite(Number(config.scale)) ? Number(config.scale) : 1;
    const offset = Number.isFinite(Number(config.offset)) ? Number(config.offset) : 0;
    const values = [];
    for (let index = 0; index < size; index++) {
        values.push(Number((offset + random() * scale).toFixed(6)));
    }
    return values;
}

function outputSummary(document, artifact, tensor) {
    const values = generateOutputValues(document, artifact, tensor);
    return {
        name: tensor.name,
        dtype: tensor.dtype,
        shape: tensor.shape,
        values,
        preview: {
            elementCount: values.length,
            sampleCount: Math.min(values.length, 8),
            sampleValues: values.slice(0, 8),
            truncated: values.length > 8
        },
        summary: summarizeValues(values)
    };
}

function createDevIrProvider(options = {}) {
    const providerId = normalizeFormatId(options.id || 'dev-ir-a');
    const sessions = new Map();
    const artifacts = new Map();
    const importedInputs = new Map();
    let sessionSeq = 0;
    let artifactSeq = 0;
    let inputSeq = 0;

    function getSession(sessionId) {
        const session = sessions.get(sessionId);
        if (!session) {
            throw new Error(`Dev IR session '${sessionId}' not found.`);
        }
        return session;
    }

    function getArtifact(artifactId) {
        const artifact = artifacts.get(artifactId);
        if (!artifact) {
            throw new Error(`Dev IR artifact '${artifactId}' not found.`);
        }
        return artifact;
    }

    function modelMetadata(session) {
        return {
            format: providerId,
            fileName: session.document.fileName,
            filePath: session.document.filePath
        };
    }

    function createIoSignature(graph) {
        return {
            inputs: graph.inputs.map((name) => tensorSpec(graph, name)),
            outputs: graph.outputs.map((name) => tensorSpec(graph, name))
        };
    }

    return {
        id: providerId,
        label: options.label || `Dev IR ${providerId.replace(/^dev-ir-/, '').toUpperCase()}`,
        capabilities: {
            crop: true,
            exportArtifact: true,
            inference: true,
            compare: true,
            textExportContext: true,
            inputImport: true
        },
        canOpen(uri) {
            const filePath = filePathOf(uri);
            return readFormatId(filePath) === providerId;
        },
        async loadModel(uri) {
            const filePath = filePathOf(uri);
            const document = loadDevIrDocument(filePath, providerId);
            sessionSeq += 1;
            const sessionId = `${providerId}-session-${sessionSeq}`;
            const session = {
                id: sessionId,
                format: providerId,
                filePath,
                document,
                snapshot: {
                    sessionId,
                    fileName: path.basename(filePath),
                    filePath,
                    format: providerId,
                    graph: graphToSnapshot(document.graph, providerId)
                }
            };
            sessions.set(session.id, session);
            return session;
        },
        async createCropArtifact({ sessionId, startKeys, endKeys }) {
            const session = getSession(sessionId);
            artifactSeq += 1;
            const graphInfo = createGraphInfo(session.document.graph);
            const cropResult = cropGraphInfo(graphInfo, startKeys, endKeys);
            const graph = buildCroppedGraph(session.document.graph, cropResult);
            const artifact = {
                id: `${providerId}-artifact-${artifactSeq}`,
                modelSessionId: session.id,
                createdAt: new Date().toISOString(),
                stale: false,
                inputKeys: graph.inputs.slice(),
                outputKeys: graph.outputs.slice(),
                ioSignature: createIoSignature(graph),
                selectedNodeIds: Array.from(cropResult.selectedNodeIds),
                graph,
                summary: {
                    modelName: session.document.fileName,
                    graphName: graph.name,
                    nodeCount: graph.nodes.length,
                    inputCount: graph.inputs.length,
                    outputCount: graph.outputs.length
                },
                cropGraphSnapshot: graphToSnapshot(graph, providerId)
            };
            artifacts.set(artifact.id, artifact);
            return artifact;
        },
        getCropTarget(artifactId) {
            const artifact = getArtifact(artifactId);
            const session = getSession(artifact.modelSessionId);
            return buildCropTargetFromCoreGraph({
                providerId,
                model: modelMetadata(session),
                artifact,
                graph: artifact.graph
            });
        },
        buildTextExportContext(artifactId) {
            const artifact = getArtifact(artifactId);
            const session = getSession(artifact.modelSessionId);
            return buildTextExportContextFromCoreGraph({
                providerId,
                model: modelMetadata(session),
                artifact,
                graph: artifact.graph
            });
        },
        getExportTarget(artifactId) {
            const artifact = getArtifact(artifactId);
            return {
                artifactId: artifact.id,
                defaultFileName: `${artifact.id}.netronir.json`,
                filters: { 'Dev IR': ['json'] },
                title: 'Export Dev IR Crop',
                stage: 'Export Dev IR crop',
                message: 'Exporting Dev IR crop artifact...',
                options: {}
            };
        },
        async exportArtifact(artifactId, targetPath) {
            const artifact = getArtifact(artifactId);
            const session = getSession(artifact.modelSessionId);
            const payload = {
                kind: DEV_IR_KIND,
                formatId: providerId,
                label: session.document.label,
                graph: artifact.graph,
                runtime: session.document.runtime,
                exportedArtifact: {
                    id: artifact.id,
                    createdAt: artifact.createdAt
                }
            };
            fs.writeFileSync(targetPath, `${JSON.stringify(payload, null, 2)}\n`);
            return {
                filePath: targetPath,
                artifactId: artifact.id,
                providerId
            };
        },
        getCompareSlot(artifactId) {
            const artifact = getArtifact(artifactId);
            return {
                providerId,
                artifactId: artifact.id,
                modelSessionId: artifact.modelSessionId,
                ioSignature: artifact.ioSignature,
                summary: artifact.summary,
                createdAt: artifact.createdAt
            };
        },
        async runCompareArtifact({ artifactId }) {
            const artifact = getArtifact(artifactId);
            const session = getSession(artifact.modelSessionId);
            return {
                outputsSummary: artifact.graph.outputs.map((name) => outputSummary(
                    session.document,
                    artifact,
                    artifact.graph.tensors.find((tensor) => tensor.name === name)
                ))
            };
        },
        async runInference(options = {}) {
            const artifactId = options.artifactId || `${providerId}-full-graph`;
            let artifact = artifacts.get(artifactId);
            if (!artifact) {
                artifact = await this.createCropArtifact({ sessionId: options.sessionId });
            }
            return {
                runId: `dev-ir-run-${Date.now()}`,
                ...(await this.runCompareArtifact({ artifactId: artifact.id }))
            };
        },
        async importInputFile(filePath) {
            const raw = readJsonFile(filePath);
            inputSeq += 1;
            const token = `${providerId}-input-${inputSeq}`;
            importedInputs.set(token, raw);
            const preview = Object.entries(raw).map(([name, value]) => ({
                name,
                dtype: value && (value.dtype || value.type) ? (value.dtype || value.type) : 'float32',
                shape: value && Array.isArray(value.shape) ? value.shape : []
            }));
            return { token, preview };
        },
        resolveImportedInput(token) {
            return importedInputs.get(token) || null;
        }
    };
}

module.exports = {
    DEV_IR_EXTENSION,
    DEV_IR_KIND,
    createDevIrProvider,
    isDevIrFileName,
    loadDevIrDocument
};
