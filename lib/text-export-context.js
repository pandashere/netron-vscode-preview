const crypto = require('crypto');
const path = require('path');

function stableHash(value) {
    return crypto.createHash('sha1').update(JSON.stringify(value)).digest('hex').slice(0, 12);
}

function normalizeDType(value) {
    const text = String(value || 'unknown').toLowerCase();
    const map = new Map([
        ['float', 'float32'],
        ['double', 'float64'],
        ['tensor(float)', 'float32']
    ]);
    return map.get(text) || text || 'unknown';
}

function normalizeShape(shape) {
    if (!Array.isArray(shape)) {
        return null;
    }
    return shape.map((dimension) => {
        if (typeof dimension === 'number' && Number.isFinite(dimension)) {
            return dimension;
        }
        if (dimension === null || dimension === undefined) {
            return '?';
        }
        const numeric = Number(dimension);
        return Number.isFinite(numeric) ? numeric : String(dimension);
    });
}

function tensorKind(name, artifact, graph) {
    if (artifact.inputKeys && artifact.inputKeys.includes(name)) {
        return 'input';
    }
    if (artifact.outputKeys && artifact.outputKeys.includes(name)) {
        return 'output';
    }
    const value = graph.values && graph.values[name] ? graph.values[name] : null;
    if (value && value.initializer) {
        return 'initializer';
    }
    return 'activation';
}

function attributeToContext(attribute) {
    if (!attribute || !attribute.name) {
        return { omit: null };
    }
    if (attribute.value === '[Tensor]') {
        return { omit: { name: attribute.name, reason: 'tensor-data' } };
    }
    if (attribute.value === '[Graph]') {
        return { omit: { name: attribute.name, reason: 'subgraph' } };
    }
    const value = attribute.value;
    if (value === undefined || typeof value === 'function') {
        return { omit: { name: attribute.name, reason: 'unsupported-type' } };
    }
    if (Array.isArray(value) && value.length > 128) {
        return { omit: { name: attribute.name, reason: 'large-array' } };
    }
    return { name: attribute.name, value };
}

function buildGraphId(session, artifact, nodes) {
    const source = session && session.graphInfo && session.graphInfo.name ? session.graphInfo.name : path.basename(session.filePath || 'model');
    const payload = {
        source,
        inputs: artifact.inputKeys || [],
        outputs: artifact.outputKeys || [],
        nodes: nodes.map((node) => ({
            id: node.id,
            type: node.type,
            inputs: node.inputs.map((item) => item.tensor),
            outputs: node.outputs.map((item) => item.tensor)
        }))
    };
    const safeSource = String(source || 'graph').replace(/[^a-z0-9_.-]+/gi, '-').replace(/^-+|-+$/g, '') || 'graph';
    return `${safeSource}:crop:${stableHash(payload)}`;
}

function buildContextGraph(session, artifact) {
    if (!session) {
        throw new Error('Model session not found.');
    }
    if (!artifact) {
        throw new Error('No confirmed crop artifact available.');
    }
    if (artifact.stale) {
        throw new Error('Current crop is stale. Confirm crop again.');
    }
    const snapshot = artifact.cropGraphSnapshot;
    if (!snapshot || !snapshot.nodes || !snapshot.values) {
        throw new Error('Crop graph snapshot is not available.');
    }
    const tensorNames = new Set();
    const nodes = snapshot.nodes.map((node) => {
        const attributes = {};
        const omittedAttributes = [];
        for (const attribute of node.attributes || []) {
            const result = attributeToContext(attribute);
            if (result.omit) {
                omittedAttributes.push(result.omit);
            } else if (result.name) {
                attributes[result.name] = result.value;
            }
        }
        const inputs = [];
        for (const input of node.inputs || []) {
            for (const name of input.values || []) {
                if (!name) {
                    continue;
                }
                tensorNames.add(name);
                inputs.push({ name: input.name || '', tensor: name });
            }
        }
        const outputs = [];
        for (const output of node.outputs || []) {
            for (const name of output.values || []) {
                if (!name) {
                    continue;
                }
                tensorNames.add(name);
                outputs.push({ name: output.name || '', tensor: name });
            }
        }
        return {
            id: node.id,
            name: node.name || node.id,
            type: node.type && node.type.name ? node.type.name : 'Unknown',
            domain: node.type && node.type.module ? node.type.module : '',
            inputs,
            outputs,
            attributes,
            omittedAttributes
        };
    });
    const graphInputs = [];
    for (const item of snapshot.inputs || []) {
        for (const name of item.values || []) {
            if (name) {
                graphInputs.push(name);
                tensorNames.add(name);
            }
        }
    }
    const graphOutputs = [];
    for (const item of snapshot.outputs || []) {
        for (const name of item.values || []) {
            if (name) {
                graphOutputs.push(name);
                tensorNames.add(name);
            }
        }
    }
    const tensors = Array.from(tensorNames).sort().map((name) => {
        const value = snapshot.values[name] || {};
        const type = value.type || {};
        const dtype = normalizeDType(type.dataType);
        return {
            name,
            dtype,
            rawDtype: type.dataType || dtype,
            shape: normalizeShape(type.shape),
            kind: tensorKind(name, artifact, snapshot)
        };
    });
    const graphId = buildGraphId(session, artifact, nodes);
    return {
        id: graphId,
        name: graphId,
        inputs: graphInputs,
        outputs: graphOutputs,
        nodes,
        tensors
    };
}

function buildCropTargetFromGraph(session, artifact, graph) {
    return buildCropTargetFromCoreGraph({
        providerId: String(session.format || 'unknown').toLowerCase(),
        model: {
            format: String(session.format || 'unknown').toLowerCase(),
            fileName: path.basename(session.filePath),
            filePath: session.filePath
        },
        artifact,
        graph
    });
}

function normalizeCorePort(port) {
    if (!port || typeof port !== 'object') {
        return null;
    }
    const tensor = typeof port.tensor === 'string' && port.tensor ? port.tensor : '';
    if (!tensor) {
        return null;
    }
    return {
        name: typeof port.name === 'string' ? port.name : '',
        tensor
    };
}

function normalizeCoreNode(node, index) {
    if (!node || typeof node !== 'object') {
        throw new Error(`Graph node at index ${index} must be an object.`);
    }
    const id = typeof node.id === 'string' && node.id ? node.id : `node-${index}`;
    return {
        id,
        name: typeof node.name === 'string' && node.name ? node.name : id,
        type: typeof node.type === 'string' && node.type ? node.type : 'Unknown',
        domain: typeof node.domain === 'string' ? node.domain : '',
        inputs: Array.isArray(node.inputs) ? node.inputs.map(normalizeCorePort).filter(Boolean) : [],
        outputs: Array.isArray(node.outputs) ? node.outputs.map(normalizeCorePort).filter(Boolean) : [],
        attributes: node.attributes && typeof node.attributes === 'object' && !Array.isArray(node.attributes) ? { ...node.attributes } : {},
        omittedAttributes: Array.isArray(node.omittedAttributes) ? node.omittedAttributes.map((item) => ({ ...item })) : []
    };
}

function normalizeCoreTensor(tensor, index) {
    if (!tensor || typeof tensor !== 'object') {
        throw new Error(`Graph tensor at index ${index} must be an object.`);
    }
    const name = typeof tensor.name === 'string' && tensor.name ? tensor.name : '';
    if (!name) {
        throw new Error(`Graph tensor at index ${index} is missing name.`);
    }
    const dtype = normalizeDType(tensor.dtype || tensor.rawDtype);
    return {
        name,
        dtype,
        rawDtype: tensor.rawDtype || dtype,
        shape: normalizeShape(tensor.shape),
        kind: typeof tensor.kind === 'string' && tensor.kind ? tensor.kind : 'activation'
    };
}

function normalizeCoreGraph(graph) {
    if (!graph || typeof graph !== 'object') {
        throw new Error('Core graph must be an object.');
    }
    const id = typeof graph.id === 'string' && graph.id ? graph.id : '';
    if (!id) {
        throw new Error('Core graph id is required.');
    }
    const nodes = Array.isArray(graph.nodes) ? graph.nodes.map(normalizeCoreNode) : [];
    const tensors = Array.isArray(graph.tensors) ? graph.tensors.map(normalizeCoreTensor) : [];
    const tensorNames = new Set(tensors.map((tensor) => tensor.name));
    const inputs = Array.isArray(graph.inputs) ? graph.inputs.filter((name) => typeof name === 'string' && name) : [];
    const outputs = Array.isArray(graph.outputs) ? graph.outputs.filter((name) => typeof name === 'string' && name) : [];
    for (const name of [...inputs, ...outputs]) {
        if (!tensorNames.has(name)) {
            throw new Error(`Boundary tensor '${name}' is missing from graph.tensors.`);
        }
    }
    for (const node of nodes) {
        for (const port of [...node.inputs, ...node.outputs]) {
            if (!tensorNames.has(port.tensor)) {
                throw new Error(`Node tensor '${port.tensor}' is missing from graph.tensors.`);
            }
        }
    }
    return {
        id,
        name: typeof graph.name === 'string' && graph.name ? graph.name : id,
        inputs,
        outputs,
        nodes,
        tensors
    };
}

function buildCropTargetFromCoreGraph({ providerId, model, artifact, graph }) {
    if (!artifact) {
        throw new Error('No confirmed crop artifact available.');
    }
    if (artifact.stale) {
        throw new Error('Current crop is stale. Confirm crop again.');
    }
    const normalizedGraph = normalizeCoreGraph(graph);
    const normalizedProviderId = String(providerId || (model && model.format) || 'unknown').toLowerCase();
    const filePath = model && model.filePath ? model.filePath : '';
    return {
        kind: 'crop-target',
        schemaVersion: 1,
        providerId: normalizedProviderId,
        target: 'crop',
        model: {
            format: String((model && model.format) || normalizedProviderId || 'unknown').toLowerCase(),
            fileName: (model && model.fileName) || path.basename(filePath || 'model'),
            filePath
        },
        artifact: {
            id: artifact.id,
            createdAt: artifact.createdAt,
            stale: !!artifact.stale
        },
        graph: {
            id: normalizedGraph.id,
            name: normalizedGraph.name
        }
    };
}

function buildTextExportContextFromCoreGraph({ providerId, model, artifact, graph }) {
    const normalizedGraph = normalizeCoreGraph(graph);
    const target = buildCropTargetFromCoreGraph({ providerId, model, artifact, graph: normalizedGraph });
    return {
        kind: 'text-export-context',
        schemaVersion: 1,
        target: 'crop',
        model: target.model,
        artifact: {
            id: target.artifact.id,
            createdAt: target.artifact.createdAt
        },
        graph: normalizedGraph
    };
}

function buildTextExportContext(session, artifact) {
    const graph = buildContextGraph(session, artifact);
    return buildTextExportContextFromCoreGraph({
        providerId: String(session.format || 'unknown').toLowerCase(),
        model: {
            format: String(session.format || 'unknown').toLowerCase(),
            fileName: path.basename(session.filePath),
            filePath: session.filePath
        },
        artifact,
        graph
    });
}

function buildCropTarget(session, artifact) {
    const graph = buildContextGraph(session, artifact);
    return buildCropTargetFromGraph(session, artifact, graph);
}

module.exports = {
    buildCropTarget,
    buildCropTargetFromCoreGraph,
    buildTextExportContext,
    buildTextExportContextFromCoreGraph,
    normalizeCoreGraph,
    stableHash
};
