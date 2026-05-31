const {
    buildCompareRows,
    buildSharedInputPlan,
    resolveCompareBindings,
    summarizeCompareRows
} = require('./compare-core');

function nowIso() {
    return new Date().toISOString();
}

function product(shape) {
    return shape.reduce((total, dimension) => total * Number(dimension), 1);
}

function generateInputData(spec, inputMode, importedData, canonicalNames) {
    if (inputMode === 'import') {
        if (!importedData) {
            throw new Error('Import mode requires imported input data.');
        }
        for (const key of canonicalNames) {
            if (importedData[key]) {
                const imported = importedData[key];
                return {
                    dtype: imported.dtype || imported.type || spec.dtype,
                    shape: imported.shape || spec.shape,
                    data: imported.data
                };
            }
        }
        throw new Error(`Imported input is missing '${canonicalNames[0]}'.`);
    }
    const size = spec.shape.length > 0 ? product(spec.shape) : 1;
    if (size <= 0) {
        throw new Error(`Invalid input shape for '${spec.name}'.`);
    }
    const supported = new Set([
        'float32',
        'float64',
        'float16',
        'bfloat16',
        'uint8',
        'int8',
        'uint16',
        'int16',
        'int32',
        'uint32',
        'bool',
        'int64',
        'uint64'
    ]);
    if (!supported.has(spec.dtype)) {
        throw new Error(`Unsupported input dtype '${spec.dtype}'.`);
    }
    const values = new Array(size);
    for (let index = 0; index < size; index++) {
        if (inputMode === 'ones') {
            values[index] = 1;
        } else if (inputMode === 'random') {
            values[index] = Math.random();
        } else {
            values[index] = 0;
        }
    }
    return {
        dtype: spec.dtype,
        shape: spec.shape,
        data: values
    };
}

function boolArrayToBuffer(values) {
    const buffer = Buffer.alloc(values.length);
    for (let index = 0; index < values.length; index++) {
        buffer[index] = values[index] ? 1 : 0;
    }
    return buffer;
}

function float32ToFloat16(value) {
    const floatView = new Float32Array(1);
    const intView = new Uint32Array(floatView.buffer);
    floatView[0] = value;
    const bits = intView[0];
    const sign = (bits >>> 16) & 0x8000;
    let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
    let mantissa = bits & 0x7fffff;
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa = (mantissa | 0x800000) >>> (1 - exponent);
        return sign | ((mantissa + 0x1000) >>> 13);
    }
    if (exponent >= 0x1f) {
        return sign | 0x7c00;
    }
    return sign | (exponent << 10) | ((mantissa + 0x1000) >>> 13);
}

function typedArrayToBuffer(values, dataType) {
    switch (dataType) {
        case 'float32':
            return Buffer.from(new Float32Array(values).buffer);
        case 'float64':
            return Buffer.from(new Float64Array(values).buffer);
        case 'uint8':
            return Buffer.from(new Uint8Array(values).buffer);
        case 'int8':
            return Buffer.from(new Int8Array(values).buffer);
        case 'uint16':
            return Buffer.from(new Uint16Array(values).buffer);
        case 'int16':
            return Buffer.from(new Int16Array(values).buffer);
        case 'int32':
            return Buffer.from(new Int32Array(values).buffer);
        case 'uint32':
            return Buffer.from(new Uint32Array(values).buffer);
        case 'int64': {
            const buffer = Buffer.alloc(values.length * 8);
            values.forEach((item, index) => buffer.writeBigInt64LE(BigInt(Math.trunc(item)), index * 8));
            return buffer;
        }
        case 'uint64': {
            const buffer = Buffer.alloc(values.length * 8);
            values.forEach((item, index) => buffer.writeBigUInt64LE(BigInt(Math.trunc(item)), index * 8));
            return buffer;
        }
        case 'bool':
            return boolArrayToBuffer(values);
        case 'float16': {
            const buffer = Buffer.alloc(values.length * 2);
            values.forEach((item, index) => buffer.writeUInt16LE(float32ToFloat16(Number(item)), index * 2));
            return buffer;
        }
        default:
            throw new Error(`Unsupported .npy export dtype '${dataType}'.`);
    }
}

function encodeNpy(values, dataType, shape) {
    const descriptor = {
        float32: '<f4',
        float64: '<f8',
        uint8: '|u1',
        int8: '|i1',
        uint16: '<u2',
        int16: '<i2',
        int32: '<i4',
        uint32: '<u4',
        int64: '<i8',
        uint64: '<u8',
        bool: '|b1',
        float16: '<f2'
    }[dataType];
    if (!descriptor) {
        throw new Error(`Unsupported .npy export dtype '${dataType}'.`);
    }
    const payload = typedArrayToBuffer(values, dataType);
    const dims = Array.isArray(shape) ? shape.map((item) => Number(item)) : [];
    const shapeLiteral = dims.length === 0 ? '' : dims.length === 1 ? `${dims[0]},` : dims.join(', ');
    let header = `{'descr': '${descriptor}', 'fortran_order': False, 'shape': (${shapeLiteral}), }`;
    const preambleLength = 10;
    const padding = (16 - ((preambleLength + Buffer.byteLength(header, 'latin1') + 1) % 16)) % 16;
    header += ' '.repeat(padding) + '\n';
    const headerBuffer = Buffer.from(header, 'latin1');
    const prefix = Buffer.alloc(10);
    prefix.write('\x93NUMPY', 0, 'binary');
    prefix[6] = 1;
    prefix[7] = 0;
    prefix.writeUInt16LE(headerBuffer.length, 8);
    return Buffer.concat([prefix, headerBuffer, payload]);
}

function sanitizeFileName(value) {
    const normalized = String(value || 'tensor').replace(/[^a-z0-9._-]+/gi, '_').replace(/^_+|_+$/g, '');
    return normalized || 'tensor';
}

function exportCompareResultAsJson(compareState) {
    return JSON.stringify(compareState && compareState.compareResult ? compareState.compareResult : {}, null, 2);
}

function exportCompareResultAsCsv(compareState) {
    const result = compareState && compareState.compareResult;
    if (!result || !Array.isArray(result.rows)) {
        return 'sourceName,targetName,status,reason,shape,dtype,maxAbs,meanAbs,rmse,maxRelativeDiff,cosineSimilarity,pearsonCorrelation';
    }
    const lines = [
        'sourceName,targetName,status,reason,shape,dtype,maxAbs,meanAbs,rmse,maxRelativeDiff,cosineSimilarity,pearsonCorrelation'
    ];
    for (const row of result.rows) {
        lines.push([
            row.sourceName || '',
            row.targetName || '',
            row.status || '',
            row.reason || '',
            Array.isArray(row.shape) ? JSON.stringify(row.shape) : '',
            row.dtype || '',
            row.maxAbs ?? '',
            row.meanAbs ?? '',
            row.rmse ?? '',
            row.maxRelativeDiff ?? '',
            row.cosineSimilarity ?? '',
            row.pearsonCorrelation ?? ''
        ].map((item) => String(item).replace(/"/g, '""')).map((item) => `"${item}"`).join(','));
    }
    return lines.join('\n');
}

function exportCompareOutputAsNpy(compareState, rawOutputs, options = {}) {
    const result = compareState && compareState.compareResult;
    if (!result || !result.rawOutputRef) {
        throw new Error('No compare outputs are available to export.');
    }
    const cached = rawOutputs && rawOutputs.get ? rawOutputs.get(result.rawOutputRef) : null;
    if (!cached) {
        throw new Error('Compare output cache is not available.');
    }
    const side = options.side === 'B' ? 'B' : 'A';
    const outputName = side === 'B' ? options.targetName : options.sourceName;
    if (!outputName) {
        throw new Error('Output name is required for NPY export.');
    }
    const outputs = side === 'B' ? cached.sideB : cached.sideA;
    const output = Array.isArray(outputs) ? outputs.find((item) => item.name === outputName) : null;
    if (!output || !Array.isArray(output.values)) {
        throw new Error(`Output '${outputName}' is not available for NPY export.`);
    }
    const subgraph = result.subgraphs && result.subgraphs[side] ? result.subgraphs[side] : null;
    const fileName = `${sanitizeFileName(subgraph && subgraph.modelName ? subgraph.modelName.replace(/\.onnx$/i, '') : `slot-${side.toLowerCase()}`)}-${sanitizeFileName(outputName)}.npy`;
    return {
        fileName,
        bytes: encodeNpy(output.values, output.dtype, output.shape)
    };
}

function getProvider(providerRegistry, providerId) {
    const provider = providerRegistry && typeof providerRegistry.get === 'function' ? providerRegistry.get(providerId) : null;
    if (!provider) {
        throw new Error(`Compare provider '${providerId}' is not registered.`);
    }
    if (typeof provider.runCompareArtifact !== 'function') {
        throw new Error(`Compare provider '${providerId}' does not support compare execution.`);
    }
    return provider;
}

function buildCompareResult({ slotA, slotB, rows, sharedInputPlan, inputBindings, outputBindings, inputMode, rawOutputRef, createdAt }) {
    const { validRows, summary } = summarizeCompareRows(rows);
    return {
        createdAt,
        inputMode,
        resolvedShapes: sharedInputPlan.map((input) => ({ name: input.sourceName, shape: input.shape, dtype: input.dtype, targetName: input.targetName })),
        rawOutputRef,
        subgraphs: {
            A: {
                providerId: slotA.providerId,
                artifactId: slotA.artifactId,
                modelSessionId: slotA.modelSessionId,
                ...slotA.summary
            },
            B: {
                providerId: slotB.providerId,
                artifactId: slotB.artifactId,
                modelSessionId: slotB.modelSessionId,
                ...slotB.summary
            }
        },
        compareStats: {
            inputBindingCount: inputBindings.length,
            outputBindingCount: outputBindings.length,
            rowCount: rows.length,
            okCount: validRows.length,
            skippedCount: rows.length - validRows.length
        },
        rows,
        summary
    };
}

async function runCrossProviderCompare(compareState, providerRegistry, options = {}) {
    const slotA = compareState && compareState.slotA;
    const slotB = compareState && compareState.slotB;
    if (!slotA || !slotB) {
        throw new Error('Compare slots are incomplete.');
    }
    const providerA = getProvider(providerRegistry, slotA.providerId);
    const providerB = getProvider(providerRegistry, slotB.providerId);
    const inputMode = options.inputMode || 'zeros';
    const { inputBindings, outputBindings } = resolveCompareBindings(compareState);
    const sharedInputPlan = buildSharedInputPlan(inputBindings, slotB, options.inputShapes || {});
    const sharedFeeds = {};
    for (const input of sharedInputPlan) {
        sharedFeeds[input.sourceName] = generateInputData(
            { name: input.sourceName, dtype: input.dtype, shape: input.shape },
            inputMode,
            options.importedInput || null,
            [input.sourceName, input.targetName]
        );
    }

    if (typeof options.onStage === 'function') {
        options.onStage('A');
    }
    const resultA = await providerA.runCompareArtifact({
        artifactId: slotA.artifactId,
        sharedFeeds,
        inputBindings,
        side: 'A'
    });
    if (typeof options.onStage === 'function') {
        options.onStage('B');
    }
    const resultB = await providerB.runCompareArtifact({
        artifactId: slotB.artifactId,
        sharedFeeds,
        inputBindings,
        side: 'B'
    });

    const outputsA = resultA && Array.isArray(resultA.outputsSummary) ? resultA.outputsSummary : [];
    const outputsB = resultB && Array.isArray(resultB.outputsSummary) ? resultB.outputsSummary : [];
    const rows = buildCompareRows(outputBindings, outputsA, outputsB);
    const rawOutputRef = typeof options.createRunId === 'function' ? options.createRunId() : `compare-${Date.now()}`;
    const createdAt = typeof options.now === 'function' ? options.now() : nowIso();
    return {
        compareState: {
            ...compareState,
            compareRunStatus: { status: 'idle', stage: '', message: '', updatedAt: createdAt },
            compareResult: buildCompareResult({
                slotA,
                slotB,
                rows,
                sharedInputPlan,
                inputBindings,
                outputBindings,
                inputMode,
                rawOutputRef,
                createdAt
            })
        },
        rawOutputs: {
            sideA: outputsA,
            sideB: outputsB
        }
    };
}

module.exports = {
    exportCompareOutputAsNpy,
    exportCompareResultAsCsv,
    exportCompareResultAsJson,
    generateInputData,
    runCrossProviderCompare
};
