const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const JSZip = require('jszip');
const ort = require('onnxruntime-node');
const { onnx } = require('onnx-proto');

const DATA_TYPE = onnx.TensorProto.DataType;
const DATA_LOCATION = onnx.TensorProto.DataLocation || { DEFAULT: 0, EXTERNAL: 1 };

const DATA_TYPE_NAME = new Map([
    [DATA_TYPE.UNDEFINED, 'undefined'],
    [DATA_TYPE.FLOAT, 'float32'],
    [DATA_TYPE.UINT8, 'uint8'],
    [DATA_TYPE.INT8, 'int8'],
    [DATA_TYPE.UINT16, 'uint16'],
    [DATA_TYPE.INT16, 'int16'],
    [DATA_TYPE.INT32, 'int32'],
    [DATA_TYPE.INT64, 'int64'],
    [DATA_TYPE.STRING, 'string'],
    [DATA_TYPE.BOOL, 'bool'],
    [DATA_TYPE.FLOAT16, 'float16'],
    [DATA_TYPE.DOUBLE, 'float64'],
    [DATA_TYPE.UINT32, 'uint32'],
    [DATA_TYPE.UINT64, 'uint64'],
    [DATA_TYPE.BFLOAT16, 'bfloat16']
]);

const NUMERIC_TYPED_ARRAY = new Map([
    ['float32', Float32Array],
    ['float64', Float64Array],
    ['uint8', Uint8Array],
    ['int8', Int8Array],
    ['uint16', Uint16Array],
    ['int16', Int16Array],
    ['int32', Int32Array],
    ['uint32', Uint32Array]
]);

const ELEMENT_SIZE = new Map([
    ['float32', 4],
    ['float64', 8],
    ['uint8', 1],
    ['int8', 1],
    ['uint16', 2],
    ['int16', 2],
    ['int32', 4],
    ['uint32', 4],
    ['int64', 8],
    ['uint64', 8],
    ['bool', 1],
    ['float16', 2],
    ['bfloat16', 2]
]);

const CATEGORY_FALLBACK = new Map([
    ['Conv', 'Layer'],
    ['ConvTranspose', 'Layer'],
    ['Gemm', 'Layer'],
    ['MatMul', 'Layer'],
    ['Relu', 'Activation'],
    ['Sigmoid', 'Activation'],
    ['Tanh', 'Activation'],
    ['Softmax', 'Activation'],
    ['LeakyRelu', 'Activation'],
    ['BatchNormalization', 'Normalization'],
    ['LayerNormalization', 'Normalization'],
    ['InstanceNormalization', 'Normalization'],
    ['MaxPool', 'Pool'],
    ['AveragePool', 'Pool'],
    ['GlobalAveragePool', 'Pool'],
    ['GlobalMaxPool', 'Pool'],
    ['Reshape', 'Shape'],
    ['Transpose', 'Transform'],
    ['Concat', 'Tensor'],
    ['Slice', 'Tensor'],
    ['Gather', 'Tensor'],
    ['Unsqueeze', 'Shape'],
    ['Squeeze', 'Shape'],
    ['Flatten', 'Shape'],
    ['Cast', 'Transform'],
    ['QuantizeLinear', 'Quantization'],
    ['DequantizeLinear', 'Quantization'],
    ['Attention', 'Attention'],
    ['Constant', 'Constant'],
    ['Identity', 'Control'],
    ['If', 'Control'],
    ['Loop', 'Control'],
    ['Add', 'Tensor'],
    ['Sub', 'Tensor'],
    ['Mul', 'Tensor'],
    ['Div', 'Tensor']
]);


function nowIso() {
    return new Date().toISOString();
}

function createId(prefix) {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function stableHash(value) {
    return crypto.createHash('sha1').update(JSON.stringify(value)).digest('hex').slice(0, 12);
}

function isOnnxFileName(fileName) {
    return typeof fileName === 'string' && /\.(onnx|ort|ort\.onnx)$/i.test(fileName);
}

function cloneProto(messageType, value) {
    return messageType.decode(messageType.encode(value).finish());
}

function asPosixName(value) {
    return String(value || '').split('\\').join('/');
}

function ensureArray(value) {
    return Array.isArray(value) ? value : [];
}

function parseTensorShape(typeProto) {
    if (!typeProto || !typeProto.tensorType || !typeProto.tensorType.shape) {
        return null;
    }
    const dims = [];
    for (const dim of ensureArray(typeProto.tensorType.shape.dim)) {
        if (dim.dimValue !== null && dim.dimValue !== undefined) {
            const raw = typeof dim.dimValue === 'object' && typeof dim.dimValue.toNumber === 'function'
                ? dim.dimValue.toNumber()
                : dim.dimValue;
            dims.push(Number(raw));
        } else if (dim.dimParam) {
            dims.push(null);
        } else {
            dims.push(null);
        }
    }
    return dims;
}

function parseTypeProto(typeProto) {
    if (!typeProto || !typeProto.tensorType) {
        return {
            kind: 'unknown',
            dataType: 'unknown',
            shape: null,
            rank: null,
            dynamicDims: [],
            optional: false
        };
    }
    const dataType = DATA_TYPE_NAME.get(typeProto.tensorType.elemType) || `dtype_${typeProto.tensorType.elemType}`;
    const shape = parseTensorShape(typeProto);
    const rank = Array.isArray(shape) ? shape.length : null;
    const dynamicDims = Array.isArray(shape)
        ? shape.map((dimension, index) => dimension === null ? index : -1).filter((index) => index >= 0)
        : [];
    return {
        kind: 'tensor',
        dataType,
        shape,
        rank,
        dynamicDims,
        optional: false
    };
}

function inferTensorTypeFromInitializer(tensor) {
    const dataType = DATA_TYPE_NAME.get(tensor.dataType) || `dtype_${tensor.dataType}`;
    const dims = ensureArray(tensor.dims).map((value) => Number(value));
    return {
        kind: 'tensor',
        dataType,
        shape: dims,
        rank: dims.length,
        dynamicDims: [],
        optional: false
    };
}

function normalizeDimValue(value) {
    if (value === null || value === undefined || value === '?') {
        return null;
    }
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

function toTypeFromOrtMetadata(entry) {
    if (!entry || !entry.isTensor) {
        return {
            kind: 'unknown',
            dataType: 'unknown',
            shape: null,
            rank: null,
            dynamicDims: [],
            optional: false
        };
    }
    const shape = Array.isArray(entry.shape) ? entry.shape.map((dimension) => normalizeDimValue(dimension)) : null;
    return {
        kind: 'tensor',
        dataType: entry.type || 'unknown',
        shape,
        rank: Array.isArray(shape) ? shape.length : null,
        dynamicDims: Array.isArray(shape) ? shape.map((dimension, index) => dimension === null ? index : -1).filter((index) => index >= 0) : [],
        optional: false
    };
}

function iterateOrtMetadataEntries(metadata) {
    if (Array.isArray(metadata)) {
        return metadata.filter((entry) => !!entry);
    }
    if (!metadata || typeof metadata !== 'object') {
        return [];
    }
    return Object.entries(metadata)
        .map(([name, entry]) => {
            if (!entry || typeof entry !== 'object') {
                return null;
            }
            return entry.name ? entry : { ...entry, name };
        })
        .filter((entry) => !!entry);
}

function shouldInferTensorType(type) {
    return !type
        || type.kind !== 'tensor'
        || type.dataType === 'unknown'
        || !Array.isArray(type.shape)
        || type.rank === null
        || type.rank === undefined;
}

function sameStaticShape(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
        return false;
    }
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== null && b[i] !== null && a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}

function summarizeAttribute(attribute) {
    if (!attribute) {
        return null;
    }
    const type = attribute.type;
    let value = null;
    if (attribute.s && attribute.s.length > 0) {
        value = Buffer.from(attribute.s).toString('utf8');
    } else if (attribute.f !== null && attribute.f !== undefined) {
        value = attribute.f;
    } else if (attribute.i !== null && attribute.i !== undefined) {
        value = Number(attribute.i);
    } else if (ensureArray(attribute.floats).length > 0) {
        value = ensureArray(attribute.floats).slice(0, 8);
    } else if (ensureArray(attribute.ints).length > 0) {
        value = ensureArray(attribute.ints).slice(0, 8).map((item) => Number(item));
    } else if (ensureArray(attribute.strings).length > 0) {
        value = ensureArray(attribute.strings).slice(0, 8).map((item) => Buffer.from(item).toString('utf8'));
    } else if (attribute.t) {
        value = '[Tensor]';
    } else if (attribute.g) {
        value = '[Graph]';
    }
    return {
        name: attribute.name || '',
        type: `attr_${type}`,
        value
    };
}

function createThumbnail(summary) {
    const escape = (value) => String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    const lines = [
        summary.modelName || 'ONNX Crop',
        `${summary.nodeCount || 0} nodes`,
        `${summary.inputCount || 0} in / ${summary.outputCount || 0} out`
    ];
    const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="360" height="200" viewBox="0 0 360 200">
  <rect width="360" height="200" rx="16" fill="#1f2430"/>
  <rect x="16" y="16" width="328" height="168" rx="12" fill="#2d3445" stroke="#586074"/>
  <text x="28" y="52" fill="#e6edf3" font-size="20" font-family="Segoe UI, Arial">${escape(lines[0])}</text>
  <text x="28" y="98" fill="#9fb0c3" font-size="16" font-family="Segoe UI, Arial">${escape(lines[1])}</text>
  <text x="28" y="130" fill="#9fb0c3" font-size="16" font-family="Segoe UI, Arial">${escape(lines[2])}</text>
  <text x="28" y="164" fill="#7aa2f7" font-size="14" font-family="Segoe UI, Arial">Confirmed Crop Artifact</text>
</svg>`;
    return `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
}

function tensorLocationInfo(modelDir, tensor) {
    if (tensor.dataLocation !== DATA_LOCATION.EXTERNAL || !ensureArray(tensor.externalData).length) {
        return null;
    }
    const entries = new Map(ensureArray(tensor.externalData).map((entry) => [entry.key, entry.value]));
    if (!entries.has('location')) {
        return null;
    }
    const location = asPosixName(entries.get('location'));
    return {
        filePath: path.resolve(modelDir, location),
        location,
        offset: entries.has('offset') ? Number(entries.get('offset')) : 0,
        length: entries.has('length') ? Number(entries.get('length')) : -1
    };
}

function product(values) {
    if (!Array.isArray(values) || values.length === 0) {
        return 0;
    }
    return values.reduce((result, value) => result * Number(value), 1);
}

function float32ToFloat16(value) {
    const floatView = new Float32Array(1);
    const intView = new Uint32Array(floatView.buffer);
    floatView[0] = value;
    const x = intView[0];
    const sign = (x >> 31) & 0x1;
    const exponent = (x >> 23) & 0xff;
    const mantissa = x & 0x7fffff;
    if (exponent === 255) {
        return (sign << 15) | 0x7c00 | (mantissa ? 0x200 : 0);
    }
    const halfExp = exponent - 127 + 15;
    if (halfExp >= 31) {
        return (sign << 15) | 0x7c00;
    }
    if (halfExp <= 0) {
        if (halfExp < -10) {
            return sign << 15;
        }
        const shifted = (mantissa | 0x800000) >> (1 - halfExp);
        return (sign << 15) | ((shifted + 0x1000) >> 13);
    }
    return (sign << 15) | (halfExp << 10) | ((mantissa + 0x1000) >> 13);
}

function bfloat16FromFloat32(value) {
    const floatView = new Float32Array(1);
    const intView = new Uint32Array(floatView.buffer);
    floatView[0] = value;
    return intView[0] >>> 16;
}

function float16ToFloat32(value) {
    const sign = (value & 0x8000) ? -1 : 1;
    const exponent = (value >> 10) & 0x1f;
    const fraction = value & 0x03ff;
    if (exponent === 0) {
        return fraction === 0 ? sign * 0 : sign * Math.pow(2, -14) * (fraction / 1024);
    }
    if (exponent === 0x1f) {
        return fraction === 0 ? sign * Infinity : NaN;
    }
    return sign * Math.pow(2, exponent - 15) * (1 + (fraction / 1024));
}

function bfloat16ToFloat32(value) {
    const buffer = Buffer.allocUnsafe(4);
    buffer.writeUInt16LE(0, 0);
    buffer.writeUInt16LE(value, 2);
    return buffer.readFloatLE(0);
}

function countTensorElements(shape) {
    if (!Array.isArray(shape)) {
        return 0;
    }
    if (shape.length === 0) {
        return 1;
    }
    let total = 1;
    for (const dimension of shape) {
        const numeric = Number(dimension);
        if (!Number.isFinite(numeric) || numeric < 0) {
            return 0;
        }
        total *= numeric;
    }
    return total;
}

function normalizePreviewScalar(value) {
    if (typeof value === 'bigint') {
        const numeric = Number(value);
        return Number.isSafeInteger(numeric) ? numeric : value.toString();
    }
    if (value && typeof value.toNumber === 'function') {
        try {
            const numeric = value.toNumber();
            return Number.isSafeInteger(numeric) ? numeric : String(numeric);
        } catch {
            return value.toString();
        }
    }
    return value;
}

function summarizeNumericSample(values) {
    const numeric = values.filter((item) => typeof item === 'number' && Number.isFinite(item));
    if (numeric.length === 0) {
        return null;
    }
    const total = numeric.reduce((sum, item) => sum + item, 0);
    return {
        min: Number(Math.min(...numeric).toFixed(6)),
        max: Number(Math.max(...numeric).toFixed(6)),
        mean: Number((total / numeric.length).toFixed(6))
    };
}

function decodeTensorSampleFromBuffer(buffer, dataType, limit) {
    const values = [];
    if (!Buffer.isBuffer(buffer) || buffer.length === 0 || limit <= 0) {
        return values;
    }
    const elementSize = ELEMENT_SIZE.get(dataType) || 0;
    if (!elementSize) {
        return values;
    }
    const count = Math.min(limit, Math.floor(buffer.length / elementSize));
    for (let index = 0; index < count; index++) {
        const offset = index * elementSize;
        switch (dataType) {
            case 'float32':
                values.push(buffer.readFloatLE(offset));
                break;
            case 'float64':
                values.push(buffer.readDoubleLE(offset));
                break;
            case 'uint8':
                values.push(buffer.readUInt8(offset));
                break;
            case 'int8':
                values.push(buffer.readInt8(offset));
                break;
            case 'uint16':
                values.push(buffer.readUInt16LE(offset));
                break;
            case 'int16':
                values.push(buffer.readInt16LE(offset));
                break;
            case 'uint32':
                values.push(buffer.readUInt32LE(offset));
                break;
            case 'int32':
                values.push(buffer.readInt32LE(offset));
                break;
            case 'int64':
                values.push(normalizePreviewScalar(buffer.readBigInt64LE(offset)));
                break;
            case 'uint64':
                values.push(normalizePreviewScalar(buffer.readBigUInt64LE(offset)));
                break;
            case 'bool':
                values.push(buffer.readUInt8(offset) !== 0);
                break;
            case 'float16':
                values.push(float16ToFloat32(buffer.readUInt16LE(offset)));
                break;
            case 'bfloat16':
                values.push(bfloat16ToFloat32(buffer.readUInt16LE(offset)));
                break;
            default:
                return values;
        }
    }
    return values;
}

async function readTensorSampleBuffer(session, tensor, bytesToRead) {
    if (tensor.rawData && tensor.rawData.length > 0) {
        return Buffer.from(tensor.rawData).subarray(0, bytesToRead);
    }
    const location = tensorLocationInfo(session.modelDir, tensor);
    if (!location) {
        return Buffer.alloc(0);
    }
    const requestedLength = location.length && location.length > 0 ? Math.min(location.length, bytesToRead) : bytesToRead;
    const fileHandle = await fs.promises.open(location.filePath, 'r');
    try {
        const buffer = Buffer.alloc(Math.max(0, requestedLength));
        const result = await fileHandle.read(buffer, 0, buffer.length, Math.max(0, location.offset || 0));
        return buffer.subarray(0, result.bytesRead);
    } finally {
        await fileHandle.close();
    }
}

function createEmbeddedTensorPreview(tensor, limit = 64) {
    if (!tensor) {
        return null;
    }
    const source = tensor.values && tensor.indices ? tensor.values : tensor;
    if (!source || source.dataLocation === DATA_LOCATION.EXTERNAL) {
        return null;
    }
    const shape = ensureArray(source.dims).map((value) => Number(value));
    const dataType = DATA_TYPE_NAME.get(source.dataType) || `dtype_${source.dataType}`;
    let sampleValues = [];
    if (ensureArray(source.floatData).length > 0) {
        sampleValues = ensureArray(source.floatData).slice(0, limit).map((value) => Number(value));
    } else if (ensureArray(source.doubleData).length > 0) {
        sampleValues = ensureArray(source.doubleData).slice(0, limit).map((value) => Number(value));
    } else if (ensureArray(source.int32Data).length > 0) {
        sampleValues = ensureArray(source.int32Data).slice(0, limit).map((value) => dataType === 'bool' ? value !== 0 : Number(value));
    } else if (ensureArray(source.int64Data).length > 0) {
        sampleValues = ensureArray(source.int64Data).slice(0, limit).map((value) => normalizePreviewScalar(typeof value === 'object' && typeof value.toBigInt === 'function' ? value.toBigInt() : value));
    } else if (ensureArray(source.uint64Data).length > 0) {
        sampleValues = ensureArray(source.uint64Data).slice(0, limit).map((value) => normalizePreviewScalar(typeof value === 'object' && typeof value.toBigInt === 'function' ? value.toBigInt() : value));
    } else if (ensureArray(source.stringData).length > 0) {
        sampleValues = ensureArray(source.stringData).slice(0, limit).map((value) => decodeUtf8Bytes(value));
    } else if (source.rawData && source.rawData.length > 0) {
        const bytesPerElement = ELEMENT_SIZE.get(dataType) || 4;
        sampleValues = decodeTensorSampleFromBuffer(Buffer.from(source.rawData), dataType, Math.min(limit, Math.floor(source.rawData.length / bytesPerElement)));
    }
    return {
        name: source.name || '',
        dataType,
        shape,
        layout: tensor.values && tensor.indices ? 'sparse' : 'dense',
        location: source.dataLocation === DATA_LOCATION.EXTERNAL ? 'external' : 'inline',
        elementCount: countTensorElements(shape),
        sampleCount: sampleValues.length,
        sampleValues,
        truncated: countTensorElements(shape) > sampleValues.length,
        stats: summarizeNumericSample(sampleValues)
    };
}

function boolArrayToBuffer(values) {
    const buffer = Buffer.alloc(values.length);
    for (let i = 0; i < values.length; i++) {
        buffer[i] = values[i] ? 1 : 0;
    }
    return buffer;
}

function typedArrayToBuffer(value, elementType) {
    if (Buffer.isBuffer(value)) {
        return value;
    }
    if (value instanceof Uint8Array) {
        return Buffer.from(value.buffer, value.byteOffset, value.byteLength);
    }
    if (ArrayBuffer.isView(value)) {
        return Buffer.from(value.buffer, value.byteOffset, value.byteLength);
    }
    if (Array.isArray(value)) {
        switch (elementType) {
            case 'int64': {
                const buffer = Buffer.alloc(value.length * 8);
                value.forEach((item, index) => buffer.writeBigInt64LE(BigInt(item), index * 8));
                return buffer;
            }
            case 'uint64': {
                const buffer = Buffer.alloc(value.length * 8);
                value.forEach((item, index) => buffer.writeBigUInt64LE(BigInt(item), index * 8));
                return buffer;
            }
            case 'bool':
                return boolArrayToBuffer(value);
            case 'float16': {
                const buffer = Buffer.alloc(value.length * 2);
                value.forEach((item, index) => buffer.writeUInt16LE(float32ToFloat16(Number(item)), index * 2));
                return buffer;
            }
            case 'bfloat16': {
                const buffer = Buffer.alloc(value.length * 2);
                value.forEach((item, index) => buffer.writeUInt16LE(bfloat16FromFloat32(Number(item)), index * 2));
                return buffer;
            }
            default: {
                const TypedArray = NUMERIC_TYPED_ARRAY.get(elementType);
                if (!TypedArray) {
                    throw new Error(`Unsupported array conversion for '${elementType}'.`);
                }
                return Buffer.from(new TypedArray(value).buffer);
            }
        }
    }
    throw new Error(`Unsupported tensor buffer value '${typeof value}'.`);
}

async function materializeTensorBuffer(session, tensor) {
    if (tensor.rawData && tensor.rawData.length > 0) {
        return Buffer.from(tensor.rawData);
    }
    const elementType = DATA_TYPE_NAME.get(tensor.dataType) || 'unknown';
    if (ensureArray(tensor.floatData).length > 0) {
        return typedArrayToBuffer(ensureArray(tensor.floatData), 'float32');
    }
    if (ensureArray(tensor.doubleData).length > 0) {
        return typedArrayToBuffer(ensureArray(tensor.doubleData), 'float64');
    }
    if (ensureArray(tensor.int32Data).length > 0) {
        switch (elementType) {
            case 'uint8':
            case 'int8':
            case 'uint16':
            case 'int16':
            case 'int32':
            case 'uint32':
            case 'bool':
            case 'float16':
            case 'bfloat16':
                return typedArrayToBuffer(ensureArray(tensor.int32Data), elementType);
            default:
                return typedArrayToBuffer(ensureArray(tensor.int32Data), 'int32');
        }
    }
    if (ensureArray(tensor.int64Data).length > 0) {
        return typedArrayToBuffer(ensureArray(tensor.int64Data), 'int64');
    }
    if (ensureArray(tensor.uint64Data).length > 0) {
        return typedArrayToBuffer(ensureArray(tensor.uint64Data), 'uint64');
    }
    const location = tensorLocationInfo(session.modelDir, tensor);
    if (location) {
        const fileBuffer = await fs.promises.readFile(location.filePath);
        const start = Math.max(0, location.offset || 0);
        const end = location.length && location.length > 0 ? start + location.length : fileBuffer.length;
        return fileBuffer.subarray(start, end);
    }
    return Buffer.alloc(0);
}

function decodeUtf8Bytes(bytes) {
    return Buffer.from(bytes).toString('utf8');
}

function parseJsonInput(text) {
    const data = JSON.parse(text);
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
        throw new Error('Imported JSON must be an object keyed by input name.');
    }
    return data;
}

function parseNpy(buffer) {
    const magic = buffer.subarray(0, 6).toString('binary');
    if (magic !== '\x93NUMPY') {
        throw new Error('Invalid .npy file header.');
    }
    const major = buffer[6];
    const headerLength = major <= 1 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8);
    const headerStart = major <= 1 ? 10 : 12;
    const headerText = buffer.subarray(headerStart, headerStart + headerLength).toString('latin1').trim();
    const descrMatch = headerText.match(/'descr'\s*:\s*'([^']+)'/);
    const shapeMatch = headerText.match(/'shape'\s*:\s*\(([^\)]*)\)/);
    const fortranMatch = headerText.match(/'fortran_order'\s*:\s*(True|False)/);
    if (!descrMatch || !shapeMatch || !fortranMatch) {
        throw new Error('Unsupported .npy header.');
    }
    if (fortranMatch[1] !== 'False') {
        throw new Error('Fortran-order .npy is not supported.');
    }
    const descr = descrMatch[1];
    const rawShape = shapeMatch[1].split(',').map((item) => item.trim()).filter((item) => item.length > 0);
    const shape = rawShape.map((item) => Number(item));
    const dataOffset = headerStart + headerLength;
    const payload = buffer.subarray(dataOffset);
    const map = {
        '<f4': { type: 'float32', view: Float32Array },
        '|u1': { type: 'uint8', view: Uint8Array },
        '|i1': { type: 'int8', view: Int8Array },
        '<u2': { type: 'uint16', view: Uint16Array },
        '<i2': { type: 'int16', view: Int16Array },
        '<i4': { type: 'int32', view: Int32Array },
        '<u4': { type: 'uint32', view: Uint32Array },
        '<f8': { type: 'float64', view: Float64Array },
        '|b1': { type: 'bool', view: Uint8Array }
    };
    if (!map[descr]) {
        throw new Error(`Unsupported .npy dtype '${descr}'.`);
    }
    const view = map[descr].view;
    const data = new view(payload.buffer, payload.byteOffset, Math.floor(payload.byteLength / view.BYTES_PER_ELEMENT));
    return {
        type: map[descr].type,
        shape,
        data: Array.from(data)
    };
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

async function parseNpzInput(buffer) {
    const archive = await JSZip.loadAsync(buffer);
    const result = {};
    const entries = Object.keys(archive.files).filter((name) => name.endsWith('.npy'));
    for (const name of entries) {
        const item = archive.files[name];
        const content = await item.async('nodebuffer');
        const parsed = parseNpy(content);
        result[path.basename(name, '.npy')] = { dtype: parsed.type, shape: parsed.shape, data: parsed.data };
    }
    return result;
}

function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let index = 0; index < a.length; index++) {
        dot += a[index] * b[index];
        normA += a[index] * a[index];
        normB += b[index] * b[index];
    }
    if (normA === 0 || normB === 0) {
        return null;
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function pearsonCorrelation(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
        return null;
    }
    let meanA = 0;
    let meanB = 0;
    for (let index = 0; index < a.length; index++) {
        meanA += Number(a[index]);
        meanB += Number(b[index]);
    }
    meanA /= a.length;
    meanB /= b.length;

    let numerator = 0;
    let varianceA = 0;
    let varianceB = 0;
    for (let index = 0; index < a.length; index++) {
        const centeredA = Number(a[index]) - meanA;
        const centeredB = Number(b[index]) - meanB;
        numerator += centeredA * centeredB;
        varianceA += centeredA * centeredA;
        varianceB += centeredB * centeredB;
    }
    if (varianceA === 0 || varianceB === 0) {
        return null;
    }
    return numerator / Math.sqrt(varianceA * varianceB);
}

function computeNumericDiff(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
        return null;
    }
    let maxAbs = 0;
    let meanAbs = 0;
    let mse = 0;
    let maxRelative = 0;
    for (let index = 0; index < a.length; index++) {
        const delta = Math.abs(Number(a[index]) - Number(b[index]));
        meanAbs += delta;
        mse += delta * delta;
        maxAbs = Math.max(maxAbs, delta);
        const denominator = Math.max(Math.abs(Number(a[index])), 1e-12);
        maxRelative = Math.max(maxRelative, delta / denominator);
    }
    meanAbs /= a.length;
    mse /= a.length;
    return {
        maxAbs,
        meanAbs,
        rmse: Math.sqrt(mse),
        maxRelativeDiff: maxRelative,
        cosineSimilarity: cosineSimilarity(a.map(Number), b.map(Number)),
        pearsonCorrelation: pearsonCorrelation(a.map(Number), b.map(Number))
    };
}

function summarizeOutputTensor(name, tensor) {
    const dims = Array.isArray(tensor.dims) ? tensor.dims.slice() : [];
    const dataType = tensor.type;
    const data = tensor.data;
    let values = null;
    if (ArrayBuffer.isView(data)) {
        values = Array.from(data);
    } else if (Array.isArray(data)) {
        values = data.slice();
    }
    const elementCount = countTensorElements(dims);
    const previewValues = Array.isArray(values) ? values.slice(0, 24).map((value) => normalizePreviewScalar(value)) : [];
    let summary = null;
    if (values && values.length > 0) {
        const numeric = values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
        if (numeric.length > 0) {
            let min = Number.POSITIVE_INFINITY;
            let max = Number.NEGATIVE_INFINITY;
            let mean = 0;
            let absMean = 0;
            let l2NormSquared = 0;
            let nonZeroCount = 0;
            for (const value of numeric) {
                min = Math.min(min, value);
                max = Math.max(max, value);
                mean += value;
                absMean += Math.abs(value);
                l2NormSquared += value * value;
                if (value !== 0) {
                    nonZeroCount += 1;
                }
            }
            mean /= numeric.length;
            absMean /= numeric.length;
            summary = {
                min,
                max,
                mean,
                absMean,
                l2Norm: Math.sqrt(l2NormSquared),
                nonZeroCount
            };
        }
    }
    return {
        name,
        dtype: dataType,
        shape: dims,
        elementCount,
        values,
        preview: {
            elementCount,
            sampleCount: previewValues.length,
            sampleValues: previewValues,
            truncated: elementCount > previewValues.length
        },
        summary
    };
}

class ONNXWorkbench {
    constructor(context, logger) {
        this.context = context;
        this.logger = logger;
        this.metadata = this._loadOnnxMetadata();
        this.sessions = new Map();
        this.artifacts = new Map();
        this.inferenceResults = new Map();
        this.sessionCache = new Map();
        this.tensorPreviewCache = new Map();
        this.compareState = {
            slotA: null,
            slotB: null,
            inputBindings: [],
            outputBindings: [],
            compareRunStatus: { status: 'idle', stage: '', message: '', updatedAt: nowIso() },
            compareResult: null,
            importedInput: null
        };
        this.listeners = new Set();
        this.tempRoot = path.join(os.tmpdir(), 'netron-vscode-workbench');
        fs.mkdirSync(this.tempRoot, { recursive: true });
    }

    onChange(listener) {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    _emitChange() {
        for (const listener of this.listeners) {
            try {
                listener(this.getCompareState());
            } catch (error) {
                this._log('warn', 'compare state listener failed', { message: error.message });
            }
        }
    }

    _log(level, message, detail) {
        if (this.logger && typeof this.logger === 'function') {
            this.logger(level, message, detail);
        }
    }

    isOnnxUri(uri) {
        return uri && isOnnxFileName(uri.fsPath || uri.path || uri.toString());
    }

    async loadModel(uri, options = {}) {
        const filePath = uri.fsPath || uri.path;
        const notifyStage = typeof options.onStage === 'function' ? options.onStage : () => {};
        notifyStage('读取文件', { filePath });
        const bytes = await fs.promises.readFile(filePath);
        notifyStage('解析 ONNX', { sizeBytes: bytes.byteLength });
        const model = onnx.ModelProto.decode(bytes);
        const session = this._createSession(uri, bytes, model);
        notifyStage('构建图快照', { sessionId: session.id });
        session.snapshot = this._createModelSnapshot(session);
        this.sessions.set(session.id, session);
        return session;
    }

    _createSession(uri, bytes, model) {
        const sessionId = createId('session');
        const filePath = uri.fsPath || uri.path;
        const modelDir = path.dirname(filePath);
        const graphInfo = this._analyzeGraph(model, modelDir);
        return {
            id: sessionId,
            uri,
            filePath,
            modelDir,
            sizeBytes: bytes.byteLength,
            createdAt: nowIso(),
            model,
            modelBytes: bytes,
            graphInfo,
            format: 'ONNX',
            producer: model.producerName || '',
            snapshot: null,
            latestArtifactId: null
        };
    }

    _analyzeGraph(model, modelDir) {
        const graph = model.graph;
        const values = new Map();
        const initializers = new Map();
        const valueInfo = new Map();
        const graphInputNames = [];
        const graphOutputNames = [];
        const graphMetadata = ensureArray(graph.metadataProps).map((entry) => ({ name: entry.key, value: entry.value }));

        const ensureValue = (name) => {
            const key = String(name || '');
            if (!values.has(key)) {
                values.set(key, {
                    name: key,
                    type: null,
                    initializer: null,
                    producer: null,
                    consumers: new Set(),
                    description: '',
                    metadata: []
                });
            }
            return values.get(key);
        };

        for (const value of ensureArray(graph.valueInfo)) {
            valueInfo.set(value.name, value);
            const item = ensureValue(value.name);
            item.type = parseTypeProto(value.type);
            item.description = value.docString || '';
        }

        for (const tensor of ensureArray(graph.initializer)) {
            const item = ensureValue(tensor.name);
            item.initializer = tensor;
            item.type = item.type || inferTensorTypeFromInitializer(tensor);
            initializers.set(tensor.name, tensor);
        }

        for (const tensor of ensureArray(graph.sparseInitializer)) {
            const name = tensor.values ? tensor.values.name : '';
            const item = ensureValue(name);
            item.initializer = tensor;
            item.type = item.type || inferTensorTypeFromInitializer(tensor.values || tensor);
            initializers.set(name, tensor);
        }

        for (const input of ensureArray(graph.input)) {
            const item = ensureValue(input.name);
            item.type = item.type || parseTypeProto(input.type);
            item.description = input.docString || '';
            item.metadata = ensureArray(input.metadataProps).map((entry) => ({ name: entry.key, value: entry.value }));
            graphInputNames.push(input.name);
        }

        for (const output of ensureArray(graph.output)) {
            const item = ensureValue(output.name);
            item.type = item.type || parseTypeProto(output.type);
            item.description = output.docString || '';
            item.metadata = ensureArray(output.metadataProps).map((entry) => ({ name: entry.key, value: entry.value }));
            graphOutputNames.push(output.name);
        }

        const nodes = ensureArray(graph.node).map((node, index) => {
            const id = `n${index}`;
            const inputNames = ensureArray(node.input).map((name) => String(name || ''));
            const outputNames = ensureArray(node.output).map((name) => String(name || ''));
            for (const name of inputNames) {
                if (!name) {
                    continue;
                }
                ensureValue(name).consumers.add(id);
            }
            for (const name of outputNames) {
                if (!name) {
                    continue;
                }
                const value = ensureValue(name);
                if (!value.producer) {
                    value.producer = id;
                }
            }
            return {
                id,
                index,
                raw: node,
                name: node.name || '',
                domain: node.domain || 'ai.onnx',
                opType: node.opType || 'Unknown',
                inputs: inputNames,
                outputs: outputNames,
                attributes: ensureArray(node.attribute).map(summarizeAttribute).filter(Boolean)
            };
        });

        const orderedInputs = graphInputNames.filter((name) => {
            const value = values.get(name);
            return value && !value.initializer;
        });

        return {
            name: graph.name || 'graph',
            description: graph.docString || '',
            metadata: graphMetadata,
            values,
            valueInfo,
            initializers,
            nodes,
            graphInputNames: orderedInputs,
            graphOutputNames,
            modelDir,
            sourceGraph: graph
        };
    }


    _loadOnnxMetadata() {
        try {
            const candidates = [];
            if (this.context && this.context.extensionPath) {
                candidates.push(path.join(this.context.extensionPath, 'netron', 'source', 'onnx-metadata.json'));
            }
            candidates.push(path.join(__dirname, '..', 'netron', 'source', 'onnx-metadata.json'));
            for (const filePath of candidates) {
                if (!filePath || !fs.existsSync(filePath)) {
                    continue;
                }
                const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                const map = new Map();
                for (const entry of data) {
                    const key = `${entry.module || 'ai.onnx'}:${entry.name}`;
                    if (!map.has(key)) {
                        map.set(key, []);
                    }
                    map.get(key).push(entry);
                }
                return map;
            }
        } catch (error) {
            this._log('warn', 'failed to load onnx metadata', { message: error.message });
        }
        return new Map();
    }

    _resolveNodeTypeInfo(domain, opType) {
        const normalizedDomain = domain || 'ai.onnx';
        const key = `${normalizedDomain}:${opType}`;
        const candidates = this.metadata.get(key) || this.metadata.get(`ai.onnx:${opType}`) || [];
        const metadata = candidates.length > 0 ? candidates[candidates.length - 1] : null;
        const category = (metadata && metadata.category) || CATEGORY_FALLBACK.get(opType) || undefined;
        return {
            name: opType,
            module: normalizedDomain,
            identifier: normalizedDomain && normalizedDomain !== 'ai.onnx' ? `${normalizedDomain}.${opType}` : opType,
            category,
            description: metadata && metadata.description ? metadata.description : ''
        };
    }
    _createValueSnapshot(value) {
        return {
            name: value.name,
            type: value.type || {
                kind: 'tensor',
                dataType: 'unknown',
                shape: null,
                rank: null,
                dynamicDims: [],
                optional: false
            },
            initializer: value.initializer
                ? {
                    name: value.name,
                    category: 'Initializer',
                    type: value.type || inferTensorTypeFromInitializer(value.initializer.values || value.initializer),
                    location: value.initializer.dataLocation === DATA_LOCATION.EXTERNAL ? 'external' : 'inline',
                    preview: createEmbeddedTensorPreview(value.initializer)
                }
                : null,
            description: value.description || ''
        };
    }

    _createGraphSnapshot(graphInfo, options = {}) {
        const selectedNodeIds = options.selectedNodeIds ? new Set(options.selectedNodeIds) : new Set(graphInfo.nodes.map((node) => node.id));
        const inputKeys = options.inputKeys ? Array.from(options.inputKeys) : Array.from(graphInfo.graphInputNames);
        const outputKeys = options.outputKeys ? Array.from(options.outputKeys) : Array.from(graphInfo.graphOutputNames);

        const values = {};
        const touchValue = (name) => {
            const entry = graphInfo.values.get(name);
            if (entry && !values[name]) {
                values[name] = this._createValueSnapshot(entry);
            }
        };

        const nodes = graphInfo.nodes
            .filter((node) => selectedNodeIds.has(node.id))
            .map((node) => {
                for (const name of node.inputs) {
                    if (name) {
                        touchValue(name);
                    }
                }
                for (const name of node.outputs) {
                    if (name) {
                        touchValue(name);
                    }
                }
                return {
                    id: node.id,
                    name: node.name,
                    type: this._resolveNodeTypeInfo(node.domain, node.opType),
                    inputs: node.inputs.map((name, index) => ({ name: `${index}`, values: name ? [name] : [] })),
                    outputs: node.outputs.map((name, index) => ({ name: `${index}`, values: name ? [name] : [] })),
                    attributes: node.attributes || []
                };
            });

        for (const name of inputKeys) {
            touchValue(name);
        }
        for (const name of outputKeys) {
            touchValue(name);
        }

        return {
            name: options.name || graphInfo.name,
            description: graphInfo.description,
            groups: false,
            values,
            inputs: inputKeys.map((name) => ({ name, values: name ? [name] : [] })),
            outputs: outputKeys.map((name) => ({ name, values: name ? [name] : [] })),
            nodes
        };
    }

    _createModelSnapshot(session) {
        return {
            sessionId: session.id,
            fileName: path.basename(session.filePath),
            filePath: session.filePath,
            format: session.format,
            producer: session.producer,
            graph: this._createGraphSnapshot(session.graphInfo),
            metadata: ensureArray(session.model.metadataProps).map((entry) => ({ name: entry.key, value: entry.value })),
            sourceGraphId: 'graph:main'
        };
    }

    async _inferTensorTypesFromOriginalModel(session, tensorNames) {
        const pendingNames = Array.from(new Set(ensureArray(tensorNames).map((name) => String(name || '')).filter((name) => name.length > 0)));
        if (pendingNames.length === 0) {
            return new Map();
        }
        const pendingNameSet = new Set(pendingNames);
        const model = cloneProto(onnx.ModelProto, session.model);
        const existingOutputs = new Set(ensureArray(model.graph.output).map((value) => value.name));
        for (const name of pendingNames) {
            if (existingOutputs.has(name)) {
                continue;
            }
            model.graph.output.push(onnx.ValueInfoProto.create({ name }));
            existingOutputs.add(name);
        }
        const bytes = Buffer.from(onnx.ModelProto.encode(model).finish());
        const ortSession = await ort.InferenceSession.create(bytes);
        const inferred = new Map();
        for (const entry of iterateOrtMetadataEntries(ortSession.inputMetadata)) {
            if (entry && entry.name && pendingNameSet.has(entry.name)) {
                inferred.set(entry.name, toTypeFromOrtMetadata(entry));
            }
        }
        for (const entry of iterateOrtMetadataEntries(ortSession.outputMetadata)) {
            if (entry && entry.name && pendingNameSet.has(entry.name)) {
                inferred.set(entry.name, toTypeFromOrtMetadata(entry));
            }
        }
        return inferred;
    }

    async _inferArtifactEndpointTypes(artifactLike) {
        const session = this.getSession(artifactLike.modelSessionId);
        if (!session) {
            return { inputTypes: new Map(), outputTypes: new Map() };
        }
        const inputTypes = new Map();
        const outputTypes = new Map();
        const unresolvedNames = new Set();
        const collectTypes = (names, target) => {
            for (const name of ensureArray(names)) {
                const value = session.graphInfo.values.get(name);
                if (value && !shouldInferTensorType(value.type)) {
                    target.set(name, value.type);
                } else {
                    unresolvedNames.add(name);
                }
            }
        };
        collectTypes(artifactLike.inputKeys, inputTypes);
        collectTypes(artifactLike.outputKeys, outputTypes);
        try {
            if (unresolvedNames.size > 0) {
                const inferred = await this._inferTensorTypesFromOriginalModel(session, Array.from(unresolvedNames));
                for (const name of ensureArray(artifactLike.inputKeys)) {
                    if (!inputTypes.has(name) && inferred.has(name)) {
                        inputTypes.set(name, inferred.get(name));
                    }
                }
                for (const name of ensureArray(artifactLike.outputKeys)) {
                    if (!outputTypes.has(name) && inferred.has(name)) {
                        outputTypes.set(name, inferred.get(name));
                    }
                }
            }
            return { inputTypes, outputTypes };
        } catch (error) {
            this._log('warn', 'artifact endpoint type inference failed', { message: error.message });
            return { inputTypes, outputTypes };
        }
    }

    _applyTypeToGraphValue(graphInfo, name, type) {
        const value = graphInfo && graphInfo.values ? graphInfo.values.get(name) : null;
        if (!value || !type || type.kind !== 'tensor') {
            return;
        }
        value.type = {
            kind: type.kind,
            dataType: type.dataType,
            shape: Array.isArray(type.shape) ? type.shape.slice() : null,
            rank: type.rank,
            dynamicDims: Array.isArray(type.dynamicDims) ? type.dynamicDims.slice() : [],
            optional: !!type.optional
        };
    }

    _applyTypeToSnapshotValue(snapshot, name, type) {
        if (!snapshot || !snapshot.values || !snapshot.values[name] || !type || type.kind !== 'tensor') {
            return;
        }
        snapshot.values[name].type = {
            kind: type.kind,
            dataType: type.dataType,
            shape: Array.isArray(type.shape) ? type.shape.slice() : null,
            rank: type.rank,
            dynamicDims: Array.isArray(type.dynamicDims) ? type.dynamicDims.slice() : [],
            optional: !!type.optional
        };
        if (snapshot.values[name].initializer) {
            snapshot.values[name].initializer.type = snapshot.values[name].type;
        }
    }

    _applyTypesToIOSignature(ioSignature, typeMap) {
        const patch = (items) => items.map((item) => {
            const type = typeMap.get(item.name);
            if (!type || type.kind !== 'tensor') {
                return item;
            }
            return {
                ...item,
                dtype: type.dataType,
                rank: type.rank,
                shape: Array.isArray(type.shape) ? type.shape.slice() : null,
                dynamicDims: Array.isArray(type.dynamicDims) ? type.dynamicDims.slice() : []
            };
        });
        return {
            inputs: patch(ioSignature.inputs || []),
            outputs: patch(ioSignature.outputs || [])
        };
    }

    getSession(sessionId) {
        return this.sessions.get(sessionId) || null;
    }

    getArtifact(artifactId) {
        return this.artifacts.get(artifactId) || null;
    }

    async createCropArtifact({ sessionId, startKeys, endKeys }) {
        const session = this.getSession(sessionId);
        if (!session) {
            throw new Error('Model session not found.');
        }
        const cropResult = this._cropGraph(session.graphInfo, new Set(startKeys), new Set(endKeys));
        const effectiveNodeIds = this._pruneConstantInitializerNodes(session.graphInfo, cropResult.selectedNodeIds);
        const typeHints = await this._inferArtifactEndpointTypes({
            modelSessionId: sessionId,
            selectedNodeIds: Array.from(effectiveNodeIds),
            inputKeys: Array.from(cropResult.inputKeys),
            outputKeys: Array.from(cropResult.outputKeys)
        });
        for (const [name, type] of new Map([...typeHints.inputTypes, ...typeHints.outputTypes])) {
            this._applyTypeToGraphValue(session.graphInfo, name, type);
            if (session.snapshot && session.snapshot.graph) {
                this._applyTypeToSnapshotValue(session.snapshot.graph, name, type);
            }
        }
        const cropGraphSnapshot = this._createGraphSnapshot(session.graphInfo, {
            name: `${session.graphInfo.name}::crop`,
            selectedNodeIds: effectiveNodeIds,
            inputKeys: cropResult.inputKeys,
            outputKeys: cropResult.outputKeys
        });
        for (const [name, type] of typeHints.inputTypes) {
            this._applyTypeToSnapshotValue(cropGraphSnapshot, name, type);
        }
        for (const [name, type] of typeHints.outputTypes) {
            this._applyTypeToSnapshotValue(cropGraphSnapshot, name, type);
        }
        const ioSignature = this._applyTypesToIOSignature(
            this._createIOSignature(session.graphInfo, cropResult.inputKeys, cropResult.outputKeys),
            new Map([...typeHints.inputTypes, ...typeHints.outputTypes])
        );
        const artifactId = createId('artifact');
        const artifact = {
            id: artifactId,
            modelSessionId: sessionId,
            sourceGraphId: 'graph:main',
            selection: {
                startKeys: Array.from(startKeys),
                endKeys: Array.from(endKeys)
            },
            selectedNodeIds: Array.from(effectiveNodeIds),
            inputKeys: Array.from(cropResult.inputKeys),
            outputKeys: Array.from(cropResult.outputKeys),
            ioSignature,
            thumbnail: createThumbnail({
                modelName: path.basename(session.filePath),
                nodeCount: effectiveNodeIds.size,
                inputCount: cropResult.inputKeys.size,
                outputCount: cropResult.outputKeys.size
            }),
            createdAt: nowIso(),
            cropGraphSnapshot,
            exportCache: null,
            inferenceCacheKeys: [],
            stale: false,
            summary: {
                modelName: path.basename(session.filePath),
                graphName: session.graphInfo.name,
                nodeCount: effectiveNodeIds.size,
                inputCount: cropResult.inputKeys.size,
                outputCount: cropResult.outputKeys.size
            }
        };
        if (session.latestArtifactId && this.artifacts.has(session.latestArtifactId)) {
            this.artifacts.get(session.latestArtifactId).stale = true;
        }
        session.latestArtifactId = artifactId;
        this.artifacts.set(artifactId, artifact);
        this._emitChange();
        return artifact;
    }

    _createIOSignature(graphInfo, inputKeys, outputKeys) {
        const build = (keys) => Array.from(keys).map((name) => {
            const value = graphInfo.values.get(name);
            const type = value && value.type ? value.type : {
                kind: 'tensor',
                dataType: 'unknown',
                shape: null,
                rank: null,
                dynamicDims: [],
                optional: false
            };
            return {
                name,
                dtype: type.dataType,
                rank: type.rank,
                shape: Array.isArray(type.shape) ? type.shape.slice() : null,
                dynamicDims: Array.isArray(type.dynamicDims) ? type.dynamicDims.slice() : [],
                optional: !!type.optional
            };
        });
        return {
            inputs: build(inputKeys),
            outputs: build(outputKeys)
        };
    }

    _cropGraph(graphInfo, startKeys, endKeys) {
        const nodeMap = new Map(graphInfo.nodes.map((node) => [node.id, node]));
        const startNodes = new Set();
        for (const key of startKeys) {
            const tensor = graphInfo.values.get(key);
            if (!tensor) {
                continue;
            }
            for (const consumer of tensor.consumers) {
                startNodes.add(consumer);
            }
        }
        const endNodes = new Set();
        for (const key of endKeys) {
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
            const node = nodeMap.get(nodeId);
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
            const node = nodeMap.get(nodeId);
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
            const node = nodeMap.get(nodeId);
            if (!node) {
                continue;
            }
            for (const key of node.inputs) {
                const tensor = graphInfo.values.get(key);
                if (!tensor) {
                    continue;
                }
                const isInitializer = !!tensor.initializer;
                if ((!tensor.producer || !selectedNodeIds.has(tensor.producer)) && !isInitializer) {
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
                if (!hasInside || hasOutside || endKeys.has(key) || graphInfo.graphOutputNames.includes(key)) {
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
        return {
            selectedNodeIds,
            inputKeys,
            outputKeys
        };
    }

    _pruneConstantInitializerNodes(graphInfo, selectedNodeIds) {
        const filtered = new Set();
        for (const node of graphInfo.nodes) {
            if (!selectedNodeIds.has(node.id)) {
                continue;
            }
            const type = String(node.opType || '').toLowerCase();
            if (type !== 'constant') {
                filtered.add(node.id);
                continue;
            }
            const hasRealInputs = node.inputs.some((key) => !!key);
            if (hasRealInputs) {
                filtered.add(node.id);
                continue;
            }
            const allOutputsInitializer = node.outputs.every((key) => {
                const tensor = graphInfo.values.get(key);
                return tensor && tensor.initializer;
            });
            if (!allOutputsInitializer) {
                filtered.add(node.id);
            }
        }
        return filtered.size > 0 ? filtered : selectedNodeIds;
    }

    _cloneValueInfo(graphInfo, name) {
        if (graphInfo.valueInfo.has(name)) {
            return cloneProto(onnx.ValueInfoProto, graphInfo.valueInfo.get(name));
        }
        const value = graphInfo.values.get(name);
        const info = onnx.ValueInfoProto.create({ name });
        if (value && value.type && value.type.kind === 'tensor') {
            info.type = onnx.TypeProto.create({
                tensorType: onnx.TypeProto.Tensor.create({
                    elemType: this._dataTypeEnumFromName(value.type.dataType),
                    shape: onnx.TensorShapeProto.create({
                        dim: ensureArray(value.type.shape).map((dimension) => dimension === null
                            ? onnx.TensorShapeProto.Dimension.create({ dimParam: '?' })
                            : onnx.TensorShapeProto.Dimension.create({ dimValue: dimension }))
                    })
                })
            });
        }
        return info;
    }

    _dataTypeEnumFromName(name) {
        for (const [key, value] of DATA_TYPE_NAME.entries()) {
            if (value === name) {
                return key;
            }
        }
        return DATA_TYPE.FLOAT;
    }

    _createArtifactModel(artifact, options = {}) {
        const session = this.getSession(artifact.modelSessionId);
        if (!session) {
            throw new Error('Artifact session not found.');
        }
        const graphInfo = session.graphInfo;
        const original = session.model;
        const graph = onnx.GraphProto.create({
            name: `${graphInfo.name}::crop`,
            docString: graphInfo.description || '',
            node: [],
            initializer: [],
            sparseInitializer: [],
            input: [],
            output: [],
            valueInfo: [],
            quantizationAnnotation: [],
            metadataProps: cloneProto(onnx.GraphProto, graphInfo.sourceGraph).metadataProps || []
        });
        const selectedNodeIds = new Set(artifact.selectedNodeIds);
        const neededValues = new Set([...artifact.inputKeys, ...artifact.outputKeys]);
        for (const node of graphInfo.nodes) {
            if (!selectedNodeIds.has(node.id)) {
                continue;
            }
            graph.node.push(cloneProto(onnx.NodeProto, node.raw));
            for (const name of node.inputs) {
                if (name) {
                    neededValues.add(name);
                }
            }
            for (const name of node.outputs) {
                if (name) {
                    neededValues.add(name);
                }
            }
        }

        for (const name of artifact.inputKeys) {
            graph.input.push(this._cloneValueInfo(graphInfo, name));
        }
        for (const name of artifact.outputKeys) {
            graph.output.push(this._cloneValueInfo(graphInfo, name));
        }

        for (const name of neededValues) {
            const value = graphInfo.values.get(name);
            if (!value) {
                continue;
            }
            if (value.initializer) {
                if (value.initializer.values && value.initializer.indices) {
                    graph.sparseInitializer.push(cloneProto(onnx.SparseTensorProto, value.initializer));
                } else {
                    graph.initializer.push(cloneProto(onnx.TensorProto, value.initializer));
                }
            } else if (!artifact.inputKeys.includes(name) && !artifact.outputKeys.includes(name) && graphInfo.valueInfo.has(name)) {
                graph.valueInfo.push(this._cloneValueInfo(graphInfo, name));
            }
        }

        const quantization = ensureArray(graphInfo.sourceGraph.quantizationAnnotation).filter((entry) => neededValues.has(entry.tensorName));
        graph.quantizationAnnotation = quantization.map((entry) => cloneProto(onnx.TensorAnnotation, entry));

        const model = onnx.ModelProto.create({
            irVersion: original.irVersion,
            producerName: original.producerName,
            producerVersion: original.producerVersion,
            domain: original.domain,
            modelVersion: original.modelVersion,
            docString: original.docString,
            opsetImport: ensureArray(original.opsetImport).map((entry) => cloneProto(onnx.OperatorSetIdProto, entry)),
            metadataProps: ensureArray(original.metadataProps).map((entry) => cloneProto(onnx.StringStringEntryProto, entry)),
            graph
        });
        if (options.inlineWeights !== false) {
            for (const tensor of ensureArray(graph.initializer)) {
                tensor.dataLocation = DATA_LOCATION.DEFAULT;
                tensor.externalData = [];
            }
        }
        return model;
    }

    async exportArtifact(artifactId, targetPath, options = {}) {
        const artifact = this.getArtifact(artifactId);
        if (!artifact) {
            throw new Error('Artifact not found.');
        }
        const session = this.getSession(artifact.modelSessionId);
        if (!session) {
            throw new Error('Artifact session not found.');
        }
        const model = this._createArtifactModel(artifact, { inlineWeights: options.inlineWeights !== false });
        const outputDir = path.dirname(targetPath);
        await fs.promises.mkdir(outputDir, { recursive: true });
        const useExternal = options.externalData === true;
        if (useExternal) {
            const sidecarName = `${path.basename(targetPath, path.extname(targetPath))}.weights.bin`;
            const sidecarPath = path.join(outputDir, sidecarName);
            let offset = 0;
            const chunks = [];
            for (const tensor of ensureArray(model.graph.initializer)) {
                const original = session.graphInfo.initializers.get(tensor.name);
                const buffer = await materializeTensorBuffer(session, original || tensor);
                tensor.rawData = Buffer.alloc(0);
                tensor.floatData = [];
                tensor.int32Data = [];
                tensor.int64Data = [];
                tensor.uint64Data = [];
                tensor.doubleData = [];
                tensor.dataLocation = DATA_LOCATION.EXTERNAL;
                tensor.externalData = [
                    onnx.StringStringEntryProto.create({ key: 'location', value: sidecarName }),
                    onnx.StringStringEntryProto.create({ key: 'offset', value: String(offset) }),
                    onnx.StringStringEntryProto.create({ key: 'length', value: String(buffer.length) })
                ];
                chunks.push(buffer);
                offset += buffer.length;
            }
            await fs.promises.writeFile(sidecarPath, Buffer.concat(chunks));
        }
        const bytes = Buffer.from(onnx.ModelProto.encode(model).finish());
        await fs.promises.writeFile(targetPath, bytes);
        artifact.exportCache = {
            filePath: targetPath,
            externalDataPath: useExternal ? path.join(outputDir, `${path.basename(targetPath, path.extname(targetPath))}.weights.bin`) : null,
            updatedAt: nowIso()
        };
        return artifact.exportCache;
    }

    async ensureTempArtifactExport(artifactId) {
        const artifact = this.getArtifact(artifactId);
        if (!artifact) {
            throw new Error('Artifact not found.');
        }
        if (artifact.exportCache && artifact.exportCache.filePath && fs.existsSync(artifact.exportCache.filePath)) {
            return artifact.exportCache;
        }
        const targetDir = path.join(this.tempRoot, artifactId);
        await fs.promises.mkdir(targetDir, { recursive: true });
        const targetPath = path.join(targetDir, `${artifactId}.onnx`);
        return this.exportArtifact(artifactId, targetPath, { inlineWeights: true, externalData: false });
    }

    _makeFullGraphArtifact(sessionId) {
        const session = this.getSession(sessionId);
        if (!session) {
            throw new Error('Model session not found.');
        }
        return {
            id: `full-${session.id}`,
            modelSessionId: sessionId,
            inputKeys: session.graphInfo.graphInputNames.slice(),
            outputKeys: session.graphInfo.graphOutputNames.slice(),
            ioSignature: this._createIOSignature(session.graphInfo, session.graphInfo.graphInputNames, session.graphInfo.graphOutputNames),
            selectedNodeIds: session.graphInfo.nodes.map((node) => node.id),
            summary: {
                modelName: path.basename(session.filePath),
                graphName: session.graphInfo.name,
                nodeCount: session.graphInfo.nodes.length,
                inputCount: session.graphInfo.graphInputNames.length,
                outputCount: session.graphInfo.graphOutputNames.length
            }
        };
    }

    _buildResolvedInputSpecs(ioSignature, inputShapes = {}) {
        return ioSignature.inputs.map((input) => {
            const requested = inputShapes[input.name];
            const resolvedShape = Array.isArray(input.shape)
                ? input.shape.map((dimension, index) => {
                    if (dimension !== null && dimension !== undefined) {
                        return Number(dimension);
                    }
                    if (Array.isArray(requested) && requested[index] !== undefined) {
                        return Number(requested[index]);
                    }
                    return 1;
                })
                : Array.isArray(requested)
                    ? requested.map(Number)
                    : [];
            return {
                name: input.name,
                dtype: input.dtype,
                shape: resolvedShape,
                rank: input.rank
            };
        });
    }

    _generateInputData(spec, inputMode, importedData, canonicalNames) {
        if (inputMode === 'import') {
            if (!importedData) {
                throw new Error('Import mode requires an imported JSON or NPZ input.');
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
        switch (spec.dtype) {
            case 'float32':
            case 'float64':
            case 'float16':
            case 'bfloat16':
            case 'uint8':
            case 'int8':
            case 'uint16':
            case 'int16':
            case 'int32':
            case 'uint32':
            case 'bool':
            case 'int64':
            case 'uint64': {
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
            default:
                throw new Error(`Unsupported input dtype '${spec.dtype}'.`);
        }
    }

    _toOrtTensor(value) {
        const dtype = value.dtype;
        if (dtype === 'float16') {
            const data = Uint16Array.from(value.data.map((item) => float32ToFloat16(Number(item))));
            return new ort.Tensor(dtype, data, value.shape);
        }
        if (dtype === 'bfloat16') {
            const data = Uint16Array.from(value.data.map((item) => bfloat16FromFloat32(Number(item))));
            return new ort.Tensor(dtype, data, value.shape);
        }
        if (dtype === 'int64') {
            const data = BigInt64Array.from(value.data.map((item) => BigInt(Math.trunc(item))));
            return new ort.Tensor(dtype, data, value.shape);
        }
        if (dtype === 'uint64') {
            const data = BigUint64Array.from(value.data.map((item) => BigInt(Math.trunc(item))));
            return new ort.Tensor(dtype, data, value.shape);
        }
        if (dtype === 'bool') {
            const data = Uint8Array.from(value.data.map((item) => item ? 1 : 0));
            return new ort.Tensor(dtype, data, value.shape);
        }
        const TypedArray = NUMERIC_TYPED_ARRAY.get(dtype) || Float32Array;
        return new ort.Tensor(dtype, new TypedArray(value.data), value.shape);
    }

    async importInputFile(filePath) {
        const bytes = await fs.promises.readFile(filePath);
        let parsed;
        if (/\.json$/i.test(filePath)) {
            parsed = parseJsonInput(bytes.toString('utf8'));
        } else if (/\.npz$/i.test(filePath)) {
            parsed = await parseNpzInput(bytes);
        } else {
            throw new Error('Only .json and .npz input files are supported.');
        }
        const token = createId('input');
        const preview = Object.entries(parsed).map(([name, value]) => ({
            name,
            dtype: value.dtype || typeof value,
            shape: Array.isArray(value.shape) ? value.shape : null
        }));
        this.inferenceResults.set(token, parsed);
        return { token, preview };
    }

    _resolveImportedInput(token) {
        if (!token) {
            return null;
        }
        return this.inferenceResults.get(token) || null;
    }

    async getTensorPreview(sessionId, tensorName, options = {}) {
        const session = this.getSession(sessionId);
        if (!session) {
            throw new Error('Model session not found.');
        }
        if (!tensorName) {
            throw new Error('Tensor name is required for preview.');
        }
        const cacheKey = `${sessionId}:${tensorName}`;
        if (this.tensorPreviewCache.has(cacheKey)) {
            return this.tensorPreviewCache.get(cacheKey);
        }
        const originalTensor = session.graphInfo.initializers.get(tensorName)
            || (session.graphInfo.values.get(tensorName) ? session.graphInfo.values.get(tensorName).initializer : null);
        if (!originalTensor) {
            throw new Error(`Initializer '${tensorName}' was not found.`);
        }
        const tensor = originalTensor.values && originalTensor.indices ? originalTensor.values : originalTensor;
        const shape = ensureArray(tensor.dims).map((value) => Number(value));
        const dataType = DATA_TYPE_NAME.get(tensor.dataType) || `dtype_${tensor.dataType}`;
        const limit = Math.max(1, Number(options.limit) || 64);
        let sampleValues = [];
        if (ensureArray(tensor.floatData).length > 0) {
            sampleValues = ensureArray(tensor.floatData).slice(0, limit).map((value) => Number(value));
        } else if (ensureArray(tensor.doubleData).length > 0) {
            sampleValues = ensureArray(tensor.doubleData).slice(0, limit).map((value) => Number(value));
        } else if (ensureArray(tensor.int32Data).length > 0) {
            sampleValues = ensureArray(tensor.int32Data).slice(0, limit).map((value) => dataType === 'bool' ? value !== 0 : Number(value));
        } else if (ensureArray(tensor.int64Data).length > 0) {
            sampleValues = ensureArray(tensor.int64Data).slice(0, limit).map((value) => normalizePreviewScalar(typeof value === 'object' && typeof value.toBigInt === 'function' ? value.toBigInt() : value));
        } else if (ensureArray(tensor.uint64Data).length > 0) {
            sampleValues = ensureArray(tensor.uint64Data).slice(0, limit).map((value) => normalizePreviewScalar(typeof value === 'object' && typeof value.toBigInt === 'function' ? value.toBigInt() : value));
        } else if (ensureArray(tensor.stringData).length > 0) {
            sampleValues = ensureArray(tensor.stringData).slice(0, limit).map((value) => decodeUtf8Bytes(value));
        } else {
            const bytesPerElement = ELEMENT_SIZE.get(dataType) || 4;
            const sampleBuffer = await readTensorSampleBuffer(session, tensor, bytesPerElement * limit);
            sampleValues = decodeTensorSampleFromBuffer(sampleBuffer, dataType, limit);
        }
        const preview = {
            name: tensorName,
            dataType,
            shape,
            layout: originalTensor.values && originalTensor.indices ? 'sparse' : 'dense',
            location: tensor.dataLocation === DATA_LOCATION.EXTERNAL ? 'external' : 'inline',
            elementCount: countTensorElements(shape),
            sampleCount: sampleValues.length,
            sampleValues,
            truncated: countTensorElements(shape) > sampleValues.length,
            stats: summarizeNumericSample(sampleValues)
        };
        this.tensorPreviewCache.set(cacheKey, preview);
        return preview;
    }

    async runInference(options) {
        const useFullGraph = !!options.useFullGraph;
        const session = this.getSession(options.sessionId || (options.artifactId ? this.getArtifact(options.artifactId).modelSessionId : null));
        if (!session) {
            throw new Error('Model session not found.');
        }
        const artifact = useFullGraph ? this._makeFullGraphArtifact(session.id) : this.getArtifact(options.artifactId);
        if (!artifact) {
            throw new Error('Artifact not found.');
        }
        const exportInfo = useFullGraph
            ? { filePath: session.filePath, externalDataPath: null }
            : await this.ensureTempArtifactExport(artifact.id);
        const cacheKey = stableHash({ file: exportInfo.filePath, provider: 'cpu' });
        let ortSession = this.sessionCache.get(cacheKey);
        if (!ortSession) {
            ortSession = await ort.InferenceSession.create(exportInfo.filePath);
            this.sessionCache.set(cacheKey, ortSession);
        }
        const imported = this._resolveImportedInput(options.importToken || (this.compareState.importedInput && this.compareState.importedInput.token));
        const inputSpecs = this._buildResolvedInputSpecs(artifact.ioSignature, options.inputShapes || {});
        const feeds = {};
        const feedSummary = {};
        for (const input of inputSpecs) {
            const prepared = this._generateInputData(input, options.inputMode || 'zeros', imported, [input.name]);
            feedSummary[input.name] = prepared;
            feeds[input.name] = this._toOrtTensor(prepared);
        }
        const outputs = await ortSession.run(feeds);
        const outputsSummary = Object.entries(outputs).map(([name, tensor]) => summarizeOutputTensor(name, tensor));
        const runId = createId('run');
        const result = {
            runId,
            artifactId: artifact.id,
            inputMode: options.inputMode || 'zeros',
            resolvedShapes: inputSpecs.map((item) => ({ name: item.name, shape: item.shape, dtype: item.dtype })),
            outputsSummary,
            rawOutputRef: runId,
            cacheKey,
            createdAt: nowIso()
        };
        this.inferenceResults.set(runId, outputsSummary);
        return result;
    }

    async _exportFullGraphTemp(session) {
        const artifact = this._makeFullGraphArtifact(session.id);
        const targetDir = path.join(this.tempRoot, artifact.id);
        await fs.promises.mkdir(targetDir, { recursive: true });
        const targetPath = path.join(targetDir, `${artifact.id}.onnx`);
        const model = this._createArtifactModel(artifact, { inlineWeights: true });
        const bytes = Buffer.from(onnx.ModelProto.encode(model).finish());
        await fs.promises.writeFile(targetPath, bytes);
        return { filePath: targetPath, externalDataPath: null };
    }

    assignCompareSlot(slot, artifactId) {
        const artifact = this.getArtifact(artifactId);
        if (!artifact) {
            throw new Error('Artifact not found.');
        }
        if (slot === 'A') {
            this.compareState.slotA = this._toCompareSlot(artifact);
        } else if (slot === 'B') {
            this.compareState.slotB = this._toCompareSlot(artifact);
        } else {
            throw new Error(`Unsupported compare slot '${slot}'.`);
        }
        this._recomputeBindings();
        this._emitChange();
        return this.getCompareState();
    }

    clearCompare() {
        this.compareState = {
            slotA: null,
            slotB: null,
            inputBindings: [],
            outputBindings: [],
            compareRunStatus: { status: 'idle', stage: '', message: '', updatedAt: nowIso() },
            compareResult: null,
            importedInput: null
        };
        this._emitChange();
    }

    setCompareImportedInput(imported) {
        this.compareState.importedInput = imported ? { token: imported.token, preview: imported.preview || [] } : null;
        this._emitChange();
        return this.getCompareState();
    }

    setCompareBinding(kind, sourceName, targetName) {
        const listName = kind === 'output' ? 'outputBindings' : 'inputBindings';
        const binding = this.compareState[listName].find((item) => item.sourceName === sourceName);
        if (!binding) {
            throw new Error('Binding source not found.');
        }
        binding.targetName = targetName || null;
        binding.confirmed = !!targetName;
        binding.reason = binding.confirmed ? 'manual' : binding.reason;
        this._emitChange();
        return this.getCompareState();
    }

    _toCompareSlot(artifact) {
        return {
            artifactId: artifact.id,
            modelSessionId: artifact.modelSessionId,
            ioSignature: artifact.ioSignature,
            summary: artifact.summary,
            thumbnail: artifact.thumbnail,
            createdAt: artifact.createdAt
        };
    }

    _recomputeBindings() {
        const slotA = this.compareState.slotA;
        const slotB = this.compareState.slotB;
        if (!slotA || !slotB) {
            this.compareState.inputBindings = [];
            this.compareState.outputBindings = [];
            return;
        }
        this.compareState.inputBindings = this._buildBindings(slotA.ioSignature.inputs, slotB.ioSignature.inputs);
        this.compareState.outputBindings = this._buildBindings(slotA.ioSignature.outputs, slotB.ioSignature.outputs);
    }

    _buildBindings(sourcePorts, targetPorts) {
        return sourcePorts.map((source) => {
            const optionList = targetPorts.map((target) => ({ name: target.name, dtype: target.dtype, rank: target.rank, shape: target.shape }));
            const sameName = targetPorts.filter((target) => target.name === source.name && target.dtype === source.dtype && target.rank === source.rank);
            if (sameName.length === 1) {
                return {
                    sourceName: source.name,
                    sourcePort: source,
                    targetName: sameName[0].name,
                    targetPort: sameName[0],
                    confirmed: true,
                    reason: 'auto-name',
                    candidates: optionList
                };
            }
            const candidates = targetPorts.filter((target) => target.dtype === source.dtype && target.rank === source.rank && sameStaticShape(source.shape, target.shape));
            if (candidates.length === 1) {
                return {
                    sourceName: source.name,
                    sourcePort: source,
                    targetName: candidates[0].name,
                    targetPort: candidates[0],
                    confirmed: true,
                    reason: 'auto-unique',
                    candidates: optionList
                };
            }
            return {
                sourceName: source.name,
                sourcePort: source,
                targetName: null,
                targetPort: null,
                confirmed: false,
                reason: candidates.length === 0 ? 'unpaired' : 'manual',
                candidates: optionList
            };
        });
    }

    getCompareState() {
        return JSON.parse(JSON.stringify(this.compareState));
    }

    _resolveCompareBindings() {
        const slotA = this.compareState.slotA;
        const slotB = this.compareState.slotB;
        if (!slotA || !slotB) {
            throw new Error('Compare slots A/B are not ready.');
        }
        const inputBindings = this.compareState.inputBindings.filter((item) => item.targetName);
        const outputBindings = this.compareState.outputBindings.filter((item) => item.targetName);
        if (inputBindings.length !== slotA.ioSignature.inputs.length) {
            throw new Error('Not all compare inputs are bound.');
        }
        if (outputBindings.length === 0) {
            throw new Error('At least one compare output binding is required.');
        }
        return { inputBindings, outputBindings };
    }

    async runCompare(options = {}) {
        const slotA = this.compareState.slotA;
        const slotB = this.compareState.slotB;
        if (!slotA || !slotB) {
            throw new Error('Compare slots are incomplete.');
        }
        const { inputBindings, outputBindings } = this._resolveCompareBindings();
        this.compareState.compareRunStatus = { status: 'running', stage: '校验/生成共享输入', message: '', updatedAt: nowIso() };
        this._emitChange();

        try {
            const inputShapes = options.inputShapes || {};
            const imported = this._resolveImportedInput(options.importToken || (this.compareState.importedInput && this.compareState.importedInput.token));
            const sharedFeeds = {};
            const sharedShapeSummary = [];
            for (const binding of inputBindings) {
                const aPort = binding.sourcePort;
                const bPort = slotB.ioSignature.inputs.find((item) => item.name === binding.targetName);
                if (!bPort) {
                    throw new Error(`Target input '${binding.targetName}' not found.`);
                }
                if (aPort.dtype !== bPort.dtype || aPort.rank !== bPort.rank) {
                    throw new Error(`Incompatible bound inputs '${aPort.name}' and '${bPort.name}'.`);
                }
                const resolvedShape = ensureArray(aPort.shape).map((dimension, index) => {
                    const other = Array.isArray(bPort.shape) ? bPort.shape[index] : null;
                    if (dimension !== null && other !== null && dimension !== other) {
                        throw new Error(`Bound inputs '${aPort.name}' and '${bPort.name}' have incompatible static shapes.`);
                    }
                    if (dimension !== null) {
                        return dimension;
                    }
                    if (other !== null) {
                        return other;
                    }
                    const requested = Array.isArray(inputShapes[aPort.name]) ? inputShapes[aPort.name][index] : undefined;
                    return requested !== undefined ? Number(requested) : 1;
                });
                const prepared = this._generateInputData({ name: aPort.name, dtype: aPort.dtype, shape: resolvedShape }, options.inputMode || 'zeros', imported, [aPort.name, bPort.name]);
                sharedFeeds[aPort.name] = prepared;
                sharedShapeSummary.push({ name: aPort.name, shape: resolvedShape, dtype: aPort.dtype, targetName: bPort.name });
            }

            this.compareState.compareRunStatus = { status: 'running', stage: '执行 A', message: '', updatedAt: nowIso() };
            this._emitChange();
            const resultA = await this._runCompareSide(slotA.artifactId, false, sharedFeeds, inputBindings, 'A');
            this.compareState.compareRunStatus = { status: 'running', stage: '执行 B', message: '', updatedAt: nowIso() };
            this._emitChange();
            const resultB = await this._runCompareSide(slotB.artifactId, false, sharedFeeds, inputBindings, 'B');
            this.compareState.compareRunStatus = { status: 'running', stage: '计算差异', message: '', updatedAt: nowIso() };
            this._emitChange();

            const compareRunId = createId('compare');
            this.inferenceResults.set(compareRunId, {
                sideA: resultA.outputsSummary,
                sideB: resultB.outputsSummary
            });
            const rows = [];
            for (const binding of outputBindings) {
                const outputA = resultA.outputsSummary.find((item) => item.name === binding.sourceName);
                const outputB = resultB.outputsSummary.find((item) => item.name === binding.targetName);
                const rowBase = {
                    sourceName: binding.sourceName,
                    targetName: binding.targetName,
                    sourceStats: outputA ? outputA.summary || null : null,
                    targetStats: outputB ? outputB.summary || null : null,
                    sourcePreview: outputA ? outputA.preview || null : null,
                    targetPreview: outputB ? outputB.preview || null : null
                };
                if (!outputA || !outputB) {
                    rows.push({
                        ...rowBase,
                        status: 'skipped',
                        reason: 'missing-output'
                    });
                    continue;
                }
                if (JSON.stringify(outputA.shape) !== JSON.stringify(outputB.shape)) {
                    rows.push({
                        ...rowBase,
                        status: 'skipped',
                        reason: 'shape-mismatch',
                        sourceShape: outputA.shape,
                        targetShape: outputB.shape,
                        dtype: outputA.dtype
                    });
                    continue;
                }
                if (outputA.dtype !== outputB.dtype) {
                    rows.push({
                        ...rowBase,
                        status: 'skipped',
                        reason: 'dtype-mismatch',
                        sourceShape: outputA.shape,
                        targetShape: outputB.shape,
                        sourceDtype: outputA.dtype,
                        targetDtype: outputB.dtype
                    });
                    continue;
                }
                const metrics = computeNumericDiff(outputA.values, outputB.values);
                if (!metrics) {
                    rows.push({
                        ...rowBase,
                        status: 'skipped',
                        reason: 'non-numeric',
                        shape: outputA.shape,
                        dtype: outputA.dtype
                    });
                    continue;
                }
                rows.push({
                    ...rowBase,
                    status: 'ok',
                    shape: outputA.shape,
                    dtype: outputA.dtype,
                    ...metrics
                });
            }
            const validRows = rows.filter((row) => row.status === 'ok');
            const summary = validRows.length > 0
                ? validRows.reduce((best, row) => (!best || row.maxAbs > best.maxAbs) ? row : best, null)
                : null;
            this.compareState.compareResult = {
                createdAt: nowIso(),
                inputMode: options.inputMode || 'zeros',
                resolvedShapes: sharedShapeSummary,
                rawOutputRef: compareRunId,
                subgraphs: {
                    A: {
                        artifactId: slotA.artifactId,
                        modelSessionId: slotA.modelSessionId,
                        ...slotA.summary
                    },
                    B: {
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
                summary: summary
                    ? {
                        maxDiffOutput: summary.sourceName,
                        maxAbs: summary.maxAbs,
                        meanAbs: summary.meanAbs,
                        rmse: summary.rmse
                    }
                    : null
            };
            this.compareState.compareRunStatus = { status: 'idle', stage: '', message: '', updatedAt: nowIso() };
            this._emitChange();
            return this.getCompareState();
        } catch (error) {
            this.compareState.compareRunStatus = {
                status: 'failed',
                stage: '',
                message: error && error.message ? error.message : String(error),
                updatedAt: nowIso()
            };
            this._emitChange();
            throw error;
        }
    }

    async _runCompareSide(artifactId, useFullGraph, sharedFeeds, inputBindings, side = 'A') {
        const artifact = this.getArtifact(artifactId);
        const session = this.getSession(artifact.modelSessionId);
        const exportInfo = await this.ensureTempArtifactExport(artifact.id);
        const cacheKey = stableHash({ file: exportInfo.filePath, provider: 'cpu', side });
        let ortSession = this.sessionCache.get(cacheKey);
        if (!ortSession) {
            ortSession = await ort.InferenceSession.create(exportInfo.filePath);
            this.sessionCache.set(cacheKey, ortSession);
        }
        const feeds = {};
        for (const binding of inputBindings) {
            const prepared = sharedFeeds[binding.sourceName];
            const tensor = this._toOrtTensor(prepared);
            const feedName = side === 'B' ? binding.targetName : binding.sourceName;
            feeds[feedName] = tensor;
        }
        const outputs = await ortSession.run(feeds);
        return {
            outputsSummary: Object.entries(outputs).map(([name, tensor]) => summarizeOutputTensor(name, tensor))
        };
    }

    exportCompareOutputAsNpy(options = {}) {
        const result = this.compareState.compareResult;
        if (!result || !result.rawOutputRef) {
            throw new Error('No compare outputs are available to export.');
        }
        const cached = this.inferenceResults.get(result.rawOutputRef);
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

    exportCompareResultAsJson() {
        return JSON.stringify(this.compareState.compareResult || {}, null, 2);
    }

    exportCompareResultAsCsv() {
        const result = this.compareState.compareResult;
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
}

module.exports = {
    ONNXWorkbench,
    isOnnxFileName
};
