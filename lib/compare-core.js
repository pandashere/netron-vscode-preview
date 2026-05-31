function isStaticDimension(value) {
    return typeof value === 'number' && Number.isFinite(value);
}

function sameStaticShape(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
        return false;
    }
    for (let index = 0; index < a.length; index++) {
        if (isStaticDimension(a[index]) && isStaticDimension(b[index]) && a[index] !== b[index]) {
            return false;
        }
    }
    return true;
}

function buildCompareBindings(sourcePorts, targetPorts) {
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

function resolveCompareBindings(compareState) {
    const slotA = compareState && compareState.slotA;
    const slotB = compareState && compareState.slotB;
    if (!slotA || !slotB) {
        throw new Error('Compare slots A/B are not ready.');
    }
    const inputBindings = compareState.inputBindings.filter((item) => item.targetName);
    const outputBindings = compareState.outputBindings.filter((item) => item.targetName);
    if (inputBindings.length !== slotA.ioSignature.inputs.length) {
        throw new Error('Not all compare inputs are bound.');
    }
    if (outputBindings.length === 0) {
        throw new Error('At least one compare output binding is required.');
    }
    return { inputBindings, outputBindings };
}

function buildSharedInputPlan(inputBindings, slotB, inputShapes = {}) {
    return inputBindings.map((binding) => {
        const sourcePort = binding.sourcePort;
        const targetPort = slotB.ioSignature.inputs.find((item) => item.name === binding.targetName);
        if (!targetPort) {
            throw new Error(`Target input '${binding.targetName}' not found.`);
        }
        if (sourcePort.dtype !== targetPort.dtype || sourcePort.rank !== targetPort.rank) {
            throw new Error(`Incompatible bound inputs '${sourcePort.name}' and '${targetPort.name}'.`);
        }
        const shape = Array.isArray(sourcePort.shape) ? sourcePort.shape.map((dimension, index) => {
            const other = Array.isArray(targetPort.shape) ? targetPort.shape[index] : null;
            if (isStaticDimension(dimension) && isStaticDimension(other) && dimension !== other) {
                throw new Error(`Bound inputs '${sourcePort.name}' and '${targetPort.name}' have incompatible static shapes.`);
            }
            if (isStaticDimension(dimension)) {
                return dimension;
            }
            if (isStaticDimension(other)) {
                return other;
            }
            const requested = Array.isArray(inputShapes[sourcePort.name]) ? inputShapes[sourcePort.name][index] : undefined;
            return requested !== undefined ? Number(requested) : 1;
        }) : [];
        return {
            sourceName: sourcePort.name,
            targetName: targetPort.name,
            dtype: sourcePort.dtype,
            shape
        };
    });
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

function buildCompareRows(outputBindings, outputsA, outputsB) {
    const rows = [];
    for (const binding of outputBindings) {
        const outputA = outputsA.find((item) => item.name === binding.sourceName);
        const outputB = outputsB.find((item) => item.name === binding.targetName);
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
    return rows;
}

function summarizeCompareRows(rows) {
    const validRows = rows.filter((row) => row.status === 'ok');
    const summary = validRows.length > 0
        ? validRows.reduce((best, row) => (!best || row.maxAbs > best.maxAbs) ? row : best, null)
        : null;
    return {
        validRows,
        summary: summary
            ? {
                maxDiffOutput: summary.sourceName,
                maxAbs: summary.maxAbs,
                meanAbs: summary.meanAbs,
                rmse: summary.rmse
            }
            : null
    };
}

module.exports = {
    buildCompareBindings,
    buildCompareRows,
    buildSharedInputPlan,
    computeNumericDiff,
    isStaticDimension,
    resolveCompareBindings,
    sameStaticShape,
    summarizeCompareRows
};
