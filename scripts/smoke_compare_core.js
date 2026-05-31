#!/usr/bin/env node
const {
    buildCompareBindings,
    buildCompareRows,
    buildSharedInputPlan,
    computeNumericDiff,
    resolveCompareBindings,
    summarizeCompareRows
} = require('../lib/compare-core');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function main() {
    const sourceInputs = [
        { name: 'x', dtype: 'float32', rank: 2, shape: [1, 'N'] },
        { name: 'mask', dtype: 'float32', rank: 1, shape: ['N'] }
    ];
    const targetInputs = [
        { name: 'x', dtype: 'float32', rank: 2, shape: [1, 4] },
        { name: 'mask_b', dtype: 'float32', rank: 1, shape: [4] }
    ];
    const inputBindings = buildCompareBindings(sourceInputs, targetInputs);
    assert(inputBindings[0].confirmed && inputBindings[0].reason === 'auto-name', 'Expected same-name input binding.');
    assert(inputBindings[1].confirmed && inputBindings[1].targetName === 'mask_b', 'Expected static-compatible input binding.');

    const outputBindings = buildCompareBindings(
        [{ name: 'y', dtype: 'float32', rank: 1, shape: [4] }],
        [{ name: 'y_b', dtype: 'float32', rank: 1, shape: [4] }]
    );
    assert(outputBindings[0].confirmed && outputBindings[0].reason === 'auto-unique', 'Expected unique output binding.');

    const compareState = {
        slotA: { ioSignature: { inputs: sourceInputs, outputs: [] } },
        slotB: { ioSignature: { inputs: targetInputs, outputs: [] } },
        inputBindings,
        outputBindings
    };
    const resolved = resolveCompareBindings(compareState);
    assert(resolved.inputBindings.length === 2, 'Expected two resolved input bindings.');
    assert(resolved.outputBindings.length === 1, 'Expected one resolved output binding.');

    const plan = buildSharedInputPlan(inputBindings, compareState.slotB, { mask: [8] });
    assert(JSON.stringify(plan[0].shape) === JSON.stringify([1, 4]), 'Expected target static dimension to resolve symbolic source dimension.');
    assert(JSON.stringify(plan[1].shape) === JSON.stringify([4]), 'Expected target static shape to win over requested shape.');

    const metrics = computeNumericDiff([1, 2, 3], [1, 4, 2]);
    assert(metrics && metrics.maxAbs === 2 && metrics.meanAbs === 1, 'Unexpected numeric diff metrics.');

    const rows = buildCompareRows(outputBindings, [
        { name: 'y', dtype: 'float32', shape: [4], values: [1, 2, 3, 4], summary: { min: 1 }, preview: { sampleCount: 4 } }
    ], [
        { name: 'y_b', dtype: 'float32', shape: [4], values: [1, 1, 3, 6], summary: { min: 1 }, preview: { sampleCount: 4 } }
    ]);
    assert(rows.length === 1 && rows[0].status === 'ok', 'Expected one comparable row.');
    assert(rows[0].maxAbs === 2, 'Unexpected row maxAbs.');
    const summarized = summarizeCompareRows(rows);
    assert(summarized.validRows.length === 1, 'Expected one valid row.');
    assert(summarized.summary && summarized.summary.maxDiffOutput === 'y', 'Unexpected summary output.');

    const skipped = buildCompareRows(outputBindings, [
        { name: 'y', dtype: 'float32', shape: [4], values: [1, 2, 3, 4] }
    ], [
        { name: 'y_b', dtype: 'int32', shape: [4], values: [1, 2, 3, 4] }
    ]);
    assert(skipped[0].status === 'skipped' && skipped[0].reason === 'dtype-mismatch', 'Expected dtype mismatch skip.');

    console.log('compare core ok', { bindings: inputBindings.length, rows: rows.length });
}

try {
    main();
} catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
}
