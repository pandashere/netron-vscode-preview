#!/usr/bin/env node
const { buildCompareBindings } = require('../lib/compare-core');
const {
    exportCompareOutputAsNpy,
    exportCompareResultAsCsv,
    exportCompareResultAsJson,
    runCrossProviderCompare
} = require('../lib/compare-engine');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

async function main() {
    const inputA = [{ name: 'x', dtype: 'float32', rank: 1, shape: ['N'] }];
    const inputB = [{ name: 'x_b', dtype: 'float32', rank: 1, shape: [3] }];
    const outputA = [{ name: 'y', dtype: 'float32', rank: 1, shape: [3] }];
    const outputB = [{ name: 'y_b', dtype: 'float32', rank: 1, shape: [3] }];
    const calls = [];
    const providers = new Map([
        ['format-a', {
            id: 'format-a',
            async runCompareArtifact(options) {
                calls.push({ provider: 'format-a', side: options.side, keys: Object.keys(options.sharedFeeds) });
                assert(options.side === 'A', 'Provider A should run side A.');
                assert(options.sharedFeeds.x.shape[0] === 3, 'Expected resolved shared feed shape.');
                return {
                    outputsSummary: [
                        { name: 'y', dtype: 'float32', shape: [3], values: [1, 2, 3] }
                    ]
                };
            }
        }],
        ['format-b', {
            id: 'format-b',
            async runCompareArtifact(options) {
                calls.push({ provider: 'format-b', side: options.side, keys: Object.keys(options.sharedFeeds) });
                assert(options.side === 'B', 'Provider B should run side B.');
                assert(options.inputBindings[0].targetName === 'x_b', 'Expected target binding for side B.');
                return {
                    outputsSummary: [
                        { name: 'y_b', dtype: 'float32', shape: [3], values: [1, 4, 2] }
                    ]
                };
            }
        }]
    ]);
    const compareState = {
        slotA: {
            providerId: 'format-a',
            artifactId: 'artifact-a',
            modelSessionId: 'session-a',
            ioSignature: { inputs: inputA, outputs: outputA },
            summary: { modelName: 'a.private' }
        },
        slotB: {
            providerId: 'format-b',
            artifactId: 'artifact-b',
            modelSessionId: 'session-b',
            ioSignature: { inputs: inputB, outputs: outputB },
            summary: { modelName: 'b.private' }
        },
        inputBindings: buildCompareBindings(inputA, inputB),
        outputBindings: buildCompareBindings(outputA, outputB),
        compareRunStatus: { status: 'idle', stage: '', message: '', updatedAt: 'now' },
        compareResult: null,
        importedInput: null
    };
    const result = await runCrossProviderCompare(compareState, {
        get(id) {
            return providers.get(id) || null;
        }
    }, {
        inputMode: 'ones',
        createRunId() {
            return 'compare-run-1';
        },
        now() {
            return '2026-05-31T00:00:00.000Z';
        }
    });
    assert(calls.length === 2, 'Expected both providers to run.');
    assert(result.compareState.compareResult.rawOutputRef === 'compare-run-1', 'Expected deterministic run id.');
    assert(result.compareState.compareResult.subgraphs.A.providerId === 'format-a', 'Expected provider A provenance.');
    assert(result.compareState.compareResult.subgraphs.B.providerId === 'format-b', 'Expected provider B provenance.');
    assert(result.compareState.compareResult.rows.length === 1, 'Expected one compare row.');
    assert(result.compareState.compareResult.rows[0].maxAbs === 2, 'Unexpected compare metric.');
    assert(result.rawOutputs.sideA[0].name === 'y', 'Expected side A raw outputs.');
    assert(result.rawOutputs.sideB[0].name === 'y_b', 'Expected side B raw outputs.');
    const json = exportCompareResultAsJson(result.compareState);
    assert(json.includes('compare-run-1'), 'Expected JSON export content.');
    const csv = exportCompareResultAsCsv(result.compareState);
    assert(csv.includes('sourceName,targetName'), 'Expected CSV header.');
    const rawOutputCache = new Map([[result.compareState.compareResult.rawOutputRef, result.rawOutputs]]);
    const npy = exportCompareOutputAsNpy(result.compareState, rawOutputCache, { side: 'A', sourceName: 'y' });
    assert(npy.fileName.includes('a.private-y.npy'), 'Expected NPY file name.');
    assert(npy.bytes[0] === 0x93 && Buffer.from(npy.bytes).subarray(1, 6).toString('ascii') === 'NUMPY', 'Expected NPY bytes.');
    console.log('compare engine ok', { providers: calls.map((item) => item.provider).join(','), rows: result.compareState.compareResult.rows.length });
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
