#!/usr/bin/env node
const {
    analysisCancelling,
    analysisFailed,
    analysisStarted,
    analysisSucceeded,
    createInitialAiAnalysisState
} = require('../lib/ai-analysis-state');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function mergeState(state, patch) {
    return {
        ...state,
        ...patch,
        updatedAt: '2026-05-31T00:00:00.000Z'
    };
}

function main() {
    const source = {
        modelFile: 'model.onnx',
        modelPath: '/tmp/model.onnx',
        artifactId: 'artifact-1',
        graphId: 'graph-1',
        exporterId: 'formatter',
        analyzerId: 'analyzer',
        time: '2026-05-31T00:00:00.000Z'
    };
    let state = createInitialAiAnalysisState();
    assert(state.status === 'idle', 'Initial AI state should be idle.');
    assert(state.result === null && state.previousResult === null, 'Initial AI state should not have results.');

    state = mergeState(state, analysisStarted(source));
    assert(state.status === 'running', 'Running state expected.');
    assert(state.result === null, 'Running should clear the visible result.');
    assert(state.resultStale === false, 'Running should clear stale state.');

    state = mergeState(state, analysisSucceeded(source, 'analysis output', '2026-05-31T00:00:01.000Z'));
    assert(state.status === 'succeeded', 'Succeeded state expected.');
    assert(state.result && state.result.text === 'analysis output', 'Succeeded result text missing.');
    assert(state.previousResult === state.result, 'Succeeded result should become previousResult.');
    assert(state.resultStale === false, 'Succeeded result should not be stale.');

    const previousResult = state.previousResult;
    state = mergeState(state, analysisStarted(source));
    assert(state.result === null, 'A new running task should clear current result.');
    assert(state.previousResult === previousResult, 'Running should preserve previous successful result.');

    state = mergeState(state, analysisFailed(state, {
        source,
        message: 'analyzer failed',
        stage: 'analyzer',
        stderr: 'stack trace'
    }));
    assert(state.status === 'failed', 'Failed state expected.');
    assert(state.result === previousResult, 'Failure should restore previous successful result.');
    assert(state.resultStale === true, 'Failure should mark restored result as stale.');
    assert(state.error.stage === 'analyzer', 'Failure should preserve failed stage.');
    assert(state.error.stderr === 'stack trace', 'Failure should preserve stderr.');

    state = mergeState(state, analysisStarted(source));
    state = mergeState(state, analysisCancelling());
    assert(state.status === 'running' && /cancelling/i.test(state.message), 'Cancelling should keep running status with cancelling message.');

    state = mergeState(state, analysisFailed(state, {
        cancelled: true,
        source,
        message: 'Analyzer exited with code null (SIGTERM).',
        stage: 'analyzer'
    }));
    assert(state.status === 'cancelled', 'Cancelled state expected.');
    assert(state.result === previousResult, 'Cancellation should restore previous successful result.');
    assert(state.resultStale === true, 'Cancellation should mark restored result as stale.');

    state = mergeState(createInitialAiAnalysisState(), analysisFailed(createInitialAiAnalysisState(), {
        source,
        message: 'first run failed',
        stage: 'exporter'
    }));
    assert(state.status === 'failed', 'First-run failure should be failed.');
    assert(state.result === null, 'First-run failure should not fabricate a stale result.');
    assert(state.resultStale === false, 'First-run failure without previous result should not be stale.');

    console.log('ai analysis state ok');
}

try {
    main();
} catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
}
