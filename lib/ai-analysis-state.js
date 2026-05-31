function createInitialAiAnalysisState() {
    return {
        status: 'idle',
        message: 'No analysis has run.',
        source: null,
        result: null,
        resultStale: false,
        previousResult: null,
        error: null,
        updatedAt: null
    };
}

function analysisStarted(source) {
    return {
        status: 'running',
        message: 'Running analysis...',
        source,
        result: null,
        resultStale: false,
        error: null
    };
}

function analysisSucceeded(source, text, createdAt) {
    const result = {
        text,
        createdAt,
        source
    };
    return {
        status: 'succeeded',
        message: 'Analysis completed.',
        source,
        result,
        previousResult: result,
        resultStale: false,
        error: null
    };
}

function analysisFailed(currentState, options = {}) {
    const previous = currentState && currentState.previousResult ? currentState.previousResult : null;
    const cancelled = !!options.cancelled;
    return {
        status: cancelled ? 'cancelled' : 'failed',
        message: cancelled
            ? 'Analysis cancelled. Showing stale result from previous successful analysis if available.'
            : 'Analysis failed. Showing stale result from previous successful analysis if available.',
        source: options.source || (currentState && currentState.source) || null,
        result: previous,
        resultStale: !!previous,
        error: {
            message: options.message || '',
            stage: options.stage || '',
            stderr: options.stderr || ''
        }
    };
}

function analysisCancelling() {
    return {
        status: 'running',
        message: 'Cancelling analysis...'
    };
}

module.exports = {
    createInitialAiAnalysisState,
    analysisStarted,
    analysisSucceeded,
    analysisFailed,
    analysisCancelling
};
