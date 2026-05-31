const {
    buildCropTargetFromCoreGraph,
    buildTextExportContextFromCoreGraph
} = require('../../lib/text-export-context');

function createMockPrivateProvider(options = {}) {
    const providerId = options.id || 'mock-private';
    const extension = options.extension || '.mockmodel';
    const inputName = options.inputName || 'x';
    const outputName = options.outputName || 'y';
    const outputValues = options.outputValues || [1, 2, 3];
    const sessions = new Map();
    const artifacts = new Map();
    const importedInputs = new Map();
    let sessionSeq = 0;
    let artifactSeq = 0;
    let inputSeq = 0;

    function getArtifact(artifactId) {
        const artifact = artifacts.get(artifactId);
        if (!artifact) {
            throw new Error(`Artifact '${artifactId}' not found.`);
        }
        return artifact;
    }

    function coreGraphForArtifact(artifact) {
        return {
            id: `${providerId}:crop:${artifact.id}`,
            name: `${providerId}:crop:${artifact.id}`,
            inputs: [inputName],
            outputs: [outputName],
            nodes: [{
                id: `${providerId}-node-0`,
                name: `${providerId}_Node_0`,
                type: 'MockOp',
                domain: providerId,
                inputs: [{ name: 'input', tensor: inputName }],
                outputs: [{ name: 'output', tensor: outputName }],
                attributes: {},
                omittedAttributes: []
            }],
            tensors: [
                { name: inputName, dtype: 'float32', rawDtype: 'FLOAT', shape: [3], kind: 'input' },
                { name: outputName, dtype: 'float32', rawDtype: 'FLOAT', shape: [3], kind: 'output' }
            ]
        };
    }

    return {
        id: providerId,
        label: options.label || providerId,
        capabilities: {
            crop: true,
            exportArtifact: true,
            inference: true,
            compare: true,
            textExportContext: true,
            inputImport: true
        },
        canOpen(uri) {
            const fileName = uri && (uri.fsPath || uri.path || String(uri));
            return typeof fileName === 'string' && fileName.endsWith(extension);
        },
        async loadModel(uri) {
            sessionSeq += 1;
            const filePath = uri.fsPath || uri.path || String(uri);
            const session = {
                id: `${providerId}-session-${sessionSeq}`,
                format: providerId,
                filePath,
                snapshot: {
                    format: providerId,
                    graph: {
                        name: `${providerId}-graph`,
                        inputs: [{ name: 'input', values: [inputName] }],
                        outputs: [{ name: 'output', values: [outputName] }],
                        nodes: [{
                            id: `${providerId}-node-0`,
                            name: `${providerId}_Node_0`,
                            type: { name: 'MockOp', module: providerId },
                            inputs: [{ name: 'input', values: [inputName] }],
                            outputs: [{ name: 'output', values: [outputName] }],
                            attributes: []
                        }],
                        values: {
                            [inputName]: { type: { dataType: 'float32', shape: [3] } },
                            [outputName]: { type: { dataType: 'float32', shape: [3] } }
                        }
                    }
                }
            };
            sessions.set(session.id, session);
            return session;
        },
        async createCropArtifact({ sessionId }) {
            const session = sessions.get(sessionId);
            if (!session) {
                throw new Error(`Session '${sessionId}' not found.`);
            }
            artifactSeq += 1;
            const artifact = {
                id: `${providerId}-artifact-${artifactSeq}`,
                modelSessionId: session.id,
                createdAt: '2026-05-31T00:00:00.000Z',
                stale: false,
                ioSignature: {
                    inputs: [{ name: inputName, dtype: 'float32', rank: 1, shape: [3] }],
                    outputs: [{ name: outputName, dtype: 'float32', rank: 1, shape: [3] }]
                },
                summary: {
                    modelName: session.filePath.split('/').pop(),
                    graphName: `${providerId}-graph`,
                    nodeCount: 1,
                    inputCount: 1,
                    outputCount: 1
                },
                cropGraphSnapshot: session.snapshot.graph
            };
            artifacts.set(artifact.id, artifact);
            return artifact;
        },
        getCropTarget(artifactId) {
            const artifact = getArtifact(artifactId);
            const session = sessions.get(artifact.modelSessionId);
            return buildCropTargetFromCoreGraph({
                providerId,
                model: {
                    format: providerId,
                    fileName: session.filePath.split('/').pop(),
                    filePath: session.filePath
                },
                artifact,
                graph: coreGraphForArtifact(artifact)
            });
        },
        buildTextExportContext(artifactId) {
            const artifact = getArtifact(artifactId);
            const session = sessions.get(artifact.modelSessionId);
            return buildTextExportContextFromCoreGraph({
                providerId,
                model: {
                    format: providerId,
                    fileName: session.filePath.split('/').pop(),
                    filePath: session.filePath
                },
                artifact,
                graph: coreGraphForArtifact(artifact)
            });
        },
        getExportTarget(artifactId) {
            const artifact = getArtifact(artifactId);
            return {
                artifactId: artifact.id,
                defaultFileName: `${artifact.id}.crop.${extension.replace(/^\./, '')}`,
                filters: { Private: [extension.replace(/^\./, '')] },
                title: 'Export Private Crop',
                stage: 'Export private crop',
                message: 'Exporting private crop artifact...',
                options: { format: providerId }
            };
        },
        async exportArtifact(artifactId, targetPath, options = {}) {
            const artifact = getArtifact(artifactId);
            return {
                filePath: targetPath,
                artifactId: artifact.id,
                providerId,
                options
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
        async runCompareArtifact() {
            return {
                outputsSummary: [{
                    name: outputName,
                    dtype: 'float32',
                    shape: [3],
                    values: outputValues,
                    preview: { elementCount: 3, sampleCount: 3, sampleValues: outputValues, truncated: false },
                    summary: {
                        min: Math.min(...outputValues),
                        max: Math.max(...outputValues),
                        mean: outputValues.reduce((sum, item) => sum + item, 0) / outputValues.length
                    }
                }]
            };
        },
        async runInference() {
            return this.runCompareArtifact();
        },
        async importInputFile() {
            inputSeq += 1;
            const token = `${providerId}-input-${inputSeq}`;
            const parsed = { [inputName]: { dtype: 'float32', shape: [3], data: [1, 1, 1] } };
            importedInputs.set(token, parsed);
            return { token, preview: [{ name: inputName, dtype: 'float32', shape: [3] }] };
        },
        resolveImportedInput(token) {
            return importedInputs.get(token) || null;
        }
    };
}

module.exports = {
    createMockPrivateProvider
};
