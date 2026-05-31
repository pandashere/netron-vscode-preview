const BASE_METHODS = ['canOpen', 'loadModel'];

const CAPABILITY_METHODS = {
    crop: ['createCropArtifact', 'getCropTarget'],
    exportArtifact: ['exportArtifact'],
    inference: ['runInference'],
    compare: ['getCompareSlot', 'runCompareArtifact'],
    textExportContext: ['getCropTarget', 'buildTextExportContext'],
    tensorPreview: ['getTensorPreview'],
    inputImport: ['importInputFile', 'resolveImportedInput']
};

function providerDiagnostics(provider) {
    const errors = [];
    const warnings = [];
    if (!provider || typeof provider !== 'object') {
        return { errors: ['Provider must be an object.'], warnings };
    }
    if (typeof provider.id !== 'string' || provider.id.trim().length === 0) {
        errors.push('Provider id is required.');
    }
    for (const method of BASE_METHODS) {
        if (typeof provider[method] !== 'function') {
            errors.push(`Provider '${provider.id || '(unknown)'}' must implement ${method}().`);
        }
    }
    const capabilities = provider.capabilities || {};
    for (const [capability, methods] of Object.entries(CAPABILITY_METHODS)) {
        if (capabilities[capability] !== true) {
            continue;
        }
        for (const method of methods) {
            if (typeof provider[method] !== 'function') {
                errors.push(`Provider '${provider.id || '(unknown)'}' declares capability '${capability}' but does not implement ${method}().`);
            }
        }
    }
    if (provider.capabilities && typeof provider.capabilities !== 'object') {
        errors.push(`Provider '${provider.id || '(unknown)'}' capabilities must be an object.`);
    }
    if (!provider.capabilities) {
        warnings.push(`Provider '${provider.id || '(unknown)'}' does not declare capabilities.`);
    }
    return { errors, warnings };
}

class FormatProviderRegistry {
    constructor() {
        this.providers = [];
        this.diagnostics = new Map();
    }

    register(provider) {
        const diagnostics = providerDiagnostics(provider);
        if (diagnostics.errors.length > 0) {
            throw new Error(diagnostics.errors.join(' '));
        }
        if (this.providers.some((item) => item.id === provider.id)) {
            throw new Error(`Duplicate provider id: ${provider.id}`);
        }
        this.providers.push(provider);
        this.diagnostics.set(provider.id, diagnostics);
    }

    unregister(id) {
        const index = this.providers.findIndex((provider) => provider.id === id);
        if (index < 0) {
            return false;
        }
        this.providers.splice(index, 1);
        this.diagnostics.delete(id);
        return true;
    }

    list() {
        return this.providers.slice();
    }

    get(id) {
        return this.providers.find((provider) => provider.id === id) || null;
    }

    getDiagnostics(id) {
        if (id) {
            return this.diagnostics.get(id) || { errors: [], warnings: [`Provider '${id}' is not registered.`] };
        }
        return this.providers.map((provider) => ({
            id: provider.id,
            ...(this.diagnostics.get(provider.id) || { errors: [], warnings: [] })
        }));
    }

    resolve(uri) {
        const matches = this.providers.filter((provider) => {
            if (typeof provider.canOpen !== 'function') {
                return false;
            }
            try {
                return !!provider.canOpen(uri);
            } catch {
                return false;
            }
        });
        if (matches.length === 0) {
            return {
                ok: false,
                reason: 'No registered provider can open this file.',
                provider: null
            };
        }
        if (matches.length > 1) {
            return {
                ok: false,
                reason: `Multiple providers can open this file: ${matches.map((item) => item.id).join(', ')}`,
                provider: null
            };
        }
        return {
            ok: true,
            reason: '',
            provider: matches[0]
        };
    }
}

function createOnnxProvider(workbench, isOnnxFileName) {
    return {
        id: 'onnx',
        label: 'ONNX',
        capabilities: {
            crop: true,
            exportArtifact: true,
            inference: true,
            compare: true,
            textExportContext: true,
            tensorPreview: true,
            inputImport: true
        },
        canOpen(uri) {
            const fileName = uri && (uri.fsPath || uri.path || uri.toString());
            return isOnnxFileName(fileName);
        },
        async loadModel(uri, options) {
            return workbench.loadModel(uri, options);
        },
        getSession(sessionId) {
            return workbench.getSession(sessionId);
        },
        getArtifact(artifactId) {
            return workbench.getArtifact(artifactId);
        },
        getCropTarget(artifactId) {
            return workbench.getCropTarget(artifactId);
        },
        buildTextExportContext(artifactId) {
            return workbench.buildTextExportContext(artifactId);
        },
        createCropArtifact(options) {
            return workbench.createCropArtifact(options);
        },
        exportArtifact(artifactId, targetPath, options) {
            return workbench.exportArtifact(artifactId, targetPath, options);
        },
        getExportTarget(artifactId, options = {}) {
            const artifact = workbench.getArtifact(artifactId);
            if (!artifact) {
                throw new Error('No confirmed crop artifact available.');
            }
            const session = workbench.getSession(artifact.modelSessionId);
            if (!session) {
                throw new Error('Artifact session not found.');
            }
            const useExternal = /external/i.test(options.weightMode || '') || (session.graphInfo.initializers.size > 0 && workbench.hasExternalData(session));
            const baseName = `${session.filePath.split(/[\\/]/).pop().replace(/\.[^.]*$/, '')}.${artifact.id}.crop.onnx`;
            return {
                artifactId: artifact.id,
                defaultFileName: baseName,
                filters: { ONNX: ['onnx'] },
                title: 'Export Crop ONNX',
                stage: '重建 ONNX',
                message: 'Exporting crop ONNX...',
                options: {
                    externalData: useExternal,
                    inlineWeights: !useExternal
                }
            };
        },
        importInputFile(filePath) {
            return workbench.importInputFile(filePath);
        },
        resolveImportedInput(token) {
            return workbench.resolveImportedInput(token);
        },
        runInference(options) {
            return workbench.runInference(options);
        },
        getCompareSlot(artifactId) {
            return workbench.getCompareSlot(artifactId);
        },
        assignCompareSlot(slot, artifactId) {
            return workbench.assignCompareSlot(slot, artifactId);
        },
        clearCompare() {
            return workbench.clearCompare();
        },
        setCompareImportedInput(imported) {
            return workbench.setCompareImportedInput(imported);
        },
        setCompareBinding(kind, sourceName, targetName) {
            return workbench.setCompareBinding(kind, sourceName, targetName);
        },
        runCompare(options) {
            return workbench.runCompare(options);
        },
        runCompareArtifact(options) {
            return workbench.runCompareArtifact(options);
        },
        getCompareState() {
            return workbench.getCompareState();
        },
        exportCompareResultAsJson() {
            return workbench.exportCompareResultAsJson();
        },
        exportCompareResultAsCsv() {
            return workbench.exportCompareResultAsCsv();
        },
        exportCompareOutputAsNpy(options) {
            return workbench.exportCompareOutputAsNpy(options);
        },
        getTensorPreview(sessionId, tensorName, options) {
            return workbench.getTensorPreview(sessionId, tensorName, options);
        }
    };
}

module.exports = {
    FormatProviderRegistry,
    createOnnxProvider,
    providerDiagnostics
};
