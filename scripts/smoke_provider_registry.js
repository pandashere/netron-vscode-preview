#!/usr/bin/env node
const { FormatProviderRegistry, createOnnxProvider, providerDiagnostics } = require('../lib/format-providers');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function main() {
    const registry = new FormatProviderRegistry();
    registry.register({
        id: 'onnx',
        loadModel() {
            return {};
        },
        canOpen(uri) {
            return /\.onnx$/i.test(uri.fsPath || uri.path || '');
        }
    });
    const resolved = registry.resolve({ fsPath: '/tmp/model.onnx' });
    assert(resolved.ok && resolved.provider.id === 'onnx', 'Expected ONNX provider.');
    assert(registry.get('onnx') === resolved.provider, 'Expected provider lookup by id.');
    assert(registry.get('missing') === null, 'Missing provider lookup should return null.');
    const unsupported = registry.resolve({ fsPath: '/tmp/model.bin' });
    assert(!unsupported.ok && /No registered provider/.test(unsupported.reason), 'Expected unsupported reason.');
    let duplicateFailed = false;
    try {
        registry.register({ id: 'onnx', loadModel() { return {}; }, canOpen() { return false; } });
    } catch (error) {
        duplicateFailed = /Duplicate provider id/.test(error.message);
    }
    assert(duplicateFailed, 'Duplicate provider id should fail.');

    let invalidFailed = false;
    try {
        registry.register({ id: 'invalid', canOpen() { return true; } });
    } catch (error) {
        invalidFailed = /loadModel/.test(error.message);
    }
    assert(invalidFailed, 'Provider without loadModel should fail.');

    let capabilityFailed = false;
    try {
        registry.register({
            id: 'bad-compare',
            capabilities: { compare: true },
            canOpen() { return false; },
            loadModel() { return {}; }
        });
    } catch (error) {
        capabilityFailed = /getCompareSlot/.test(error.message) && /runCompareArtifact/.test(error.message);
    }
    assert(capabilityFailed, 'Provider with incomplete compare capability should fail.');

    const diagnostics = providerDiagnostics({
        id: 'diag',
        capabilities: { textExportContext: true },
        canOpen() { return false; },
        loadModel() { return {}; },
        getCropTarget() { return {}; }
    });
    assert(diagnostics.errors.some((item) => item.includes('buildTextExportContext')), 'Expected capability diagnostic.');

    const ambiguous = new FormatProviderRegistry();
    ambiguous.register({ id: 'a', loadModel() { return {}; }, canOpen() { return true; } });
    ambiguous.register({ id: 'b', loadModel() { return {}; }, canOpen() { return true; } });
    const ambiguousResult = ambiguous.resolve({ fsPath: '/tmp/model.any' });
    assert(!ambiguousResult.ok && /Multiple providers/.test(ambiguousResult.reason), 'Expected ambiguous provider reason.');

    const calls = [];
    const fakeWorkbench = {
        loadModel() { calls.push('loadModel'); return 'loaded'; },
        getSession() { calls.push('getSession'); return 'session'; },
        getArtifact() { calls.push('getArtifact'); return 'artifact'; },
        getCropTarget() { calls.push('getCropTarget'); return 'target'; },
        buildTextExportContext() { calls.push('buildTextExportContext'); return 'context'; },
        createCropArtifact() { calls.push('createCropArtifact'); return 'crop'; },
        exportArtifact() { calls.push('exportArtifact'); return 'export'; },
        importInputFile() { calls.push('importInputFile'); return 'input'; },
        resolveImportedInput() { calls.push('resolveImportedInput'); return 'resolved-input'; },
        runInference() { calls.push('runInference'); return 'inference'; },
        getCompareSlot() { calls.push('getCompareSlot'); return 'compare-slot'; },
        assignCompareSlot() { calls.push('assignCompareSlot'); return 'slot'; },
        clearCompare() { calls.push('clearCompare'); return 'clear'; },
        setCompareImportedInput() { calls.push('setCompareImportedInput'); return 'imported'; },
        setCompareBinding() { calls.push('setCompareBinding'); return 'binding'; },
        runCompare() { calls.push('runCompare'); return 'compare'; },
        runCompareArtifact() { calls.push('runCompareArtifact'); return 'compare-artifact'; },
        getCompareState() { calls.push('getCompareState'); return 'state'; },
        exportCompareResultAsJson() { calls.push('exportCompareResultAsJson'); return 'json'; },
        exportCompareResultAsCsv() { calls.push('exportCompareResultAsCsv'); return 'csv'; },
        exportCompareOutputAsNpy() { calls.push('exportCompareOutputAsNpy'); return 'npy'; },
        getTensorPreview() { calls.push('getTensorPreview'); return 'preview'; }
    };
    const onnxProvider = createOnnxProvider(fakeWorkbench, (fileName) => /\.onnx$/i.test(fileName));
    assert(onnxProvider.canOpen({ fsPath: '/tmp/a.onnx' }), 'ONNX provider should match .onnx files.');
    assert(!onnxProvider.canOpen({ fsPath: '/tmp/a.bin' }), 'ONNX provider should not match unsupported files.');
    assert(onnxProvider.capabilities.compare === true, 'ONNX provider should declare compare support.');
    const onnxDiagnostics = providerDiagnostics(onnxProvider);
    assert(onnxDiagnostics.errors.length === 0, `ONNX provider diagnostics should be clean: ${onnxDiagnostics.errors.join('; ')}`);
    const onnxRegistry = new FormatProviderRegistry();
    onnxRegistry.register(onnxProvider);
    assert(onnxRegistry.getDiagnostics('onnx').warnings.length === 0, 'ONNX provider should not have registry warnings.');
    assert(onnxProvider.getCropTarget('artifact') === 'target', 'Expected getCropTarget forwarding.');
    assert(onnxProvider.buildTextExportContext('artifact') === 'context', 'Expected buildTextExportContext forwarding.');
    assert(onnxProvider.resolveImportedInput('token') === 'resolved-input', 'Expected resolveImportedInput forwarding.');
    assert(onnxProvider.getCompareSlot('artifact') === 'compare-slot', 'Expected getCompareSlot forwarding.');
    assert(onnxProvider.clearCompare() === 'clear', 'Expected clearCompare forwarding.');
    assert(onnxProvider.setCompareBinding('input', 'a', 'b') === 'binding', 'Expected setCompareBinding forwarding.');
    assert(onnxProvider.runCompare({}) === 'compare', 'Expected runCompare forwarding.');
    assert(onnxProvider.runCompareArtifact({}) === 'compare-artifact', 'Expected runCompareArtifact forwarding.');
    assert(onnxProvider.exportCompareResultAsJson() === 'json', 'Expected JSON export forwarding.');
    assert(onnxProvider.exportCompareResultAsCsv() === 'csv', 'Expected CSV export forwarding.');
    assert(onnxProvider.exportCompareOutputAsNpy({}) === 'npy', 'Expected NPY export forwarding.');
    for (const expected of [
        'getCropTarget',
        'buildTextExportContext',
        'resolveImportedInput',
        'getCompareSlot',
        'clearCompare',
        'setCompareBinding',
        'runCompare',
        'runCompareArtifact',
        'exportCompareResultAsJson',
        'exportCompareResultAsCsv',
        'exportCompareOutputAsNpy'
    ]) {
        assert(calls.includes(expected), `Missing forwarded call: ${expected}`);
    }
    console.log('provider registry ok');
}

try {
    main();
} catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
}
