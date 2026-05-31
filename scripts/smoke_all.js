#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');
const requiredFixtures = [
    'testdata/generated/dual-io-compare-a.onnx',
    'testdata/generated/dual-io-compare-b.onnx'
];

const checks = [
    ['package-json', ['-e', "JSON.parse(require('fs').readFileSync('package.json','utf8')); console.log('package ok')"]],
    ['package-manifest', ['scripts/smoke_package_manifest.js']],
    ['extension-activation', ['scripts/smoke_extension_activation.js']],
    ['extension-provider-api', ['scripts/smoke_extension_provider_api.js']],
    ['provider-artifact-export', ['scripts/smoke_provider_artifact_export.js']],
    ['webview-contract', ['scripts/smoke_webview_contract.js']],
    ['provider-registry', ['scripts/smoke_provider_registry.js']],
    ['private-ir-guide-examples', ['scripts/smoke_private_ir_guide_examples.js']],
    ['cli-tools', ['scripts/smoke_cli_tools.js']],
    ['manual-ai-tools', ['scripts/smoke_manual_ai_tools.js']],
    ['graph-deepseek-tools', ['scripts/smoke_graph_deepseek_tools.js']],
    ['copy-export-host-flow', ['scripts/smoke_copy_export_host_flow.js']],
    ['ai-analysis-state', ['scripts/smoke_ai_analysis_state.js']],
    ['ai-panel-contract', ['scripts/smoke_ai_panel_contract.js']],
    ['ai-task-host-flow', ['scripts/smoke_ai_task_host_flow.js']],
    ['core-text-export-context', ['scripts/smoke_core_text_export_context.js']],
    ['text-export-context', ['scripts/smoke_text_export_context.js']],
    ['compare-core', ['scripts/smoke_compare_core.js']],
    ['host-compare-state', ['scripts/smoke_host_compare_state.js']],
    ['compare-engine', ['scripts/smoke_compare_engine.js']],
    ['private-provider-contract', ['scripts/smoke_private_provider_contract.js']],
    ['dev-ir-provider', ['scripts/smoke_dev_ir_provider.js']],
    ['compare-internal', ['scripts/smoke_compare_internal_io.js']]
];

function assertFixtures() {
    const missing = requiredFixtures.filter((item) => !fs.existsSync(path.join(root, item)));
    if (missing.length > 0) {
        throw new Error(`Missing generated ONNX fixtures: ${missing.join(', ')}. Run 'npm run generate:testmodels' first.`);
    }
}

function runCheck(name, args) {
    console.log(`\n[smoke] ${name}`);
    const result = spawnSync(process.execPath, args, {
        cwd: root,
        stdio: 'inherit'
    });
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0) {
        throw new Error(`${name} failed with exit code ${result.status}.`);
    }
}

function main() {
    assertFixtures();
    for (const [name, args] of checks) {
        runCheck(name, args);
    }
    console.log('\nsmoke all ok');
}

try {
    main();
} catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
}
