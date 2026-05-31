#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');

function parseArgs(argv) {
    const result = {
        root: path.join(os.homedir(), '.netron', 'vscode-preview')
    };
    for (let index = 0; index < argv.length; index++) {
        const item = argv[index];
        if (item === '--root' && argv[index + 1]) {
            result.root = path.resolve(argv[index + 1]);
            index += 1;
        }
    }
    return result;
}

function writeJson(filePath, data) {
    fs.writeFileSync(filePath, `${JSON.stringify(data, null, 2)}\n`);
}

function writeExecutable(filePath, content) {
    fs.writeFileSync(filePath, content);
    fs.chmodSync(filePath, 0o755);
}

function installTool(root, kind, name, manifest, scriptContent) {
    const dir = path.join(root, `${kind}s`, name);
    fs.mkdirSync(dir, { recursive: true });
    const scriptPath = path.join(dir, `${name}.js`);
    writeExecutable(scriptPath, scriptContent);
    writeJson(path.join(dir, `${kind}.json`), {
        ...manifest,
        command: process.execPath,
        args: [scriptPath]
    });
    return dir;
}

function installManualAiTools(root) {
    const exporterDir = installTool(root, 'exporter', 'crop-json-summary', {
        id: 'crop-json-summary',
        label: 'Crop JSON Summary',
        timeoutMs: 30000
    }, `#!/usr/bin/env node
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  const context = JSON.parse(input || '{}');
  const graph = context.graph || {};
  const lines = [
    'Model: ' + ((context.model && context.model.fileName) || '(unknown)'),
    'Artifact: ' + ((context.artifact && context.artifact.id) || '(none)'),
    'Graph: ' + (graph.id || '(none)'),
    'Inputs: ' + (Array.isArray(graph.inputs) ? graph.inputs.join(', ') : ''),
    'Outputs: ' + (Array.isArray(graph.outputs) ? graph.outputs.join(', ') : ''),
    'Nodes: ' + (Array.isArray(graph.nodes) ? graph.nodes.length : 0),
    'Tensors: ' + (Array.isArray(graph.tensors) ? graph.tensors.length : 0)
  ];
  process.stdout.write(lines.join('\\n'));
});
`);

    const analyzerDir = installTool(root, 'analyzer', 'line-count-analysis', {
        id: 'line-count-analysis',
        label: 'Line Count Analysis',
        timeoutMs: 30000
    }, `#!/usr/bin/env node
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  const lines = input.split(/\\r?\\n/).filter((line) => line.trim().length > 0);
  process.stdout.write([
    'Analysis Result',
    'Input lines: ' + lines.length,
    'First line: ' + (lines[0] || '(empty)')
  ].join('\\n'));
});
`);

    return {
        root,
        exporterDir,
        analyzerDir
    };
}

function main() {
    const options = parseArgs(process.argv.slice(2));
    const result = installManualAiTools(options.root);
    console.log('Manual AI tools installed', result);
}

if (require.main === module) {
    try {
        main();
    } catch (error) {
        console.error(error && error.stack ? error.stack : String(error));
        process.exit(1);
    }
}

module.exports = {
    installManualAiTools,
    parseArgs
};
