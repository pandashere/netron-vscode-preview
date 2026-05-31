# Private IR Integration Guide

This guide is written for a coding agent that should integrate a private model format without reading the rest of this repository.

Follow the steps in order. Do not invent a different extension mechanism unless the owner asks for it. The current design is:

- Model formats are integrated as internal JavaScript providers registered by the VS Code extension host.
- Text export and AI analysis are external CLI tools discovered from a global user directory.
- Compare supports different providers when both confirmed crops expose compatible input/output signatures.

## 0. What To Build

You need to implement one provider object for your private IR.

Minimum useful provider:

```js
{
  id: 'my-ir',
  label: 'My Private IR',
  capabilities: {
    crop: true,
    exportArtifact: true,
    inference: true,
    compare: true,
    textExportContext: true
  },
  canOpen(uri) {},
  async loadModel(uri, options) {},
  async createCropArtifact({ sessionId, startKeys, endKeys }) {},
  getCropTarget(artifactId) {},
  buildTextExportContext(artifactId) {},
  getExportTarget(artifactId, options) {},
  async exportArtifact(artifactId, targetPath, options) {},
  getCompareSlot(artifactId) {},
  async runCompareArtifact({ artifactId, input }) {},
  async runInference(options) {}
}
```

If you only want preview first, implement only:

```js
{
  id: 'my-ir',
  label: 'My Private IR',
  capabilities: {},
  canOpen(uri) {},
  async loadModel(uri) {}
}
```

Then add crop, export, inference, compare, and AI support one by one.

## 1. Files To Copy

Use the reference files already in this repository:

- Provider template: `examples/private-ir/my-ir-provider.js`
- Sample private model: `examples/private-ir/sample.myir.json`
- Exporter examples:
  - `examples/tools/exporters/crop-json-summary/`
  - `examples/tools/exporters/graph-edge-list/`
- Analyzer examples:
  - `examples/tools/analyzers/line-count-analysis/`
  - `examples/tools/analyzers/deepseek-graph-analysis/`
  - `examples/tools/analyzers/codex-one-shot-analysis/`

The DeepSeek example is desensitized. It contains no API key. It reads the key from `DEEPSEEK_API_KEY` or `~/.netron/vscode-preview/secrets/deepseek_api_key`.
The Codex example calls `codex exec` once with the exported graph text on stdin. It requires a working Codex CLI login on the same machine where the VS Code extension host runs.

## 2. Register A Provider

This is the shortest copy-paste path. Do these steps first, then replace the parser/runtime internals.

1. Copy the provider template:

```bash
cp examples/private-ir/my-ir-provider.js lib/my-ir-provider.js
```

2. In the copied file, change this import:

```js
} = require('../../lib/text-export-context');
```

to:

```js
} = require('./text-export-context');
```

3. In `extension.js`, add this near the other `require()` lines:

```js
const { createMyIrProvider } = require('./lib/my-ir-provider');
```

4. In `activate(context)`, find the provider registration block and add your provider after ONNX:

```js
state.providerRegistry.register(createOnnxProvider(state.workbench, isOnnxFileName));
state.providerRegistry.register(createMyIrProvider());
```

5. Start with this minimal provider if you do not want crop/runtime yet:

```js
const fs = require('fs');
const path = require('path');

function createMyIrProvider() {
  const sessions = new Map();

  return {
    id: 'my-ir',
    label: 'My Private IR',
    capabilities: {},
    canOpen(uri) {
      const filePath = uri && (uri.fsPath || uri.path || String(uri));
      return typeof filePath === 'string' && filePath.endsWith('.myir.json');
    },
    async loadModel(uri) {
      const filePath = uri.fsPath || uri.path || String(uri);
      const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const sessionId = `my-ir-session-${sessions.size + 1}`;
      const graph = raw.graph || { inputs: [], outputs: [], nodes: [], tensors: [] };
      const snapshot = {
        sessionId,
        format: 'my-ir',
        fileName: path.basename(filePath),
        filePath,
        graph: {
          name: graph.name || sessionId,
          inputs: (graph.inputs || []).map((name) => ({ name, values: [name] })),
          outputs: (graph.outputs || []).map((name) => ({ name, values: [name] })),
          nodes: [],
          values: {}
        }
      };
      const session = { id: sessionId, format: 'my-ir', filePath, snapshot };
      sessions.set(sessionId, session);
      return session;
    }
  };
}

module.exports = { createMyIrProvider };
```

This minimal version should open a private file and show graph inputs/outputs. It does not support crop, export, inference, compare, or AI yet.

6. Run:

```bash
node --check lib/my-ir-provider.js
node --check extension.js
npm run smoke:extension-provider-api
npm run smoke:provider-registry
```

7. Package and install:

```bash
npm run package:vsix
code --install-extension dist/netron-vscode-workbench-0.1.0.vsix --force
```

8. In VS Code, run `Developer: Reload Window`.

## 3. Provider Lifecycle

The extension host calls methods in this order:

1. `canOpen(uri)` decides whether the provider owns the file.
2. `loadModel(uri)` parses the file and returns a renderable model snapshot.
3. User selects start/end tensor edges in the UI.
4. `createCropArtifact({ sessionId, startKeys, endKeys })` creates a confirmed crop artifact.
5. Optional actions use that artifact:
   - Copy Export Text calls `buildTextExportContext(artifactId)`.
   - AI Analyze runs formatter/exporter, then analyzer.
   - Export Crop calls `getExportTarget()` and `exportArtifact()`.
   - Run Inference calls `runInference()`.
   - Set As A/B calls `getCompareSlot()`.
   - Compare calls `runCompareArtifact()`.

Keep all state inside the provider instance:

```js
const sessions = new Map();   // sessionId -> parsed model/session
const artifacts = new Map();  // artifactId -> confirmed crop
```

Never store tensor payload data in text export context. Store tensor metadata only.

## 3.1 No-Brainer Implementation Order

Do not implement everything at once. Use this order and verify each step before moving on.

Step A: open file and show any graph.

```js
capabilities: {}
canOpen(uri) {
  return String(uri.fsPath || uri.path || uri).endsWith('.myir.json');
}
async loadModel(uri) {
  // Parse your file and return snapshot.sessionId + snapshot.graph.
}
```

Verify:

```bash
node --check lib/my-ir-provider.js
npm run smoke:provider-registry
```

Manual check: open `.myir.json` in VS Code. If it does not open, debug `canOpen()` and registration first.

Step B: show correct nodes, colors, tensors, and initializers.

```js
type: {
  name: node.type,
  module: 'my.ir',
  identifier: `my.ir.${node.type}`,
  category: node.category || 'Custom'
}
```

Use initializer objects:

```js
values.weight = {
  type: { dataType: 'float32', shape: [64, 3, 3, 3] },
  initializer: {
    name: 'weight',
    category: 'Initializer',
    type: { dataType: 'float32', shape: [64, 3, 3, 3] },
    location: 'runtime',
    preview: null
  }
};
```

Step C: implement real crop.

```js
capabilities: { crop: true }
async createCropArtifact({ sessionId, startKeys, endKeys }) {
  const session = getSession(sessionId);
  const croppedGraph = cropGraph(session.graph, startKeys, endKeys);
  return makeArtifact(session, croppedGraph);
}
getCropTarget(artifactId) {
  return buildCropTargetFromCoreGraph({ providerId: 'my-ir', model, artifact, graph: artifact.graph });
}
```

Manual check: after Confirm Crop, the displayed graph must become smaller when the selected path is smaller.

Step D: add text export and AI.

```js
capabilities: { crop: true, textExportContext: true }
buildTextExportContext(artifactId) {
  return buildTextExportContextFromCoreGraph({ providerId: 'my-ir', model, artifact, graph: artifact.graph });
}
```

Manual check: Copy Export Text works with `Graph Edge List`, then AI Analyze works with `Line Count Analysis`.

Step E: add export, inference, and compare.

```js
capabilities: {
  crop: true,
  textExportContext: true,
  exportArtifact: true,
  inference: true,
  compare: true
}
```

Use fake deterministic outputs for the first compare smoke. Replace them with real runtime outputs only after the UI path is stable.

## 4. Model Snapshot Schema

`loadModel()` must return:

```js
{
  id: 'my-ir-session-1',
  format: 'my-ir',
  filePath: '/absolute/path/model.myir.json',
  snapshot: {
    sessionId: 'my-ir-session-1',
    format: 'my-ir',
    fileName: 'model.myir.json',
    filePath: '/absolute/path/model.myir.json',
    graph: {
      name: 'graph name',
      inputs: [{ name: 'input', values: ['input_tensor'] }],
      outputs: [{ name: 'output', values: ['output_tensor'] }],
      nodes: [/* node snapshots */],
      values: {/* tensor name -> value snapshot */}
    }
  }
}
```

Node snapshot:

```js
{
  id: 'conv0',
  name: 'Conv_0',
  type: {
    name: 'Conv',
    module: 'my.ir',
    identifier: 'my.ir.Conv',
    category: 'Layer'
  },
  inputs: [{ name: 'X', values: ['input_tensor'] }],
  outputs: [{ name: 'Y', values: ['output_tensor'] }],
  attributes: [{ name: 'kernel', value: [3, 3] }]
}
```

Tensor value snapshot:

```js
{
  type: {
    dataType: 'float32',
    shape: [1, 3, 224, 224]
  },
  initializer: null
}
```

Initializer tensor snapshot:

```js
{
  type: {
    dataType: 'float32',
    shape: [64, 3, 3, 3]
  },
  initializer: {
    name: 'conv_weight',
    category: 'Initializer',
    type: {
      dataType: 'float32',
      shape: [64, 3, 3, 3]
    },
    location: 'inline',
    preview: null
  }
}
```

Important:

- `snapshot.sessionId` must equal the returned session id. The crop UI depends on it.
- `type.category` controls Netron node colors. Common categories: `Layer`, `Activation`, `Pool`, `Normalization`, `Shape`, `Tensor`, `Transform`, `Data`, `Quantization`, `Attention`, `Constant`, `Control`, `Custom`.
- Use initializer objects, not booleans. A boolean initializer will not display correctly.

## 5. Core Graph Schema

Crop, text export, compare, and artifact export should use this provider-neutral core graph:

```js
{
  id: 'my-ir:graph0',
  name: 'my-ir:graph0',
  inputs: ['input_tensor'],
  outputs: ['output_tensor'],
  nodes: [
    {
      id: 'conv0',
      name: 'Conv_0',
      type: 'Conv',
      domain: 'my.ir',
      inputs: [{ name: 'X', tensor: 'input_tensor' }],
      outputs: [{ name: 'Y', tensor: 'output_tensor' }],
      attributes: { kernel: [3, 3] },
      omittedAttributes: []
    }
  ],
  tensors: [
    {
      name: 'input_tensor',
      dtype: 'float32',
      rawDtype: 'FLOAT',
      shape: [1, 3, 224, 224],
      kind: 'input'
    }
  ]
}
```

Tensor `kind` values should be one of:

- `input`
- `output`
- `activation`
- `initializer`
- `constant`
- `weight`

Do not include raw tensor data in this graph. Shape and dtype are enough.

## 5.1 Parser Template: JavaScript Only

Use this when the private format is JSON or can be parsed directly in Node.js.

```js
const fs = require('fs');
const path = require('path');
const { normalizeCoreGraph, stableHash } = require('./text-export-context');

function readPrivateModel(filePath) {
  const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  const graph = raw.graph || {};
  const tensors = (graph.tensors || []).map((tensor) => ({
    name: String(tensor.name),
    dtype: String(tensor.dtype || 'float32').toLowerCase(),
    rawDtype: tensor.rawDtype || tensor.dtype || 'float32',
    shape: Array.isArray(tensor.shape) ? tensor.shape : [],
    kind: tensor.kind || 'activation'
  }));
  const nodes = (graph.nodes || []).map((node, index) => ({
    id: node.id || `node-${index}`,
    name: node.name || node.id || `node-${index}`,
    type: node.type || 'PrivateOp',
    domain: node.domain || 'my.ir',
    category: node.category || 'Custom',
    inputs: (node.inputs || []).map((input) => ({ name: input.name || '', tensor: input.tensor })),
    outputs: (node.outputs || []).map((output) => ({ name: output.name || '', tensor: output.tensor })),
    attributes: node.attributes || {},
    omittedAttributes: []
  }));
  const inputs = graph.inputs || [];
  const outputs = graph.outputs || [];
  const id = graph.id || `my-ir:${path.basename(filePath)}:${stableHash({ inputs, outputs, nodes })}`;
  return normalizeCoreGraph({ id, name: graph.name || id, inputs, outputs, nodes, tensors });
}
```

Then call it from `loadModel()`:

```js
async loadModel(uri) {
  const filePath = uri.fsPath || uri.path || String(uri);
  const graph = readPrivateModel(filePath);
  const sessionId = `my-ir-session-${sessions.size + 1}`;
  const session = {
    id: sessionId,
    format: 'my-ir',
    filePath,
    graph,
    snapshot: {
      sessionId,
      format: 'my-ir',
      fileName: path.basename(filePath),
      filePath,
      graph: graphToSnapshot(graph)
    }
  };
  sessions.set(sessionId, session);
  return session;
}
```

## 5.2 Parser Template: Python Implementation Called From JS

Use this when your private format already has a Python parser, vendor SDK, or runtime package.

Create `lib/my-ir-python/parse_model.py`:

```python
#!/usr/bin/env python3
import json
import os
import sys


def tensor(name, dtype="float32", shape=None, kind="activation"):
    return {
        "name": name,
        "dtype": dtype,
        "rawDtype": dtype,
        "shape": shape or [],
        "kind": kind,
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: parse_model.py <model-path>")

    model_path = sys.argv[1]

    # Replace this block with your real parser:
    #   import my_private_runtime
    #   parsed = my_private_runtime.load(model_path)
    with open(model_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    graph = raw.get("graph", {})
    result = {
        "id": graph.get("id") or "my-ir:" + os.path.basename(model_path),
        "name": graph.get("name") or os.path.basename(model_path),
        "inputs": graph.get("inputs", ["input"]),
        "outputs": graph.get("outputs", ["output"]),
        "nodes": graph.get("nodes", []),
        "tensors": graph.get("tensors", [
            tensor("input", shape=[1, 3, 224, 224], kind="input"),
            tensor("output", shape=[1, 1000], kind="output"),
        ]),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

Create a JS helper in `lib/my-ir-python-bridge.js`:

```js
const path = require('path');
const { spawnFileJson } = require('./my-ir-spawn');
const { normalizeCoreGraph } = require('./text-export-context');

async function parseWithPython(filePath) {
  const python = process.env.MY_IR_PYTHON || '/opt/my-ir/venv/bin/python';
  const script = process.env.MY_IR_PARSE_SCRIPT
    || path.join(__dirname, 'my-ir-python', 'parse_model.py');
  const graph = await spawnFileJson(python, [script, filePath], {
    env: {
      MY_IR_HOME: process.env.MY_IR_HOME || '/opt/my-ir',
      PYTHONPATH: process.env.MY_IR_PYTHONPATH || ''
    },
    timeoutMs: 120000
  });
  return normalizeCoreGraph(graph);
}

module.exports = { parseWithPython };
```

Create the reusable process helper `lib/my-ir-spawn.js`:

```js
const { spawn } = require('child_process');

function spawnFileJson(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    const child = spawn(command, args, {
      cwd: options.cwd || process.cwd(),
      shell: false,
      env: {
        ...process.env,
        ...(options.env || {})
      }
    });
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      const error = new Error(`${command} timed out after ${options.timeoutMs || 120000}ms`);
      error.stderr = stderr;
      reject(error);
    }, options.timeoutMs || 120000);
    child.stdout.on('data', (chunk) => stdout += chunk.toString('utf8'));
    child.stderr.on('data', (chunk) => stderr += chunk.toString('utf8'));
    child.on('error', (error) => {
      clearTimeout(timer);
      error.stderr = stderr;
      reject(error);
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        const error = new Error(`${command} exited with code ${code}`);
        error.stdout = stdout;
        error.stderr = stderr;
        reject(error);
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        error.message = `Failed to parse JSON from ${command}: ${error.message}`;
        error.stdout = stdout.slice(0, 2000);
        error.stderr = stderr.slice(0, 2000);
        reject(error);
      }
    });
  });
}

module.exports = { spawnFileJson };
```

Use the Python bridge in the provider:

```js
const { parseWithPython } = require('./my-ir-python-bridge');

async loadModel(uri) {
  const filePath = uri.fsPath || uri.path || String(uri);
  const graph = await parseWithPython(filePath);
  // Then build session + snapshot exactly like the JS-only path.
}
```

Test Python directly before testing VS Code:

```bash
/opt/my-ir/venv/bin/python lib/my-ir-python/parse_model.py examples/private-ir/sample.myir.json
MY_IR_PYTHON=/opt/my-ir/venv/bin/python node -e "require('./lib/my-ir-python-bridge').parseWithPython('examples/private-ir/sample.myir.json').then(console.log)"
```

## 6. Crop Implementation

`createCropArtifact({ sessionId, startKeys, endKeys })` must return a real subgraph, not the full graph.

Algorithm:

1. Build an index:
   - tensor -> producer node
   - tensor -> consumer nodes
2. Start nodes are consumers of all selected start tensors.
3. End nodes are producers of all selected end tensors.
4. Walk forward from start nodes.
5. Walk backward from end nodes.
6. Selected crop nodes are the intersection.
7. Crop inputs are selected-node inputs whose producer is outside the crop and are not initializers.
8. Crop outputs are selected-node outputs that have no inside consumer, have outside consumers, are selected end tensors, or are original graph outputs.
9. Keep initializer tensors used by selected nodes.

Return artifact:

```js
{
  id: 'my-ir-artifact-1',
  modelSessionId: 'my-ir-session-1',
  createdAt: new Date().toISOString(),
  stale: false,
  graph: croppedCoreGraph,
  inputKeys: croppedCoreGraph.inputs.slice(),
  outputKeys: croppedCoreGraph.outputs.slice(),
  ioSignature: {
    inputs: [{ name: 'input_tensor', dtype: 'float32', rank: 4, shape: [1, 3, 224, 224], optional: false }],
    outputs: [{ name: 'output_tensor', dtype: 'float32', rank: 4, shape: [1, 64, 222, 222], optional: false }]
  },
  summary: {
    modelName: 'model.myir.json',
    graphName: croppedCoreGraph.name,
    nodeCount: croppedCoreGraph.nodes.length,
    inputCount: croppedCoreGraph.inputs.length,
    outputCount: croppedCoreGraph.outputs.length
  },
  cropGraphSnapshot: graphToSnapshot(croppedCoreGraph)
}
```

If crop appears to do nothing, inspect `createCropArtifact()` first. The UI only sends `startKeys/endKeys`; the provider is responsible for creating the real subgraph.

## 7. Text Export Context

Use helper functions from `lib/text-export-context.js`.

```js
const {
  buildCropTargetFromCoreGraph,
  buildTextExportContextFromCoreGraph
} = require('./text-export-context');
```

`getCropTarget(artifactId)`:

```js
return buildCropTargetFromCoreGraph({
  providerId: 'my-ir',
  model: {
    format: 'my-ir',
    fileName: 'model.myir.json',
    filePath: '/absolute/path/model.myir.json'
  },
  artifact,
  graph: artifact.graph
});
```

`buildTextExportContext(artifactId)`:

```js
return buildTextExportContextFromCoreGraph({
  providerId: 'my-ir',
  model,
  artifact,
  graph: artifact.graph
});
```

Exporter scripts receive this JSON on stdin:

```js
{
  kind: 'text-export-context',
  schemaVersion: 1,
  target: 'crop',
  model: { format, fileName, filePath },
  artifact: { id, createdAt },
  graph: coreGraph
}
```

Exporter scripts must write text to stdout. Empty stdout is treated as failure.

## 8. Artifact Export

`getExportTarget()` tells the host how to show the save dialog:

```js
getExportTarget(artifactId) {
  return {
    artifactId,
    defaultFileName: `${artifactId}.myir.json`,
    filters: { 'My IR': ['json'] },
    title: 'Export My IR Crop',
    stage: 'Export My IR crop',
    message: 'Exporting My IR crop...',
    options: {}
  };
}
```

`exportArtifact()` writes the real exported model:

```js
async exportArtifact(artifactId, targetPath, options = {}) {
  const artifact = getArtifact(artifactId);
  fs.writeFileSync(targetPath, JSON.stringify({
    kind: 'my-ir',
    graph: artifact.graph
  }, null, 2));
  return { filePath: targetPath, artifactId, providerId: 'my-ir' };
}
```

For production, this must write a model that can run in your private runtime. JSON graph export is only a test format unless your runtime accepts it.

## 9. Inference And Compare

Compare is provider-neutral. It only requires both sides to expose compatible IO signatures and numeric output summaries.

`getCompareSlot(artifactId)`:

```js
return {
  providerId: 'my-ir',
  artifactId,
  modelSessionId: artifact.modelSessionId,
  ioSignature: artifact.ioSignature,
  summary: artifact.summary,
  createdAt: artifact.createdAt
};
```

`runCompareArtifact({ artifactId, input })`:

```js
return {
  outputsSummary: [
    {
      name: 'output_tensor',
      dtype: 'float32',
      shape: [1, 4],
      values: [0.1, 0.2, 0.3, 0.4],
      preview: {
        elementCount: 4,
        sampleCount: 4,
        sampleValues: [0.1, 0.2, 0.3, 0.4],
        truncated: false
      },
      summary: { min: 0.1, max: 0.4, mean: 0.25 }
    }
  ]
};
```

`values` must be an array of finite numbers for normal numeric diff. If dtype or shape differs, the compare row will not be comparable.

`runInference(options)` may call the same runtime code as `runCompareArtifact()`.

For early testing, fake deterministic outputs are acceptable. For real validation against ONNX or another provider, use the real private runtime.

## 10. Input Import

Input import is optional. Declare it only if implemented:

```js
capabilities: {
  inputImport: true
}
```

Required methods:

```js
async importInputFile(filePath) {
  return {
    token: 'my-ir-input-1',
    preview: [{ name: 'input_tensor', dtype: 'float32', shape: [1, 3, 224, 224] }]
  };
}

resolveImportedInput(token) {
  return parsedInputObjectOrNull;
}
```

The compare engine passes imported input data back to the provider runtime. Keep large input data out of UI preview.

## 11. Exporter And Analyzer Tools

Tools are not registered in code. They are discovered from global directories:

```text
~/.netron/vscode-preview/exporters/<tool-name>/exporter.json
~/.netron/vscode-preview/analyzers/<tool-name>/analyzer.json
```

Manifest schema:

```json
{
  "id": "graph-edge-list",
  "label": "Graph Edge List",
  "command": "node",
  "args": ["graph-edge-list.js"],
  "timeoutMs": 30000,
  "env": {
    "OPTIONAL_NAME": "optional value"
  }
}
```

Rules:

- `id`, `label`, and `command` are required.
- `args` must be an array of strings.
- `env` must be an object whose values are strings.
- Duplicate ids are disabled and shown as conflicts.
- Invalid manifests are shown in the UI as disabled entries with reasons.
- The registry watches the directories and updates the dropdowns after file changes.

Minimal Node exporter:

```js
#!/usr/bin/env node
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  const context = JSON.parse(input);
  const graph = context.graph || {};
  const lines = [];
  for (const node of graph.nodes || []) {
    for (const port of node.inputs || []) {
      lines.push(`${port.tensor} -> ${node.name || node.id}`);
    }
    for (const port of node.outputs || []) {
      lines.push(`${node.name || node.id} -> ${port.tensor}`);
    }
  }
  process.stdout.write(lines.join('\n') || '(empty graph)');
});
```

Minimal Python analyzer:

```python
#!/usr/bin/env python3
import os
import sys

text = sys.stdin.read()
if not text.strip():
    raise SystemExit("empty analyzer input")

debug = os.environ.get("MY_IR_DEBUG") == "1"
if debug:
    print(f"[analyzer] bytes={len(text.encode('utf-8'))}", file=sys.stderr)

lines = [line for line in text.splitlines() if line.strip()]
print("Python Analysis Result")
print(f"Input lines: {len(lines)}")
print(f"First line: {lines[0] if lines else '(empty)'}")
```

Manifest for that Python analyzer:

```json
{
  "id": "my-python-analysis",
  "label": "My Python Analysis",
  "command": "/bin/bash",
  "args": ["run-python-analysis.sh"],
  "timeoutMs": 180000,
  "env": {
    "MY_IR_DEBUG": "1",
    "MY_IR_PYTHON": "/opt/my-ir/venv/bin/python"
  }
}
```

`run-python-analysis.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/opt/my-ir/python:${PYTHONPATH:-}"
exec "${MY_IR_PYTHON:-/opt/my-ir/venv/bin/python}" my_python_analysis.py
```

Install examples:

```bash
mkdir -p ~/.netron/vscode-preview/exporters
mkdir -p ~/.netron/vscode-preview/analyzers
cp -R examples/tools/exporters/crop-json-summary ~/.netron/vscode-preview/exporters/
cp -R examples/tools/exporters/graph-edge-list ~/.netron/vscode-preview/exporters/
cp -R examples/tools/analyzers/line-count-analysis ~/.netron/vscode-preview/analyzers/
cp -R examples/tools/analyzers/deepseek-graph-analysis ~/.netron/vscode-preview/analyzers/
cp -R examples/tools/analyzers/codex-one-shot-analysis ~/.netron/vscode-preview/analyzers/
```

Configure DeepSeek without storing secrets in this repository:

```bash
mkdir -p ~/.netron/vscode-preview/secrets
chmod 700 ~/.netron/vscode-preview/secrets
printf '%s' 'YOUR_DEEPSEEK_KEY' > ~/.netron/vscode-preview/secrets/deepseek_api_key
chmod 600 ~/.netron/vscode-preview/secrets/deepseek_api_key
```

Alternative:

```bash
export DEEPSEEK_API_KEY=<your-key-from-secret-manager>
```

## 11.1 Environment Variables In Provider Code

Provider code runs in the VS Code extension host. Read environment variables with `process.env`.

```js
const runtimeHome = process.env.MY_IR_HOME || '/opt/my-ir';
const python = process.env.MY_IR_PYTHON || `${runtimeHome}/venv/bin/python`;
const debug = process.env.MY_IR_DEBUG === '1';

if (debug) {
  console.error('[my-ir] runtimeHome=', runtimeHome);
  console.error('[my-ir] python=', python);
}
```

Use env values to choose runtime paths, not to store large data:

```js
const vendorLibPath = process.env.MY_IR_VENDOR_LIB || '/opt/my-ir/lib/libmyir.so';
```

When you need secrets, prefer a file under the user home:

```js
const fs = require('fs');
const os = require('os');
const path = require('path');

function readSecret() {
  if (process.env.MY_IR_TOKEN) {
    return process.env.MY_IR_TOKEN.trim();
  }
  const filePath = process.env.MY_IR_TOKEN_FILE
    || path.join(os.homedir(), '.netron', 'vscode-preview', 'secrets', 'my_ir_token');
  return fs.readFileSync(filePath, 'utf8').trim();
}
```

## 11.2 Environment Variables In Exporter/Analyzer Scripts

`exporter.json` and `analyzer.json` can inject env values:

```json
{
  "id": "my-analyzer",
  "label": "My Analyzer",
  "command": "/bin/bash",
  "args": ["run-analyzer.sh"],
  "timeoutMs": 180000,
  "env": {
    "MY_IR_HOME": "/opt/my-ir",
    "MY_IR_PYTHON": "/opt/my-ir/venv/bin/python",
    "MY_IR_DEBUG": "1"
  }
}
```

Node script:

```js
const home = process.env.MY_IR_HOME || '/opt/my-ir';
const debug = process.env.MY_IR_DEBUG === '1';
```

Python script:

```python
import os

home = os.environ.get("MY_IR_HOME", "/opt/my-ir")
debug = os.environ.get("MY_IR_DEBUG") == "1"
```

Bash wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail
export MY_IR_HOME="${MY_IR_HOME:-/opt/my-ir}"
export PYTHONPATH="${MY_IR_HOME}/python:${PYTHONPATH:-}"
exec "${MY_IR_PYTHON:-${MY_IR_HOME}/venv/bin/python}" analyze.py
```

Do not put real API keys in the repository. Use `env`, shell exports, or secret files created outside git.

## 11.3 Dependency Resolution Rules

Use absolute paths for anything not guaranteed by VS Code:

```json
{
  "command": "/bin/bash",
  "args": ["run-analyzer.sh"]
}
```

Do not rely on `python`, `node`, `codex`, or vendor CLIs being in PATH unless you verified the VS Code extension host sees the same PATH.

Provider JS dependencies:

```bash
npm install --save <runtime-js-package>
```

Then import them from provider code:

```js
const runtime = require('<runtime-js-package>');
```

If the package is private or machine-local, keep it outside the extension bundle and load it through an absolute path:

```js
const runtimePath = process.env.MY_IR_NODE_RUNTIME || '/opt/my-ir/node-runtime';
const runtime = require(runtimePath);
```

Python dependencies:

```bash
python3 -m venv /opt/my-ir/venv
/opt/my-ir/venv/bin/pip install -r /opt/my-ir/requirements.txt
```

Call that exact interpreter:

```js
const python = process.env.MY_IR_PYTHON || '/opt/my-ir/venv/bin/python';
```

Conda dependencies:

```bash
#!/usr/bin/env bash
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate my-ir
exec python analyze.py
```

Native/CUDA/vendor libraries:

```bash
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="/opt/my-ir/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec /opt/my-ir/venv/bin/python run_runtime.py
```

Recommended rule: make every provider/runtime/script command pass when copied into a plain SSH shell first. Then point the manifest/provider to the same absolute commands.

## 12. Remote SSH Environment

When VS Code is connected through Remote SSH, the extension host runs on the remote server.

Implications:

- Provider code runs on the remote server.
- Exporter/analyzer scripts run on the remote server.
- `~/.netron/vscode-preview` means the remote user's home directory.
- Private runtime packages must be installed on the remote server.
- If your provider needs Python/CUDA/vendor libraries, use absolute commands or wrapper scripts that activate the right environment.

Recommended analyzer/exporter wrapper style:

```json
{
  "id": "my-analyzer",
  "label": "My Analyzer",
  "command": "/bin/bash",
  "args": ["run-analyzer.sh"],
  "timeoutMs": 180000
}
```

`run-analyzer.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source /opt/my-env/bin/activate
python3 analyze.py
```

The tool receives stdin and must write stdout.

## 13. Testing Checklist

Run these after changing provider code:

```bash
node --check lib/my-ir-provider.js
node --check extension.js
npm run smoke:extension-provider-api
npm run smoke:provider-registry
npm run smoke:private-provider-contract
npm run smoke:all
```

If you used the examples in this guide:

```bash
npm run smoke:private-ir-guide-examples
```

Manual UI test:

1. Package and install the extension.
2. Reload VS Code.
3. Open a private IR sample file.
4. Confirm nodes have category colors.
5. Confirm initializer tensors display as initializers.
6. Select start and end tensor edges.
7. Confirm Crop.
8. Verify the graph changes to the cropped subgraph.
9. Copy Export Text with `Graph Edge List`.
10. Run AI Analyze with `Graph Edge List` + `Line Count Analysis`.
11. Set private crop as Compare A.
12. Set another compatible crop as Compare B.
13. Run Compare and inspect binding/output rows.

## 14. Common Failures

## 14.0 Where To Debug

Use this table before changing code.

| Symptom | Most likely component | What to debug first |
|---|---|---|
| File does not open from Explorer | Provider registration or `canOpen()` | Confirm provider is registered in `extension.js`; log the file path seen by `canOpen()` |
| Blank or tiny graph opens | `loadModel()` snapshot | Print returned `snapshot.graph.nodes.length`, inputs, outputs, and values |
| Nodes have wrong colors | Render snapshot | Check `node.type.category` in `graphToSnapshot()` |
| Initializers do not display | Render snapshot values | Check `values[name].initializer` is an object, not `true` |
| Confirm Crop does nothing | Provider crop implementation | Debug `createCropArtifact()`, `cropGraphSnapshot`, and artifact `graph.nodes.length` |
| Copy Export Text disabled | Provider capability or artifact state | Check `capabilities.textExportContext`, confirmed crop, and `buildTextExportContext()` |
| Exporter fails | User exporter script | Run the exporter manually with saved stdin JSON |
| Analyzer exits with code 1 | User analyzer script | Run the analyzer manually with the exact exporter stdout |
| Compare cannot bind | Provider `ioSignature` | Print both slots' input/output name, dtype, rank, shape |
| Compare runs but values are nonsense | Runtime script | Confirm `runCompareArtifact()` returns real numeric values from the intended runtime |
| Works locally but not Remote SSH | Environment/dependencies | Check the remote server has the same files, venv, secrets, PATH, and library paths |

Provider debug stub:

```js
function debugLog(...args) {
  if (process.env.MY_IR_DEBUG === '1') {
    console.error('[my-ir]', ...args);
  }
}

async loadModel(uri) {
  const filePath = uri.fsPath || uri.path || String(uri);
  debugLog('loadModel', filePath);
  const graph = await parseWithPython(filePath);
  debugLog('graph', {
    inputs: graph.inputs,
    outputs: graph.outputs,
    nodes: graph.nodes.length,
    tensors: graph.tensors.length
  });
  // build session...
}
```

Exporter/analyzer debug stub:

```js
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  if (process.env.MY_IR_DEBUG === '1') {
    console.error('[tool] stdin bytes:', Buffer.byteLength(input));
    console.error('[tool] cwd:', process.cwd());
  }
  process.stdout.write('debug ok');
});
```

Python runtime debug stub:

```python
import json
import os
import sys

def debug(*items):
    if os.environ.get("MY_IR_DEBUG") == "1":
        print("[my-ir-python]", *items, file=sys.stderr)

debug("python", sys.executable)
debug("cwd", os.getcwd())
debug("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
debug("argv", sys.argv)

print(json.dumps({"ok": True}))
```

Manual script replay:

```bash
# 1. Save provider export context from Copy Export Text or a smoke fixture.
node examples/tools/exporters/graph-edge-list/graph-edge-list.js < /tmp/context.json > /tmp/graph.txt

# 2. Replay analyzer exactly like the plugin does.
MY_IR_DEBUG=1 node examples/tools/analyzers/line-count-analysis/line-count-analysis.js < /tmp/graph.txt
```

Crop button is disabled:

- `snapshot.sessionId` is missing or does not match the session id.
- Provider did not declare `capabilities.crop: true`.
- Provider declared crop but is missing `createCropArtifact()` or `getCropTarget()`.
- No start/end tensor edge has been selected.

Crop confirms but graph does not change:

- `createCropArtifact()` is returning the full graph.
- `cropGraphSnapshot` is built from the full graph instead of the cropped graph.
- `inputKeys/outputKeys` are not updated to cropped boundaries.

Nodes are all the same color:

- `node.type.category` is missing in the render snapshot.
- Use categories such as `Layer`, `Activation`, `Transform`, or `Custom`.

Constants or weights do not display:

- Tensor value `initializer` is `true` or `false`.
- It must be an object with `name`, `category`, `type`, `location`, and `preview`.

Copy Export Text is unavailable:

- No confirmed crop exists.
- Crop is stale after selection changed.
- Provider did not declare `textExportContext: true`.
- Provider is missing `buildTextExportContext()`.
- No ready exporter exists under `~/.netron/vscode-preview/exporters`.

Analyzer is unavailable:

- No ready analyzer exists under `~/.netron/vscode-preview/analyzers`.
- Manifest has invalid JSON.
- Duplicate analyzer ids exist.
- Environment variables in manifest are not strings.

Codex One-Shot Analysis fails:

- `codex` is not installed or is not in PATH for the VS Code extension host.
- Codex is not logged in on the same local or Remote SSH machine.
- The analyzer timed out while `codex exec` was running.
- Set `CODEX_COMMAND`, `CODEX_MODEL`, `CODEX_WORKDIR`, or `CODEX_EXTRA_ARGS` in `analyzer.json` if your environment needs a wrapper or a specific profile.

Compare cannot bind:

- Input/output names differ and shapes differ.
- Dtype differs.
- Rank differs.
- `ioSignature` does not match the cropped graph boundaries.

Compare runs but numeric diff is meaningless:

- One side uses fake runtime outputs.
- Use real runtime outputs for correctness testing.

## 15. Completion Definition

Private IR integration is complete only when all items are true:

- Opening a private IR file renders a graph.
- Node colors match categories.
- Initializers display as initializers.
- Crop creates a smaller graph when boundaries select a subpath.
- Export Crop writes a real runnable model artifact or a clearly labeled test artifact.
- Copy Export Text works with at least one exporter.
- AI Analyze works with at least one formatter/analyzer pipeline.
- Run Inference returns output summaries.
- Compare A/B works against another compatible provider.
- Smoke tests pass.
- VSIX is packaged and installed.
- Manual VS Code reload test passes.
