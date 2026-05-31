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

The DeepSeek example is desensitized. It contains no API key. It reads the key from `DEEPSEEK_API_KEY` or `~/.netron/vscode-preview/secrets/deepseek_api_key`.

## 2. Register A Provider

1. Copy `examples/private-ir/my-ir-provider.js` to `lib/my-ir-provider.js`.

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

5. Run:

```bash
node --check lib/my-ir-provider.js
node --check extension.js
npm run smoke:extension-provider-api
npm run smoke:provider-registry
```

6. Package and install:

```bash
npm run package:vsix
code --install-extension dist/netron-vscode-workbench-0.1.0.vsix --force
```

7. In VS Code, run `Developer: Reload Window`.

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

Install examples:

```bash
mkdir -p ~/.netron/vscode-preview/exporters
mkdir -p ~/.netron/vscode-preview/analyzers
cp -R examples/tools/exporters/crop-json-summary ~/.netron/vscode-preview/exporters/
cp -R examples/tools/exporters/graph-edge-list ~/.netron/vscode-preview/exporters/
cp -R examples/tools/analyzers/line-count-analysis ~/.netron/vscode-preview/analyzers/
cp -R examples/tools/analyzers/deepseek-graph-analysis ~/.netron/vscode-preview/analyzers/
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
