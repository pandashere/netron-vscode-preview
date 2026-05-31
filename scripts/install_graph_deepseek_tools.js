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

const edgeListExporterScript = `#!/usr/bin/env node
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function valueText(value) {
  if (value === null || value === undefined) {
    return '?';
  }
  if (Array.isArray(value)) {
    return '[' + value.map(valueText).join(',') + ']';
  }
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}

function tensorLabel(tensor) {
  if (!tensor) {
    return 'unknown';
  }
  const shape = tensor.shape === null ? '?' : valueText(tensor.shape);
  return (tensor.dtype || tensor.rawDtype || '?') + shape;
}

function nodeRef(node) {
  return node.id || node.name || '(unnamed-node)';
}

function portRef(node, port) {
  return nodeRef(node) + '.' + (port.name || '?');
}

function boundaryRef(kind, name) {
  return kind + '[' + name + ']';
}

function edgeLine(from, to, tensor, tensors) {
  const meta = tensors.get(tensor) || { name: tensor };
  return from + ' -> ' + to + ' | tensor=' + tensor + ' | ' + tensorLabel(meta);
}

function buildEdgeList(context) {
  const graph = context.graph || {};
  const nodes = asArray(graph.nodes);
  const graphInputs = new Set(asArray(graph.inputs));
  const graphOutputs = new Set(asArray(graph.outputs));
  const tensors = new Map(asArray(graph.tensors).map((tensor) => [tensor.name, tensor]));
  const producers = new Map();
  const edges = [];

  for (const node of nodes) {
    for (const output of asArray(node.outputs)) {
      if (output && output.tensor) {
        producers.set(output.tensor, { node, port: output });
      }
    }
  }

  for (const node of nodes) {
    for (const input of asArray(node.inputs)) {
      if (!input || !input.tensor) {
        continue;
      }
      const tensorName = input.tensor;
      const producer = producers.get(tensorName);
      if (producer) {
        edges.push(edgeLine(portRef(producer.node, producer.port), portRef(node, input), tensorName, tensors));
        continue;
      }
      const tensor = tensors.get(tensorName) || {};
      const kind = String(tensor.kind || '').toLowerCase();
      const source = graphInputs.has(tensorName)
        ? boundaryRef('GRAPH_INPUT', tensorName)
        : /initializer|constant|weight/.test(kind)
          ? boundaryRef('CONST', tensorName)
          : boundaryRef('UNKNOWN_SOURCE', tensorName);
      edges.push(edgeLine(source, portRef(node, input), tensorName, tensors));
    }
  }

  for (const outputName of graphOutputs) {
    const producer = producers.get(outputName);
    if (producer) {
      edges.push(edgeLine(portRef(producer.node, producer.port), boundaryRef('GRAPH_OUTPUT', outputName), outputName, tensors));
    } else {
      edges.push(edgeLine(boundaryRef('UNKNOWN_SOURCE', outputName), boundaryRef('GRAPH_OUTPUT', outputName), outputName, tensors));
    }
  }

  const lines = [
    '# Netron Crop Graph Edge List',
    'model.format=' + ((context.model && context.model.format) || ''),
    'model.fileName=' + ((context.model && context.model.fileName) || ''),
    'model.filePath=' + ((context.model && context.model.filePath) || ''),
    'artifact.id=' + ((context.artifact && context.artifact.id) || ''),
    'artifact.createdAt=' + ((context.artifact && context.artifact.createdAt) || ''),
    'graph.id=' + (graph.id || ''),
    'graph.name=' + (graph.name || graph.id || ''),
    '',
    '[inputs]',
    ...Array.from(graphInputs).map((name) => name + ' | ' + tensorLabel(tensors.get(name))),
    '',
    '[outputs]',
    ...Array.from(graphOutputs).map((name) => name + ' | ' + tensorLabel(tensors.get(name))),
    '',
    '[nodes]',
    ...nodes.map((node) => [
      nodeRef(node),
      node.type || '',
      node.domain || '',
      node.name || nodeRef(node)
    ].join(' | ')),
    '',
    '[edges]',
    ...(edges.length > 0 ? edges : ['(no edges)'])
  ];
  return lines.join('\\n');
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  try {
    const context = JSON.parse(input || '{}');
    process.stdout.write(buildEdgeList(context));
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
  }
});
`;

const deepseekAnalyzerScript = `#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const https = require('https');

function readStdin() {
  return new Promise((resolve) => {
    let input = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => input += chunk);
    process.stdin.on('end', () => resolve(input));
  });
}

function readApiKey() {
  if (process.env.DEEPSEEK_API_KEY) {
    return process.env.DEEPSEEK_API_KEY.trim();
  }
  const keyFile = process.env.DEEPSEEK_API_KEY_FILE
    || path.join(os.homedir(), '.netron', 'vscode-preview', 'secrets', 'deepseek_api_key');
  try {
    return fs.readFileSync(keyFile, 'utf8').trim();
  } catch {
    return '';
  }
}

function requestJson(urlText, apiKey, payload, timeoutMs) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlText);
    const body = JSON.stringify(payload);
    const transport = url.protocol === 'http:' ? http : https;
    const request = transport.request({
      method: 'POST',
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || undefined,
      path: url.pathname + url.search,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey,
        'Content-Length': Buffer.byteLength(body)
      },
      timeout: timeoutMs
    }, (response) => {
      let responseBody = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => responseBody += chunk);
      response.on('end', () => {
        let parsed = null;
        try {
          parsed = responseBody ? JSON.parse(responseBody) : null;
        } catch {
          // keep raw response body
        }
        if (response.statusCode < 200 || response.statusCode >= 300) {
          const error = new Error('DeepSeek request failed with HTTP ' + response.statusCode + '.');
          error.responseBody = responseBody.slice(0, 2000);
          reject(error);
          return;
        }
        resolve(parsed);
      });
    });
    request.on('timeout', () => {
      request.destroy(new Error('DeepSeek request timed out.'));
    });
    request.on('error', reject);
    request.end(body);
  });
}

async function main() {
  const graphText = await readStdin();
  if (!graphText.trim()) {
    throw new Error('Analyzer input is empty.');
  }
  const apiKey = readApiKey();
  if (!apiKey) {
    throw new Error('DEEPSEEK_API_KEY is required, or create ~/.netron/vscode-preview/secrets/deepseek_api_key.');
  }
  const baseUrl = (process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com').replace(/\\/+$/, '');
  const model = process.env.DEEPSEEK_MODEL || 'deepseek-v4-flash';
  const timeoutMs = Number(process.env.DEEPSEEK_TIMEOUT_MS || 120000);
  const result = await requestJson(baseUrl + '/chat/completions', apiKey, {
    model,
    messages: [
      {
        role: 'system',
        content: [
          '你是神经网络图结构分析助手。',
          '输入是 Netron crop 图的 edge-list 文本。',
          '请用中文输出，重点分析主干路径、分支/汇合、输入输出、可疑断点、可并行区域、以及对后续推理/裁剪/比较的建议。',
          '不要输出 Markdown 表格；保持短小、可直接阅读。'
        ].join('\\n')
      },
      {
        role: 'user',
        content: graphText
      }
    ],
    temperature: 0.2,
    max_tokens: 2048,
    stream: false
  }, Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 120000);
  const text = result && result.choices && result.choices[0] && result.choices[0].message
    ? result.choices[0].message.content
    : '';
  if (!text || !text.trim()) {
    throw new Error('DeepSeek returned an empty analysis result.');
  }
  process.stdout.write(text.trim());
}

main().catch((error) => {
  console.error(error && error.message ? error.message : String(error));
  if (error && error.responseBody) {
    console.error(error.responseBody);
  }
  process.exit(1);
});
`;

function installGraphDeepseekTools(root) {
    const exporterDir = installTool(root, 'exporter', 'graph-edge-list', {
        id: 'graph-edge-list',
        label: 'Graph Edge List',
        timeoutMs: 30000
    }, edgeListExporterScript);

    const analyzerDir = installTool(root, 'analyzer', 'deepseek-graph-analysis', {
        id: 'deepseek-graph-analysis',
        label: 'DeepSeek Graph Analysis',
        timeoutMs: 180000,
        env: {
            DEEPSEEK_MODEL: 'deepseek-v4-flash',
            DEEPSEEK_BASE_URL: 'https://api.deepseek.com'
        }
    }, deepseekAnalyzerScript);

    return {
        root,
        exporterDir,
        analyzerDir,
        keyFile: path.join(root, 'secrets', 'deepseek_api_key')
    };
}

function main() {
    const options = parseArgs(process.argv.slice(2));
    const result = installGraphDeepseekTools(options.root);
    console.log('Graph DeepSeek tools installed', result);
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
    installGraphDeepseekTools,
    parseArgs
};
