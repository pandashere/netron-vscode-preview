#!/usr/bin/env node
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
    edges.push(producer
      ? edgeLine(portRef(producer.node, producer.port), boundaryRef('GRAPH_OUTPUT', outputName), outputName, tensors)
      : edgeLine(boundaryRef('UNKNOWN_SOURCE', outputName), boundaryRef('GRAPH_OUTPUT', outputName), outputName, tensors));
  }

  return [
    '# Netron Crop Graph Edge List',
    'model.format=' + ((context.model && context.model.format) || ''),
    'model.fileName=' + ((context.model && context.model.fileName) || ''),
    'artifact.id=' + ((context.artifact && context.artifact.id) || ''),
    'graph.id=' + (graph.id || ''),
    '',
    '[inputs]',
    ...Array.from(graphInputs).map((name) => name + ' | ' + tensorLabel(tensors.get(name))),
    '',
    '[outputs]',
    ...Array.from(graphOutputs).map((name) => name + ' | ' + tensorLabel(tensors.get(name))),
    '',
    '[nodes]',
    ...nodes.map((node) => [nodeRef(node), node.type || '', node.domain || '', node.name || nodeRef(node)].join(' | ')),
    '',
    '[edges]',
    ...(edges.length > 0 ? edges : ['(no edges)'])
  ].join('\n');
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  input += chunk;
});
process.stdin.on('end', () => {
  try {
    process.stdout.write(buildEdgeList(JSON.parse(input || '{}')));
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
  }
});
