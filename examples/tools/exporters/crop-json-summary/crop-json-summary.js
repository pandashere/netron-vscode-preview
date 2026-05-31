#!/usr/bin/env node
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  input += chunk;
});
process.stdin.on('end', () => {
  try {
    const context = JSON.parse(input || '{}');
    const graph = context.graph || {};
    const lines = [
      'Model: ' + ((context.model && context.model.fileName) || '(unknown)'),
      'Format: ' + ((context.model && context.model.format) || '(unknown)'),
      'Artifact: ' + ((context.artifact && context.artifact.id) || '(none)'),
      'Graph: ' + (graph.id || '(none)'),
      'Inputs: ' + (Array.isArray(graph.inputs) ? graph.inputs.join(', ') : ''),
      'Outputs: ' + (Array.isArray(graph.outputs) ? graph.outputs.join(', ') : ''),
      'Nodes: ' + (Array.isArray(graph.nodes) ? graph.nodes.length : 0),
      'Tensors: ' + (Array.isArray(graph.tensors) ? graph.tensors.length : 0)
    ];
    process.stdout.write(lines.join('\n'));
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
  }
});
