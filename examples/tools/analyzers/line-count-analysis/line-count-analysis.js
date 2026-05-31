#!/usr/bin/env node
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  input += chunk;
});
process.stdin.on('end', () => {
  const lines = input.split(/\r?\n/).filter((line) => line.trim().length > 0);
  process.stdout.write([
    'Analysis Result',
    'Input lines: ' + lines.length,
    'First line: ' + (lines[0] || '(empty)')
  ].join('\n'));
});
