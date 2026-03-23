#!/usr/bin/env node
import fs from 'node:fs';
import process from 'node:process';
import { layout } from '../netron/source/dagre.js';

function normalizeAdjacency(adjacency) {
  if (!Array.isArray(adjacency)) {
    throw new Error('adjacency must be an array');
  }
  return adjacency.map((targets) => Array.isArray(targets) ? targets.map((value) => Number(value)) : []);
}

function buildGraphInput(adjacency, nodeHeights) {
  const nodes = adjacency.map((_, index) => ({
    v: String(index),
    width: 1,
    height: Array.isArray(nodeHeights) && nodeHeights[index] !== undefined ? Number(nodeHeights[index]) : 1,
  }));
  const edges = [];
  for (let from = 0; from < adjacency.length; from++) {
    for (const to of adjacency[from]) {
      if (from === to) {
        continue;
      }
      edges.push({
        v: String(from),
        w: String(to),
        minlen: 1,
        weight: 1,
        width: 0,
        height: 0,
        labeloffset: 10,
        labelpos: 'r',
      });
    }
  }
  return { nodes, edges };
}

function buildRanksFromY(nodes) {
  const sortedY = Array.from(new Set(nodes.map((node) => Number(node.y)).sort((a, b) => a - b)));
  const yToRank = new Map(sortedY.map((value, index) => [value, index]));
  const ranks = new Array(nodes.length).fill(0);
  const layers = Array.from({ length: sortedY.length }, () => []);
  for (const node of nodes) {
    const index = Number(node.v);
    const rank = yToRank.get(Number(node.y));
    ranks[index] = rank;
    layers[rank].push(index);
  }
  return {
    ranks,
    layers: layers.map((layer) => layer.sort((a, b) => a - b)),
    y: nodes.map((node) => Number(node.y)),
  };
}

function main() {
  const inputPath = process.argv[2];
  if (!inputPath) {
    throw new Error('usage: netron_height_order_js.mjs <input.json>');
  }
  const payload = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const adjacency = normalizeAdjacency(payload.adjacency);
  const nodeHeights = Array.isArray(payload.node_heights) ? payload.node_heights : null;
  const ranker = payload.ranker || (adjacency.length > (payload.large_graph_threshold || 3000) ? 'longest-path' : 'network-simplex');
  const { nodes, edges } = buildGraphInput(adjacency, nodeHeights);
  const start = process.hrtime.bigint();
  layout(nodes, edges, { rankdir: 'TB', ranksep: payload.ranksep || 20, nodesep: 20, ranker }, {});
  const elapsedMs = Number(process.hrtime.bigint() - start) / 1e6;
  const result = buildRanksFromY(nodes);
  process.stdout.write(JSON.stringify({
    ranker,
    elapsed_ms: elapsedMs,
    node_count: nodes.length,
    edge_count: edges.length,
    ...result,
  }));
}

main();
