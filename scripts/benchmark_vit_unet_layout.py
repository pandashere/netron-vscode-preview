#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx

from netron_height_order import netron_height_order
from generate_vit_unet_3200_model import ViTUNet3200Builder


def ensure_model(model_path: Path, metadata_path: Path, seed: int) -> Dict[str, object]:
    if model_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding='utf-8'))
    builder = ViTUNet3200Builder(seed=seed)
    model, metadata = builder.build()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def extract_node_adjacency(model_path: Path) -> Tuple[List[np.ndarray], int]:
    model = onnx.load(model_path)
    graph = model.graph
    producer: Dict[str, int] = {}
    for index, node in enumerate(graph.node):
        for output_name in node.output:
            if output_name:
                producer[output_name] = index

    adjacency_sets = [set() for _ in graph.node]
    for consumer_index, node in enumerate(graph.node):
        for input_name in node.input:
            if not input_name:
                continue
            source_index = producer.get(input_name)
            if source_index is not None and source_index != consumer_index:
                adjacency_sets[source_index].add(consumer_index)
    adjacency = [np.asarray(sorted(targets), dtype=np.int64) for targets in adjacency_sets]
    edge_count = sum(len(targets) for targets in adjacency)
    return adjacency, edge_count


def load_with_workbench_node_count(model_path: Path) -> int:
    code = textwrap.dedent(
        """
        const path = require('path');
        const { ONNXWorkbench } = require('./lib/onnx-workbench');
        const vscode = { Uri: { file(filePath) { return { fsPath: filePath, path: filePath }; } } };
        (async () => {
          const workbench = new ONNXWorkbench({}, () => {});
          const session = await workbench.loadModel(vscode.Uri.file(process.argv[1]));
          console.log(JSON.stringify({ node_count: session.graphInfo.nodes.length }));
        })().catch((error) => {
          console.error(error && error.stack ? error.stack : String(error));
          process.exit(1);
        });
        """
    )
    out = subprocess.check_output(
        ['node', '-e', code, str(model_path.resolve())],
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return int(json.loads(out)['node_count'])


def run_js_layout(adjacency: Sequence[np.ndarray], ranker: str | None, repeats: int) -> Dict[str, object]:
    repo = Path(__file__).resolve().parents[1]
    script = repo / 'scripts' / 'netron_height_order_js.mjs'
    payload = {
        'adjacency': [targets.astype(np.int64).tolist() for targets in adjacency],
    }
    if ranker is not None:
        payload['ranker'] = ranker
    timings: List[float] = []
    last_result: Dict[str, object] | None = None
    for _ in range(repeats):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle)
        try:
            out = subprocess.check_output(
                ['node', '--no-warnings', str(script), str(temp_path)],
                text=True,
                cwd=repo,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        last_result = json.loads(out)
        timings.append(float(last_result['elapsed_ms']))
    assert last_result is not None
    last_result['timings_ms'] = timings
    last_result['median_ms'] = statistics.median(timings)
    last_result['mean_ms'] = statistics.mean(timings)
    return last_result


def run_python_layout(adjacency: Sequence[np.ndarray], ranker: str | None, repeats: int) -> Dict[str, object]:
    timings: List[float] = []
    ranks = None
    layers = None
    y = None
    for _ in range(repeats):
        start = time.perf_counter()
        ranks, layers, y = netron_height_order(adjacency, ranker=ranker)
        timings.append((time.perf_counter() - start) * 1000.0)
    assert ranks is not None and layers is not None and y is not None
    return {
        'ranker': ranker or ('longest-path' if len(adjacency) > 3000 else 'network-simplex'),
        'elapsed_ms': timings[-1],
        'timings_ms': timings,
        'median_ms': statistics.median(timings),
        'mean_ms': statistics.mean(timings),
        'node_count': len(adjacency),
        'edge_count': int(sum(len(targets) for targets in adjacency)),
        'ranks': ranks.tolist(),
        'layers': [layer.tolist() for layer in layers],
        'y': y.tolist(),
    }


def compare_ranks(py_ranks: Sequence[int], js_ranks: Sequence[int]) -> Dict[str, object]:
    py_arr = np.asarray(py_ranks, dtype=np.int64)
    js_arr = np.asarray(js_ranks, dtype=np.int64)
    diff = py_arr - js_arr
    mismatch_indices = np.nonzero(diff != 0)[0]
    return {
        'rank_mismatch_count': int(mismatch_indices.size),
        'max_abs_rank_delta': int(np.max(np.abs(diff))) if diff.size else 0,
        'layer_count_py': int(py_arr.max()) + 1 if py_arr.size else 0,
        'layer_count_js': int(js_arr.max()) + 1 if js_arr.size else 0,
        'first_mismatch_indices': mismatch_indices[:20].tolist(),
    }


def markdown_report(
    model_path: Path,
    metadata: Dict[str, object],
    edge_count: int,
    workbench_node_count: int,
    comparison: Dict[str, object],
    py_auto: Dict[str, object],
    js_auto: Dict[str, object],
    py_simplex: Dict[str, object],
    js_simplex: Dict[str, object],
) -> str:
    return textwrap.dedent(
        f"""\
        # ViT-UNet 3200 节点布局对比报告

        ## 样本

        - 模型文件：`{model_path}`
        - ONNX 节点数：`{metadata['node_count']}`
        - Netron Workbench 读取节点数：`{workbench_node_count}`
        - 边数：`{edge_count}`
        - 结构：4 encoder stages + bottleneck + 4 decoder stages，带 U-Net skip merge，block 内部为简化 ViT attention/MLP 模式

        ## 布局差异

        - Python 版 ranker：`{py_auto['ranker']}`
        - JS / Netron ranker：`{js_auto['ranker']}`
        - rank mismatch count：`{comparison['rank_mismatch_count']}`
        - max abs rank delta：`{comparison['max_abs_rank_delta']}`
        - Python layer count：`{comparison['layer_count_py']}`
        - JS layer count：`{comparison['layer_count_js']}`
        - first mismatch indices：`{comparison['first_mismatch_indices']}`

        ## 性能

        ### 默认 Netron 策略（>3000 节点自动走 `longest-path`）

        - Python median：`{py_auto['median_ms']:.2f} ms`
        - JS median：`{js_auto['median_ms']:.2f} ms`
        - JS / Python 比值：`{(js_auto['median_ms'] / py_auto['median_ms']) if py_auto['median_ms'] else math.inf:.2f}x`

        ### 强制 `network-simplex`

        - Python median：`{py_simplex['median_ms']:.2f} ms`
        - JS median：`{js_simplex['median_ms']:.2f} ms`
        - JS / Python 比值：`{(js_simplex['median_ms'] / py_simplex['median_ms']) if py_simplex['median_ms'] else math.inf:.2f}x`

        ## 结论

        - 对这个 3200 节点 ViT-UNet 样本，Python 版层号与 Netron JS 版 **{'完全一致' if comparison['rank_mismatch_count'] == 0 else '存在差异'}**。
        - 因为节点数大于 3000，Netron 默认不会用 `network-simplex`，而是切到 `longest-path`；Python 版已复现这一策略。
        - Python 版只求“高低层 / rank”，而 JS 版执行的是完整 Dagre 布局（还包含横向排序与坐标分配），因此 JS 通常更慢，这个差距在默认策略下尤为明显。
        - 如果你的目标只是得到 Netron 风格的节点高低关系，Python 版已经足够对齐，并且更适合离线批量处理。
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a 3200-node ViT-UNet model and compare Python vs Netron JS layout ranks.')
    parser.add_argument('--model', default='testdata/generated/vit-unet-3200.onnx')
    parser.add_argument('--metadata', default='testdata/generated/vit-unet-3200.meta.json')
    parser.add_argument('--report', default='docs/vit-unet-3200-layout-report.md')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    model_path = (repo / args.model).resolve()
    metadata_path = (repo / args.metadata).resolve()
    report_path = (repo / args.report).resolve()

    metadata = ensure_model(model_path, metadata_path, args.seed)
    adjacency, edge_count = extract_node_adjacency(model_path)
    workbench_node_count = load_with_workbench_node_count(model_path)

    py_auto = run_python_layout(adjacency, ranker=None, repeats=args.repeats)
    js_auto = run_js_layout(adjacency, ranker=None, repeats=args.repeats)
    comparison = compare_ranks(py_auto['ranks'], js_auto['ranks'])

    py_simplex = run_python_layout(adjacency, ranker='network-simplex', repeats=1)
    js_simplex = run_js_layout(adjacency, ranker='network-simplex', repeats=1)

    report = markdown_report(
        model_path=model_path,
        metadata=metadata,
        edge_count=edge_count,
        workbench_node_count=workbench_node_count,
        comparison=comparison,
        py_auto=py_auto,
        js_auto=js_auto,
        py_simplex=py_simplex,
        js_simplex=js_simplex,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')

    summary = {
        'model': str(model_path),
        'metadata': str(metadata_path),
        'report': str(report_path),
        'node_count': metadata['node_count'],
        'edge_count': edge_count,
        'workbench_node_count': workbench_node_count,
        'comparison': comparison,
        'py_auto_median_ms': py_auto['median_ms'],
        'js_auto_median_ms': js_auto['median_ms'],
        'py_simplex_median_ms': py_simplex['median_ms'],
        'js_simplex_median_ms': js_simplex['median_ms'],
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
