#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.external_data_helper import convert_model_to_external_data


def save_model(model, path: Path, external: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if external:
        location = f"{path.stem}.weights.bin"
        convert_model_to_external_data(model, all_tensors_to_one_file=True, location=location, size_threshold=1024, convert_attribute=False)
    onnx.save(model, str(path))


def make_large_matmul_model(output_path: Path, input_dim: int, output_dim: int, seed: int, external: bool = False):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
    B = rng.standard_normal((output_dim,), dtype=np.float32)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, input_dim])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, output_dim])
    node = helper.make_node('Gemm', ['X', 'W', 'B'], ['Y'], name='gemm0')
    graph = helper.make_graph(
        [node],
        'large_matmul_graph',
        [X],
        [Y],
        [numpy_helper.from_array(W, name='W'), numpy_helper.from_array(B, name='B')]
    )
    model = helper.make_model(graph, producer_name='onnx-workbench-generator')
    save_model(model, output_path, external=external)


def make_branch_crop_model(output_path: Path):
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    W1 = numpy_helper.from_array(np.eye(4, dtype=np.float32), name='W1')
    W2 = numpy_helper.from_array((np.eye(4, dtype=np.float32) * 2.0), name='W2')
    B = numpy_helper.from_array(np.ones((4,), dtype=np.float32), name='B')
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['A'], name='matmul_a'),
        helper.make_node('MatMul', ['X', 'W2'], ['B0'], name='matmul_b'),
        helper.make_node('Add', ['A', 'B0'], ['C'], name='add_merge'),
        helper.make_node('Add', ['C', 'B'], ['Y'], name='add_bias')
    ]
    graph = helper.make_graph(nodes, 'branch_crop_graph', [X], [Y], [W1, W2, B])
    model = helper.make_model(graph, producer_name='onnx-workbench-generator')
    save_model(model, output_path)


def make_compare_pair(output_dir: Path):
    X = helper.make_tensor_value_info('shared_in', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('shared_out', TensorProto.FLOAT, [1, 4])
    W = numpy_helper.from_array(np.eye(4, dtype=np.float32), name='W')
    B = numpy_helper.from_array(np.ones((4,), dtype=np.float32), name='B')

    graph_a = helper.make_graph(
        [
            helper.make_node('MatMul', ['shared_in', 'W'], ['tmp'], name='a_matmul'),
            helper.make_node('Add', ['tmp', 'B'], ['shared_out'], name='a_add')
        ],
        'compare_a_graph',
        [X],
        [Y],
        [W, B]
    )
    model_a = helper.make_model(graph_a, producer_name='onnx-workbench-generator')
    save_model(model_a, output_dir / 'dual-io-compare-a.onnx')

    graph_b = helper.make_graph(
        [
            helper.make_node('Add', ['shared_in', 'B'], ['tmp0'], name='b_add0'),
            helper.make_node('MatMul', ['tmp0', 'W'], ['shared_out'], name='b_matmul')
        ],
        'compare_b_graph',
        [X],
        [Y],
        [W, B]
    )
    model_b = helper.make_model(graph_b, producer_name='onnx-workbench-generator')
    save_model(model_b, output_dir / 'dual-io-compare-b.onnx')


def main():
    parser = argparse.ArgumentParser(description='Generate offline ONNX workbench test models.')
    parser.add_argument('--output-dir', default='testdata/generated', help='Target directory for generated models.')
    parser.add_argument('--large-input-dim', type=int, default=8192)
    parser.add_argument('--large-output-dim', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    make_large_matmul_model(output_dir / 'large-matmul-singlefile.onnx', args.large_input_dim, args.large_output_dim, args.seed, external=False)
    make_large_matmul_model(output_dir / 'large-matmul-external-data.onnx', args.large_input_dim, args.large_output_dim, args.seed + 1, external=True)
    make_branch_crop_model(output_dir / 'branch-crop-small.onnx')
    make_compare_pair(output_dir)

    print(f'Generated ONNX fixtures in: {output_dir}')


if __name__ == '__main__':
    main()
