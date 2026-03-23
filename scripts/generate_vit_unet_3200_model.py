#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass
class StageSpec:
    name: str
    dim: int
    blocks: int


class ViTUNet3200Builder:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.nodes: List[onnx.NodeProto] = []
        self.initializers: Dict[str, onnx.TensorProto] = {}
        self.value_infos: Dict[str, onnx.ValueInfoProto] = {}
        self.tensor_seq = 0
        self.node_count = 0

    def _tensor(self, prefix: str) -> str:
        self.tensor_seq += 1
        return f'{prefix}_{self.tensor_seq}'

    def _ensure_value_info(self, name: str, dim: int) -> None:
        if name not in self.value_infos:
            self.value_infos[name] = helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, dim])

    def _weight(self, role: str, in_dim: int, out_dim: int) -> str:
        name = f'W_{role}_{in_dim}_{out_dim}'
        if name not in self.initializers:
            values = self.rng.standard_normal((in_dim, out_dim), dtype=np.float32) * 0.02
            self.initializers[name] = numpy_helper.from_array(values, name=name)
        return name

    def _bias(self, role: str, dim: int) -> str:
        name = f'B_{role}_{dim}'
        if name not in self.initializers:
            values = self.rng.standard_normal((dim,), dtype=np.float32) * 0.01
            self.initializers[name] = numpy_helper.from_array(values, name=name)
        return name

    def _positional(self, dim: int) -> str:
        name = f'POS_{dim}'
        if name not in self.initializers:
            values = self.rng.standard_normal((dim,), dtype=np.float32) * 0.01
            self.initializers[name] = numpy_helper.from_array(values, name=name)
        return name

    def _add_node(self, op_type: str, inputs: List[str], outputs: List[str], name: str, output_dim: int, **attrs) -> str:
        node = helper.make_node(op_type, inputs, outputs, name=name, **attrs)
        self.nodes.append(node)
        self.node_count += 1
        for output in outputs:
            self._ensure_value_info(output, output_dim)
        return outputs[0]

    def patch_embed(self, input_name: str, dim: int) -> str:
        x = self._add_node('Identity', [input_name], [self._tensor('patch_norm')], 'patch_norm', dim)
        x = self._add_node('MatMul', [x, self._weight('patch_proj', dim, dim)], [self._tensor('patch_proj_mm')], 'patch_proj_mm', dim)
        x = self._add_node('Add', [x, self._bias('patch_proj', dim)], [self._tensor('patch_proj_add')], 'patch_proj_add', dim)
        x = self._add_node('Identity', [x], [self._tensor('patch_mix_in')], 'patch_mix_in', dim)
        x = self._add_node('MatMul', [x, self._weight('patch_mix', dim, dim)], [self._tensor('patch_mix_mm')], 'patch_mix_mm', dim)
        x = self._add_node('Add', [x, self._bias('patch_mix', dim)], [self._tensor('patch_mix_add')], 'patch_mix_add', dim)
        x = self._add_node('Add', [x, self._positional(dim)], [self._tensor('patch_pos_add')], 'patch_pos_add', dim)
        x = self._add_node('Identity', [x], [self._tensor('patch_out')], 'patch_out', dim)
        return x

    def transition(self, prefix: str, input_name: str, in_dim: int, out_dim: int, skip_name: str | None = None) -> str:
        x = self._add_node('Identity', [input_name], [self._tensor(f'{prefix}_norm')], f'{prefix}_norm', in_dim)
        x = self._add_node('MatMul', [x, self._weight(f'{prefix}_proj', in_dim, out_dim)], [self._tensor(f'{prefix}_mm')], f'{prefix}_mm', out_dim)
        x = self._add_node('Add', [x, self._bias(f'{prefix}_proj', out_dim)], [self._tensor(f'{prefix}_add')], f'{prefix}_add', out_dim)
        if skip_name is not None:
            x = self._add_node('Add', [x, skip_name], [self._tensor(f'{prefix}_merge')], f'{prefix}_merge', out_dim)
        else:
            x = self._add_node('Identity', [x], [self._tensor(f'{prefix}_out')], f'{prefix}_out', out_dim)
        return x

    def transformer_block(self, prefix: str, input_name: str, dim: int, output_name: str | None = None) -> str:
        hidden_dim = dim * 2
        ln1 = self._add_node('Identity', [input_name], [self._tensor(f'{prefix}_ln1')], f'{prefix}_ln1', dim)
        q_mm = self._add_node('MatMul', [ln1, self._weight(f'{prefix}_q', dim, dim)], [self._tensor(f'{prefix}_q_mm')], f'{prefix}_q_mm', dim)
        q = self._add_node('Add', [q_mm, self._bias(f'{prefix}_q', dim)], [self._tensor(f'{prefix}_q')], f'{prefix}_q', dim)
        k_mm = self._add_node('MatMul', [ln1, self._weight(f'{prefix}_k', dim, dim)], [self._tensor(f'{prefix}_k_mm')], f'{prefix}_k_mm', dim)
        k = self._add_node('Add', [k_mm, self._bias(f'{prefix}_k', dim)], [self._tensor(f'{prefix}_k')], f'{prefix}_k', dim)
        v_mm = self._add_node('MatMul', [ln1, self._weight(f'{prefix}_v', dim, dim)], [self._tensor(f'{prefix}_v_mm')], f'{prefix}_v_mm', dim)
        v = self._add_node('Add', [v_mm, self._bias(f'{prefix}_v', dim)], [self._tensor(f'{prefix}_v')], f'{prefix}_v', dim)
        attn_logits = self._add_node('Add', [q, k], [self._tensor(f'{prefix}_attn_logits')], f'{prefix}_attn_logits', dim)
        attn_prob = self._add_node('Softmax', [attn_logits], [self._tensor(f'{prefix}_attn_prob')], f'{prefix}_attn_prob', dim, axis=-1)
        attn_ctx = self._add_node('Mul', [attn_prob, v], [self._tensor(f'{prefix}_attn_ctx')], f'{prefix}_attn_ctx', dim)
        proj_mm = self._add_node('MatMul', [attn_ctx, self._weight(f'{prefix}_proj', dim, dim)], [self._tensor(f'{prefix}_proj_mm')], f'{prefix}_proj_mm', dim)
        proj = self._add_node('Add', [proj_mm, self._bias(f'{prefix}_proj', dim)], [self._tensor(f'{prefix}_proj')], f'{prefix}_proj', dim)
        res1 = self._add_node('Add', [proj, input_name], [self._tensor(f'{prefix}_res1')], f'{prefix}_res1', dim)
        ln2 = self._add_node('Identity', [res1], [self._tensor(f'{prefix}_ln2')], f'{prefix}_ln2', dim)
        fc1_mm = self._add_node('MatMul', [ln2, self._weight(f'{prefix}_fc1', dim, hidden_dim)], [self._tensor(f'{prefix}_fc1_mm')], f'{prefix}_fc1_mm', hidden_dim)
        fc1 = self._add_node('Add', [fc1_mm, self._bias(f'{prefix}_fc1', hidden_dim)], [self._tensor(f'{prefix}_fc1')], f'{prefix}_fc1', hidden_dim)
        act = self._add_node('Relu', [fc1], [self._tensor(f'{prefix}_relu')], f'{prefix}_relu', hidden_dim)
        fc2_mm = self._add_node('MatMul', [act, self._weight(f'{prefix}_fc2', hidden_dim, dim)], [self._tensor(f'{prefix}_fc2_mm')], f'{prefix}_fc2_mm', dim)
        fc2 = self._add_node('Add', [fc2_mm, self._bias(f'{prefix}_fc2', dim)], [self._tensor(f'{prefix}_fc2')], f'{prefix}_fc2', dim)
        final_output = output_name or self._tensor(f'{prefix}_res2')
        self._add_node('Add', [fc2, res1], [final_output], f'{prefix}_res2', dim)
        return final_output

    def build(self) -> Tuple[onnx.ModelProto, Dict[str, object]]:
        encoder_specs = [
            StageSpec('enc0', 16, 18),
            StageSpec('enc1', 24, 18),
            StageSpec('enc2', 32, 18),
            StageSpec('enc3', 48, 18),
        ]
        bottleneck = StageSpec('bottleneck', 64, 14)
        decoder_specs = [
            StageSpec('dec0', 48, 18),
            StageSpec('dec1', 32, 18),
            StageSpec('dec2', 24, 18),
            StageSpec('dec3', 16, 18),
        ]

        input_name = 'X'
        output_name = 'Y'
        self._ensure_value_info(input_name, encoder_specs[0].dim)
        x = self.patch_embed(input_name, encoder_specs[0].dim)

        skips: List[str] = []
        for index, stage in enumerate(encoder_specs):
            for block_index in range(stage.blocks):
                x = self.transformer_block(f'{stage.name}_block{block_index}', x, stage.dim)
            skips.append(x)
            next_dim = bottleneck.dim if index == len(encoder_specs) - 1 else encoder_specs[index + 1].dim
            x = self.transition(f'{stage.name}_down', x, stage.dim, next_dim)

        for block_index in range(bottleneck.blocks):
            x = self.transformer_block(f'{bottleneck.name}_block{block_index}', x, bottleneck.dim)

        for stage, skip_name in zip(decoder_specs, reversed(skips), strict=True):
            input_dim = bottleneck.dim if stage.name == 'dec0' else decoder_specs[decoder_specs.index(stage) - 1].dim
            x = self.transition(f'{stage.name}_up', x, input_dim, stage.dim, skip_name=skip_name)
            for block_index in range(stage.blocks):
                is_last = stage.name == 'dec3' and block_index == stage.blocks - 1
                x = self.transformer_block(f'{stage.name}_block{block_index}', x, stage.dim, output_name=output_name if is_last else None)

        assert self.node_count == 3200, f'expected 3200 nodes, got {self.node_count}'

        graph = helper.make_graph(
            self.nodes,
            'vit_unet_3200_graph',
            [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, encoder_specs[0].dim])],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, decoder_specs[-1].dim])],
            list(self.initializers.values()),
            value_info=list(self.value_infos.values()),
        )
        model = helper.make_model(graph, producer_name='onnx-workbench-generator')
        onnx.checker.check_model(model)
        metadata = {
            'node_count': self.node_count,
            'encoder_blocks': {stage.name: stage.blocks for stage in encoder_specs},
            'bottleneck_blocks': bottleneck.blocks,
            'decoder_blocks': {stage.name: stage.blocks for stage in decoder_specs},
            'dims': {stage.name: stage.dim for stage in encoder_specs + [bottleneck] + decoder_specs},
        }
        return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a 3200-node ViT-UNet ONNX model for Netron layout benchmarks.')
    parser.add_argument('--output', default='testdata/generated/vit-unet-3200.onnx')
    parser.add_argument('--metadata', default='testdata/generated/vit-unet-3200.meta.json')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    builder = ViTUNet3200Builder(seed=args.seed)
    model, metadata = builder.build()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)

    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(json.dumps({
        'model': str(output_path),
        'metadata': str(metadata_path),
        **metadata,
    }, indent=2))


if __name__ == '__main__':
    main()
