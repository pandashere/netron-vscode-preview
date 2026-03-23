#!/usr/bin/env python3
"""
Netron-compatible vertical rank extraction for a simple sparse adjacency graph.

This ports the part of Netron's layout pipeline that decides the vertical
"which node is higher / lower" relation.

Input
-----
adjacency:
    A sequence of 1-D arrays. `adjacency[u]` stores the destination node indices
    for outgoing edges `u -> v`.

Output
------
ranks:
    Integer layer index for each node. Smaller rank means higher in Netron's
    default vertical layout.
layers:
    Nodes grouped by rank. Nodes in the same array share the same height.
y:
    Optional y center coordinate for each node. This matches Netron's final
    `position()` rule if you provide `node_heights`; otherwise it uses unit
    heights and preserves only the height ordering.

What is ported from Netron
--------------------------
1. DFS back-edge reversal (`acyclic_run`) to make the graph acyclic.
2. Rank assignment:
   - `network-simplex` for graphs with <= 3000 nodes
   - `longest-path` for graphs with > 3000 nodes
3. Rank normalization so the top-most real node starts at rank 0.
4. Final y assignment rule:
      layer_y = cumulative_previous_layer_heights + max_height(layer) / 2

What is intentionally not ported
--------------------------------
- Left/right ordering inside a layer (`order()`)
- X coordinate assignment
- Compound subgraphs / border dummy nodes
- Edge label dummy nodes

These steps do not change the vertical layer relation for a simple adjacency-only
graph with no cluster hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Edge:
    v: int
    w: int
    weight: float = 1.0
    minlen: int = 1


@dataclass
class Node:
    v: int
    in_edges: List[Edge] = field(default_factory=list)
    out_edges: List[Edge] = field(default_factory=list)
    rank: Optional[int] = None
    low: Optional[int] = None
    lim: Optional[int] = None
    parent: Optional[int] = None


class DirectedGraph:
    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}

    def add_node(self, v: int) -> None:
        if v not in self.nodes:
            self.nodes[v] = Node(v=v)

    def add_edge(self, v: int, w: int, weight: float = 1.0, minlen: int = 1) -> None:
        self.add_node(v)
        self.add_node(w)
        key = (v, w)
        if key in self.edges:
            edge = self.edges[key]
            edge.weight += weight
            edge.minlen = max(edge.minlen, minlen)
            return
        edge = Edge(v=v, w=w, weight=weight, minlen=minlen)
        self.edges[key] = edge
        self.nodes[v].out_edges.append(edge)
        self.nodes[w].in_edges.append(edge)

    def get_edge(self, v: int, w: int) -> Optional[Edge]:
        return self.edges.get((v, w))

    def remove_edge(self, edge: Edge) -> None:
        key = (edge.v, edge.w)
        if key not in self.edges:
            return
        self.nodes[edge.v].out_edges = [entry for entry in self.nodes[edge.v].out_edges if entry is not edge]
        self.nodes[edge.w].in_edges = [entry for entry in self.nodes[edge.w].in_edges if entry is not edge]
        del self.edges[key]

    def remove_node(self, v: int) -> None:
        if v not in self.nodes:
            return
        for edge in list(self.nodes[v].in_edges):
            self.remove_edge(edge)
        for edge in list(self.nodes[v].out_edges):
            self.remove_edge(edge)
        del self.nodes[v]

    def reverse_edge(self, edge: Edge) -> None:
        self.remove_edge(edge)
        self.add_edge(edge.w, edge.v, weight=edge.weight, minlen=edge.minlen)

    def successors(self, v: int) -> List[int]:
        return [edge.w for edge in self.nodes[v].out_edges]

    def predecessors(self, v: int) -> List[int]:
        return [edge.v for edge in self.nodes[v].in_edges]

    def neighbors(self, v: int) -> List[int]:
        values = {edge.v for edge in self.nodes[v].in_edges}
        values.update(edge.w for edge in self.nodes[v].out_edges)
        return list(values)

    def copy_simple(self) -> "DirectedGraph":
        graph = DirectedGraph()
        for node in self.nodes:
            graph.add_node(node)
        for edge in self.edges.values():
            graph.add_edge(edge.v, edge.w, weight=edge.weight, minlen=edge.minlen)
        return graph


class UndirectedTree:
    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[frozenset[int], Dict[str, float]] = {}

    def add_node(self, v: int) -> None:
        if v not in self.nodes:
            self.nodes[v] = Node(v=v)

    def has_node(self, v: int) -> bool:
        return v in self.nodes

    def add_edge(self, v: int, w: int) -> None:
        self.add_node(v)
        self.add_node(w)
        key = frozenset((v, w))
        if key in self.edges:
            return
        self.edges[key] = {"cutvalue": 0.0}

    def remove_edge(self, edge: Tuple[int, int]) -> None:
        key = frozenset(edge)
        if key in self.edges:
            del self.edges[key]

    def edge(self, v: int, w: int) -> Optional[Dict[str, float]]:
        return self.edges.get(frozenset((v, w)))

    def neighbors(self, v: int) -> List[int]:
        values: List[int] = []
        for edge_key in self.edges.keys():
            if v in edge_key:
                for other in edge_key:
                    if other != v:
                        values.append(other)
        return values

    def all_edges(self) -> List[Tuple[int, int, Dict[str, float]]]:
        result: List[Tuple[int, int, Dict[str, float]]] = []
        for key, label in self.edges.items():
            v, w = tuple(key)
            result.append((v, w, label))
        return result


def _normalize_adjacency(adjacency: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(targets, dtype=np.int64).reshape(-1) for targets in adjacency]


def _build_graph(adjacency: Sequence[np.ndarray]) -> DirectedGraph:
    graph = DirectedGraph()
    normalized = _normalize_adjacency(adjacency)
    for source in range(len(normalized)):
        graph.add_node(source)
    for source, targets in enumerate(normalized):
        for target in targets.tolist():
            target_idx = int(target)
            if source == target_idx:
                continue
            graph.add_edge(source, target_idx, weight=1.0, minlen=1)
    return graph


def _acyclic_run(graph: DirectedGraph) -> List[Tuple[int, int]]:
    reversed_edges: List[Tuple[int, int]] = []
    visited: set[int] = set()
    path: set[int] = set()
    for start in list(graph.nodes.keys()):
        if start in visited:
            continue
        stack: List[Tuple[int, int]] = [(start, 0)]
        visited.add(start)
        path.add(start)
        while stack:
            v, edge_index = stack[-1]
            out_edges = graph.nodes[v].out_edges
            if edge_index >= len(out_edges):
                path.remove(v)
                stack.pop()
                continue
            edge = out_edges[edge_index]
            stack[-1] = (v, edge_index + 1)
            if edge.w in path:
                reversed_edges.append((edge.v, edge.w))
            elif edge.w not in visited:
                visited.add(edge.w)
                path.add(edge.w)
                stack.append((edge.w, 0))
    for v, w in reversed_edges:
        edge = graph.get_edge(v, w)
        if edge is not None:
            graph.reverse_edge(edge)
    return reversed_edges


def _slack(graph: DirectedGraph, edge: Edge) -> int:
    assert graph.nodes[edge.v].rank is not None
    assert graph.nodes[edge.w].rank is not None
    return int(graph.nodes[edge.w].rank - graph.nodes[edge.v].rank - edge.minlen)


def _longest_path(graph: DirectedGraph) -> None:
    visited: set[int] = set()
    sources = [node for node in graph.nodes.values() if len(node.in_edges) == 0]
    stack: List[object] = [list(reversed(sources))]
    while stack:
        current = stack[-1]
        if isinstance(current, list):
            node = current.pop()
            if not current:
                stack.pop()
            if node.v not in visited:
                visited.add(node.v)
                children = [graph.nodes[edge.w] for edge in node.out_edges]
                if children:
                    stack.append(node)
                    stack.append(list(reversed(children)))
                else:
                    node.rank = 0
        else:
            node = stack.pop()
            rank = min(graph.nodes[edge.w].rank - edge.minlen for edge in node.out_edges)
            node.rank = int(rank)


def _tight_tree(tree: UndirectedTree, graph: DirectedGraph) -> int:
    stack = list(reversed(list(tree.nodes.keys())))
    while stack:
        v = stack.pop()
        graph_node = graph.nodes[v]
        for edge in graph_node.in_edges + graph_node.out_edges:
            w = edge.w if edge.v == v else edge.v
            if not tree.has_node(w) and _slack(graph, edge) == 0:
                tree.add_node(w)
                tree.add_edge(v, w)
                stack.append(w)
    return len(tree.nodes)


def _feasible_tree(graph: DirectedGraph) -> UndirectedTree:
    tree = UndirectedTree()
    start = next(iter(graph.nodes.keys()))
    tree.add_node(start)
    size = len(graph.nodes)
    while _tight_tree(tree, graph) < size:
        min_slack = np.iinfo(np.int64).max
        chosen: Optional[Edge] = None
        for edge in graph.edges.values():
            in_tree_v = tree.has_node(edge.v)
            in_tree_w = tree.has_node(edge.w)
            if in_tree_v != in_tree_w:
                current_slack = _slack(graph, edge)
                if current_slack < min_slack:
                    min_slack = current_slack
                    chosen = edge
        if chosen is None:
            break
        delta = _slack(graph, chosen) if tree.has_node(chosen.v) else -_slack(graph, chosen)
        for v in tree.nodes.keys():
            graph.nodes[v].rank = int(graph.nodes[v].rank + delta)
    return tree


def _init_low_lim_values(tree: UndirectedTree) -> None:
    start = next(iter(tree.nodes.keys()))
    visited: set[int] = set()
    next_lim = 1
    stack: List[Tuple[int, Optional[int], int]] = [(start, None, 0)]
    temp: Dict[int, Tuple[int, Optional[int]]] = {}
    while stack:
        v, parent, state = stack.pop()
        if state == 0:
            if v not in visited:
                visited.add(v)
                temp[v] = (next_lim, parent)
                stack.append((v, parent, 1))
                for w in tree.neighbors(v):
                    if w not in visited:
                        stack.append((w, v, 0))
        else:
            low, parent_value = temp[v]
            node = tree.nodes[v]
            node.low = low
            node.lim = next_lim
            node.parent = parent_value
            next_lim += 1


def _init_cut_values(tree: UndirectedTree, graph: DirectedGraph) -> None:
    order: List[int] = []
    visited: set[int] = set()
    stack: List[object] = [list(reversed(list(tree.nodes.keys())))]
    while stack:
        current = stack[-1]
        if isinstance(current, list):
            v = current.pop()
            if not current:
                stack.pop()
            if v not in visited:
                visited.add(v)
                children = tree.neighbors(v)
                if children:
                    stack.append(v)
                    stack.append(list(reversed(children)))
                else:
                    order.append(v)
        else:
            order.append(stack.pop())
    if len(order) <= 1:
        return
    for v in order[:-1]:
        child = tree.nodes[v]
        parent = child.parent
        if parent is None:
            continue
        direct = graph.get_edge(v, parent)
        child_is_tail = direct is not None
        graph_edge = direct if direct is not None else graph.get_edge(parent, v)
        if graph_edge is None:
            continue
        cut_value = graph_edge.weight
        node = graph.nodes[v]
        for edge in node.in_edges + node.out_edges:
            is_out = edge.v == v
            other = edge.w if is_out else edge.v
            if other == parent:
                continue
            points_to_head = (is_out == child_is_tail)
            cut_value += edge.weight if points_to_head else -edge.weight
            tree_edge = tree.edge(v, other)
            if tree_edge is not None:
                other_cut = tree_edge["cutvalue"]
                cut_value += -other_cut if points_to_head else other_cut
        tree.edge(v, parent)["cutvalue"] = cut_value


def _leave_edge(tree: UndirectedTree) -> Optional[Tuple[int, int]]:
    for v, w, label in tree.all_edges():
        if label["cutvalue"] < 0:
            return (v, w)
    return None


def _is_descendant(node: Node, root: Node) -> bool:
    assert node.lim is not None and root.low is not None and root.lim is not None
    return root.low <= node.lim <= root.lim


def _enter_edge(tree: UndirectedTree, graph: DirectedGraph, edge: Tuple[int, int]) -> Optional[Edge]:
    v, w = edge
    if graph.get_edge(v, w) is None:
        v, w = w, v
    v_label = tree.nodes[v]
    w_label = tree.nodes[w]
    tail_label = v_label
    flip = False
    if v_label.lim is not None and w_label.lim is not None and v_label.lim > w_label.lim:
        tail_label = w_label
        flip = True
    best_edge: Optional[Edge] = None
    best_slack = float("inf")
    for candidate in graph.edges.values():
        candidate_v = tree.nodes[candidate.v]
        candidate_w = tree.nodes[candidate.w]
        if flip == _is_descendant(candidate_v, tail_label) and flip != _is_descendant(candidate_w, tail_label):
            current_slack = _slack(graph, candidate)
            if current_slack < best_slack:
                best_slack = current_slack
                best_edge = candidate
    return best_edge


def _exchange_edges(tree: UndirectedTree, graph: DirectedGraph, leaving: Tuple[int, int], entering: Edge) -> None:
    tree.remove_edge(leaving)
    tree.add_edge(entering.v, entering.w)
    _init_low_lim_values(tree)
    _init_cut_values(tree, graph)
    roots = [node.v for node in tree.nodes.values() if node.parent is None]
    if not roots:
        return
    root = roots[0]
    visited: set[int] = set()
    ordered: List[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        ordered.append(v)
        for w in reversed(tree.neighbors(v)):
            stack.append(w)
    for v in ordered[1:]:
        parent = tree.nodes[v].parent
        if parent is None:
            continue
        edge = graph.get_edge(v, parent)
        flipped = False
        if edge is None:
            edge = graph.get_edge(parent, v)
            flipped = True
        if edge is None:
            continue
        parent_rank = graph.nodes[parent].rank
        graph.nodes[v].rank = int(parent_rank + (edge.minlen if flipped else -edge.minlen))


def _network_simplex(graph: DirectedGraph) -> None:
    _longest_path(graph)
    tree = _feasible_tree(graph)
    _init_low_lim_values(tree)
    _init_cut_values(tree, graph)
    while True:
        leaving = _leave_edge(tree)
        if leaving is None:
            break
        entering = _enter_edge(tree, graph, leaving)
        if entering is None:
            break
        _exchange_edges(tree, graph, leaving, entering)


def _add_super_root(graph: DirectedGraph) -> int:
    root = len(graph.nodes)
    graph.add_node(root)
    for node in list(graph.nodes.keys()):
        if node == root:
            continue
        graph.add_edge(root, node, weight=0.0, minlen=1)
    return root


def _normalize_real_ranks(graph: DirectedGraph, root: Optional[int]) -> None:
    ranks = [node.rank for node in graph.nodes.values() if node.v != root and node.rank is not None]
    if not ranks:
        return
    min_rank = min(ranks)
    for node in graph.nodes.values():
        if node.v != root and node.rank is not None:
            node.rank = int(node.rank - min_rank)


def _extract_layers(graph: DirectedGraph, root: Optional[int]) -> List[np.ndarray]:
    rank_to_nodes: Dict[int, List[int]] = {}
    for node in graph.nodes.values():
        if node.v == root or node.rank is None:
            continue
        rank_to_nodes.setdefault(int(node.rank), []).append(node.v)
    layers: List[np.ndarray] = []
    for rank in sorted(rank_to_nodes.keys()):
        layers.append(np.asarray(sorted(rank_to_nodes[rank]), dtype=np.int64))
    return layers


def _compute_y(layers: List[np.ndarray], num_nodes: int, node_heights: Optional[np.ndarray], ranksep: float) -> np.ndarray:
    if node_heights is None:
        node_heights = np.ones((num_nodes,), dtype=np.float64)
    else:
        node_heights = np.asarray(node_heights, dtype=np.float64).reshape(num_nodes)
    y = np.zeros((num_nodes,), dtype=np.float64)
    cursor = 0.0
    for layer in layers:
        if layer.size == 0:
            continue
        max_height = float(np.max(node_heights[layer]))
        center = cursor + max_height / 2.0
        y[layer] = center
        cursor += max_height + ranksep
    return y


def netron_height_order(
    adjacency: Sequence[np.ndarray],
    *,
    node_heights: Optional[np.ndarray] = None,
    ranksep: float = 20.0,
    ranker: Optional[str] = None,
    large_graph_threshold: int = 3000,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Compute the Netron-style vertical layer relation for a sparse adjacency list.

    Parameters
    ----------
    adjacency:
        `adjacency[u]` contains destination node indices for `u -> v`.
    node_heights:
        Optional per-node rendered heights. If omitted, all heights are treated as 1.
    ranksep:
        Gap between two adjacent layers in the final y coordinate reconstruction.
    ranker:
        `None` means Netron's default:
            - `network-simplex` when node count <= 3000
            - `longest-path` when node count > 3000
    large_graph_threshold:
        Threshold used by Netron to switch to `longest-path`.
    """
    graph = _build_graph(adjacency)
    num_nodes = len(adjacency)
    if not graph.edges:
        ranks = np.zeros((num_nodes,), dtype=np.int64)
        layers = [np.arange(num_nodes, dtype=np.int64)] if num_nodes > 0 else []
        y = _compute_y(layers, num_nodes, node_heights, ranksep)
        return ranks, layers, y

    _acyclic_run(graph)
    root = _add_super_root(graph)

    selected_ranker = ranker
    if selected_ranker is None:
        selected_ranker = "longest-path" if num_nodes > large_graph_threshold else "network-simplex"

    if selected_ranker == "longest-path":
        _longest_path(graph)
    elif selected_ranker == "network-simplex":
        _network_simplex(graph)
    else:
        raise ValueError(f"Unsupported ranker: {selected_ranker}")

    graph.remove_node(root)
    _normalize_real_ranks(graph, root=None)

    ranks = np.zeros((num_nodes,), dtype=np.int64)
    for node in graph.nodes.values():
        if node.rank is None:
            raise RuntimeError(f"Node {node.v} has no rank.")
        ranks[node.v] = int(node.rank)

    layers = _extract_layers(graph, root=None)
    y = _compute_y(layers, num_nodes, node_heights, ranksep)
    return ranks, layers, y


def _demo() -> None:
    adjacency = [
        np.array([1, 2]),
        np.array([3]),
        np.array([3]),
        np.array([], dtype=np.int64),
    ]
    ranks, layers, y = netron_height_order(adjacency)
    print("ranks:", ranks)
    print("layers:", [layer.tolist() for layer in layers])
    print("y:", y)


if __name__ == "__main__":
    _demo()
