"""Pick a small, renderable subset of the real connectome for the dashboard's
"neurons firing" animation, and extract the real edges among just that subset.

This is a visualization concern only -- it never feeds back into classification.
The full 165K-neuron graph can't be sent to a browser, so this samples a mix of
actual input (R1-R6/R8), output (DNp20/DNpe017), and high-degree "hidden" neurons
from the SAME real, signed weight matrix the simulator uses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from fly_qa.connectome_data import ConnectomeGraph

DEFAULT_N_INPUT_SAMPLE = 40
DEFAULT_N_HIDDEN = 120


@dataclass(frozen=True)
class VizNode:
    body_id: int
    role: str  # "r1r6" | "r8" | "dnp20" | "dnpe017" | "hidden"
    label: str


@dataclass(frozen=True)
class VizSubset:
    nodes: list[VizNode]
    edges: list[tuple[int, int, float]]  # (source_body_id, target_body_id, signed weight)

    def to_json_dict(self) -> dict:
        return {
            "nodes": [{"id": n.body_id, "role": n.role, "label": n.label} for n in self.nodes],
            "edges": [{"source": s, "target": t, "weight": w} for s, t, w in self.edges],
        }


def _top_degree_body_ids(
    weight_matrix: sp.csr_matrix, id_to_index: dict[int, int], exclude: set[int], n: int
) -> list[int]:
    degree = np.abs(weight_matrix).sum(axis=1).A1 + np.abs(weight_matrix).sum(axis=0).A1
    index_to_id = {idx: bid for bid, idx in id_to_index.items()}
    order = np.argsort(-degree)

    picked = []
    for idx in order:
        bid = index_to_id[int(idx)]
        if bid in exclude:
            continue
        picked.append(bid)
        if len(picked) >= n:
            break
    return picked


def pick_visualization_subset(
    graph: ConnectomeGraph,
    r1_r6_ids: list[int],
    r8_ids: list[int],
    dnp20_ids: list[int],
    dnpe017_ids: list[int],
    n_input_sample: int = DEFAULT_N_INPUT_SAMPLE,
    n_hidden: int = DEFAULT_N_HIDDEN,
    seed: int = 0,
) -> VizSubset:
    rng = random.Random(seed)

    def sample(ids: list[int], n: int) -> list[int]:
        return sorted(rng.sample(ids, min(n, len(ids))))

    r1r6_sample = sample(r1_r6_ids, n_input_sample)
    r8_sample = sample(r8_ids, n_input_sample)

    already = set(r1r6_sample) | set(r8_sample) | set(dnp20_ids) | set(dnpe017_ids)
    hidden_sample = _top_degree_body_ids(graph.weight_matrix, graph.id_to_index, already, n_hidden)

    neurons = graph.neurons if graph.neurons.index.name == "bodyId" else graph.neurons.set_index("bodyId")

    def label_for(bid: int) -> str:
        t = neurons.loc[bid, "type"] if bid in neurons.index else str(bid)
        return str(t) if t is not None else str(bid)

    nodes = (
        [VizNode(bid, "r1r6", label_for(bid)) for bid in r1r6_sample]
        + [VizNode(bid, "r8", label_for(bid)) for bid in r8_sample]
        + [VizNode(bid, "dnp20", label_for(bid)) for bid in dnp20_ids]
        + [VizNode(bid, "dnpe017", label_for(bid)) for bid in dnpe017_ids]
        + [VizNode(bid, "hidden", label_for(bid)) for bid in hidden_sample]
    )

    node_ids = [n.body_id for n in nodes]
    node_index_set = {graph.id_to_index[bid] for bid in node_ids if bid in graph.id_to_index}
    index_to_id = {idx: bid for bid, idx in graph.id_to_index.items()}

    edges: list[tuple[int, int, float]] = []
    submatrix_indices = sorted(node_index_set)
    sub = graph.weight_matrix[submatrix_indices, :][:, submatrix_indices].tocoo()
    for r, c, w in zip(sub.row, sub.col, sub.data):
        if w == 0:
            continue
        source_id = index_to_id[submatrix_indices[c]]  # column = presynaptic
        target_id = index_to_id[submatrix_indices[r]]  # row = postsynaptic
        edges.append((source_id, target_id, float(w)))

    return VizSubset(nodes=nodes, edges=edges)
