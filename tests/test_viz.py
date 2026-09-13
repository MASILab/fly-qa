import pandas as pd

from fly_qa.connectome_data import build_connectome_graph
from fly_qa.viz import pick_visualization_subset


def _small_graph():
    # 10 neurons: 2 R1-R6, 2 R8, 1 DNp20-ish pair, 1 DNpe017-ish pair, rest "hidden"
    neurons = pd.DataFrame(
        {
            "bodyId": list(range(1, 11)),
            "type": ["R1-R6", "R1-R6", "R8d", "R8y", "DNp20", "DNp20", "DNpe017", "DNpe017", "X", "X"],
            "instance": ["", "", "", "", "DNp20_L", "DNp20_R", "", "", "", ""],
            "predictedNt": ["acetylcholine"] * 10,
        }
    )
    # a chain of connections so degree varies
    roi_conn = pd.DataFrame(
        {
            "bodyId_pre": [1, 2, 3, 9, 9, 10],
            "bodyId_post": [9, 9, 10, 5, 7, 6],
            "weight": [5, 5, 5, 5, 5, 5],
        }
    )
    return build_connectome_graph(neurons, roi_conn)


def test_pick_visualization_subset_includes_all_roles():
    graph = _small_graph()
    subset = pick_visualization_subset(
        graph,
        r1_r6_ids=[1, 2],
        r8_ids=[3, 4],
        dnp20_ids=[5, 6],
        dnpe017_ids=[7, 8],
        n_input_sample=10,
        n_hidden=2,
        seed=0,
    )
    roles = {n.role for n in subset.nodes}
    assert roles == {"r1r6", "r8", "dnp20", "dnpe017", "hidden"}

    ids_by_role = {role: [n.body_id for n in subset.nodes if n.role == role] for role in roles}
    assert set(ids_by_role["r1r6"]) == {1, 2}
    assert set(ids_by_role["r8"]) == {3, 4}
    assert set(ids_by_role["dnp20"]) == {5, 6}
    assert set(ids_by_role["dnpe017"]) == {7, 8}


def test_pick_visualization_subset_edges_are_real():
    graph = _small_graph()
    subset = pick_visualization_subset(
        graph, r1_r6_ids=[1, 2], r8_ids=[3, 4], dnp20_ids=[5, 6], dnpe017_ids=[7, 8],
        n_input_sample=10, n_hidden=2, seed=0,
    )
    # node 9 should be in the hidden set (highest degree: receives from 1,2,3, sends to 5,7)
    hidden_ids = {n.body_id for n in subset.nodes if n.role == "hidden"}
    assert 9 in hidden_ids

    edge_pairs = {(s, t) for s, t, _ in subset.edges}
    assert (1, 9) in edge_pairs  # real edge from the underlying graph


def test_to_json_dict_shape():
    graph = _small_graph()
    subset = pick_visualization_subset(
        graph, r1_r6_ids=[1, 2], r8_ids=[3, 4], dnp20_ids=[5, 6], dnpe017_ids=[7, 8],
        n_input_sample=10, n_hidden=2, seed=0,
    )
    data = subset.to_json_dict()
    assert "nodes" in data and "edges" in data
    assert all({"id", "role", "label"} <= set(n.keys()) for n in data["nodes"])
    assert all({"source", "target", "weight"} <= set(e.keys()) for e in data["edges"])
