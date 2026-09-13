import numpy as np
import pandas as pd

from fly_qa.connectome_data import (
    body_ids_by_type,
    build_connectome_graph,
    neurotransmitter_sign,
)


def test_neurotransmitter_sign_inhibitory():
    assert neurotransmitter_sign("gaba") == -1.0
    assert neurotransmitter_sign("GABA") == -1.0
    assert neurotransmitter_sign("glutamate") == -1.0
    assert neurotransmitter_sign("histamine") == -1.0


def test_neurotransmitter_sign_excitatory_and_default():
    assert neurotransmitter_sign("acetylcholine") == 1.0
    assert neurotransmitter_sign("dopamine") == 1.0
    assert neurotransmitter_sign("unclear") == 1.0
    assert neurotransmitter_sign(None) == 1.0
    assert neurotransmitter_sign(float("nan")) == 1.0


def test_body_ids_by_type_exact_match_only():
    neurons = pd.DataFrame(
        {"bodyId": [1, 2, 3], "type": ["R8d", "R8", "R8y"]}
    ).set_index("bodyId")
    assert body_ids_by_type(neurons, "R8") == [2]
    assert body_ids_by_type(neurons, "R8d") == [1]
    assert body_ids_by_type(neurons, "nonexistent") == []


def _small_neurons():
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3],
            "type": ["A", "B", "C"],
            "predictedNt": ["acetylcholine", "gaba", "unclear"],
        }
    )


def test_build_graph_signs_by_presynaptic_neurotransmitter():
    neurons = _small_neurons()
    # 1 (excitatory) -> 3, weight 10; 2 (inhibitory) -> 3, weight 10
    roi_conn = pd.DataFrame(
        {"bodyId_pre": [1, 2], "bodyId_post": [3, 3], "weight": [10, 10]}
    )
    graph = build_connectome_graph(neurons, roi_conn)

    idx3 = graph.id_to_index[3]
    idx1 = graph.id_to_index[1]
    idx2 = graph.id_to_index[2]

    row = graph.weight_matrix[idx3].toarray().flatten()
    assert row[idx1] > 0  # excitatory input
    assert row[idx2] < 0  # inhibitory input


def test_build_graph_row_normalizes():
    neurons = _small_neurons()
    roi_conn = pd.DataFrame(
        {"bodyId_pre": [1, 2], "bodyId_post": [3, 3], "weight": [30, 10]}
    )
    graph = build_connectome_graph(neurons, roi_conn)
    idx3 = graph.id_to_index[3]
    row = graph.weight_matrix[idx3].toarray().flatten()
    assert abs(np.abs(row).sum() - 1.0) < 1e-9


def test_build_graph_sums_multiple_roi_rows_for_same_pair():
    neurons = _small_neurons()
    roi_conn = pd.DataFrame(
        {
            "bodyId_pre": [1, 1],
            "bodyId_post": [3, 3],
            "roi": ["ROI_A", "ROI_B"],
            "weight": [5, 7],
        }
    )
    graph = build_connectome_graph(neurons, roi_conn)
    idx3 = graph.id_to_index[3]
    idx1 = graph.id_to_index[1]
    row = graph.weight_matrix[idx3].toarray().flatten()
    # only nonzero entry, so after row-normalization it should be exactly 1.0
    assert abs(row[idx1] - 1.0) < 1e-9


def test_build_graph_ignores_bodyids_not_in_neurons():
    neurons = _small_neurons()
    roi_conn = pd.DataFrame(
        {"bodyId_pre": [1, 999], "bodyId_post": [3, 3], "weight": [10, 10]}
    )
    graph = build_connectome_graph(neurons, roi_conn)
    assert 999 not in graph.id_to_index
    assert graph.weight_matrix.shape == (3, 3)
