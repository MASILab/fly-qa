"""Loads the retained MaleCNS v1.0 connectome (fetched via fetch_traced_adjacencies,
see scripts/fetch_connectome.py) into a signed, row-normalized sparse weight matrix
for the rate simulator.

Ground rule: the connectome graph itself is used UNMODIFIED (raw synapse-count
weights, real body IDs) -- normalization here only rescales magnitudes for numerical
stability, it does not add, remove, or reweight edges based on task performance.

Sign convention (a documented simplification, not a claim of biological ground truth):
  acetylcholine            -> excitatory (+1)
  gaba, glutamate,
  histamine                -> inhibitory (-1)
  dopamine, octopamine,
  serotonin, unclear, other -> excitatory (+1), since these are neuromodulatory
                                or unknown in this fast-rate approximation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

INHIBITORY_TRANSMITTERS = {"gaba", "glutamate", "histamine"}


@dataclass(frozen=True)
class ConnectomeGraph:
    weight_matrix: sp.csr_matrix  # signed, row-normalized: W[i, j] = post i <- pre j
    id_to_index: dict[int, int]
    neurons: pd.DataFrame  # indexed by bodyId, has at least 'type' and 'predictedNt'


def neurotransmitter_sign(nt: str | float | None) -> float:
    if not isinstance(nt, str):
        return 1.0
    return -1.0 if nt.lower() in INHIBITORY_TRANSMITTERS else 1.0


def body_ids_by_type(neurons: pd.DataFrame, type_pattern: str) -> list[int]:
    """Exact match on the `type` column (not a regex) -- callers combine calls for
    multiple real subtypes (e.g. several R8 variants) rather than relying on a
    pattern that might silently match something unintended."""
    matches = neurons.index[neurons["type"] == type_pattern]
    return sorted(int(b) for b in matches)


def build_connectome_graph(neurons: pd.DataFrame, roi_conn: pd.DataFrame) -> ConnectomeGraph:
    """
    `neurons`: DataFrame with a 'bodyId' column (or index) and 'predictedNt'/'type'.
    `roi_conn`: DataFrame with bodyId_pre, bodyId_post, weight (already summed across ROIs,
                or with a 'roi' column this function will sum itself).
    """
    if "bodyId" in neurons.columns:
        neurons = neurons.set_index("bodyId")

    if "roi" in roi_conn.columns:
        roi_conn = roi_conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()

    body_ids = sorted(neurons.index.tolist())
    id_to_index = {int(bid): i for i, bid in enumerate(body_ids)}
    n = len(body_ids)

    nt_by_id = neurons["predictedNt"].to_dict() if "predictedNt" in neurons.columns else {}
    sign_by_id = {bid: neurotransmitter_sign(nt_by_id.get(bid)) for bid in body_ids}

    rows, cols, data = [], [], []
    for pre, post, weight in roi_conn[["bodyId_pre", "bodyId_post", "weight"]].itertuples(index=False):
        pre_idx = id_to_index.get(int(pre))
        post_idx = id_to_index.get(int(post))
        if pre_idx is None or post_idx is None:
            continue
        signed_weight = float(weight) * sign_by_id.get(int(pre), 1.0)
        rows.append(post_idx)  # row = postsynaptic (receives)
        cols.append(pre_idx)  # col = presynaptic (sends)
        data.append(signed_weight)

    raw = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    normalized = _row_normalize(raw)

    return ConnectomeGraph(weight_matrix=normalized, id_to_index=id_to_index, neurons=neurons)


def _row_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Scale each postsynaptic neuron's incoming weights so the sum of |weight| is 1.

    This keeps the rate model numerically stable regardless of the raw synapse-count
    scale (which varies hugely by neuron) -- each neuron's drive becomes a weighted
    average of its presynaptic partners' rates, not an unbounded sum.
    """
    row_abs_sum = np.abs(matrix).sum(axis=1).A1
    row_abs_sum[row_abs_sum == 0] = 1.0
    scale = sp.diags(1.0 / row_abs_sum)
    return (scale @ matrix).tocsr()


def load_connectome_export(export_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the CSVs written by scripts/fetch_connectome.py.

    `fetch_traced_adjacencies`'s own `neurons.csv` only has bodyId/type/instance --
    no `predictedNt` -- so we prefer `neurons_full.csv` (a supplementary fetch_neurons()
    pull with the fields sign assignment needs) when present, falling back to the
    bare version otherwise (signs will then default to excitatory for everything).

    Prefers `total-connections.csv` (bodyId_pre, bodyId_post, weight already summed
    across ROIs) over `roi-connections.csv` (one row per bodyId_pre/bodyId_post/roi)
    since it's what build_connectome_graph needs and avoids redoing that groupby
    ourselves over tens of millions of rows.
    """
    export_dir = Path(export_dir)
    full_path = export_dir / "neurons_full.csv"
    neurons = pd.read_csv(full_path) if full_path.exists() else pd.read_csv(export_dir / "neurons.csv")

    total_path = export_dir / "total-connections.csv"
    if total_path.exists():
        roi_conn = pd.read_csv(total_path)
    else:
        roi_conn = pd.read_csv(export_dir / "roi-connections.csv")
    return neurons, roi_conn
