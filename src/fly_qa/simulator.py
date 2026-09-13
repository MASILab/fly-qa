"""Section 3 (core): frozen-weight leaky rate simulation over the retained connectome.

This is an INVENTED, SIMPLIFIED stand-in for real neural dynamics -- a discrete-time
leaky rate model, not a spiking or biophysical simulation. Weights come directly from
the retained MaleCNS v1.0 connectivity (synapse counts), signed by each presynaptic
neuron's predicted neurotransmitter, and are never retrained here: the connectome is
inert with respect to "defect," per the project's ground rules. Only the encoder
(external input) and decoder (which units get read out) are task-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class SimulationResult:
    final_rates: np.ndarray
    id_to_index: dict[int, int]
    # One row per integration step, columns matching `capture_body_ids` passed to run()
    # (real activity for the dashboard's "neurons firing" animation) -- None if not requested.
    step_snapshots: np.ndarray | None = None
    capture_body_ids: list[int] | None = None

    def rate(self, body_id: int) -> float:
        idx = self.id_to_index.get(body_id)
        return float(self.final_rates[idx]) if idx is not None else 0.0


class ConnectomeSimulator:
    """Discrete-time leaky rate model: r <- (1-leak)*r + leak*relu(W @ r + I_ext).

    `weight_matrix` must already be signed and row-normalized (see connectome_data.py) --
    this class runs the dynamics, it does not build or interpret the connectivity.
    """

    def __init__(
        self,
        weight_matrix: sp.spmatrix,
        id_to_index: dict[int, int],
        leak: float = 0.2,
        steps: int = 30,
    ):
        if not (0.0 < leak <= 1.0):
            raise ValueError(f"leak must be in (0, 1], got {leak}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.W = weight_matrix.tocsr()
        self.id_to_index = id_to_index
        self.leak = leak
        self.steps = steps

    def run(
        self,
        external_input: dict[int, float],
        capture_body_ids: list[int] | None = None,
    ) -> SimulationResult:
        n = self.W.shape[0]
        r = np.zeros(n, dtype=np.float64)
        i_ext = np.zeros(n, dtype=np.float64)
        for body_id, value in external_input.items():
            idx = self.id_to_index.get(body_id)
            if idx is not None:
                i_ext[idx] = value

        capture_indices = None
        snapshots: list[np.ndarray] = []
        if capture_body_ids is not None:
            capture_indices = np.array(
                [self.id_to_index[bid] for bid in capture_body_ids if bid in self.id_to_index]
            )

        for _ in range(self.steps):
            drive = self.W @ r + i_ext
            r = (1.0 - self.leak) * r + self.leak * np.maximum(drive, 0.0)
            if capture_indices is not None:
                snapshots.append(r[capture_indices].copy())

        step_snapshots = np.stack(snapshots) if snapshots else None
        return SimulationResult(
            final_rates=r,
            id_to_index=self.id_to_index,
            step_snapshots=step_snapshots,
            capture_body_ids=capture_body_ids,
        )
