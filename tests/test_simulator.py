import numpy as np
import scipy.sparse as sp

from fly_qa.simulator import ConnectomeSimulator


def test_zero_input_stays_at_zero():
    W = sp.csr_matrix(np.zeros((5, 5)))
    sim = ConnectomeSimulator(W, {i: i for i in range(5)}, leak=0.5, steps=10)
    result = sim.run({})
    assert np.allclose(result.final_rates, 0.0)


def test_isolated_neuron_settles_to_its_own_input():
    # No recurrent connections: neuron 0 with constant external drive 1.0
    # should settle toward relu(1.0) = 1.0 as steps -> large.
    W = sp.csr_matrix(np.zeros((3, 3)))
    sim = ConnectomeSimulator(W, {10: 0, 20: 1, 30: 2}, leak=0.5, steps=200)
    result = sim.run({10: 1.0})
    assert abs(result.rate(10) - 1.0) < 1e-6
    assert result.rate(20) == 0.0


def test_excitatory_connection_propagates_activity():
    # neuron 0 -> neuron 1 excitatory
    W = sp.csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))
    sim = ConnectomeSimulator(W, {0: 0, 1: 1}, leak=0.3, steps=100)
    result = sim.run({0: 1.0})
    assert result.rate(0) > 0
    assert result.rate(1) > 0  # activity propagated from 0 to 1


def test_inhibitory_connection_suppresses_target():
    # neuron 0 excites neuron 2; neuron 1 (also externally driven) inhibits neuron 2
    W = sp.lil_matrix((3, 3))
    W[2, 0] = 1.0
    W[2, 1] = -1.0
    W = W.tocsr()
    sim = ConnectomeSimulator(W, {0: 0, 1: 1, 2: 2}, leak=0.3, steps=100)

    excite_only = sim.run({0: 1.0})
    excite_and_inhibit = sim.run({0: 1.0, 1: 1.0})

    assert excite_and_inhibit.rate(2) < excite_only.rate(2)


def test_unknown_body_id_returns_zero():
    W = sp.csr_matrix(np.zeros((2, 2)))
    sim = ConnectomeSimulator(W, {1: 0, 2: 1})
    result = sim.run({1: 1.0})
    assert result.rate(999) == 0.0


def test_invalid_leak_raises():
    W = sp.csr_matrix(np.zeros((2, 2)))
    try:
        ConnectomeSimulator(W, {1: 0, 2: 1}, leak=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_capture_body_ids_returns_per_step_snapshots():
    W = sp.csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))
    sim = ConnectomeSimulator(W, {10: 0, 20: 1}, leak=0.3, steps=15)
    result = sim.run({10: 1.0}, capture_body_ids=[10, 20])

    assert result.step_snapshots is not None
    assert result.step_snapshots.shape == (15, 2)
    # activity should be monotonically settling (increasing then plateauing) for neuron 10
    assert result.step_snapshots[-1, 0] >= result.step_snapshots[0, 0]
    # final captured value should match the final rate for the same body id
    assert abs(result.step_snapshots[-1, 0] - result.rate(10)) < 1e-9
    assert abs(result.step_snapshots[-1, 1] - result.rate(20)) < 1e-9


def test_capture_body_ids_ignores_unknown_ids():
    W = sp.csr_matrix(np.zeros((2, 2)))
    sim = ConnectomeSimulator(W, {10: 0, 20: 1}, leak=0.5, steps=5)
    result = sim.run({10: 1.0}, capture_body_ids=[10, 999])
    assert result.step_snapshots.shape == (5, 1)  # 999 silently dropped


def test_no_capture_by_default():
    W = sp.csr_matrix(np.zeros((2, 2)))
    sim = ConnectomeSimulator(W, {10: 0, 20: 1}, leak=0.5, steps=5)
    result = sim.run({10: 1.0})
    assert result.step_snapshots is None
