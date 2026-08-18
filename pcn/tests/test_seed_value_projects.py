"""Seed-ordering of value-targeting Project/Modulate in ``_single_pass``.

Historically the feedforward seed ran the predict loop FIRST and applied
value-targeting Project/Modulate only afterwards, so a chain

    clamped_raw --Project--> free_input --Predict--> h

seeded ``h`` from a ZERO input (the projected evidence landed in
``free_input`` only after ``h`` was computed). run_batch now statically
partitions value-PM conns: those whose every pre is a HARD-clamped value
layer apply BEFORE the predict loop; everything else (free / soft-clamped /
error / delayed pres) keeps the historical after-predicts placement.

These tests pin both sides of the contract.
"""
import numpy as np
import optax
import pytest

import pcn


def _one_iter_values(net, batch, data_map, records):
    """Run a single-iteration test (run_batch early-return => the values are
    exactly the ``_single_pass`` seed) and return the recorded arrays."""
    sim = pcn.Simulation(net)
    res = sim.test([batch], data_map=data_map, record_map=records,
                   iterations_per_sample=1, log_every=1,
                   values_optimizer=optax.sgd(0.0), verbose=False)
    return {k: np.asarray(res[k][0]) for k in records}


def test_gated_chain_seeds_from_projected_evidence():
    """h must seed from W @ (raw * mask), not from W @ 0."""
    D, H = 4, 3
    W0 = np.arange(H * D, dtype=np.float32).reshape(H, D) / 10.0

    def build(gated):
        net = pcn.PCNetwork(seed=0)
        net.config(use_bias=False)
        with net:
            if gated:
                l_raw = pcn.Layer(dim=D, activation=pcn.Direct(), label='raw')
                l_mask = pcn.Layer(dim=D, activation=pcn.Direct(), label='mask')
                l_x = pcn.Layer(dim=D, activation=pcn.Direct(), label='x')
                pcn.Project(l_raw, l_x, init_weight=np.eye(D, dtype=np.float32),
                            update_rule=pcn.NoLearning(), use_bias=False)
                pcn.Project(l_x, l_x, init_weight=-np.eye(D, dtype=np.float32),
                            update_rule=pcn.NoLearning(), use_bias=False)
                pcn.Modulate(l_mask, l_x, init_weight=np.eye(D, dtype=np.float32),
                             update_rule=pcn.NoLearning(), use_bias=False)
            else:
                l_raw = l_mask = None
                l_x = pcn.Layer(dim=D, activation=pcn.Direct(), label='x')
            l_h = pcn.Layer(dim=H, activation=pcn.Direct(), label='h')
            pcn.Predict(l_x, l_h, init_weight=W0)
        net.build()
        return net, l_raw, l_mask, l_x, l_h

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 1.0, size=(2, D)).astype(np.float32)
    mask = np.array([[1.0, 0.0, 0.5, 1.0],
                     [0.0, 1.0, 1.0, 0.25]], dtype=np.float32)

    # Gated net: raw + mask clamped, x free, gate delivers raw*mask.
    net, l_raw, l_mask, l_x, l_h = build(gated=True)
    got = _one_iter_values(
        net, {'raw': raw, 'mask': mask},
        data_map={l_raw: 'raw', l_mask: 'mask'},
        records={'h': ((l_h.value,), lambda v: np.asarray(v))})

    # Reference net: x clamped directly with raw*mask (the pre-mask semantics).
    ref_net, _, _, ref_x, ref_h = build(gated=False)
    ref = _one_iter_values(
        ref_net, {'xm': raw * mask},
        data_map={ref_x: 'xm'},
        records={'h': ((ref_h.value,), lambda v: np.asarray(v))})

    assert np.abs(ref['h']).max() > 1e-3          # reference is non-trivial
    np.testing.assert_allclose(got['h'], ref['h'], rtol=1e-5, atol=1e-6)


def test_free_pre_project_keeps_after_predicts_placement():
    """A Project whose pre is a FREE (predict-seeded) layer must still read
    the predict-loop-seeded value (historical GD-Project seeding)."""
    D, A = 4, 3
    W1 = (np.arange(A * D, dtype=np.float32).reshape(A, D) - 5.0) / 10.0

    net = pcn.PCNetwork(seed=0)
    net.config(use_bias=False)
    with net:
        l_in = pcn.Layer(dim=D, activation=pcn.Direct(), label='in')
        l_a = pcn.Layer(dim=A, activation=pcn.Direct(), label='a')
        l_b = pcn.Layer(dim=A, activation=pcn.Direct(), label='b')
        pcn.Predict(l_in, l_a, init_weight=W1)
        pcn.Project(l_a, l_b, init_weight=np.eye(A, dtype=np.float32),
                    update_rule=pcn.NoLearning(), use_bias=False)
    net.build()

    x = np.random.default_rng(1).uniform(0.1, 1.0, (2, D)).astype(np.float32)
    got = _one_iter_values(
        net, {'x': x}, data_map={l_in: 'x'},
        records={'a': ((l_a.value,), lambda v: np.asarray(v)),
                 'b': ((l_b.value,), lambda v: np.asarray(v))})

    assert np.abs(got['a']).max() > 1e-3
    np.testing.assert_allclose(got['b'], got['a'], rtol=1e-5, atol=1e-6)


def test_plain_net_seed_unchanged():
    """No value-PM: the seed is the ordinary feedforward pass."""
    D, H = 4, 3
    W0 = np.arange(H * D, dtype=np.float32).reshape(H, D) / 7.0
    net = pcn.PCNetwork(seed=0)
    net.config(use_bias=False)
    with net:
        l_in = pcn.Layer(dim=D, activation=pcn.Direct(), label='in')
        l_h = pcn.Layer(dim=H, activation=pcn.Direct(), label='h')
        pcn.Predict(l_in, l_h, init_weight=W0)
    net.build()

    x = np.random.default_rng(2).uniform(0.1, 1.0, (2, D)).astype(np.float32)
    got = _one_iter_values(
        net, {'x': x}, data_map={l_in: 'x'},
        records={'h': ((l_h.value,), lambda v: np.asarray(v))})
    np.testing.assert_allclose(got['h'], x @ W0.T, rtol=1e-5, atol=1e-6)
