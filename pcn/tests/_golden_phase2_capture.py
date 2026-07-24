"""Golden capture + matrix builder for the Phase-2 delay/hist refactor.

This is a SCRIPT (not a pytest module). It builds a matrix of small nets that
exercise every consumer of the ``prev_errors`` / ``prev_precisions`` one-step
carries, runs a battery of ``sim.test(return_logs=True)`` and
``sim.train(save_logs=True)`` configs on fixed-seed data, and flattens all
``values`` / ``errors`` / ``precisions`` / ``deltas`` logs (plus test energies
and post-train params) into a single flat ``dict[str, np.ndarray]`` of float32
arrays.

Run it on the CURRENT (pre-Phase-2) code to write the golden ``.npz``:

    python -m pcn.tests._golden_phase2_capture <out.npz>

``test_phase2_golden.py`` imports :func:`build_matrix` and :func:`collect_all`
from here so the golden and the post-refactor comparison build the *identical*
matrix (single source of truth). Every array must be ``np.array_equal`` (exact)
across the refactor.

Matrix (task-mandated):
  1  plain Predict stack (no carry consumers)
  2  Leaky (memory) error activation
  3  Leaky (memory) precision activation
  4  error-targeting Project + error-pre Project
  5  precision-targeting Project
  6  precision_input = value layer
  7  precision_input = [error node]
  8  precision_input = [precision node]
  9  flow-gate modulator whose pre is an error node
  10 Phase-1 delay (delay_unit='timestep') edge WITH a Leaky error
"""

import sys

import numpy as np
import jax.numpy as jnp

import pcn
from pcn.core.activations import Softplus, Direct


# ---------------------------------------------------------------------------
# Net builders. Each returns (net, input_label, input_is_temporal).
# The input layer is clamped for both test and train (uniform, deterministic).
# ---------------------------------------------------------------------------

def _n1_plain():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=3, label='c')
        pcn.Predict(a, b)
        pcn.Predict(b, c)
    net.build()
    return net, 'a', False


def _n2_leaky_error():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=3, label='c')
        pcn.Predict(a, b, error_activation=pcn.Leaky(leak=0.3))
        pcn.Predict(b, c)
    net.build()
    return net, 'a', False


def _n3_leaky_precision():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=3, label='c')
        pcn.Predict(a, b,
                    precision_activation=pcn.Leaky(base=Softplus(), leak=0.3))
        pcn.Predict(b, c)
    net.build()
    return net, 'a', False


def _n4_error_routing():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=4, label='c')
        p1 = pcn.Predict(a, b)
        pcn.Predict(b, c)
        pcn.Project(a.value, p1.error)   # error-TARGET (post is error node)
        pcn.Project(p1.error, c.value)   # error-PRE (pre is error node)
    net.build()
    return net, 'a', False


def _n5_precision_routing():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=3, label='c')
        p1 = pcn.Predict(a, b)
        pcn.Predict(b, c)
        pcn.Project(a.value, p1.precision)   # precision-TARGET
    net.build()
    return net, 'a', False


def _n6_pin_value():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, activation=pcn.Direct(), label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=3, label='c')
        ctx = pcn.Layer(dim=6, label='ctx')
        pcn.Predict(a, b, precision_input=ctx)   # value source (node_type 0)
        pcn.Predict(b, c)
    net.build()
    return net, 'a', False


def _n7_pin_error():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=3, label='c')
        p1 = pcn.Predict(a, b)
        pcn.Predict(b, c, precision_input=[p1.error])   # error source (nt 1)
    net.build()
    return net, 'a', False


def _n8_pin_precision():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=3, label='c')
        p1 = pcn.Predict(a, b)
        pcn.Predict(b, c, precision_input=[p1.precision])  # prec source (nt 2)
    net.build()
    return net, 'a', False


def _n9_flow_gate_error_pre():
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=5, label='a')
        b = pcn.Layer(dim=4, activation=pcn.Relu(), label='b')
        c = pcn.Layer(dim=4, label='c')
        p1 = pcn.Predict(a, b)
        p2 = pcn.Predict(a, c)
        # Flow gate on p1 whose gating pre is p2's error node (reads carry err).
        pcn.Modulate(p2.error, p1.flow_to_pre)
    net.build()
    return net, 'a', False


def _n10_delay_ts_leaky():
    net = pcn.PCNetwork(seed=0)
    with net:
        l_in = pcn.Layer(dim=4, label='in')
        z = pcn.Layer(dim=4, activation=pcn.Direct(), label='z')
        # Observation edge with a Leaky (memory) error -> error hist buffers.
        pcn.Predict(z, l_in, error_activation=pcn.Leaky(leak=0.3))
        # Transition edge with a timestep delay -> value hist buffer on z.
        pcn.Predict(z, z, delay=1, delay_unit='timestep', use_bias=False)
    net.build()
    return net, 'in', True


def build_matrix():
    """Return an ordered list of (name, builder) for the golden matrix."""
    return [
        ('n01_plain', _n1_plain),
        ('n02_leaky_error', _n2_leaky_error),
        ('n03_leaky_precision', _n3_leaky_precision),
        ('n04_error_routing', _n4_error_routing),
        ('n05_precision_routing', _n5_precision_routing),
        ('n06_pin_value', _n6_pin_value),
        ('n07_pin_error', _n7_pin_error),
        ('n08_pin_precision', _n8_pin_precision),
        ('n09_flow_gate_error_pre', _n9_flow_gate_error_pre),
        ('n10_delay_ts_leaky', _n10_delay_ts_leaky),
    ]


# ---------------------------------------------------------------------------
# Deterministic data + a battery of run configs per net.
# ---------------------------------------------------------------------------

_BATCH = 3


def _make_sample(net, input_label, temporal, T):
    """Fixed-seed clamp data for the input layer (keyed 'in_data')."""
    rng = np.random.default_rng(20260724)
    dim = net.structure.layer_dims[net[input_label]]
    if temporal:
        arr = (0.5 * rng.standard_normal((_BATCH, T, dim))).astype(np.float32)
    else:
        arr = (0.5 * rng.standard_normal((_BATCH, dim))).astype(np.float32)
    return {'in_data': jnp.asarray(arr)}


def _flatten_log_list(prefix, log_list, out):
    """Store a list-of-arrays (test return_logs reshaped output)."""
    for i, arr in enumerate(log_list):
        out[f"{prefix}|{i}"] = np.asarray(arr, dtype=np.float32)


def _flatten_train_logs(prefix, logs, out):
    """Store sim.logs = {'values': [per-batch [per-node arr]], ...}."""
    for kind in ('values', 'errors', 'precisions', 'deltas'):
        for bi, per_node in enumerate(logs[kind]):
            for ni, arr in enumerate(per_node):
                out[f"{prefix}|{kind}|b{bi}|n{ni}"] = np.asarray(arr, dtype=np.float32)


def _flatten_params(prefix, params, out):
    for attr in ('predict_weights', 'predict_biases',
                 'project_weights', 'project_biases',
                 'modulate_weights', 'modulate_biases',
                 'precision_weights', 'precision_biases'):
        for i, arr in enumerate(getattr(params, attr)):
            out[f"{prefix}|{attr}|{i}"] = np.asarray(arr, dtype=np.float32)


def collect_net(name, builder, out):
    """Run the full battery for one net; write flat arrays into ``out``."""
    # Temporal net uses T=2 frames; iteration counts kept multiples of T.
    _, input_label, temporal = builder()
    T = 2 if temporal else 1
    iters_test = 6 if temporal else 4
    iters_train = 6 if temporal else 4
    li2 = 2

    def fresh():
        net, in_label, _ = builder()
        return net, {net[in_label]: 'in_data'}

    # --- test configs (return_logs) ---
    for cfg, kw in (
        ('test_ff1', dict(feedforward_init=True)),
        ('test_ff0', dict(feedforward_init=False)),
        ('test_conv', dict(feedforward_init=True, convergence_threshold=0.05)),
    ):
        net, dmap = fresh()
        sample = _make_sample(net, input_label, temporal, T)
        sim = pcn.Simulation(net)
        res = sim.test([sample], data_map=dmap,
                       iterations_per_sample=iters_test, log_every=1,
                       return_logs=True, **kw)
        for kind in ('values', 'errors', 'precisions', 'deltas'):
            _flatten_log_list(f"{name}|{cfg}|{kind}", res[kind], out)
        for ei, e in enumerate(res['energies']):
            out[f"{name}|{cfg}|energies|{ei}"] = np.asarray(e, dtype=np.float32)

    # --- train configs (save_logs) ---
    for cfg, li in (('train_li0', 0), ('train_li2', li2)):
        net, dmap = fresh()
        sample = _make_sample(net, input_label, temporal, T)
        sim = pcn.Simulation(net)
        sim.train([sample], data_map=dmap, epochs=1,
                  iterations_per_sample=iters_train,
                  learning_iterations_per_sample=li,
                  log_every=1, save_logs=True)
        _flatten_train_logs(f"{name}|{cfg}", sim.logs, out)
        _flatten_params(f"{name}|{cfg}|params", sim.params, out)

    return out


def collect_all():
    """Build the whole matrix and return one flat dict[str, np.ndarray]."""
    out = {}
    for name, builder in build_matrix():
        collect_net(name, builder, out)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'golden_phase2.npz'
    print("Collecting golden matrix (run #1) ...")
    a = collect_all()
    print(f"  {len(a)} arrays")
    # Determinism self-check: a second pass must be bit-identical.
    print("Collecting golden matrix (run #2, determinism check) ...")
    b = collect_all()
    assert a.keys() == b.keys(), "key set changed between runs"
    n_nan = 0
    for k in a:
        if not np.array_equal(a[k], b[k]):
            raise SystemExit(f"NON-DETERMINISTIC array: {k}")
        if np.isnan(np.asarray(a[k])).any():
            n_nan += 1
    if n_nan:
        raise SystemExit(f"{n_nan} arrays contain NaN — adjust matrix configs")
    np.savez(out_path, **a)
    print(f"Saved {len(a)} deterministic, NaN-free arrays to {out_path}")


if __name__ == '__main__':
    main()
