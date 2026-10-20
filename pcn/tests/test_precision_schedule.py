"""Precision params follow the predict-weight update schedule.

Precision weights/biases (on a conn that learns them) step exactly when the
predict weights step: once per batch at ``learning_iterations_per_sample=0``,
once per learning iteration at li>0, once per batch in BPTT. A parameter a
conn does not learn (``learn_precision*=False``, ``use_bias=False``,
``learn_weights=False``) stays bit-exact whatever the optimizer, including
momentum carry-over and decoupled weight decay. Fixed predict weights do not
freeze precision.
"""

import numpy as np
import optax
import pytest

import pcn

_PARAM_KINDS = ("predict_weights", "predict_biases", "precision_weights", "precision_biases")


def _two_layer(**predict_kw):
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=6, label="a")
        b = pcn.Layer(dim=4, label="b")
        pcn.Predict(a, b, **predict_kw)
    net.build()
    return net, a, b


def _three_layer(**predict_kw):
    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=8, label="a")
        h = pcn.Layer(dim=6, label="h")
        b = pcn.Layer(dim=4, label="b")
        pcn.Predict(a, h, **predict_kw)
        pcn.Predict(h, b, **predict_kw)
    net.build()
    return net, a, b


def _loader(n_batches, pre_dim, post_dim, batch=5):
    # Fresh numpy dicts every batch: run_batch donates its device inputs.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(batch, pre_dim)).astype(np.float32)
    y = rng.normal(size=(batch, post_dim)).astype(np.float32)
    return [{"x": x.copy(), "y": y.copy()} for _ in range(n_batches)]


def _snapshot(params):
    return {k: [np.array(p) for p in getattr(params, k)] for k in _PARAM_KINDS}


@pytest.mark.parametrize(
    "make_opt", [lambda: optax.sgd(1e-2), lambda: optax.adam(1e-2)], ids=["sgd", "adam"]
)
def test_li_n_equals_n_single_iteration_batches(make_opt):
    """One batch at li=3 == three batches at li=1 (all layers clamped)."""
    finals = []
    for n_batches, li in ((1, 3), (3, 1)):
        net, a, b = _two_layer()
        sim = pcn.Simulation(net)
        sim.train(
            _loader(n_batches, 6, 4),
            data_map={a: "x", b: "y"},
            epochs=1,
            iterations_per_sample=0,
            learning_iterations_per_sample=li,
            params_optimizer=make_opt(),
        )
        finals.append(_snapshot(sim.params))
    for k in _PARAM_KINDS:
        for p_li3, p_li1 in zip(finals[0][k], finals[1][k], strict=True):
            np.testing.assert_allclose(p_li3, p_li1, rtol=1e-5, atol=1e-7, err_msg=k)


@pytest.mark.parametrize("li", [0, 1, 4])
def test_optimizer_steps_once_per_learning_iteration(li):
    """The params optimizer advances max(li, 1) steps per batch."""
    net, a, b = _two_layer()
    sim = pcn.Simulation(net)
    sim.train(
        _loader(1, 6, 4),
        data_map={a: "x", b: "y"},
        epochs=1,
        iterations_per_sample=2,
        learning_iterations_per_sample=li,
        params_optimizer=optax.adam(1e-3),
    )
    counts = optax.tree_utils.tree_get_all_with_path(sim._params_opt_state, "count")
    assert counts
    for _, c in counts:
        assert int(c) == max(li, 1)


def test_precision_bias_takes_one_sgd_step_per_learning_iteration():
    """Closed form: frozen predict weights -> constant errors; the Direct
    precision bias follows li hand-computed SGD steps on
    0.5 * mean_b mean_d (pi * e**2 - log pi)."""
    lr, li = 0.05, 5
    net, a, b = _two_layer(
        learn_precision_weights=False, precision_activation=pcn.Direct(), init_precision=1.0
    )
    pi0 = np.array(net.params.precision_biases[0])
    opt = net.multi_transform(
        {"predict": optax.set_to_zero(), "precision": optax.sgd(lr)},
        default_optim=optax.set_to_zero(),
    )
    sim = pcn.Simulation(net)
    sim.train(
        _loader(1, 6, 4),
        data_map={a: "x", b: "y"},
        epochs=1,
        iterations_per_sample=0,
        learning_iterations_per_sample=li,
        params_optimizer=opt,
        save_logs=True,
    )

    e = np.asarray(sim.logs["errors"][0][0][-1])  # (batch, post); constant
    d = e.shape[-1]
    pi = pi0.astype(np.float64)
    for _ in range(li):
        grad = 0.5 / d * np.mean(e.astype(np.float64) ** 2 - 1.0 / pi, axis=0)
        pi = pi - lr * grad
    np.testing.assert_allclose(np.array(sim.params.precision_biases[0]), pi, rtol=1e-5)
    assert not np.allclose(pi, pi0)


_FROZEN_CASES = [
    ({"learn_precision": False}, ("precision_weights", "precision_biases")),
    ({"learn_precision_weights": False}, ("precision_weights",)),
    ({"learn_precision_bias": False}, ("precision_biases",)),
    ({"use_bias": False}, ("predict_biases",)),
    ({"learn_weights": False}, ("predict_weights", "predict_biases")),
]


@pytest.mark.parametrize("mode", ["li0", "li6", "bptt"])
@pytest.mark.parametrize(
    "predict_kw, frozen", _FROZEN_CASES, ids=[next(iter(kw)) for kw, _ in _FROZEN_CASES]
)
def test_non_learned_params_bit_exact_under_adamw(mode, predict_kw, frozen):
    """Frozen params stay bit-exact under momentum + decoupled weight decay;
    the learned ones move."""
    net, a, b = _three_layer(**predict_kw)
    before = _snapshot(net.params)
    kw = {
        "data_map": {a: "x", b: "y"},
        "epochs": 1,
        "iterations_per_sample": 4,
        "params_optimizer": optax.adamw(1e-2, weight_decay=0.5),
    }
    if mode == "bptt":
        sim = pcn.BPTTSimulation(net)
    else:
        sim = pcn.Simulation(net)
        kw["learning_iterations_per_sample"] = 0 if mode == "li0" else 6
    sim.train(_loader(2, 8, 4), **kw)
    after = _snapshot(sim.params)

    for k in _PARAM_KINDS:
        changed = any(not np.array_equal(x, y) for x, y in zip(before[k], after[k], strict=True))
        if k in frozen:
            for x, y in zip(before[k], after[k], strict=True):
                np.testing.assert_array_equal(x, y, err_msg=k)
        else:
            assert changed, f"{k} should be learned"
