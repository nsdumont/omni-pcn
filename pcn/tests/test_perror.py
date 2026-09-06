"""Tests for the perror (precision-weighted error, pi * eps) derived node."""
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import pcn
from pcn.core.structure import _pm_get_pre


class TestPerrorNode:
    def test_property_and_ids(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            a = pcn.Layer(dim=3, activation=pcn.Direct(), label='a')
            b = pcn.Layer(dim=2, activation=pcn.Direct(), label='b')
            p = pcn.Predict(a, b)
        ref = p.perror
        assert ref.node_type == 'perror'
        assert ref.node_type_id == 5
        assert ref.owner_type == 'predict'
        assert ref.dim == 2  # post_dim, same as the error

    def test_perror_rejected_as_post(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            a = pcn.Layer(dim=3, activation=pcn.Direct(), label='a')
            b = pcn.Layer(dim=2, activation=pcn.Direct(), label='b')
            p = pcn.Predict(a, b)
            with pytest.raises(ValueError, match='read-only'):
                pcn.Project(a.value, p.perror)
            with pytest.raises(ValueError, match='read-only'):
                pcn.Modulate(a.value, p.perror)

    def test_pm_get_pre_perror(self):
        errors = (jnp.array([[1.0, -2.0]]),)
        precisions = (jnp.array([[3.0]]),)  # (B, 1) broadcast case
        out = _pm_get_pre((0,), 5, (), errors, (), precisions=precisions)
        np.testing.assert_allclose(out, [[3.0, -6.0]])
        # per-dim precision case
        precisions = (jnp.array([[3.0, 0.5]]),)
        out = _pm_get_pre((0,), 5, (), errors, (), precisions=precisions)
        np.testing.assert_allclose(out, [[3.0, -1.0]])


class TestPerrorIntegration:
    def _run(self, source_node, V):
        """3-layer chain, Project(source_node -> hidden value), 4 sgd iters."""
        LAM = 0.1
        Wb = np.array([[0.5, 0.1], [-0.2, 0.4]], np.float32)
        Wt = np.array([[0.3, -0.5], [0.7, 0.2]], np.float32)
        net = pcn.PCNetwork(seed=0)
        net.config(use_bias=False, learn_precision_weights=False,
                   learn_precision_bias=False)
        with net:
            l_in = pcn.Layer(dim=2, activation=pcn.Direct(), label='in')
            l_h = pcn.Layer(dim=2, activation=pcn.Direct(), label='h')
            l_out = pcn.Layer(dim=2, activation=pcn.Direct(), label='out')
            pcn.Predict(l_in, l_h, init_weight=Wb, learn_weights=False,
                        init_precision=2.0)
            p_top = pcn.Predict(l_h, l_out, init_weight=Wt,
                                learn_weights=False, init_precision=2.0)
            pcn.Project(source_node(p_top), l_h.value,
                        update_rule=pcn.NoLearning(), init_weight=V)
        net.build()
        x = np.array([[1.0, -0.5]], np.float32)
        y = np.array([[0.5, 1.0]], np.float32)
        sim = pcn.Simulation(net)
        res = sim.test([{'in': x, 'out': y}],
                       data_map={l_in: 'in', l_out: 'out'},
                       iterations_per_sample=4, log_every=1, return_logs=True,
                       values_optimizer=optax.sgd(LAM), verbose=False)
        return np.array(res['values'][net['h']])[0]

    def test_perror_project_equals_scaled_error_project(self):
        """With constant precision pi=2, Project(perror)@V must be
        bit-identical to Project(error)@(2V)."""
        V = np.array([[0.05, -0.08], [0.02, 0.06]], np.float32)
        vh_perror = self._run(lambda p: p.perror, V)
        vh_err2v = self._run(lambda p: p.error, 2.0 * V)
        np.testing.assert_array_equal(vh_perror, vh_err2v)

    def test_perror_differs_from_raw_error(self):
        V = np.array([[0.05, -0.08], [0.02, 0.06]], np.float32)
        vh_perror = self._run(lambda p: p.perror, V)
        vh_err = self._run(lambda p: p.error, V)
        assert not np.allclose(vh_perror, vh_err)

    def test_perror_on_unit_precision_conn(self):
        """unit_precision conns store a ones precision: perror == error."""
        LAM = 0.1
        V = np.array([[0.05, -0.08], [0.02, 0.06]], np.float32)

        def run(source):
            Wb = np.array([[0.5, 0.1], [-0.2, 0.4]], np.float32)
            Wt = np.array([[0.3, -0.5], [0.7, 0.2]], np.float32)
            net = pcn.PCNetwork(seed=0)
            net.config(use_bias=False, learn_precision_weights=False,
                       learn_precision_bias=False)
            with net:
                l_in = pcn.Layer(dim=2, activation=pcn.Direct(), label='in')
                l_h = pcn.Layer(dim=2, activation=pcn.Direct(), label='h')
                l_out = pcn.Layer(dim=2, activation=pcn.Direct(), label='out')
                pcn.Predict(l_in, l_h, init_weight=Wb, learn_weights=False)
                p_top = pcn.Predict(l_h, l_out, init_weight=Wt,
                                    learn_weights=False)  # init_precision=1
                pcn.Project(source(p_top), l_h.value,
                            update_rule=pcn.NoLearning(), init_weight=V)
            net.build()
            x = np.array([[1.0, -0.5]], np.float32)
            y = np.array([[0.5, 1.0]], np.float32)
            sim = pcn.Simulation(net)
            res = sim.test([{'in': x, 'out': y}],
                           data_map={l_in: 'in', l_out: 'out'},
                           iterations_per_sample=3, log_every=1,
                           return_logs=True,
                           values_optimizer=optax.sgd(LAM), verbose=False)
            return np.array(res['values'][net['h']])[0]

        np.testing.assert_array_equal(run(lambda p: p.perror),
                                      run(lambda p: p.error))
