"""
Tests for functional Project and Modulate connections.

Verifies:
- Project additive effect on values
- Modulate multiplicative effect on errors
- Hebbian weight learning for Project/Modulate
- Pre-sorted connection lists are built correctly
- stop_gradient prevents value grads from flowing through routing paths
- Integration with run_batch (full pipeline)
"""

import pytest
import jax
import jax.numpy as jnp
import optax
import numpy as np

import pcn
from pcn.backend.simulation import (
    _inference_step, _combined_step, _compute_energy,
    _apply_project_modulate_internal, _apply_project_modulate_values,
    ACTIVATIONS,
)
from pcn.backend import run_batch


# ── Helper: build a small network with Project/Modulate ─────────────────────

def _build_project_value_net(seed=0):
    """Network with a Project targeting a value node."""
    net = pcn.PCNetwork(seed=seed)
    with net:
        l_in = pcn.Layer(dim=4, label='input')
        l_hid = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
        l_out = pcn.Layer(dim=2, activation=pcn.Relu(), label='output')

        pcn.Predict(l_hid, l_in)
        pcn.Predict(l_out, l_hid)

        # Project: output value adds to hidden value
        pcn.Project(
            l_out.value, l_hid.value,
            update_rule=pcn.Hebbian(learning_rate=1e-2),
        )
    net.build()
    return net, (l_in, l_hid, l_out)


def _build_modulate_error_net(seed=0):
    """Network with a Modulate targeting an error node."""
    net = pcn.PCNetwork(seed=seed)
    with net:
        l_in = pcn.Layer(dim=4, label='input')
        l_hid = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
        l_out = pcn.Layer(dim=2, activation=pcn.Relu(), label='output')

        p1 = pcn.Predict(l_hid, l_in)
        pcn.Predict(l_out, l_hid)

        # Modulate: output value multiplicatively modulates p1's error
        pcn.Modulate(
            l_out.value, p1.error,
            update_rule=pcn.Hebbian(learning_rate=1e-2),
        )
    net.build()
    return net, (l_in, l_hid, l_out)


# ── Tests: NetworkStructure pre-sorted lists ────────────────────────────────

class TestPreSortedLists:
    def test_project_value_sorting(self):
        """Project targeting value should appear in project_conns_value."""
        net, _ = _build_project_value_net()
        s = net.structure
        assert len(s.project_conns_value) == 1
        assert len(s.project_conns_internal) == 0
        weight_idx, spec = s.project_conns_value[0]
        assert spec.post_node_type == 0  # value

    def test_modulate_error_sorting(self):
        """Modulate targeting error should appear in modulate_conns_internal."""
        net, _ = _build_modulate_error_net()
        s = net.structure
        assert len(s.modulate_conns_internal) == 1
        assert len(s.modulate_conns_value) == 0
        weight_idx, spec = s.modulate_conns_internal[0]
        assert spec.post_node_type == 1  # error

    def test_complex_network_sorting(self, complex_network):
        """complex_network has 1 Project(value) and 1 Modulate(error)."""
        net, _ = complex_network
        s = net.structure
        assert len(s.project_conns_value) == 1
        assert len(s.project_conns_internal) == 0
        assert len(s.modulate_conns_internal) == 1
        assert len(s.modulate_conns_value) == 0


# ── Tests: ConnSpec get_pre / get_post / apply ──────────────────────────────

class TestConnSpecMethods:
    """Test the get_pre, get_post, and apply methods on ConnSpecs."""

    def test_predict_conn_get_pre_single(self):
        """PredictConnSpec.get_pre with single pre index applies activation."""
        from pcn.core.structure import PredictConnSpec
        spec = PredictConnSpec(
            pre_idx=(0,), post_idx=1,
            has_fixed_weights=False,
            learn_precision_weights=True,
            learn_precision_bias=True,
        )
        values = (jnp.array([[-1.0, 2.0]]),)
        errors = ()
        activation_fns = tuple(ACTIVATIONS[t] for t in (1,))  # ReLU
        result = spec.get_pre(values, errors, activation_fns)
        assert jnp.allclose(result, jnp.array([[0.0, 2.0]]))

    def test_predict_conn_get_pre_multi(self):
        """PredictConnSpec.get_pre with multi-pre concatenates."""
        from pcn.core.structure import PredictConnSpec
        spec = PredictConnSpec(
            pre_idx=(0, 1), post_idx=2,
            has_fixed_weights=False,
            learn_precision_weights=True,
            learn_precision_bias=True,
        )
        values = (jnp.array([[1.0, 2.0]]), jnp.array([[3.0]]))
        errors = ()
        activation_fns = tuple(ACTIVATIONS[t] for t in (0, 0))  # Direct, Direct
        result = spec.get_pre(values, errors, activation_fns)
        assert result.shape == (1, 3)
        assert jnp.allclose(result, jnp.array([[1.0, 2.0, 3.0]]))

    def test_project_conn_get_pre_value(self):
        """ProjectConnSpec.get_pre with value nodes applies activation."""
        from pcn.core.structure import ProjectConnSpec
        spec = ProjectConnSpec(
            pre_idx=(0,), pre_node_type=0,
            post_idx=1, post_node_type=0,
            learning_rule_type=0, learning_rate=0.001,
            reward_fn_idx=-1,
        )
        values = (jnp.array([[-1.0, 2.0]]),)
        errors = ()
        activation_fns = tuple(ACTIVATIONS[t] for t in (1,))  # ReLU
        result = spec.get_pre(values, errors, activation_fns)
        assert jnp.allclose(result, jnp.array([[0.0, 2.0]]))

    def test_project_conn_get_pre_error(self):
        """ProjectConnSpec.get_pre with error nodes uses identity."""
        from pcn.core.structure import ProjectConnSpec
        spec = ProjectConnSpec(
            pre_idx=(0,), pre_node_type=1,
            post_idx=1, post_node_type=0,
            learning_rule_type=0, learning_rate=0.001,
            reward_fn_idx=-1,
        )
        values = ()
        errors = (jnp.array([[-1.0, 2.0]]),)
        activation_fns = tuple(ACTIVATIONS[t] for t in (1,))  # ReLU (irrelevant)
        result = spec.get_pre(values, errors, activation_fns)
        assert jnp.allclose(result, jnp.array([[-1.0, 2.0]]))

    def test_predict_conn_apply_linear(self):
        """PredictConnSpec.apply for linear transform = W @ f(pre)."""
        from pcn.core.structure import PredictConnSpec
        spec = PredictConnSpec(
            pre_idx=(0,), post_idx=1,
            has_fixed_weights=False,
            learn_precision_weights=True,
            learn_precision_bias=True,
        )
        pre_act = jnp.ones((2, 3))
        W = jnp.eye(4, 3)
        b = jnp.zeros(4)
        result = spec.apply(pre_act, W, b)
        assert result.shape == (2, 4)

    def test_project_conn_apply_linear(self):
        """ProjectConnSpec.apply for linear transform (no bias)."""
        from pcn.core.structure import ProjectConnSpec
        spec = ProjectConnSpec(
            pre_idx=(0,), pre_node_type=0,
            post_idx=1, post_node_type=0,
            learning_rule_type=0, learning_rate=0.001,
            reward_fn_idx=-1,
        )
        pre_act = jnp.ones((2, 3))
        W = jnp.eye(4, 3)
        result = spec.apply(pre_act, W)
        assert result.shape == (2, 4)


# ── Tests: Project additive effect on values ────────────────────────────────

class TestProjectValues:
    def test_project_changes_values(self):
        """After inference, the Project targeting hidden values should
        make hidden values different from a network without the Project."""
        net_proj, layers_proj = _build_project_value_net(seed=0)
        l_in_proj = layers_proj[0]

        # Build the same network without the Project
        net_base = pcn.PCNetwork(seed=0)
        with net_base:
            l_in_b = pcn.Layer(dim=4, label='input')
            l_hid_b = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            l_out_b = pcn.Layer(dim=2, activation=pcn.Relu(), label='output')
            pcn.Predict(l_hid_b, l_in_b)
            pcn.Predict(l_out_b, l_hid_b)
        net_base.build()

        # run_batch donates its array inputs; build a fresh sample for each call.
        sample_proj = {'input': jax.random.normal(jax.random.PRNGKey(99), (4, 4))}
        sample_base = {'input': jax.random.normal(jax.random.PRNGKey(99), (4, 4))}
        data_map_proj = ((l_in_proj._idx, 'input'),)
        data_map_base = ((l_in_b._idx, 'input'),)

        params_proj, _, _, vl_proj, _, _, _, _ = run_batch(
            sample_proj, net_proj.params, net_proj.structure,
            data_map_proj, n_iterations=10, log_every=10, learning=False)

        params_base, _, _, vl_base, _, _, _, _ = run_batch(
            sample_base, net_base.params, net_base.structure,
            data_map_base, n_iterations=10, log_every=10, learning=False)

        # Hidden values should differ due to Project influence
        hidden_proj = vl_proj[1][-1]  # (batch, 3)
        hidden_base = vl_base[1][-1]
        assert not jnp.allclose(hidden_proj, hidden_base, atol=1e-4), \
            "Project should change hidden values"


# ── Tests: Modulate multiplicative effect on errors ─────────────────────────

class TestModulateErrors:
    def test_modulate_changes_energy(self):
        """A Modulate on errors should change the energy landscape."""
        net_mod, layers_mod = _build_modulate_error_net(seed=0)
        l_in_mod = layers_mod[0]

        net_base = pcn.PCNetwork(seed=0)
        with net_base:
            l_in_b = pcn.Layer(dim=4, label='input')
            l_hid_b = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            l_out_b = pcn.Layer(dim=2, activation=pcn.Relu(), label='output')
            pcn.Predict(l_hid_b, l_in_b)
            pcn.Predict(l_out_b, l_hid_b)
        net_base.build()

        # run_batch donates its array inputs; build a fresh sample for each call.
        sample_mod = {'input': jax.random.normal(jax.random.PRNGKey(99), (4, 4))}
        sample_base = {'input': jax.random.normal(jax.random.PRNGKey(99), (4, 4))}
        data_map_mod = ((l_in_mod._idx, 'input'),)
        data_map_base = ((l_in_b._idx, 'input'),)

        _, _, _, _, _, _, _, energies_mod = run_batch(
            sample_mod, net_mod.params, net_mod.structure,
            data_map_mod, n_iterations=10, log_every=10, learning=False)

        _, _, _, _, _, _, _, energies_base = run_batch(
            sample_base, net_base.params, net_base.structure,
            data_map_base, n_iterations=10, log_every=10, learning=False)

        # Energies should differ
        assert not jnp.allclose(energies_mod[-1], energies_base[-1], atol=1e-4), \
            "Modulate should change the energy"


# ── Tests: Hebbian weight learning ─────────────────────────────────────────

class TestHebbianLearning:
    def test_project_weights_change(self):
        """Project weights should change after combined_step with learning."""
        net, layers = _build_project_value_net(seed=0)
        l_in, l_hid, l_out = layers

        # Clamp both input and output so all layers have active signals
        sample = {
            'input': jax.random.normal(jax.random.PRNGKey(99), (4, 4)),
            'output': jnp.abs(jax.random.normal(jax.random.PRNGKey(100), (4, 2))),
        }
        data_map = ((l_in._idx, 'input'), (l_out._idx, 'output'))

        pw_before = np.array(net.params.project_weights[0])

        new_params, _, _, _, _, _, _, _ = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5,
            learning=True, n_learning_iterations=3,
            params_optimizer=optax.adam(1e-3))

        pw_after = np.array(new_params.project_weights[0])
        assert not np.allclose(pw_before, pw_after, atol=1e-8), \
            "Project weights should be updated by Hebbian learning"

    def test_modulate_weights_change(self):
        """Modulate weights should change after combined_step with learning."""
        net, layers = _build_modulate_error_net(seed=0)
        l_in, l_hid, l_out = layers

        # Clamp both input and output so errors and values are non-zero
        sample = {
            'input': jax.random.normal(jax.random.PRNGKey(99), (4, 4)),
            'output': jnp.abs(jax.random.normal(jax.random.PRNGKey(100), (4, 2))),
        }
        data_map = ((l_in._idx, 'input'), (l_out._idx, 'output'))

        mw_before = np.array(net.params.modulate_weights[0])

        new_params, _, _, _, _, _, _, _ = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5,
            learning=True, n_learning_iterations=3,
            params_optimizer=optax.adam(1e-3))

        mw_after = np.array(new_params.modulate_weights[0])
        assert not np.allclose(mw_before, mw_after, atol=1e-8), \
            "Modulate weights should be updated by Hebbian learning"


# ── Tests: Oja weight learning ─────────────────────────────────────────────

class TestOjaLearning:
    def test_project_weights_change(self):
        """Project weights should change after combined_step with Oja rule."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=4, label='input')
            l_hid = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            l_out = pcn.Layer(dim=2, activation=pcn.Relu(), label='output')

            pcn.Predict(l_hid, l_in)
            pcn.Predict(l_out, l_hid)

            pcn.Project(
                l_out.value, l_hid.value,
                update_rule=pcn.Oja(learning_rate=1e-2),
            )
        net.build()

        sample = {
            'input': jax.random.normal(jax.random.PRNGKey(99), (4, 4)),
            'output': jnp.abs(jax.random.normal(jax.random.PRNGKey(100), (4, 2))),
        }
        data_map = ((l_in._idx, 'input'), (l_out._idx, 'output'))

        pw_before = np.array(net.params.project_weights[0])

        new_params, _, _, _, _, _, _, _ = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5,
            learning=True, n_learning_iterations=3,
            params_optimizer=optax.adam(1e-3))

        pw_after = np.array(new_params.project_weights[0])
        assert not np.allclose(pw_before, pw_after, atol=1e-8), \
            "Project weights should be updated by Oja learning"

    def test_oja_stays_bounded(self):
        """Oja rule should keep weights bounded unlike plain Hebbian."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=4, label='input')
            l_hid = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            l_out = pcn.Layer(dim=2, activation=pcn.Relu(), label='output')

            pcn.Predict(l_hid, l_in)
            pcn.Predict(l_out, l_hid)

            pcn.Project(
                l_out.value, l_hid.value,
                update_rule=pcn.Oja(learning_rate=1e-2),
            )
        net.build()

        sample = {
            'input': jax.random.normal(jax.random.PRNGKey(99), (4, 4)),
            'output': jnp.abs(jax.random.normal(jax.random.PRNGKey(100), (4, 2))),
        }
        data_map = ((l_in._idx, 'input'), (l_out._idx, 'output'))

        new_params, _, _, _, _, _, _, _ = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=50, log_every=50,
            learning=True, n_learning_iterations=50,
            params_optimizer=optax.adam(1e-3))

        pw = np.array(new_params.project_weights[0])
        assert np.all(np.isfinite(pw)), \
            "Oja weights should remain finite (no NaN/Inf)"


# ── Tests: Full pipeline with complex_network fixture ───────────────────────

class TestComplexNetworkPipeline:
    def test_run_batch_with_project_modulate(self, complex_network):
        """run_batch should complete without error for network with both
        Project and Modulate connections."""
        net, (l1, l2, l3, l4) = complex_network
        sample = {
            'input': jax.random.normal(jax.random.PRNGKey(0), (4, 32)),
            'output': jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (4, 4)), axis=-1),
        }
        data_map = ((l1._idx, 'input'), (l4._idx, 'output'))

        new_params, _, _, values_log, errors_log, _, _, energies = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5,
            learning=True, n_learning_iterations=3,
            params_optimizer=optax.adam(1e-3))

        # Basic sanity: shapes are correct
        assert len(values_log) == 4  # 4 layers
        assert len(errors_log) == 3  # 3 predict connections
        assert energies.shape[0] > 0

    def test_learning_updates_all_weights(self, complex_network):
        """Both Project and Modulate weights should update during learning."""
        net, (l1, l2, l3, l4) = complex_network
        sample = {
            'input': jax.random.normal(jax.random.PRNGKey(0), (4, 32)),
            'output': jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (4, 4)), axis=-1),
        }
        data_map = ((l1._idx, 'input'), (l4._idx, 'output'))

        pw_before = np.array(net.params.project_weights[0])
        mw_before = np.array(net.params.modulate_weights[0])

        new_params, _, _, _, _, _, _, _ = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5,
            learning=True, n_learning_iterations=3,
            params_optimizer=optax.adam(1e-3))

        assert not np.allclose(pw_before, np.array(new_params.project_weights[0]), atol=1e-8)
        assert not np.allclose(mw_before, np.array(new_params.modulate_weights[0]), atol=1e-8)


# ── Tests: GradientDescent learning rule ────────────────────────────────────

class TestGradientDescentRule:
    def test_gradient_descent_type_id(self):
        # loss_fn is required; type_id is still 2.
        rule = pcn.GradientDescent(loss_fn=(('x',), lambda x: jnp.sum(x ** 2)))
        assert rule.type_id == 2
        assert rule.loss_fn is not None

    def test_gradient_descent_requires_loss_fn(self):
        with pytest.raises(TypeError):
            pcn.GradientDescent()

    def test_gradient_descent_with_loss_fn(self):
        def my_loss(vals):
            return jnp.sum(vals ** 2)

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4, label='input')
            l2 = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            pcn.Predict(l2, l1)
            rule = pcn.GradientDescent(
                loss_fn=(l1.value, my_loss))
        assert rule.loss_fn is not None
        inputs, fn = rule.loss_fn
        assert fn is my_loss

    def test_gradient_descent_loss_fn_validates(self):
        """Passing a bare callable should raise ValueError."""
        def my_loss(x):
            return jnp.sum(x ** 2)

        with pytest.raises(ValueError, match="must be a .inputs, fn. tuple"):
            pcn.GradientDescent(loss_fn=my_loss)

    def test_loss_fn_collected_in_build(self):
        """loss_fn should be collected into net._loss_fns during build."""
        def my_loss(vals):
            return jnp.sum(vals ** 2)

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4, label='input')
            l2 = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            pcn.Predict(l2, l1)
            pcn.Project(
                l2.value, l1.value,
                update_rule=pcn.GradientDescent(
                    loss_fn=(l1.value, my_loss)))
        net.build()

        assert len(net._loss_fns) == 1
        resolved_inputs, fn = net._loss_fns[0]
        assert fn is my_loss
        # Resolved input should be (node_type_id=0, layer_idx=0) for l1.value
        assert resolved_inputs == (0, 0)
        # The spec should have loss_fn_idx=0
        assert net.structure.project_conns[0].loss_fn_idx == 0

    def test_loss_fn_with_sample_key(self):
        """loss_fn inputs can reference sample dict keys."""
        def my_loss(vals, labels):
            return jnp.mean((vals - labels) ** 2)

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4, label='input')
            l2 = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            pcn.Predict(l2, l1)
            pcn.Project(
                l2.value, l1.value,
                update_rule=pcn.GradientDescent(
                    loss_fn=((l1.value, 'label'), my_loss)))
        net.build()

        resolved_inputs, fn = net._loss_fns[0]
        assert fn is my_loss
        # First element: resolved NodeRef, second: sample key string
        assert resolved_inputs == ((0, 0), 'label')
        assert net.structure.loss_fn_sample_keys == ('label',)


# ── Tests: reward_fn collection ─────────────────────────────────────────────

class TestRewardFnCollection:
    def test_reward_fn_collected(self):
        def my_reward(err):
            return jnp.ones(err.shape[0])

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4, label='input')
            l2 = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            p1 = pcn.Predict(l2, l1)
            pcn.Modulate(
                l2.value, p1.error,
                update_rule=pcn.ThreeFactorHebbian(
                    learning_rate=1e-3,
                    reward_fn=((p1.error,), my_reward)))
        net.build()

        assert len(net._reward_fns) == 1
        resolved_inputs, fn = net._reward_fns[0]
        assert fn is my_reward
        # Resolved form: single node ref (node_type, idx)
        assert isinstance(resolved_inputs, tuple) and len(resolved_inputs) == 1
        assert isinstance(resolved_inputs[0], tuple)
        assert net.structure.modulate_conns[0].reward_fn_idx == 0


# ── Tests: Project/Modulate effects persist into the carried state ──────────

class TestProjectModulatePersistence:
    """Value-targeting Project/Modulate are explicit state operators that
    persist (integrating drive) across inference iterations; error- and
    precision-targeting routing is reflected in the recomputed/logged state.
    """

    def test_value_project_integrates_across_iterations(self):
        """A value->value Project with no Predict touching the layer evolves
        purely by its recurrence. Value Projects combine **Jacobi**-style: all
        pre-activations are read from the frozen pre-update state v[t], so the
        self-loop (a*m) and input drive (x) both read m[t] and sum,
        order-independently, giving m_{t+1} = (1+a) m_t + x. This proves the
        routing persists into the carried value and integrates across iterations.
        """
        a = 0.5
        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=2, activation=pcn.Direct(), label='in')
            l_aux = pcn.Layer(dim=2, activation=pcn.Direct(), label='aux')
            mem = pcn.Layer(dim=2, activation=pcn.Direct(), label='mem')
            # A Predict that does NOT touch `mem` -> mem has zero energy grad.
            pcn.Predict(l_in, l_aux)
            # B = I (input drive) and self-loop A' = a*I (additive integrator).
            pcn.Project(l_in.value, mem.value,
                        update_rule=pcn.NoLearning(), init_weight=jnp.eye(2))
            pcn.Project(mem.value, mem.value,
                        update_rule=pcn.NoLearning(), init_weight=a * jnp.eye(2))
        net.build()

        sim = pcn.Simulation(net)
        x = jnp.ones((1, 2), dtype=jnp.float32)
        res = sim.test([{'in': x}], data_map={l_in: 'in'},
                       iterations_per_sample=6, log_every=1)
        mem_traj = np.array(res['values'][net['mem']])[0, :, 0]  # (iters,)

        # Jacobi recurrence invariant between consecutive carried states (x == 1).
        assert np.allclose(mem_traj[1:], (1 + a) * mem_traj[:-1] + 1.0, atol=1e-4)
        # And it genuinely integrates (grows), not a bounded transient.
        assert mem_traj[-1] > mem_traj[0] * 3

    def test_error_project_reflected_in_logged_errors(self):
        """An error-targeting Project changes the recomputed/logged error of
        its target connection (carry/logs match what the energy consumed)."""
        def build(with_error_project):
            net = pcn.PCNetwork(seed=1)
            with net:
                l_in = pcn.Layer(dim=3, activation=pcn.Direct(), label='in')
                l_out = pcn.Layer(dim=3, activation=pcn.Direct(), label='out')
                src = pcn.Layer(dim=3, activation=pcn.Direct(), label='src')
                p = pcn.Predict(l_in, l_out)
                if with_error_project:
                    pcn.Project(src.value, p.error,
                                update_rule=pcn.NoLearning(),
                                init_weight=jnp.eye(3))
            net.build()
            return net, l_in, src, p

        x = jnp.ones((1, 3), dtype=jnp.float32)
        s = 2.0 * jnp.ones((1, 3), dtype=jnp.float32)

        net_a, lin_a, _, p_a = build(False)
        net_b, lin_b, src_b, p_b = build(True)

        res_a = pcn.Simulation(net_a).test(
            [{'in': x}], data_map={lin_a: 'in'},
            iterations_per_sample=3, log_every=1)
        res_b = pcn.Simulation(net_b).test(
            [{'in': x, 'src': s}], data_map={lin_b: 'in', src_b: 'src'},
            iterations_per_sample=3, log_every=1)

        err_a = np.array(res_a['errors'][p_a._idx])[0, -1]
        err_b = np.array(res_b['errors'][p_b._idx])[0, -1]
        # The error routing (additive +I @ src) must show up in the logged error.
        assert not np.allclose(err_a, err_b, atol=1e-4)
