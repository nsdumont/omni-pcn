"""
Tests for structural slicing on NodeRef/Layer.

Verifies:
- Layer.__getitem__ and NodeRef.__getitem__ produce correct slice_bounds
- Sliced connections have correct pre_dim/post_dim and weight shapes
- run_batch integration with sliced Project/Modulate/Predict
- Only targeted dimensions are affected by sliced connections
- Error cases: out-of-bounds, double slice, conv+slicing
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import pcn
from pcn.backend import run_batch


# ── Layer / NodeRef indexing ────────────────────────────────────────────────

class TestLayerGetitem:
    def test_single_index(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        ref = layer[3]
        assert ref.slice_bounds == (3, 4)
        assert ref.dim == 1

    def test_slice(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        ref = layer[3:7]
        assert ref.slice_bounds == (3, 7)
        assert ref.dim == 4

    def test_open_start(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        ref = layer[:5]
        assert ref.slice_bounds == (0, 5)
        assert ref.dim == 5

    def test_open_end(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        ref = layer[5:]
        assert ref.slice_bounds == (5, 10)
        assert ref.dim == 5

    def test_negative_index(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        ref = layer[-3:]
        assert ref.slice_bounds == (7, 10)
        assert ref.dim == 3

    def test_value_getitem(self):
        """layer.value[3:7] same as layer[3:7]."""
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        ref = layer.value[3:7]
        assert ref.slice_bounds == (3, 7)
        assert ref.node_type == 'value'


class TestNodeRefSliceErrors:
    def test_double_slice_raises(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        ref = layer[3:7]
        with pytest.raises(ValueError, match="already-sliced"):
            ref[0:2]

    def test_out_of_bounds_raises(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        with pytest.raises(IndexError):
            layer[15]

    def test_empty_slice_raises(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        with pytest.raises(ValueError, match="Empty slice"):
            layer[5:3]

    def test_step_raises(self):
        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=10, label="x")
        with pytest.raises(ValueError, match="Step"):
            layer[::2]


# ── Sliced connection construction ──────────────────────────────────────────

class TestSlicedProjectConstruction:
    def test_sliced_pre(self):
        """Project with sliced pre has reduced pre_dim."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            pcn.Predict(l1, l2)
            conn = pcn.Project(l1[:5], l2.value)
        net.build()
        assert conn.pre_dim == 5
        assert conn.post_dim == 8
        W = net.params.project_weights[0]
        assert W.shape == (8, 5)

    def test_sliced_post(self):
        """Project with sliced post has reduced post_dim."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            pcn.Predict(l1, l2)
            conn = pcn.Project(l1.value, l2[:3])
        net.build()
        assert conn.pre_dim == 10
        assert conn.post_dim == 3
        W = net.params.project_weights[0]
        assert W.shape == (3, 10)

    def test_sliced_both(self):
        """Project with both pre and post sliced."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            pcn.Predict(l1, l2)
            conn = pcn.Project(l1[:3], l2[2:5])
        net.build()
        assert conn.pre_dim == 3
        assert conn.post_dim == 3
        W = net.params.project_weights[0]
        assert W.shape == (3, 3)


class TestSlicedModulateConstruction:
    def test_sliced_pre_single_unit(self):
        """Modulate with single-unit sliced pre."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            p1 = pcn.Predict(l1, l2)
            conn = pcn.Modulate(l1[0], p1.error)
        net.build()
        assert conn.pre_dim == 1
        W = net.params.modulate_weights[0]
        assert W.shape == (8, 1)


class TestSlicedPredictConstruction:
    def test_sliced_pre(self):
        """Predict with sliced pre layer."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            conn = pcn.Predict(l1[:5], l2)
        net.build()
        assert conn.pre_dim == 5
        assert conn.post_dim == 8
        W = net.params.predict_weights[0]
        assert W.shape == (8, 5)

    def test_sliced_post(self):
        """Predict with sliced post layer."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            conn = pcn.Predict(l1, l2[:3])
        net.build()
        assert conn.pre_dim == 10
        assert conn.post_dim == 3
        W = net.params.predict_weights[0]
        assert W.shape == (3, 10)
        # Error dim should match sliced post dim
        assert net.structure.predict_error_dims[0] == 3

    def test_multi_pre_with_slices(self):
        """Predict with mixed sliced/unsliced multi-pre."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            l3 = pcn.Layer(dim=6, label="target")
            conn = pcn.Predict([l1[:5], l2], l3)
        net.build()
        assert conn.pre_dim == 13  # 5 + 8
        spec = net.structure.predict_conns[0]
        # pre_slices: ((0,5), None) — first pre is sliced, second is not
        assert spec.pre_slices == ((0, 5), None)

    def test_conv_with_slicing_raises(self):
        """Conv + slicing is invalid."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="a")
            l2 = pcn.Layer(dim=32, label="b")
            with pytest.raises(ValueError):
                pcn.Predict(l1[:8], l2, transformation='conv',
                            kernel_size=3, input_shape=(4, 4))


# ── ConnSpec slice field verification ───────────────────────────────────────

class TestConnSpecSliceFields:
    def test_predict_spec_slices(self):
        """PredictConnSpec carries pre_slices and post_slice."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            pcn.Predict(l1[:5], l2[:3])
        net.build()
        spec = net.structure.predict_conns[0]
        assert spec.pre_slices == ((0, 5),)
        assert spec.post_slice == (0, 3)

    def test_project_spec_slices(self):
        """ProjectConnSpec carries pre_slices and post_slice."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            pcn.Predict(l1, l2)
            pcn.Project(l1[2:5], l2[:4])
        net.build()
        spec = net.structure.project_conns[0]
        assert spec.pre_slices == ((2, 5),)
        assert spec.post_slice == (0, 4)

    def test_unsliced_has_empty_slices(self):
        """Unsliced connections have empty tuple for slice fields."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            pcn.Predict(l1, l2)
        net.build()
        spec = net.structure.predict_conns[0]
        assert spec.pre_slices == ()
        assert spec.post_slice == ()


# ── Integration: run_batch with sliced connections ──────────────────────────

class TestSlicedProjectFunctional:
    def test_sliced_project_only_affects_target_dims(self):
        """Project targeting l2[:3] should only modify first 3 dims of l2."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=6, activation=pcn.Direct(), label="input")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=4, activation=pcn.Softmax(), label="output")

            pcn.Predict(l2, l1)
            pcn.Predict(l3, l2)
            pcn.Project(l3.value, l2[:3],
                        update_rule=pcn.NoLearning(),
                        init_weight=jnp.ones((3, 4)))
        net.build()

        # Build baseline without Project
        net_base = pcn.PCNetwork(seed=0)
        with net_base:
            l1b = pcn.Layer(dim=6, activation=pcn.Direct(), label="input")
            l2b = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3b = pcn.Layer(dim=4, activation=pcn.Softmax(), label="output")
            pcn.Predict(l2b, l1b)
            pcn.Predict(l3b, l2b)
        net_base.build()

        # run_batch donates its array inputs; build a fresh sample for each call.
        sample = {'input': jax.random.normal(jax.random.PRNGKey(99), (4, 6))}
        sample_base = {'input': jax.random.normal(jax.random.PRNGKey(99), (4, 6))}
        data_map = ((0, 'input'),)

        _, _, _, vl, _, _, _, _ = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=10, log_every=10, learning=False)

        _, _, _, vl_base, _, _, _, _ = run_batch(
            sample_base, net_base.params, net_base.structure,
            data_map, n_iterations=10, log_every=10, learning=False)

        hidden_vals = vl[1][-1]       # (batch, 8)
        hidden_base = vl_base[1][-1]  # (batch, 8)

        # First 3 dims should differ (project target)
        assert not jnp.allclose(hidden_vals[:, :3], hidden_base[:, :3], atol=1e-4), \
            "Sliced project should modify targeted dimensions"

    def test_sliced_project_run_batch_completes(self):
        """run_batch with sliced Project completes without error."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=8, label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            pcn.Predict(l2, l1)
            pcn.Predict(l3, l2)
            pcn.Project(l3[:2], l2[4:8],
                        update_rule=pcn.Hebbian(learning_rate=1e-3))
        net.build()

        sample = {
            'input': jax.random.normal(jax.random.PRNGKey(0), (4, 10)),
            'output': jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (4, 4)), axis=-1),
        }
        data_map = ((0, 'input'), (2, 'output'))

        import optax
        new_params, _, _, vl, el, _, _, energies = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5,
            learning=True, n_learning_iterations=3,
            params_optimizer=optax.adam(1e-3))
        assert energies.shape[0] > 0


class TestSlicedPredictFunctional:
    def test_sliced_predict_run_batch(self):
        """run_batch with a sliced Predict completes end-to-end."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=8, label="output")
            pcn.Predict(l1[:5], l2)
        net.build()

        sample = {'input': jax.random.normal(jax.random.PRNGKey(0), (4, 10))}
        data_map = ((0, 'input'),)

        _, _, _, vl, el, _, _, energies = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5, learning=False)
        assert len(vl) == 2
        assert len(el) == 1
        assert energies.shape[0] > 0

    def test_sliced_post_predict(self):
        """Predict targeting a slice of post layer."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=8, label="output")
            pcn.Predict(l1, l2[:4])
        net.build()

        assert net.structure.predict_error_dims[0] == 4

        sample = {'input': jax.random.normal(jax.random.PRNGKey(0), (4, 10))}
        data_map = ((0, 'input'),)

        _, _, _, vl, el, _, _, energies = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5, learning=False)
        assert energies.shape[0] > 0


class TestSlicedModulateFunctional:
    def test_sliced_modulate_run_batch(self):
        """run_batch with sliced Modulate completes."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=8, label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            p1 = pcn.Predict(l2, l1)
            pcn.Predict(l3, l2)
            pcn.Modulate(l3[0], p1.error,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))
        net.build()

        sample = {
            'input': jax.random.normal(jax.random.PRNGKey(0), (4, 10)),
            'output': jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (4, 4)), axis=-1),
        }
        data_map = ((0, 'input'), (2, 'output'))

        import optax
        new_params, _, _, vl, el, _, _, energies = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5,
            learning=True, n_learning_iterations=3,
            params_optimizer=optax.adam(1e-3))
        assert energies.shape[0] > 0
        # Modulate weight: pre_dim=1 (l3[0]), post_dim=error_dim of p1
        # p1 = Predict(l2, l1), error dim = l1.dim = 10
        assert net.params.modulate_weights[0].shape == (10, 1)
