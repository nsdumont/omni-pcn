"""Tests for sparse (CSR/CSC) weights — ``pcn.core.sparse`` and ``sparse=`` on
Predict / Project / Modulate.

Kernel tests compare the custom-VJP product against the dense masked matmul;
network tests compare a ``sparse=True`` net against the dense-masked net it
replaces (same init, same data) — the two must train identically.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

import pcn
from pcn.core.sparse import (
    SparseWeight, sparse_matmul, sampled_outer, mask_to_indices, band_indices,
    indices_to_dense_mask, resolve_sparse_mode, AUTO_MAX_DENSITY, AUTO_MIN_SIZE,
)
from pcn.core.network import _make_band_mask


def _rand_mask_weight(rng, post, pre, density=0.3):
    mask = (rng.random((post, pre)) < density).astype(np.float32)
    mask[0, 0] = 1.0
    W = (rng.standard_normal((post, pre)) * mask).astype(np.float32)
    return mask, W


# ============================================================================
# Kernel
# ============================================================================

class TestKernel:
    @pytest.mark.parametrize("post,pre", [(7, 5), (5, 7), (16, 16)])
    def test_forward_and_both_grads_match_dense(self, post, pre):
        rng = np.random.default_rng(0)
        mask, Wd = _rand_mask_weight(rng, post, pre)
        w = SparseWeight.from_dense(Wd, mask)
        assert w.nse == int(mask.sum())
        np.testing.assert_allclose(np.asarray(w.todense()), Wd, atol=1e-6)
        x = jnp.asarray(rng.standard_normal((4, pre)).astype(np.float32))
        rows, cols = np.asarray(w.indices[:, 0]), np.asarray(w.indices[:, 1])

        with jax.default_matmul_precision('highest'):
            y = jax.jit(sparse_matmul)(w, x)
            ref = x @ jnp.asarray(Wd).T
            np.testing.assert_allclose(np.asarray(y), np.asarray(ref), atol=1e-5)

            def loss_s(d, x):
                return 0.5 * jnp.sum(sparse_matmul(w.with_data(d), x) ** 2)

            def loss_d(W, x):
                return 0.5 * jnp.sum((x @ W.T) ** 2)

            gd, gx = jax.jit(jax.grad(loss_s, argnums=(0, 1)))(w.data, x)
            gW, gx_ref = jax.grad(loss_d, argnums=(0, 1))(jnp.asarray(Wd), x)
        np.testing.assert_allclose(np.asarray(gx), np.asarray(gx_ref), atol=1e-4)
        np.testing.assert_allclose(np.asarray(gd), np.asarray(gW)[rows, cols], atol=1e-4)

    def test_empty_rows_and_columns(self):
        mask = np.zeros((6, 5), np.float32)
        mask[1, 0] = mask[1, 3] = mask[4, 3] = 1.0      # rows 0,2,3,5 and cols 1,2,4 empty
        W = np.random.default_rng(1).standard_normal((6, 5)).astype(np.float32) * mask
        w = SparseWeight.from_dense(W, mask)
        x = jnp.ones((3, 5))
        np.testing.assert_allclose(np.asarray(sparse_matmul(w, x)), x @ W.T, atol=1e-6)
        gx = jax.grad(lambda x: jnp.sum(sparse_matmul(w, x)))(x)
        np.testing.assert_allclose(np.asarray(gx), np.ones((3, 6)) @ W, atol=1e-6)

    @pytest.mark.parametrize("nse,chunk", [(1, 4), (2, 4), (7, 3), (8, 4), (9, 4), (20, 6)])
    def test_sampled_outer_chunking(self, nse, chunk):
        rng = np.random.default_rng(nse)
        post, pre, B = 6, 9, 5
        flat = rng.choice(post * pre, size=nse, replace=False)
        idx = jnp.asarray(np.stack([flat // pre, flat % pre], 1).astype(np.int32))
        g = jnp.asarray(rng.standard_normal((B, post)).astype(np.float32))
        x = jnp.asarray(rng.standard_normal((B, pre)).astype(np.float32))
        with jax.default_matmul_precision('highest'):
            out = jax.jit(lambda g, x, i: sampled_outer(g, x, i, chunk=chunk))(g, x, idx)
            ref = (g.T @ x)[idx[:, 0], idx[:, 1]]
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-5)

    def test_from_indices_sorts_and_rejects_duplicates(self):
        w = SparseWeight.from_indices([2, 0, 1], [1, 3, 0], (3, 4), [20., 3., 10.])
        np.testing.assert_array_equal(np.asarray(w.indices), [[0, 3], [1, 0], [2, 1]])
        np.testing.assert_array_equal(np.asarray(w.data), [3., 10., 20.])
        with pytest.raises(ValueError, match="duplicate"):
            SparseWeight.from_indices([0, 0], [1, 1], (3, 4), [1., 2.])
        with pytest.raises(ValueError, match="out of range"):
            SparseWeight.from_indices([3], [0], (3, 4), [1.])

    def test_pytree_roundtrip_and_stop_gradient(self):
        w = SparseWeight.from_indices([0, 1], [1, 0], (2, 2), [1., 2.])
        leaves, treedef = jax.tree_util.tree_flatten(w)
        assert len(leaves) == 5                      # shape is static aux
        w2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(w2, SparseWeight) and w2.shape == (2, 2)
        assert isinstance(jax.lax.stop_gradient(w), SparseWeight)
        assert 'nse=2' in repr(w)


class TestMaskFormats:
    def test_dense_tuple_scipy_bcoo_agree(self):
        rng = np.random.default_rng(3)
        mask, _ = _rand_mask_weight(rng, 9, 7)
        rows_ref, cols_ref = np.nonzero(mask)
        for fmt in ('dense', 'tuple', 'scipy', 'bcoo', 'bcsr'):
            if fmt == 'dense':
                m = mask
            elif fmt == 'tuple':
                perm = rng.permutation(len(rows_ref))          # unsorted, plus a duplicate
                m = (np.concatenate([rows_ref[perm], rows_ref[:1]]),
                     np.concatenate([cols_ref[perm], cols_ref[:1]]))
            elif fmt == 'scipy':
                import scipy.sparse
                m = scipy.sparse.csr_matrix(mask)
            elif fmt == 'bcoo':
                from jax.experimental import sparse as jsparse
                m = jsparse.BCOO.fromdense(jnp.asarray(mask))
            else:
                from jax.experimental import sparse as jsparse
                m = jsparse.BCSR.fromdense(jnp.asarray(mask))
            rows, cols = mask_to_indices(m, (9, 7))
            np.testing.assert_array_equal(rows, rows_ref, err_msg=fmt)
            np.testing.assert_array_equal(cols, cols_ref, err_msg=fmt)
        with pytest.raises(ValueError, match="shape"):
            mask_to_indices(mask, (7, 9))

    def test_band_indices_match_make_band_mask(self):
        for m, n, nb in [(5, 5, 1), (4, 9, 2), (9, 4, 0), (6, 3, 10)]:
            rows, cols = band_indices(m, n, nb)
            dense = indices_to_dense_mask(rows, cols, (m, n))
            np.testing.assert_array_equal(dense, np.asarray(_make_band_mask(m, n, nb)))

    def test_resolve_sparse_mode(self):
        big = (2048, 2048)                                   # 2**22 >= AUTO_MIN_SIZE
        assert resolve_sparse_mode(True, 5, (4, 4))
        assert not resolve_sparse_mode(False, 5, (4, 4))
        assert resolve_sparse_mode('auto', int(0.01 * 2 ** 22), big)
        assert not resolve_sparse_mode('auto', int(0.2 * 2 ** 22), big)
        assert not resolve_sparse_mode('auto', 3, (8, 8))    # too small, however sparse
        assert AUTO_MAX_DENSITY == 0.05 and AUTO_MIN_SIZE == 2 ** 20


# ============================================================================
# Network construction
# ============================================================================

class TestBuild:
    def test_predict_sparse_masked(self):
        rng = np.random.default_rng(0)
        mask, _ = _rand_mask_weight(rng, 6, 8)
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8)
            l2 = pcn.Layer(dim=6)
            conn = pcn.Predict(l1, l2, transformation='masked', weight_mask=mask, sparse=True)
        net.build()
        assert conn.is_sparse and conn.is_masked and conn.weight_mask is None
        assert net.structure.predict_conns[0].is_sparse
        w = net.params.predict_weights[0]
        assert isinstance(w, SparseWeight) and w.shape == (6, 8) and w.nse == int(mask.sum())
        np.testing.assert_array_equal(np.asarray(w.dense_mask()), mask)
        assert net.predict_weight_masks[0].ndim == 0            # no dense mask carried
        assert isinstance(net.dense_weights('predict')[0], jax.Array)

    def test_banded_linear_mask_and_activation_variants(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=7)
            l3 = pcn.Layer(dim=5)
            c_band = pcn.Predict(l1, l2, transformation='banded2', sparse=True)
            c_lin = pcn.Predict(l2, l3, weight_mask=np.eye(5, 7, dtype=np.float32), sparse=True)
            c_act = pcn.Predict(l1, l3, transformation='masked-sigmoid',
                                weight_mask=(np.arange(5)[:, None] < np.arange(10)[None, :]),
                                sparse=True)
        net.build()
        for c in (c_band, c_lin, c_act):
            assert c.is_sparse
        rows, cols = band_indices(7, 10, 2)
        np.testing.assert_array_equal(np.asarray(net.params.predict_weights[0].indices),
                                      np.stack([rows, cols], 1))
        assert c_band.n_bands == 2 and not c_band.is_masked
        assert net.params.predict_weights[1].nse == 5
        assert net.structure.predict_conns[2].post_activation_type == pcn.Sigmoid().type_id

    def test_invalid_combinations_raise(self):
        with pcn.PCNetwork(seed=0):
            l1 = pcn.Layer(dim=8)
            l2 = pcn.Layer(dim=8)
            with pytest.raises(ValueError, match="requires a sparsity structure"):
                pcn.Predict(l1, l2, sparse=True)
            with pytest.raises(ValueError, match="empty"):
                pcn.Predict(l1, l2, transformation='masked',
                            weight_mask=np.zeros((8, 8), np.float32), sparse=True)
            with pytest.raises(ValueError, match="sparse= must be"):
                pcn.Predict(l1, l2, transformation='masked',
                            weight_mask=np.eye(8, dtype=np.float32), sparse='yes')
        with pcn.PCNetwork(seed=0):
            l1 = pcn.Layer(dim=16)
            l2 = pcn.Layer(dim=16)
            with pytest.raises(ValueError, match="conv"):
                pcn.Predict(l1, l2, transformation='conv', kernel_size=3,
                            input_shape=(4, 4), sparse=True)

    def test_auto_falls_back_to_dense_on_small_matrices(self):
        mask = np.eye(6, 8, dtype=np.float32)
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8)
            l2 = pcn.Layer(dim=6)
            conn = pcn.Predict(l1, l2, transformation='masked', weight_mask=mask, sparse='auto')
        net.build()
        assert not conn.is_sparse and conn.is_masked
        np.testing.assert_array_equal(conn.weight_mask, mask)      # densified index set
        assert not net.structure.predict_conns[0].is_sparse
        W = np.asarray(net.params.predict_weights[0])
        assert W.shape == (6, 8) and np.allclose(W * (1 - mask), 0.0)
        assert net.predict_weight_masks[0].ndim == 2

    def test_index_format_mask_without_sparse_is_densified(self):
        import scipy.sparse
        mask = np.eye(5, 4, dtype=np.float32)
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4)
            l2 = pcn.Layer(dim=5)
            conn = pcn.Predict(l1, l2, transformation='masked',
                               weight_mask=scipy.sparse.coo_matrix(mask))
        net.build()
        assert conn.is_masked and not conn.is_sparse
        np.testing.assert_array_equal(conn.weight_mask, mask)

    def test_project_and_modulate_sparse_init(self):
        rng = np.random.default_rng(0)
        mask, _ = _rand_mask_weight(rng, 6, 6)
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=6)
            l2 = pcn.Layer(dim=6)
            pcn.Predict(l1, l2)
            pcn.Project(l1, l2, transformation='masked', weight_mask=mask,
                        update_rule=pcn.Hebbian(1e-3), sparse=True)
            pcn.Modulate(l1, l2, transformation='masked', weight_mask=mask,
                         update_rule=pcn.NoLearning(), sparse=True, use_bias=False)
        net.build()
        wp, wm = net.params.project_weights[0], net.params.modulate_weights[0]
        assert isinstance(wp, SparseWeight) and isinstance(wm, SparseWeight)
        assert net.structure.project_conns[0].is_sparse and net.structure.modulate_conns[0].is_sparse
        # no-bias Modulate initialises near identity (1 + 0.01 noise) on the structure
        assert np.allclose(np.asarray(wm.data), 1.0, atol=0.1)


# ============================================================================
# Training equivalence: sparse vs the dense-masked path it replaces
# ============================================================================

def _pair_of_nets(rng, hebb_rule=None, mod_rule=None, gd_loss=False, seed=0):
    """Two identical 3-layer nets, one dense-masked, one sparse, sharing init."""
    d_in, d_h, d_out = 8, 12, 4
    mask, W0 = _rand_mask_weight(rng, d_in, d_h, 0.4)          # hidden -> input predict
    mask_p, Wp0 = _rand_mask_weight(rng, d_h, d_in, 0.4)      # input -> hidden project/modulate
    Wp0 = (Wp0 * 0.05).astype(np.float32)                    # small: value drives integrate
    nets, layers = [], []
    for sparse in (False, True):
        net = pcn.PCNetwork(seed=seed)
        with net:
            l_in = pcn.Layer(dim=d_in, activation=pcn.Direct(), label='in')
            l_h = pcn.Layer(dim=d_h, activation=pcn.LeakyRelu(), label='h')
            l_out = pcn.Layer(dim=d_out, activation=pcn.Direct(), label='out')
            p = pcn.Predict(l_h, l_in, transformation='masked', weight_mask=mask,
                            init_weight=W0, learn_weights=True, sparse=sparse,
                            learn_precision_weights=False, learn_precision_bias=False)
            pcn.Predict(l_out, l_h, learn_precision_weights=False, learn_precision_bias=False)
            if hebb_rule is not None:
                pcn.Project(l_in, l_h, transformation='masked', weight_mask=mask_p,
                            init_weight=Wp0, update_rule=hebb_rule(), sparse=sparse)
            if mod_rule is not None:
                pcn.Modulate(l_in, l_h, transformation='masked', weight_mask=mask_p,
                             init_weight=Wp0, update_rule=mod_rule(), sparse=sparse,
                             use_bias=True)       # 1 + W f(pre): near-identity gain
            if gd_loss:
                pcn.Project(l_h, p.error, transformation='masked', weight_mask=mask,
                            init_weight=W0,
                            update_rule=pcn.GradientDescent(
                                loss_fn=((p.error,), lambda e: jnp.sum(e ** 2))),
                            sparse=sparse)
        net.build()
        nets.append(net); layers.append((l_in, l_h, l_out))
    return nets, layers, mask


def _loader(rng, n=3, B=5):
    return [{'x': rng.standard_normal((B, 8)).astype(np.float32),
             'y': rng.standard_normal((B, 4)).astype(np.float32)} for _ in range(n)]


def _dense(w):
    return np.asarray(w.todense() if isinstance(w, SparseWeight) else w)


class TestTrainingEquivalence:
    @pytest.mark.parametrize("opt", ['sgd', 'adam'])
    def test_predict_sparse_trains_like_dense_masked(self, opt):
        rng = np.random.default_rng(0)
        (net_d, net_s), (ld, ls), mask = _pair_of_nets(rng)
        loader = _loader(rng)
        params_opt = optax.sgd(0.05) if opt == 'sgd' else optax.adam(1e-2)
        energies = []
        with jax.default_matmul_precision('highest'):
            for net, (l_in, l_h, l_out) in ((net_d, ld), (net_s, ls)):
                sim = pcn.Simulation(net)
                sim.train(loader, data_map={l_in: 'x', l_out: 'y'}, epochs=2,
                          iterations_per_sample=5, values_optimizer=optax.sgd(0.1),
                          params_optimizer=params_opt, verbose=False)
                energies.append(np.asarray([np.asarray(e[-1]) for e in sim.train_energies]))
        assert isinstance(net_s.params.predict_weights[0], SparseWeight)
        Wd = _dense(net_d.params.predict_weights[0])
        Ws = _dense(net_s.params.predict_weights[0])
        np.testing.assert_allclose(Ws, Wd, atol=2e-4, rtol=1e-4)
        np.testing.assert_allclose(Ws * (1 - mask), 0.0)
        np.testing.assert_allclose(energies[1], energies[0], rtol=1e-4, atol=1e-5)
        # the unmasked conn is unaffected either way
        np.testing.assert_allclose(_dense(net_s.params.predict_weights[1]),
                                   _dense(net_d.params.predict_weights[1]), atol=2e-4)

    def test_multi_transform_and_weight_decay_on_sparse_leaf(self):
        rng = np.random.default_rng(1)
        (net_d, net_s), (ld, ls), mask = _pair_of_nets(rng)
        loader = _loader(rng)
        with jax.default_matmul_precision('highest'):
            for net, (l_in, l_h, l_out) in ((net_d, ld), (net_s, ls)):
                opt = net.multi_transform(
                    {'predict_weights': optax.chain(optax.add_decayed_weights(1e-2), optax.sgd(0.05))},
                    default_optim=optax.sgd(0.01))
                pcn.Simulation(net).train(
                    loader, data_map={l_in: 'x', l_out: 'y'}, epochs=1,
                    iterations_per_sample=4, values_optimizer=optax.sgd(0.1),
                    params_optimizer=opt, verbose=False)
        np.testing.assert_allclose(_dense(net_s.params.predict_weights[0]),
                                   _dense(net_d.params.predict_weights[0]), atol=2e-4, rtol=1e-4)

    @pytest.mark.parametrize("rule", ['hebbian', 'oja', 'threefactor'])
    def test_project_hebbian_family_sparse_matches_dense(self, rule):
        rng = np.random.default_rng(2)
        rules = {
            'hebbian': lambda: pcn.Hebbian(learning_rate=1e-2),
            'oja': lambda: pcn.Oja(learning_rate=1e-2),
            'threefactor': lambda: pcn.ThreeFactorHebbian(
                learning_rate=1e-2, reward_fn=(('y',), lambda y: jnp.sum(y ** 2, axis=-1))),
        }
        (net_d, net_s), (ld, ls), _ = _pair_of_nets(rng, hebb_rule=rules[rule], mod_rule=rules[rule])
        init_p = _dense(net_d.params.project_weights[0]).copy()
        loader = _loader(rng)
        with jax.default_matmul_precision('highest'):
            for net, (l_in, l_h, l_out) in ((net_d, ld), (net_s, ls)):
                pcn.Simulation(net).train(
                    loader, data_map={l_in: 'x', l_out: 'y'}, epochs=1,
                    iterations_per_sample=4, values_optimizer=optax.sgd(0.1),
                    params_optimizer=optax.sgd(0.01), verbose=False)
        for kind in ('project', 'modulate'):
            ws, wd = getattr(net_s.params, f'{kind}_weights')[0], getattr(net_d.params, f'{kind}_weights')[0]
            assert isinstance(ws, SparseWeight)
            assert np.all(np.isfinite(_dense(wd))) and np.all(np.isfinite(_dense(ws)))
            assert not np.allclose(_dense(wd), init_p, atol=1e-6), f"{kind} weights did not learn"
            np.testing.assert_allclose(_dense(ws), _dense(wd), atol=2e-4, rtol=1e-4, err_msg=kind)
        np.testing.assert_allclose(_dense(net_s.params.predict_weights[0]),
                                   _dense(net_d.params.predict_weights[0]), atol=2e-4, rtol=1e-4)

    def test_gd_loss_project_sparse_matches_dense(self):
        rng = np.random.default_rng(4)
        (net_d, net_s), (ld, ls), _ = _pair_of_nets(rng, gd_loss=True)
        loader = _loader(rng)
        with jax.default_matmul_precision('highest'):
            for net, (l_in, l_h, l_out) in ((net_d, ld), (net_s, ls)):
                pcn.Simulation(net).train(
                    loader, data_map={l_in: 'x', l_out: 'y'}, epochs=1,
                    iterations_per_sample=4, values_optimizer=optax.sgd(0.1),
                    params_optimizer=optax.sgd(0.01), verbose=False)
        ws, wd = net_s.params.project_weights[0], net_d.params.project_weights[0]
        assert isinstance(ws, SparseWeight)
        assert net_s.structure.gd_loss_project
        np.testing.assert_allclose(_dense(ws), _dense(wd), atol=2e-4, rtol=1e-4)

    def test_repeated_train_test_calls_keep_structure_alive(self):
        rng = np.random.default_rng(5)
        (_, net_s), (_, (l_in, l_h, l_out)), mask = _pair_of_nets(rng)
        loader = _loader(rng, n=2)
        sim = pcn.Simulation(net_s)
        kw = dict(data_map={l_in: 'x', l_out: 'y'}, epochs=1, iterations_per_sample=3,
                  values_optimizer=optax.sgd(0.1), params_optimizer=optax.adam(1e-3), verbose=False)
        for _ in range(3):
            sim.train(loader, **kw)
            sim.test(loader, data_map={l_in: 'x'}, iterations_per_sample=3,
                     values_optimizer=optax.sgd(0.1), verbose=False)
        # (like the dense masked test: ``sim.params`` is the live copy after a
        # test() call — net.params is only synced back by train())
        w = sim.params.predict_weights[0]
        assert isinstance(w, SparseWeight) and not w.indices.is_deleted()
        np.testing.assert_array_equal(np.asarray(w.dense_mask()), mask)

    def test_bptt_sparse_matches_dense(self):
        rng = np.random.default_rng(6)
        (net_d, net_s), (ld, ls), _ = _pair_of_nets(rng)
        loader = _loader(rng, n=2)
        with jax.default_matmul_precision('highest'):
            for net, (l_in, l_h, l_out) in ((net_d, ld), (net_s, ls)):
                pcn.BPTTSimulation(net).train(
                    loader, data_map={l_in: 'x', l_out: 'y'}, epochs=1,
                    iterations_per_sample=3, values_optimizer=optax.sgd(0.1),
                    params_optimizer=optax.sgd(0.01), verbose=False)
        assert isinstance(net_s.params.predict_weights[0], SparseWeight)
        np.testing.assert_allclose(_dense(net_s.params.predict_weights[0]),
                                   _dense(net_d.params.predict_weights[0]), atol=2e-4, rtol=1e-4)

    def test_backprop_simulation_rejects_sparse(self):
        rng = np.random.default_rng(7)
        (_, net_s), (_, (l_in, l_h, l_out)), _ = _pair_of_nets(rng)
        with pytest.raises(NotImplementedError, match="sparse"):
            pcn.BackpropSimulation(net_s, objective_fn=lambda values, sample: 0.0)


class TestSaveLoad:
    def test_roundtrip_and_mismatch(self, tmp_path):
        rng = np.random.default_rng(8)
        (net_d, net_s), (ld, ls), _ = _pair_of_nets(rng)
        path = tmp_path / 'sparse.h5'
        net_s.save(path)
        # fresh identical (sparse) net loads the SparseWeight back bit-for-bit
        (_, net_s2), _, _ = _pair_of_nets(np.random.default_rng(8))
        net_s2.load(path)
        w, w2 = net_s.params.predict_weights[0], net_s2.params.predict_weights[0]
        assert isinstance(w2, SparseWeight) and w2.shape == w.shape
        for fld in ('data', 'indices', 'indptr', 't_indptr', 't_perm'):
            np.testing.assert_array_equal(np.asarray(getattr(w2, fld)), np.asarray(getattr(w, fld)))
        # the dense-masked twin refuses the sparse file (and vice versa)
        with pytest.raises(ValueError, match="sparse= mismatch"):
            net_d.load(path)
        path_d = tmp_path / 'dense.h5'
        net_d.save(path_d)
        with pytest.raises(ValueError, match="sparse= mismatch"):
            net_s.load(path_d)
