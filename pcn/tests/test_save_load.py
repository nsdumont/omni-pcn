"""Tests for PCNetwork save/load functionality."""

import tempfile
import os
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

import pcn


@pytest.fixture
def built_network():
    """A simple built network for save/load testing."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8, label="input")
        l2 = pcn.Layer(dim=4, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=2, activation=pcn.Softmax(), label="output")
        pcn.Predict(l2, l1)
        pcn.Predict(l3, l2)
    net.build()
    return net


class TestSave:
    def test_save_creates_file(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        result = built_network.save(path)
        assert result == path
        assert path.exists()

    def test_save_default_path(self, built_network, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = built_network.save()
        assert result == Path("saved_models/input_output.h5")
        assert result.exists()

    def test_save_metadata(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        built_network.save(path)
        with h5py.File(path, "r") as f:
            assert "saved_at" in f["metadata"].attrs
            assert "code_version" in f["metadata"].attrs
            # ISO format timestamp
            assert "T" in f["metadata"].attrs["saved_at"]

    def test_save_structure(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        built_network.save(path)
        with h5py.File(path, "r") as f:
            sg = f["structure"]
            assert list(sg.attrs["layer_dims"]) == [8, 4, 2]
            # 3 layers stored
            assert len(sg["layers"]) == 3
            assert sg["layers"]["0"].attrs["label"] == "input"
            assert sg["layers"]["2"].attrs["label"] == "output"
            # 2 predict connections
            assert len(sg["predict_conns"]) == 2

    def test_save_params(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        built_network.save(path)
        with h5py.File(path, "r") as f:
            pg = f["params"]
            assert len(pg["predict_weights"]) == 2
            # Weight shapes: (post_dim, pre_dim)
            assert pg["predict_weights"]["0"].shape == (8, 4)  # l2->l1: (l1.dim, l2.dim)
            assert pg["predict_weights"]["1"].shape == (4, 2)  # l3->l2: (l2.dim, l3.dim) wait...
            # Predict(l2, l1): pre=l2(dim=4), post=l1(dim=8) -> W shape (8, 4)
            # Predict(l3, l2): pre=l3(dim=2), post=l2(dim=4) -> W shape (4, 2)
            assert len(pg["precision_biases"]) == 2
            assert len(pg["precision_weights"]) == 2

    def test_save_with_simulation_results(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        results = {
            "train_energies": [1.0, 0.5, 0.3],
            "test_energies": [0.8, 0.4],
            "accuracy": [0.9, 0.95],
        }
        built_network.save(path, simulation_results=results)
        with h5py.File(path, "r") as f:
            rg = f["results"]
            np.testing.assert_array_almost_equal(rg["train_energies"][...], [1.0, 0.5, 0.3])
            np.testing.assert_array_almost_equal(rg["test_energies"][...], [0.8, 0.4])
            np.testing.assert_array_almost_equal(rg["accuracy"][...], [0.9, 0.95])

    def test_save_without_results_has_no_results_group(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        built_network.save(path)
        with h5py.File(path, "r") as f:
            assert "results" not in f

    def test_save_raises_if_not_built(self):
        net = pcn.PCNetwork(seed=0)
        with pytest.raises(RuntimeError, match="must be built"):
            net.save("foo.h5")


class TestLoad:
    def test_load_restores_params(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        original_weights = [np.array(w) for w in built_network.params.predict_weights]
        built_network.save(path)

        # Rebuild with same structure but different seed -> different weights
        net2 = pcn.PCNetwork(seed=999)
        with net2:
            pcn.Layer(dim=8, label="input")
            pcn.Layer(dim=4, activation=pcn.Relu(), label="hidden")
            pcn.Layer(dim=2, activation=pcn.Softmax(), label="output")
            pcn.Predict(net2._layers[1], net2._layers[0])
            pcn.Predict(net2._layers[2], net2._layers[1])
        net2.build()

        # Weights should differ before load
        assert not np.allclose(
            np.array(net2.params.predict_weights[0]),
            original_weights[0]
        )

        net2.load(path)

        # After load, weights should match
        for orig, loaded in zip(original_weights, net2.params.predict_weights):
            np.testing.assert_array_almost_equal(np.array(loaded), orig)

    def test_load_restores_log_precisions(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        built_network.save(path)

        net2 = pcn.PCNetwork(seed=999)
        with net2:
            pcn.Layer(dim=8, label="input")
            pcn.Layer(dim=4, activation=pcn.Relu(), label="hidden")
            pcn.Layer(dim=2, activation=pcn.Softmax(), label="output")
            pcn.Predict(net2._layers[1], net2._layers[0])
            pcn.Predict(net2._layers[2], net2._layers[1])
        net2.build()
        net2.load(path)

        for orig, loaded in zip(built_network.params.precision_biases, net2.params.precision_biases):
            np.testing.assert_array_almost_equal(np.array(loaded), np.array(orig))
        for orig, loaded in zip(built_network.params.precision_weights, net2.params.precision_weights):
            np.testing.assert_array_almost_equal(np.array(loaded), np.array(orig))

    def test_load_raises_if_not_built(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        built_network.save(path)
        net2 = pcn.PCNetwork(seed=0)
        with pytest.raises(RuntimeError, match="must be built"):
            net2.load(path)

    def test_load_raises_on_dim_mismatch(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        built_network.save(path)

        # Build a network with different dims
        net2 = pcn.PCNetwork(seed=0)
        with net2:
            pcn.Layer(dim=16, label="input")
            pcn.Layer(dim=4, label="hidden")
            pcn.Layer(dim=2, label="output")
            pcn.Predict(net2._layers[1], net2._layers[0])
            pcn.Predict(net2._layers[2], net2._layers[1])
        net2.build()

        with pytest.raises(ValueError, match="Layer dims mismatch"):
            net2.load(path)

    def test_load_returns_self(self, built_network, tmp_path):
        path = tmp_path / "model.h5"
        built_network.save(path)
        result = built_network.load(path)
        assert result is built_network
