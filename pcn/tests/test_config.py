"""Tests for the JSON-based configuration system."""

import json
import os
import tempfile

import pytest

import pcn
from pcn.config import DEFAULTS, load_config, _DEFAULT
from pcn.core.activations import activation_from_name, ACTIVATION_REGISTRY, Relu, Tanh, Direct


# --- load_config ---

def test_load_default_config():
    """load_config() with no args returns the built-in defaults."""
    cfg = load_config()
    assert "model" in cfg
    assert "train" in cfg
    assert "test" in cfg
    assert cfg["model"]["dynamics_rate"] == 0.1



def test_load_custom_config():
    """A custom JSON file partially overrides defaults."""
    custom = {"model": {"dynamics_rate": 0.5}, "train": {"iterations_per_sample": 30}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(custom, f)
        path = f.name
    try:
        cfg = load_config(path)
        # Overridden
        assert cfg["model"]["dynamics_rate"] == 0.5
        assert cfg["train"]["iterations_per_sample"] == 30
        # Non-overridden defaults preserved
        assert cfg["test"]["iterations_per_sample"] == 50
    finally:
        os.unlink(path)


# --- activation_from_name ---

def test_activation_from_name():
    """All registered names resolve correctly."""
    for name, cls in ACTIVATION_REGISTRY.items():
        act = activation_from_name(name)
        assert isinstance(act, cls)
    # Case-insensitive
    assert isinstance(activation_from_name("RELU"), Relu)
    assert isinstance(activation_from_name("Tanh"), Tanh)


def test_activation_from_name_invalid():
    """Unknown activation name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown activation"):
        activation_from_name("nonexistent")


# --- PCNetwork.config ---

def test_network_defaults_from_json():
    """PCNetwork._defaults are populated from the JSON at construction."""
    net = pcn.PCNetwork(seed=0)
    assert net._defaults["dynamics_rate"] == 0.1



def test_network_config_kwargs():
    """net.config(**kwargs) overrides specific defaults."""
    net = pcn.PCNetwork(seed=0)
    net.config(dynamics_rate=0.05, init_precision=2.0)
    assert net._defaults["dynamics_rate"] == 0.05
    assert net._defaults["init_precision"] == 2.0
    # Others unchanged
    assert net._defaults["learn_precision_weights"] == True
    assert net._defaults["learn_precision_bias"] == True


def test_network_config_init_log_precision_bc():
    """Legacy ``init_log_precision`` kwarg is converted via exp."""
    import math
    net = pcn.PCNetwork(seed=0)
    net.config(init_log_precision=1.0)
    assert net._defaults["init_precision"] == pytest.approx(math.e)


def test_network_config_from_json():
    """net.config(config_file=...) loads the model section."""
    custom = {"model": {"dynamics_rate": 0.2, "init_precision": 1.5}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(custom, f)
        path = f.name
    try:
        net = pcn.PCNetwork(seed=0)
        net.config(config_file=path)
        assert net._defaults["dynamics_rate"] == 0.2
        assert net._defaults["init_precision"] == 1.5
    finally:
        os.unlink(path)


def test_network_config_from_json_init_log_precision_bc():
    """Legacy ``init_log_precision`` in JSON is converted via exp."""
    import math
    custom = {"model": {"init_log_precision": 1.0}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(custom, f)
        path = f.name
    try:
        net = pcn.PCNetwork(seed=0)
        net.config(config_file=path)
        assert net._defaults["init_precision"] == pytest.approx(math.e)
    finally:
        os.unlink(path)


def test_network_config_kwargs_override_json():
    """Explicit kwargs win over JSON values."""
    custom = {"model": {"dynamics_rate": 0.2}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(custom, f)
        path = f.name
    try:
        net = pcn.PCNetwork(seed=0)
        net.config(config_file=path, dynamics_rate=0.99)
        assert net._defaults["dynamics_rate"] == 0.99
    finally:
        os.unlink(path)


def test_network_config_string_activation():
    """String activation in config() is converted to Activation instance."""
    net = pcn.PCNetwork(seed=0)
    net.config(activation="relu")
    assert isinstance(net._defaults["activation"], Relu)


# --- Predict uses config defaults ---

def test_predict_uses_config_defaults():
    """Predict picks up init_precision from config."""
    net = pcn.PCNetwork(seed=0)
    net.config(init_precision=2.0)
    with net:
        l1 = pcn.Layer(dim=4, label="a")
        l2 = pcn.Layer(dim=4, label="b")
        p = pcn.Predict(l1, l2)
    assert p.init_precision == 2.0


def test_predict_explicit_overrides_config():
    """Explicit args to Predict() override config defaults."""
    net = pcn.PCNetwork(seed=0)
    net.config(init_precision=2.0)
    with net:
        l1 = pcn.Layer(dim=4, label="a")
        l2 = pcn.Layer(dim=4, label="b")
        p = pcn.Predict(l1, l2, init_precision=0.5)
    assert p.init_precision == 0.5


def test_predict_init_log_precision_bc():
    """Legacy ``init_log_precision`` kwarg still works on Predict()."""
    import math
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=4, label="a")
        l2 = pcn.Layer(dim=4, label="b")
        p = pcn.Predict(l1, l2, init_log_precision=1.0)
    assert p.init_precision == pytest.approx(math.e)
    assert p.init_log_precision == pytest.approx(1.0)


# --- Layer uses config defaults ---

def test_layer_uses_config_activation():
    """Layer picks up activation from config when not specified."""
    net = pcn.PCNetwork(seed=0)
    net.config(activation="tanh")
    with net:
        l = pcn.Layer(dim=4, label="x")
    assert isinstance(l.activation, Tanh)


def test_layer_explicit_activation_overrides():
    """Explicit activation on Layer wins over config."""
    net = pcn.PCNetwork(seed=0)
    net.config(activation="tanh")
    with net:
        l = pcn.Layer(dim=4, activation=Tanh(), label="x")
    assert isinstance(l.activation, Tanh)


# --- Simulation.config ---

def test_simulation_config():
    """sim.config() stores values in train/test defaults."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=4, label="a")
        l2 = pcn.Layer(dim=4, label="b")
        pcn.Predict(l1, l2)
    net.build()
    sim = pcn.Simulation(net)
    sim.config(iterations_per_sample=50)
    assert sim._train_defaults["iterations_per_sample"] == 50
    assert sim._test_defaults["iterations_per_sample"] == 50



def test_simulation_config_convergence_threshold():
    """sim.config() sets convergence_threshold in both train and test."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=4, label="a")
        l2 = pcn.Layer(dim=4, label="b")
        pcn.Predict(l1, l2)
    net.build()
    sim = pcn.Simulation(net)
    sim.config(convergence_threshold=1e-4)
    assert sim._train_defaults["convergence_threshold"] == 1e-4
    assert sim._test_defaults["convergence_threshold"] == 1e-4




def test_simulation_config_from_json():
    """sim.config(config_file=...) loads train and test sections."""
    custom = {"train": {"iterations_per_sample": 200}, "test": {"iterations_per_sample": 5}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(custom, f)
        path = f.name
    try:
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4, label="a")
            l2 = pcn.Layer(dim=4, label="b")
            pcn.Predict(l1, l2)
        net.build()
        sim = pcn.Simulation(net)
        sim.config(config_file=path)
        assert sim._train_defaults["iterations_per_sample"] == 200
        assert sim._test_defaults["iterations_per_sample"] == 5
    finally:
        os.unlink(path)
