"""
Test that all PCN modules import correctly.
"""

import pytest


class TestPackageImports:
    """Test top-level package imports."""

    def test_import_pcn(self):
        """Test basic package import."""
        import pcn
        assert pcn.__version__ == "0.1.0"

    def test_import_core_classes(self):
        """Test importing core classes from pcn."""
        from pcn import (
            PCNetwork,
            Layer,
            NodeRef,
            Predict,
            Project,
            Modulate,
        )
        assert PCNetwork is not None
        assert Layer is not None
        assert NodeRef is not None
        assert Predict is not None
        assert Project is not None
        assert Modulate is not None

    def test_import_activations(self):
        """Test importing activation classes."""
        from pcn import (
            Activation,
            Direct,
            Relu,
            Softmax,
            Tanh,
            Sigmoid,
        )
        assert Activation is not None
        assert Direct is not None
        assert Relu is not None
        assert Softmax is not None
        assert Tanh is not None
        assert Sigmoid is not None

    def test_import_learning_rules(self):
        """Test importing learning rules."""
        from pcn import (
            LearningRule,
            Hebbian,
            ThreeFactorHebbian,
        )
        assert LearningRule is not None
        assert Hebbian is not None
        assert ThreeFactorHebbian is not None

    def test_import_state_classes(self):
        """Test importing state and params classes."""
        from pcn import (
            NetworkState,
            NetworkParams,
            NetworkStructure,
        )
        assert NetworkState is not None
        assert NetworkParams is not None
        assert NetworkStructure is not None

    def test_import_simulation(self):
        """Test importing Simulation class."""
        from pcn import Simulation
        assert Simulation is not None


class TestSubmoduleImports:
    """Test submodule imports."""

    def test_import_core_module(self):
        """Test importing core module."""
        from pcn import core
        assert hasattr(core, 'PCNetwork')
        assert hasattr(core, 'Layer')
        assert hasattr(core, 'Predict')
        assert hasattr(core, 'NetworkState')
        assert hasattr(core, 'NetworkStructure')

    def test_import_backend_module(self):
        """Test importing backend module."""
        from pcn import backend
        assert hasattr(backend, 'run_batch')
        assert hasattr(backend, 'ACTIVATIONS')
        


class TestBackendImports:
    """Test direct backend submodule imports."""

    def test_import_simulation_backend(self):
        """Test importing consolidated simulation functions."""
        from pcn.backend.simulation import (
            run_batch,
            ACTIVATIONS,
        )
        assert callable(run_batch)
        assert isinstance(ACTIVATIONS, tuple)



class TestCoreImports:
    """Test direct core submodule imports."""

    def test_import_network(self):
        """Test importing network module."""
        from pcn.core.network import PCNetwork, _get_current_network
        assert PCNetwork is not None
        assert callable(_get_current_network)

    def test_import_layer(self):
        """Test importing layer module."""
        from pcn.core.layer import Layer, NodeRef
        assert Layer is not None
        assert NodeRef is not None

    def test_import_connections(self):
        """Test importing connections module."""
        from pcn.core.connections import Predict, Project, Modulate
        assert Predict is not None
        assert Project is not None
        assert Modulate is not None

    def test_import_state(self):
        """Test importing state module."""
        from pcn.core.state import NetworkState, NetworkParams
        assert NetworkState is not None
        assert NetworkParams is not None

    def test_import_structure(self):
        """Test importing structure module."""
        from pcn.core.structure import (
            NetworkStructure,
            LayerSpec,
            PredictConnSpec,
            ProjectConnSpec,
            ModulateConnSpec,
        )
        assert NetworkStructure is not None
        assert LayerSpec is not None
        assert PredictConnSpec is not None
        assert ProjectConnSpec is not None
        assert ModulateConnSpec is not None
