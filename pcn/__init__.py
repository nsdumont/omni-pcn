"""
PCN: Predictive Coding Networks

A JAX-based Python package for defining, simulating, and training
predictive coding networks on graphs.

Basic Usage:
    import pcn

    # Define network
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=784, label="input")
        l2 = pcn.Layer(dim=256, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=10, activation=pcn.Softmax(), label="output")

        pcn.Predict(l2, l1, learning_rate=1e-3)
        pcn.Predict(l3, l2, learning_rate=1e-3)

    net.build()

    # Create simulation
    sim = pcn.Simulation(
        net, dataloader,
        learning=True,
        data_map={l1: 'image', l3: 'label'}
    )

    # Run training
    sim.run(epochs=10, iterations_per_sample=50, lr=0.1)

Modules:
    pcn.core: Network definition classes and data structures
    pcn.backend: JAX-compiled inference and learning functions
    pcn.simulation: High-level Simulation class

Connection Types:
    Predict: Standard PC connection (layer -> layer)
    Project: Additive non-PC connection (node -> node)
    Modulate: Multiplicative non-PC connection (node -> node)

Learning Rules:
    Hebbian: dW = lr * post @ pre.T
    Oja: dW = lr * (post @ pre.T - post^2 * W)
    ThreeFactorHebbian: dW = lr * reward * post @ pre.T
"""

__version__ = "0.1.0"

# Core classes
from .core.network import PCNetwork
from .core.layer import Layer, NodeRef
from .core.connections import Predict, PredictRes, PredictConv, PredictConvPool, PredictTransConv, Project, Modulate
from .core.memory import Memory
from .core.sensory import (
    SensoryTransform, Sequential, SensoryInput, VisualInput, AuditoryInput,
    DoGCenterSurround, DivisiveNormalization, GaborBank, ComplexEnergy,
    ColorOpponent, GaussianBlur, SpatialPool, ChannelStandardize,
    ChannelSelect, ParallelPathways,
    MelPower, PowerCompression, LateralInhibition, LeakyIntegrator, STRFBank,
)
from .core.activations import (
    Activation, Direct, Relu, Softmax, Tanh, HardTanh, Sigmoid, LeakyRelu, Gelu, Elu,
    LayerNorm, NWTA, Poisson, MemoryActivation, Leaky, StochasticActivation, Stochastic,
)
from .core.learning_rules import LearningRule, Hebbian, Oja, ThreeFactorHebbian, GradientDescent, NoLearning
from .core.state import NetworkState, NetworkParams
from .core.structure import NetworkStructure
from .core.regularization import L1Norm, L2Norm, UnitNorm, SIGReg, SupConLoss, SumReg

# Configuration
from .config import load_config
from .core.activations import activation_from_name, ACTIVATION_REGISTRY

# High-level API
from .simulation import Simulation
from .backprop_simulation import BackpropSimulation
from .bptt_simulation import BPTTSimulation

# Optimizers
from .core.optimizers import natural_gradient_precision

# Submodules
from . import core
from . import backend

__all__ = [
    # Version
    '__version__',
    # Core classes
    'PCNetwork',
    'Layer',
    'NodeRef',
    'Predict',
    'PredictRes',
    'PredictConv',
    'PredictConvPool',
    'PredictTransConv',
    'Project',
    'Modulate',
    'Memory',
    # Sensory front-ends
    'SensoryTransform',
    'Sequential',
    'SensoryInput',
    'VisualInput',
    'AuditoryInput',
    'DoGCenterSurround',
    'DivisiveNormalization',
    'GaborBank',
    'ComplexEnergy',
    'ColorOpponent',
    'GaussianBlur',
    'SpatialPool',
    'ChannelStandardize',
    'ChannelSelect',
    'ParallelPathways',
    'MelPower',
    'PowerCompression',
    'LateralInhibition',
    'LeakyIntegrator',
    'STRFBank',
    # Activations
    'Activation',
    'Direct',
    'Relu',
    'Softmax',
    'Tanh',
    'HardTanh',
    'Sigmoid',
    'LeakyRelu',
    'Gelu',
    'LayerNorm',
    'NWTA',
    'Elu',
    'Poisson',
    'MemoryActivation',
    'Leaky',
    'StochasticActivation',
    'Stochastic',
    # Regularization
    'L1Norm',
    'L2Norm',
    'UnitNorm',
    'SIGReg',
    'SupConLoss',
    'SumReg',
    # Learning rules
    'LearningRule',
    'Hebbian',
    'Oja',
    'ThreeFactorHebbian',
    'GradientDescent',
    'NoLearning',
    # State and structure
    'NetworkState',
    'NetworkParams',
    'NetworkStructure',
    # Configuration
    'load_config',
    'activation_from_name',
    'ACTIVATION_REGISTRY',
    # High-level API
    'Simulation',
    'BackpropSimulation',
    'BPTTSimulation',
    # Optimizers
    'natural_gradient_precision',
    # Submodules
    'core',
    'backend',
]
