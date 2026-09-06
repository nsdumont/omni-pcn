"""
PCN Core Module - Network definition classes and data structures.

This module provides the user-facing API for defining predictive coding networks:
- PCNetwork: Context manager for network definition
- Layer: Layer class with value, error, and precision nodes
- Predict, Project, Modulate: Connection classes
- Activation functions: Direct, Relu, Softmax, Tanh, Sigmoid
- Learning rules: Hebbian, Oja, ThreeFactorHebbian
"""

from .state import NetworkState, NetworkParams
from .sparse import SparseWeight
from .structure import (
    NetworkStructure,
    LayerSpec,
    PredictConnSpec,
    ProjectConnSpec,
    ModulateConnSpec,
)
from .activations import Activation, Direct, Relu, Softmax, Tanh, Sigmoid
from .learning_rules import LearningRule, Hebbian, Oja, ThreeFactorHebbian
from .layer import Layer, NodeRef
from .connections import Predict, Project, Modulate
from .network import PCNetwork
from .regularization import L1Norm, L2Norm, UnitNorm, SIGReg, SupConLoss, SumReg
from .optimizers import natural_gradient_precision

__all__ = [
    # State and params
    'NetworkState',
    'NetworkParams',
    'SparseWeight',
    # Structure
    'NetworkStructure',
    'LayerSpec',
    'PredictConnSpec',
    'ProjectConnSpec',
    'ModulateConnSpec',
    # Activations
    'Activation',
    'Direct',
    'Relu',
    'Softmax',
    'Tanh',
    'Sigmoid',
    # Learning rules
    'LearningRule',
    'Hebbian',
    'Oja',
    'ThreeFactorHebbian',
    # Layer and nodes
    'Layer',
    'NodeRef',
    # Connections
    'Predict',
    'Project',
    'Modulate',
    # Network
    'PCNetwork',
    # Regularization
    'L1Norm',
    'L2Norm',
    'UnitNorm',
    'SIGReg',
    'SupConLoss',
    'SumReg',
    # Optimizers
    'natural_gradient_precision',
]
