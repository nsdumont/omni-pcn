"""
PCN Core Module - Network definition classes and data structures.

This module provides the user-facing API for defining predictive coding networks:
- PCNetwork: Context manager for network definition
- Layer: Layer class with value, error, and precision nodes
- Predict, Project, Modulate: Connection classes
- Activation functions: Direct, Relu, Softmax, Tanh, Sigmoid
- Learning rules: Hebbian, Oja, ThreeFactorHebbian
"""

from .activations import Activation, Direct, Relu, Sigmoid, Softmax, Tanh
from .connections import Modulate, Predict, Project
from .layer import Layer, NodeRef
from .learning_rules import Hebbian, LearningRule, Oja, ThreeFactorHebbian
from .network import PCNetwork
from .optimizers import natural_gradient_precision
from .regularization import L1Norm, L2Norm, SIGReg, SumReg, SupConLoss, UnitNorm
from .sparse import SparseWeight
from .state import NetworkParams, NetworkState
from .structure import (
    LayerSpec,
    ModulateConnSpec,
    NetworkStructure,
    PredictConnSpec,
    ProjectConnSpec,
)

__all__ = [
    # State and params
    "NetworkState",
    "NetworkParams",
    "SparseWeight",
    # Structure
    "NetworkStructure",
    "LayerSpec",
    "PredictConnSpec",
    "ProjectConnSpec",
    "ModulateConnSpec",
    # Activations
    "Activation",
    "Direct",
    "Relu",
    "Softmax",
    "Tanh",
    "Sigmoid",
    # Learning rules
    "LearningRule",
    "Hebbian",
    "Oja",
    "ThreeFactorHebbian",
    # Layer and nodes
    "Layer",
    "NodeRef",
    # Connections
    "Predict",
    "Project",
    "Modulate",
    # Network
    "PCNetwork",
    # Regularization
    "L1Norm",
    "L2Norm",
    "UnitNorm",
    "SIGReg",
    "SupConLoss",
    "SumReg",
    # Optimizers
    "natural_gradient_precision",
]
