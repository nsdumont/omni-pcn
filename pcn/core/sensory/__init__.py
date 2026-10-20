"""Fixed biological sensory front-ends for OmniPCN.

``VisualInput`` (retina→V1) and ``AuditoryInput`` (cochlea→cortex) are
:class:`~pcn.core.layer.Layer` wrappers that apply a fixed, approximately
invertible feature transform to a modality's raw input *outside* the
predictive-coding energy loop. See :mod:`pcn.core.sensory.base`.
"""

from .audio import (
    AuditoryInput,
    LateralInhibition,
    LeakyIntegrator,
    MelPower,
    PowerCompression,
    STRFBank,
)
from .base import (
    SensoryInput,
    SensoryTransform,
    Sequential,
    to_flat,
    to_shaped,
)
from .vision import (
    ChannelSelect,
    ChannelStandardize,
    ColorOpponent,
    ComplexEnergy,
    DivisiveNormalization,
    DoGCenterSurround,
    GaborBank,
    GaussianBlur,
    ParallelPathways,
    SpatialPool,
    VisualInput,
)

__all__ = [
    "SensoryTransform",
    "Sequential",
    "SensoryInput",
    "to_shaped",
    "to_flat",
    # vision
    "VisualInput",
    "DoGCenterSurround",
    "DivisiveNormalization",
    "GaborBank",
    "ComplexEnergy",
    "ColorOpponent",
    "GaussianBlur",
    "SpatialPool",
    "ChannelStandardize",
    "ChannelSelect",
    "ParallelPathways",
    # audio
    "AuditoryInput",
    "MelPower",
    "PowerCompression",
    "LateralInhibition",
    "LeakyIntegrator",
    "STRFBank",
]
