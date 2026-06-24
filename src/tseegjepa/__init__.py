"""tseegjepa: multi-scale, hardware-agnostic EEG foundation model (JEPA)."""

from .config import ModelConfig, MaskConfig, PretrainConfig
from .tokenizer import EEGTokenizer
from .encoder import MultiScaleEncoder
from .predictor import Predictor
from .jepa import EEGJepa
from .jepa_hier import HierarchicalEEGJepa
from .pooling import SpatialAnchorPool

__all__ = [
    "ModelConfig",
    "MaskConfig",
    "PretrainConfig",
    "EEGTokenizer",
    "MultiScaleEncoder",
    "Predictor",
    "EEGJepa",
    "HierarchicalEEGJepa",
    "SpatialAnchorPool",
]
