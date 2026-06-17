"""tseegjepa: multi-scale, hardware-agnostic EEG foundation model (JEPA)."""

from .config import ModelConfig, MaskConfig, PretrainConfig
from .tokenizer import EEGTokenizer
from .encoder import MultiScaleEncoder
from .predictor import Predictor
from .jepa import EEGJepa

__all__ = [
    "ModelConfig",
    "MaskConfig",
    "PretrainConfig",
    "EEGTokenizer",
    "MultiScaleEncoder",
    "Predictor",
    "EEGJepa",
]
