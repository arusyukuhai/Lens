"""Lens: deterministic old-Replace, exact stochastic law, unbounded-capture DP surrogate.

The two differentiable engines share the discrete wildcard language but differ
in probability semantics and computational cost. See README.md.
"""
from .model import Config, Lens, SerialRule, discrete_features
from .old_replace import replace_with_count, serial_features
from .exact_pushforward import (
    ExactLaw, ExactSerialPushforward, RuleDistribution, RuleSupport,
    TrainableRule, TrainableSerial, deterministic_text, length_distribution,
)

__all__ = ["Config", "Lens", "SerialRule", "discrete_features",
           "replace_with_count", "serial_features", "ExactLaw",
           "ExactSerialPushforward", "RuleDistribution", "RuleSupport",
           "TrainableRule", "TrainableSerial", "deterministic_text",
           "length_distribution"]
