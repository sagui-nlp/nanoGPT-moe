"""Convenience imports for the model package.

Allows scripts to simply do:

        from model import GPT, GPTConfig, MoELayer

instead of importing from individual modules. This also ensures that
relative imports work correctly when the package is imported from the
repository root (e.g. in training scripts).
"""

from .gpt import GPT, GPTConfig  # noqa: F401
from .moe_layer import Expert, MoELayer  # noqa: F401

__all__ = [
    "GPT",
    "GPTConfig",
    "MoELayer",
    "Expert",
]
