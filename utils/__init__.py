"""Utility package bootstrap.

Avoid importing heavy optional dependencies at package import time.
Import submodules directly as needed, e.g. `from utils.llm_client import LLMClient`.
"""

__version__ = "1.0.0"

# Light re-exports only.
from .logger import setup_logger, log  # noqa: F401

__all__ = [
    'setup_logger',
    'log',
]
