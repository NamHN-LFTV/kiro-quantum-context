"""kiro-quantum-context: Quantum-inspired pipeline for LLM context processing."""

from kiro_quantum.core.superposition import entangle
from kiro_quantum.core.interference import interfere
from kiro_quantum.core.wave_function import collapse
from kiro_quantum.types import (
    SemanticDensityChunk,
    InformationSuperpositionMatrix,
    QuantumContextError,
    EntanglementError,
    InterferenceError,
    CollapseError,
)

__all__ = [
    "entangle",
    "interfere",
    "collapse",
    "SemanticDensityChunk",
    "InformationSuperpositionMatrix",
    "QuantumContextError",
    "EntanglementError",
    "InterferenceError",
    "CollapseError",
]
