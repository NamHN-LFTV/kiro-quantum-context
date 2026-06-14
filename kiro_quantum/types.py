"""Data models and exception hierarchy for kiro-quantum-context."""

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --- Exception Hierarchy ---


class QuantumContextError(Exception):
    """Base exception for all kiro-quantum-context errors."""

    def __init__(self, message: str, stage: str = "unknown"):
        self.stage = stage
        super().__init__(message)


class EntanglementError(QuantumContextError):
    """Failure during the entangle() stage."""

    def __init__(self, message: str, input_length: int = 0):
        self.input_length = input_length
        super().__init__(message, stage="entangle")


class InterferenceError(QuantumContextError):
    """Failure during the interfere() stage."""

    def __init__(self, message: str):
        super().__init__(message, stage="interfere")


class CollapseError(QuantumContextError):
    """Failure during the collapse() stage."""

    def __init__(self, message: str, chunk_count: int = 0):
        self.chunk_count = chunk_count
        super().__init__(message, stage="collapse")


class DeserializationError(QuantumContextError):
    """Failure during deserialization of JSON data."""

    def __init__(self, field_name: str, reason: str):
        self.field_name = field_name
        self.reason = reason
        super().__init__(
            f"Deserialization failed for '{field_name}': {reason}",
            stage="deserialization",
        )


# --- Pydantic Data Models ---


class SemanticDensityChunk(BaseModel):
    """A single chunk produced by semantic density segmentation."""

    model_config = ConfigDict(strict=True)

    text: str = Field(..., min_length=1, description="Chunk text content")
    amplitude: float = Field(
        ..., ge=0.0, le=1.0, description="Quantum amplitude (information density)"
    )
    phase: float = Field(
        default=0.0, ge=0.0, le=2 * math.pi, description="Phase angle in radians"
    )
    effective_amplitude: float = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validate_default=True,
        description="Amplitude after interference",
    )
    error_state: bool = Field(default=False, description="Whether chunk processing failed")
    chunk_index: int = Field(..., ge=0, description="Original position in text")

    @field_validator("effective_amplitude", mode="before")
    @classmethod
    def default_effective_amplitude(cls, v, info):
        if v is None:
            return info.data.get("amplitude", 0.0)
        return v


class InformationSuperpositionMatrix(BaseModel):
    """Matrix holding all chunks in parallel quantum-inspired states."""

    model_config = ConfigDict(strict=True)

    chunks: list[SemanticDensityChunk] = Field(default_factory=list)
    timed_out_indices: list[int] = Field(
        default_factory=list, description="Indices of chunks that timed out"
    )
    original_text_length: int = Field(default=0, ge=0)
    pipeline_stage: str = Field(
        default="created",
        description="Current pipeline stage: created | entangled | interfered | collapsed",
    )

    def to_json(self) -> str:
        """Serialize to JSON string for persistence."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "InformationSuperpositionMatrix":
        """Deserialize from JSON string."""
        if len(json_str.encode("utf-8")) > 50 * 1024 * 1024:
            raise DeserializationError(
                field_name="json_str",
                reason="JSON string exceeds 50 MB size limit",
            )
        return cls.model_validate_json(json_str)
