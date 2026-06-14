"""Smoke tests — verify package imports, public API, and pipeline."""

import asyncio
import sys

import pytest
from hypothesis import given, settings

sys.path.insert(0, "tests")
from conftest import valid_chunk


class TestPackageImports:
    """Verify that all package imports work correctly."""

    def test_import_kiro_quantum(self):
        import kiro_quantum
        assert kiro_quantum is not None

    def test_public_functions_accessible(self):
        from kiro_quantum import entangle, interfere, collapse
        assert callable(entangle)
        assert callable(interfere)
        assert callable(collapse)

    def test_public_models_accessible(self):
        from kiro_quantum import SemanticDensityChunk, InformationSuperpositionMatrix
        assert SemanticDensityChunk is not None
        assert InformationSuperpositionMatrix is not None

    def test_exception_hierarchy_accessible(self):
        from kiro_quantum import (
            QuantumContextError, EntanglementError, InterferenceError, CollapseError,
        )
        assert issubclass(EntanglementError, QuantumContextError)
        assert issubclass(InterferenceError, QuantumContextError)
        assert issubclass(CollapseError, QuantumContextError)

    def test_submodule_imports(self):
        from kiro_quantum.core import superposition, interference, wave_function
        assert superposition is not None
        assert interference is not None
        assert wave_function is not None

    def test_all_exports(self):
        import kiro_quantum
        expected = {
            "entangle", "interfere", "collapse",
            "SemanticDensityChunk", "InformationSuperpositionMatrix",
            "QuantumContextError", "EntanglementError", "InterferenceError", "CollapseError",
        }
        assert expected.issubset(set(kiro_quantum.__all__))


class TestConftest:
    """Verify that conftest strategies produce valid instances."""

    @given(chunk=valid_chunk())
    @settings(max_examples=5)
    def test_valid_chunk_strategy(self, chunk):
        from kiro_quantum import SemanticDensityChunk
        assert isinstance(chunk, SemanticDensityChunk)
        assert 0.0 <= chunk.amplitude <= 1.0
        assert len(chunk.text) >= 1


class TestPipelineSmoke:
    """Minimal end-to-end pipeline smoke test."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        from kiro_quantum import entangle, interfere, collapse

        text = "The sky is blue. Water flows downhill. Cats are friendly."
        matrix = await entangle(text)
        assert matrix.pipeline_stage == "entangled"
        assert len(matrix.chunks) > 0

        result = interfere(matrix, "The sky is blue.")
        assert result.pipeline_stage == "interfered"

        output = collapse(result, threshold=0.0)
        assert isinstance(output, str)
        assert len(output) > 0
