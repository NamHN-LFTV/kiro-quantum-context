"""Wavefunction Collapser — threshold filtering and amplitude-ordered concatenation."""

from kiro_quantum.types import CollapseError, InformationSuperpositionMatrix


def collapse(matrix: InformationSuperpositionMatrix, *, threshold: float = 0.1) -> str:
    """Collapse the interference-processed matrix into a single conflict-free string.

    Filters chunks below the amplitude threshold, orders remaining by
    effective_amplitude descending, and joins their text with newline separators.

    Args:
        matrix: An interference-processed InformationSuperpositionMatrix.
        threshold: Minimum effective_amplitude to include a chunk (default 0.1).
            Must be in [0.0, 1.0].

    Returns:
        A newline-joined string of chunk texts ordered by descending amplitude,
        or an empty string if all chunks are filtered out.

    Raises:
        CollapseError: If threshold is outside [0.0, 1.0], or matrix is None or
            has an empty chunks list.
    """
    # Validate threshold range
    if not (0.0 <= threshold <= 1.0):
        raise CollapseError(
            message=f"threshold must be in [0.0, 1.0], got {threshold}",
            chunk_count=0,
        )

    # Validate matrix is not None
    if matrix is None:
        raise CollapseError(
            message="matrix must not be None",
            chunk_count=0,
        )

    # Validate matrix has chunks
    if not matrix.chunks:
        raise CollapseError(
            message="matrix must contain at least one chunk",
            chunk_count=0,
        )

    # Filter chunks with effective_amplitude >= threshold
    surviving_chunks = [
        chunk for chunk in matrix.chunks if chunk.effective_amplitude >= threshold
    ]

    # All chunks filtered out → return empty string (not an error)
    if not surviving_chunks:
        return ""

    # Sort by effective_amplitude descending (stable sort preserves order for equal amplitudes)
    surviving_chunks.sort(key=lambda c: c.effective_amplitude, reverse=True)

    # Join chunk texts with newline separator, preserving text content exactly
    result = "\n".join(chunk.text for chunk in surviving_chunks)

    # Update pipeline_stage to "collapsed"
    matrix.pipeline_stage = "collapsed"

    return result
