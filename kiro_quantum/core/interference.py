"""Interference Engine — claim extraction, alignment scoring, and phase assignment."""

import logging
import math
import re

from kiro_quantum.types import (
    InformationSuperpositionMatrix,
    InterferenceError,
    SemanticDensityChunk,
)

logger = logging.getLogger(__name__)

# Phase constants
_PHASE_CONSTRUCTIVE = 0.0  # θ=0: aligned or not addressed
_PHASE_DESTRUCTIVE = math.pi  # θ=π: contradicts steering_base

# Negation indicators for contradiction detection
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor", "none",
    "cannot", "can't", "won't", "wouldn't", "shouldn't",
    "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
})

# Words that indicate superseded/outdated information
_CONTRADICTION_INDICATORS = frozenset({
    "incorrect", "outdated", "deprecated", "obsolete", "replaced",
    "superseded", "no longer", "previously", "formerly", "was",
    "old", "removed", "discontinued", "invalid", "wrong",
    "changed", "updated", "revised", "corrected",
})

# Sentence boundary regex (same as superposition.py for consistency)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

# Word tokenization regex
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def interfere(
    matrix: InformationSuperpositionMatrix, steering_base: str
) -> InformationSuperpositionMatrix:
    """Apply quantum interference to chunks based on steering_base authority.

    Maps each chunk to a complex number representation r·e^(iθ) where:
    - r is the quantum amplitude (information density)
    - θ is the phase (0 for constructive, π for destructive)

    Constructive interference (θ=0): chunk aligns with or is not addressed by steering_base.
    Destructive interference (θ=π): chunk contradicts a claim in steering_base.

    Args:
        matrix: The Information Superposition Matrix from entangle().
        steering_base: The authority document text for conflict resolution.

    Returns:
        A new InformationSuperpositionMatrix with phase and effective_amplitude
        assigned to each chunk, and pipeline_stage set to "interfered".

    Raises:
        InterferenceError: If an unrecoverable error occurs during processing.
    """
    # Handle empty matrix → return empty matrix with pipeline_stage="interfered"
    if not matrix.chunks:
        return InformationSuperpositionMatrix(
            chunks=[],
            timed_out_indices=matrix.timed_out_indices,
            original_text_length=matrix.original_text_length,
            pipeline_stage="interfered",
        )

    # Handle empty/None steering_base → log warning, treat all as valid (θ=0)
    if not steering_base or not steering_base.strip():
        logger.warning(
            "No steering_base provided to interfere() — "
            "treating all chunks as valid (Phase θ=0)"
        )
        # Return matrix with all chunks at θ=0, effective_amplitude = amplitude
        interfered_chunks = [
            SemanticDensityChunk(
                text=chunk.text,
                amplitude=chunk.amplitude,
                phase=_PHASE_CONSTRUCTIVE,
                effective_amplitude=chunk.amplitude,
                error_state=chunk.error_state,
                chunk_index=chunk.chunk_index,
            )
            for chunk in matrix.chunks
        ]
        return InformationSuperpositionMatrix(
            chunks=interfered_chunks,
            timed_out_indices=matrix.timed_out_indices,
            original_text_length=matrix.original_text_length,
            pipeline_stage="interfered",
        )

    # Extract claims from steering_base
    try:
        claims = _extract_claims(steering_base)
    except Exception as e:
        raise InterferenceError(
            message=f"Failed to extract claims from steering_base: {e}"
        ) from e

    # Assign phase to each chunk based on alignment with claims
    try:
        interfered_chunks = []
        for chunk in matrix.chunks:
            phase = _determine_phase(chunk.text, claims)
            effective_amplitude = _compute_effective_amplitude(
                chunk.amplitude, phase
            )
            interfered_chunks.append(
                SemanticDensityChunk(
                    text=chunk.text,
                    amplitude=chunk.amplitude,
                    phase=phase,
                    effective_amplitude=effective_amplitude,
                    error_state=chunk.error_state,
                    chunk_index=chunk.chunk_index,
                )
            )
    except Exception as e:
        raise InterferenceError(
            message=f"Failed during chunk phase assignment: {e}"
        ) from e

    return InformationSuperpositionMatrix(
        chunks=interfered_chunks,
        timed_out_indices=matrix.timed_out_indices,
        original_text_length=matrix.original_text_length,
        pipeline_stage="interfered",
    )


def _extract_claims(steering_base: str) -> list[tuple[set[str], str, bool]]:
    """Extract subject-assertion pairs from steering_base text.

    Parses the steering_base into sentences and extracts claims as tuples of:
    - subject_words: set of key words identifying the subject
    - assertion_text: the full assertion sentence
    - has_negation: whether the claim contains negation/contradiction indicators

    Returns:
        List of (subject_words, assertion_text, has_negation) tuples.
    """
    # Split steering_base into sentences
    sentences = _SENTENCE_BOUNDARY_RE.split(steering_base.strip())

    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Tokenize sentence into words
        words = _tokenize(sentence)
        if not words:
            continue

        # Subject words: content words (excluding very common stopwords)
        subject_words = _extract_subject_words(words)
        if not subject_words:
            continue

        # Determine if this claim contains negation
        has_negation = _has_negation(words, sentence)

        claims.append((subject_words, sentence, has_negation))

    return claims


def _determine_phase(chunk_text: str, claims: list[tuple[set[str], str, bool]]) -> float:
    """Determine the phase for a chunk based on its alignment with claims.

    Logic:
    1. Find claims whose subjects overlap with the chunk's content
    2. If no claims relate to this chunk → θ=0 (not addressed, treat as valid)
    3. If chunk aligns with a related claim → θ=0 (constructive)
    4. If chunk contradicts a related claim → θ=π (destructive)

    A contradiction is detected when:
    - The chunk and claim share subject words (same topic)
    - But one has negation/contradiction indicators while the other doesn't
      (opposing assertions)

    Returns:
        Phase angle: 0.0 for constructive, π for destructive.
    """
    chunk_words = _tokenize(chunk_text)
    chunk_word_set = set(chunk_words)
    chunk_has_negation = _has_negation(chunk_words, chunk_text)

    # Track whether any claim addresses this chunk's subject
    has_related_claim = False
    has_contradiction = False

    for subject_words, claim_text, claim_has_negation in claims:
        # Check if this claim relates to the chunk (subject overlap)
        overlap = subject_words & chunk_word_set
        # Require meaningful overlap (at least 2 content words or 1 if subject is small)
        min_overlap = 1 if len(subject_words) <= 2 else 2
        if len(overlap) < min_overlap:
            continue

        has_related_claim = True

        # Determine alignment vs contradiction
        # Contradiction: same subject but opposing assertions
        # Detected by differing negation status
        if chunk_has_negation != claim_has_negation:
            has_contradiction = True
            break

    # If no claims address this chunk's subject → treat as valid (θ=0)
    if not has_related_claim:
        return _PHASE_CONSTRUCTIVE

    # If contradicts → destructive (θ=π)
    if has_contradiction:
        return _PHASE_DESTRUCTIVE

    # Aligned → constructive (θ=0)
    return _PHASE_CONSTRUCTIVE


def _compute_effective_amplitude(amplitude: float, phase: float) -> float:
    """Compute effective amplitude after interference: r_effective = r × cos(θ).

    For θ=0 (constructive): cos(0) = 1, so r_effective = r
    For θ=π (destructive): cos(π) = -1, so r_effective = r × (-1) = -r

    Since effective_amplitude has ge=0.0 constraint, we clamp to 0.0:
    r_effective = max(0.0, r × cos(θ))

    Args:
        amplitude: The original quantum amplitude [0.0, 1.0].
        phase: The assigned phase angle in radians.

    Returns:
        The effective amplitude after interference, clamped to [0.0, 1.0].
    """
    r_effective = amplitude * math.cos(phase)
    # Clamp to valid range [0.0, 1.0]
    return max(0.0, min(1.0, r_effective))


def _extract_subject_words(words: list[str]) -> set[str]:
    """Extract meaningful subject words from a tokenized sentence.

    Filters out common English stopwords to retain content-bearing terms
    that identify what the sentence is about.
    """
    stopwords = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those", "what", "which", "who", "whom",
        "and", "or", "but", "if", "then", "than", "when", "while",
        "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "up", "about", "into", "over", "after", "before",
        "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "only", "own", "same", "so", "very",
    })
    return {w for w in words if w not in stopwords and len(w) > 1}


def _has_negation(words: list[str], original_text: str) -> bool:
    """Detect if text contains negation or contradiction indicators.

    Checks both individual negation words and multi-word contradiction phrases.

    Args:
        words: Tokenized lowercase words from the text.
        original_text: The original text (for multi-word phrase matching).

    Returns:
        True if negation/contradiction indicators are present.
    """
    # Check for negation words in token list
    word_set = set(words)
    if word_set & _NEGATION_WORDS:
        return True

    # Check for multi-word contradiction indicators in original text
    lower_text = original_text.lower()
    for indicator in _CONTRADICTION_INDICATORS:
        if indicator in lower_text:
            return True

    return False


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return _WORD_RE.findall(text.lower())
