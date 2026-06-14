"""Superposition Processor — semantic density chunking and amplitude scoring."""

import asyncio
import math
import re
from collections import Counter

from kiro_quantum.types import (
    EntanglementError,
    InformationSuperpositionMatrix,
    SemanticDensityChunk,
)

# Maximum input length: 1,000,000 characters
_MAX_INPUT_LENGTH = 1_000_000

# Regex for sentence boundaries: period, exclamation, or question mark
# followed by whitespace or end of string
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:\s|$)")


def chunk_text(text: str) -> list[SemanticDensityChunk]:
    """Segment text into semantic density chunks with TF-IDF-based amplitude scoring.

    This function splits text at sentence boundaries, assigns each chunk a quantum
    amplitude based on its information density relative to the full document, and
    returns a list of SemanticDensityChunk objects.

    Args:
        text: The input text to chunk.

    Returns:
        A list of SemanticDensityChunk objects. Empty list if input is empty/whitespace.

    Raises:
        EntanglementError: If input exceeds 1,000,000 characters.
    """
    # Edge case: empty or whitespace-only input → empty list
    if not text or text.isspace():
        return []

    # Edge case: input too large
    if len(text) > _MAX_INPUT_LENGTH:
        raise EntanglementError(
            message=f"Input text exceeds maximum length of {_MAX_INPUT_LENGTH} characters",
            input_length=len(text),
        )

    # Find sentence boundary positions
    boundaries = _find_sentence_boundaries(text)

    # Split text into raw segments at boundaries
    raw_segments = _split_at_boundaries(text, boundaries)

    # Filter out segments that are purely whitespace (Requirement 1.7)
    # But we must preserve the original text exactly, so whitespace-only segments
    # get merged into adjacent segments
    segments = _merge_whitespace_segments(raw_segments)

    # Single chunk case: fewer than 2 boundaries means no meaningful split
    if len(segments) <= 1:
        # Return entire text as single chunk
        amplitude = 1.0  # Single chunk gets maximum amplitude
        return [
            SemanticDensityChunk(
                text=text,
                amplitude=amplitude,
                chunk_index=0,
            )
        ]

    # Compute TF-IDF-inspired amplitude for each segment
    amplitudes = _compute_amplitudes(segments)

    # Build chunk objects
    chunks = []
    for i, (segment_text, amplitude) in enumerate(zip(segments, amplitudes)):
        chunks.append(
            SemanticDensityChunk(
                text=segment_text,
                amplitude=amplitude,
                chunk_index=i,
            )
        )

    return chunks


def _find_sentence_boundaries(text: str) -> list[int]:
    """Find positions in text where sentence boundaries occur.

    A sentence boundary is after a period, exclamation mark, or question mark
    followed by whitespace or end of string.

    Returns:
        Sorted list of character positions where splits should occur.
    """
    boundaries = []
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        pos = match.start()
        # Only add if it's not at the very end (would produce empty trailing segment)
        if pos < len(text):
            boundaries.append(pos)
    return boundaries


def _split_at_boundaries(text: str, boundaries: list[int]) -> list[str]:
    """Split text at the given boundary positions, preserving all characters.

    Each boundary position indicates where to cut. The character at the boundary
    position starts the next segment.
    """
    if not boundaries:
        return [text]

    segments = []
    prev = 0
    for pos in boundaries:
        if pos > prev:
            segments.append(text[prev:pos])
            prev = pos
    # Append remaining text
    if prev < len(text):
        segments.append(text[prev:])

    return segments


def _merge_whitespace_segments(segments: list[str]) -> list[str]:
    """Merge whitespace-only segments into adjacent segments.

    This ensures every chunk has at least one non-whitespace character
    (Requirement 1.7) while preserving exact text reconstruction.
    """
    if not segments:
        return []

    merged = []
    for segment in segments:
        if segment.strip():
            # Segment has non-whitespace content
            merged.append(segment)
        else:
            # Whitespace-only segment: merge into previous or hold for next
            if merged:
                merged[-1] = merged[-1] + segment
            else:
                # No previous segment yet, hold it for the next
                merged.append(segment)

    # Final pass: if first segment is whitespace-only, merge into next
    if len(merged) > 1 and not merged[0].strip():
        merged[1] = merged[0] + merged[1]
        merged.pop(0)

    # If after merging we have a single whitespace-only segment, return empty
    # (this shouldn't happen since we check isspace() at the top, but safety check)
    if len(merged) == 1 and not merged[0].strip():
        return []

    return merged


def _compute_amplitudes(segments: list[str]) -> list[float]:
    """Compute TF-IDF-inspired amplitude scores for each segment.

    Higher amplitude indicates the segment contains more unique/rare terms
    relative to the full document. Amplitudes are normalized to [0.0, 1.0].

    The approach:
    1. Tokenize each segment into lowercase words
    2. Compute document frequency (DF) for each term across all segments
    3. For each segment, compute a TF-IDF score as sum of (tf * idf) for its terms
    4. Normalize scores to [0.0, 1.0] range
    """
    n_segments = len(segments)
    if n_segments == 0:
        return []

    # Tokenize segments
    segment_tokens = [_tokenize(seg) for seg in segments]

    # Compute document frequency: how many segments contain each term
    doc_freq: Counter = Counter()
    for tokens in segment_tokens:
        unique_terms = set(tokens)
        for term in unique_terms:
            doc_freq[term] += 1

    # Compute TF-IDF score for each segment
    scores = []
    for tokens in segment_tokens:
        if not tokens:
            scores.append(0.0)
            continue

        tf_counter = Counter(tokens)
        score = 0.0
        for term, tf in tf_counter.items():
            # IDF: log(N / df) where N = number of segments
            df = doc_freq[term]
            idf = math.log(n_segments / df) if df > 0 else 0.0
            score += tf * idf

        # Normalize by segment length to avoid bias toward longer segments
        score = score / len(tokens) if tokens else 0.0
        scores.append(score)

    # Normalize to [0.0, 1.0]
    if not scores:
        return []

    max_score = max(scores)
    min_score = min(scores)

    if max_score == min_score:
        # All segments have equal density — assign uniform amplitude
        return [1.0] * n_segments

    normalized = []
    for score in scores:
        # Linear normalization to [0.1, 1.0] to ensure no chunk gets zero amplitude
        # (zero amplitude would mean "no information" which is wrong for non-empty chunks)
        norm = 0.1 + 0.9 * (score - min_score) / (max_score - min_score)
        normalized.append(round(norm, 6))

    return normalized


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens.

    Uses simple word boundary splitting, filtering out pure punctuation
    and very short tokens.
    """
    # Split on non-alphanumeric characters
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return words


async def _process_chunk(
    chunk: SemanticDensityChunk,
    semaphore: asyncio.Semaphore | None,
    timeout: float,
) -> SemanticDensityChunk:
    """Process a single chunk asynchronously with optional semaphore and timeout.

    This is an async wrapper around the chunk data. The actual processing
    (TF-IDF scoring) is already done in chunk_text(). This function enables
    the async concurrency/timeout/error-handling semantics required by entangle().

    Args:
        chunk: The chunk to process.
        semaphore: Optional semaphore for concurrency limiting.
        timeout: Maximum seconds allowed for this task.

    Returns:
        The processed chunk (unchanged since scoring is already done).

    Raises:
        asyncio.TimeoutError: If processing exceeds timeout.
        Exception: Any other failure during processing.
    """

    async def _inner() -> SemanticDensityChunk:
        if semaphore is not None:
            async with semaphore:
                # Yield control to allow true concurrent scheduling
                await asyncio.sleep(0)
                return chunk
        else:
            await asyncio.sleep(0)
            return chunk

    return await asyncio.wait_for(_inner(), timeout=timeout)


async def entangle(
    text: str,
    *,
    concurrency_limit: int | None = None,
    timeout: float = 30.0,
) -> InformationSuperpositionMatrix:
    """Process text into an Information Superposition Matrix via async parallel chunking.

    Segments text into semantic density chunks and processes them concurrently.
    Each chunk is treated as an independent quantum state in superposition.

    Args:
        text: The input text to process.
        concurrency_limit: Optional maximum number of concurrent chunk tasks.
            If None, all chunks are processed simultaneously.
        timeout: Maximum seconds allowed per individual chunk task (default 30s).

    Returns:
        An InformationSuperpositionMatrix containing all processed chunks.

    Raises:
        EntanglementError: If input exceeds 1M chars, or if all chunks fail to process.
    """
    # Validate concurrency_limit if provided
    if concurrency_limit is not None and concurrency_limit < 1:
        raise EntanglementError(
            message="concurrency_limit must be >= 1",
            input_length=len(text) if text else 0,
        )

    # Step 1: chunk_text handles empty/whitespace and >1M validation
    chunks = chunk_text(text)

    # Step 2: empty chunks → return empty matrix with pipeline_stage="entangled"
    if not chunks:
        return InformationSuperpositionMatrix(
            chunks=[],
            timed_out_indices=[],
            original_text_length=len(text) if text else 0,
            pipeline_stage="entangled",
        )

    # Step 3: Process chunks concurrently
    semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_limit is not None else None

    # Create tasks for each chunk
    tasks = [
        asyncio.create_task(_process_chunk(chunk, semaphore, timeout))
        for chunk in chunks
    ]

    # Gather results, allowing individual failures
    results: list[SemanticDensityChunk | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )

    # Step 4-6: Process results, handling timeouts and failures
    processed_chunks: list[SemanticDensityChunk] = []
    timed_out_indices: list[int] = []
    failure_count = 0

    for i, result in enumerate(results):
        if isinstance(result, asyncio.TimeoutError):
            # Partial timeout: record index
            timed_out_indices.append(i)
            failure_count += 1
        elif isinstance(result, BaseException):
            # Single chunk failure: mark error_state=True, amplitude=0, continue
            error_chunk = SemanticDensityChunk(
                text=chunks[i].text,
                amplitude=0.0,
                phase=chunks[i].phase,
                effective_amplitude=0.0,
                error_state=True,
                chunk_index=chunks[i].chunk_index,
            )
            processed_chunks.append(error_chunk)
            failure_count += 1
        else:
            # Success
            processed_chunks.append(result)

    # Step 6: Total failure — all chunks failed
    if failure_count == len(chunks):
        raise EntanglementError(
            message=(
                f"All {len(chunks)} chunks failed during entanglement "
                f"(stage=entangle, input_length={len(text)})"
            ),
            input_length=len(text),
        )

    # Step 7: Return matrix with results
    return InformationSuperpositionMatrix(
        chunks=processed_chunks,
        timed_out_indices=timed_out_indices,
        original_text_length=len(text),
        pipeline_stage="entangled",
    )
