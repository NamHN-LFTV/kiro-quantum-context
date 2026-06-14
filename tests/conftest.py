"""Test configuration and custom Hypothesis strategies for kiro-quantum-context."""

import json
import math
import string

from hypothesis import settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hypothesis profile configuration
# ---------------------------------------------------------------------------
settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")


# ---------------------------------------------------------------------------
# Strategy: multi_sentence_text
# ---------------------------------------------------------------------------
_SENTENCE_ENDINGS = [".", "!", "?"]

_sentence_body = st.text(
    alphabet=st.sampled_from(
        list(string.ascii_letters + string.digits + " ,;:-'\"")
    ),
    min_size=3,
    max_size=80,
).map(lambda s: s.strip() or "word")

_sentence = st.tuples(_sentence_body, st.sampled_from(_SENTENCE_ENDINGS)).map(
    lambda t: f"{t[0]}{t[1]}"
)

multi_sentence_text = st.lists(_sentence, min_size=2, max_size=10).map(
    lambda sentences: " ".join(sentences)
)
"""Strategy that generates text with 2+ sentences."""


# ---------------------------------------------------------------------------
# Strategy: whitespace_only_text
# ---------------------------------------------------------------------------
whitespace_only_text = st.text(
    alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"]),
    min_size=1,
    max_size=100,
)
"""Strategy that generates strings containing only whitespace characters."""


# ---------------------------------------------------------------------------
# Strategy: valid_chunk
# ---------------------------------------------------------------------------
_valid_amplitude = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_valid_phase = st.floats(min_value=0.0, max_value=2 * math.pi, allow_nan=False, allow_infinity=False)
_non_empty_text = st.text(
    alphabet=st.sampled_from(list(string.ascii_letters + string.digits + " .,!?")),
    min_size=1,
    max_size=200,
).filter(lambda t: len(t.strip()) > 0)


@st.composite
def valid_chunk(draw):
    """Generate a valid SemanticDensityChunk instance."""
    from kiro_quantum.types import SemanticDensityChunk

    text = draw(_non_empty_text)
    amplitude = draw(_valid_amplitude)
    phase = draw(_valid_phase)
    chunk_index = draw(st.integers(min_value=0, max_value=1000))
    error_state = draw(st.booleans())

    use_custom_effective = draw(st.booleans())
    if use_custom_effective:
        effective_amplitude = draw(_valid_amplitude)
    else:
        effective_amplitude = None

    return SemanticDensityChunk(
        text=text,
        amplitude=amplitude,
        phase=phase,
        effective_amplitude=effective_amplitude,
        error_state=error_state,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# Strategy: valid_matrix
# ---------------------------------------------------------------------------
_PIPELINE_STAGES = ["created", "entangled", "interfered", "collapsed"]


@st.composite
def valid_matrix(draw):
    """Generate a valid InformationSuperpositionMatrix instance."""
    from kiro_quantum.types import InformationSuperpositionMatrix

    chunks = draw(st.lists(valid_chunk(), min_size=0, max_size=10))
    num_chunks = len(chunks)

    if num_chunks > 0:
        timed_out_indices = draw(
            st.lists(
                st.integers(min_value=0, max_value=num_chunks - 1),
                min_size=0,
                max_size=min(num_chunks, 5),
                unique=True,
            )
        )
    else:
        timed_out_indices = []

    original_text_length = draw(st.integers(min_value=0, max_value=100000))
    pipeline_stage = draw(st.sampled_from(_PIPELINE_STAGES))

    return InformationSuperpositionMatrix(
        chunks=chunks,
        timed_out_indices=timed_out_indices,
        original_text_length=original_text_length,
        pipeline_stage=pipeline_stage,
    )


# ---------------------------------------------------------------------------
# Strategy: out_of_range_amplitude
# ---------------------------------------------------------------------------
out_of_range_amplitude = st.one_of(
    st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.001, max_value=1e10, allow_nan=False, allow_infinity=False),
)

# ---------------------------------------------------------------------------
# Strategy: out_of_range_phase
# ---------------------------------------------------------------------------
out_of_range_phase = st.one_of(
    st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),
    st.floats(min_value=2 * math.pi + 0.001, max_value=1e10, allow_nan=False, allow_infinity=False),
)


# ---------------------------------------------------------------------------
# Strategy: mutated_json
# ---------------------------------------------------------------------------
_MUTATION_TYPES = ["remove_field", "wrong_type", "out_of_range", "extra_field", "corrupt_syntax"]


@st.composite
def mutated_json(draw):
    """Generate a valid InformationSuperpositionMatrix JSON string with random mutations."""
    from kiro_quantum.types import SemanticDensityChunk

    num_chunks = draw(st.integers(min_value=1, max_value=5))
    chunks = []
    for i in range(num_chunks):
        chunk = {
            "text": draw(_non_empty_text),
            "amplitude": draw(_valid_amplitude),
            "phase": draw(_valid_phase),
            "effective_amplitude": draw(_valid_amplitude),
            "error_state": draw(st.booleans()),
            "chunk_index": i,
        }
        chunks.append(chunk)

    matrix_dict = {
        "chunks": chunks,
        "timed_out_indices": [],
        "original_text_length": draw(st.integers(min_value=0, max_value=100000)),
        "pipeline_stage": draw(st.sampled_from(_PIPELINE_STAGES)),
    }

    mutation = draw(st.sampled_from(_MUTATION_TYPES))

    if mutation == "remove_field" and chunks:
        target_chunk = draw(st.integers(min_value=0, max_value=len(chunks) - 1))
        field_to_remove = draw(st.sampled_from(["text", "amplitude", "chunk_index"]))
        del matrix_dict["chunks"][target_chunk][field_to_remove]

    elif mutation == "wrong_type" and chunks:
        target_chunk = draw(st.integers(min_value=0, max_value=len(chunks) - 1))
        field = draw(st.sampled_from(["amplitude", "phase", "chunk_index"]))
        matrix_dict["chunks"][target_chunk][field] = "not_a_number"

    elif mutation == "out_of_range" and chunks:
        target_chunk = draw(st.integers(min_value=0, max_value=len(chunks) - 1))
        mutation_choice = draw(st.booleans())
        if mutation_choice:
            matrix_dict["chunks"][target_chunk]["amplitude"] = draw(out_of_range_amplitude)
        else:
            matrix_dict["chunks"][target_chunk]["phase"] = draw(out_of_range_phase)

    elif mutation == "extra_field":
        matrix_dict["unexpected_field"] = "surprise"

    elif mutation == "corrupt_syntax":
        json_str = json.dumps(matrix_dict)
        corrupt_pos = draw(st.integers(min_value=max(1, len(json_str) // 4), max_value=max(1, len(json_str) - 2)))
        return json_str[:corrupt_pos] + json_str[corrupt_pos + 1:]

    return json.dumps(matrix_dict)
