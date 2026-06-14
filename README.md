# kiro-quantum-context

Quantum-inspired pipeline for processing and resolving conflicts in large text contexts before feeding them to LLMs.

## Vấn đề giải quyết

Package này giải quyết 2 hạn chế phổ biến của LLM:

- **Lost in the Middle** — thông tin nằm giữa context dài bị bỏ qua → giải quyết bằng xử lý song song (không bias vị trí) và sắp xếp theo amplitude
- **Conflict Context** — dữ liệu mâu thuẫn trong context window → giải quyết bằng interference dựa trên phase (triệt tiêu toán học thông tin contradicts)

## Cài đặt

```bash
pip install -e .

# Dev dependencies (testing)
pip install -e ".[dev]"
```

Yêu cầu Python >= 3.11

## Quick Start

```python
import asyncio
from kiro_quantum import entangle, interfere, collapse

async def main():
    # 1. Entangle: chunk text theo semantic density + xử lý song song
    text = """
    The Earth is flat. The Earth revolves around the Sun.
    Water boils at 100°C at sea level. Python is a compiled language.
    """
    matrix = await entangle(text)

    # 2. Interfere: so sánh với steering_base authority để phân xử xung đột
    steering_base = "The Earth revolves around the Sun. Python is an interpreted language."
    result = interfere(matrix, steering_base)

    # 3. Collapse: lọc chunks contradicts, trả về context sạch
    clean_context = collapse(result, threshold=0.1)
    print(clean_context)

asyncio.run(main())
```

## Pipeline Architecture

```
Text Input → entangle() → interfere() → collapse() → Clean Context
              │                │               │
              ▼                ▼               ▼
    Semantic Chunking    Phase Assignment   Threshold Filter
    + Amplitude Score    θ=0 (valid)        + Amplitude Sort
    + Async Parallel     θ=π (contradicts)  + Join Output
```

## API Reference

### `entangle(text, *, concurrency_limit=None, timeout=30.0)`

Async function. Chunk text theo semantic density boundaries và xử lý song song.

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|--------|
| text | str | required | Text input (max 1M chars) |
| concurrency_limit | int \| None | None | Giới hạn tasks song song |
| timeout | float | 30.0 | Timeout mỗi chunk (giây) |

**Returns:** `InformationSuperpositionMatrix`

**Raises:** `EntanglementError` nếu input > 1M chars hoặc tất cả chunks fail.

### `interfere(matrix, steering_base)`

Sync function. Gán phase cho mỗi chunk dựa trên alignment với steering_base.

| Parameter | Type | Mô tả |
|-----------|------|--------|
| matrix | InformationSuperpositionMatrix | Output từ entangle() |
| steering_base | str | Tài liệu authority để phân xử xung đột |

**Returns:** `InformationSuperpositionMatrix` với phase và effective_amplitude đã gán.

**Logic:**
- Chunk aligned với steering_base → θ=0, amplitude giữ nguyên
- Chunk contradicts steering_base → θ=π, effective_amplitude=0
- Chunk không liên quan → θ=0, amplitude giữ nguyên

### `collapse(matrix, *, threshold=0.1)`

Sync function. Lọc và join chunks thành string sạch.

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|--------|
| matrix | InformationSuperpositionMatrix | required | Output từ interfere() |
| threshold | float | 0.1 | Min effective_amplitude [0.0, 1.0] |

**Returns:** `str` — newline-joined text, sắp xếp theo amplitude giảm dần.

**Raises:** `CollapseError` nếu threshold ngoài range hoặc matrix rỗng.

## Data Models

```python
from kiro_quantum import SemanticDensityChunk, InformationSuperpositionMatrix

# Mỗi chunk có:
# - text: nội dung chunk
# - amplitude: [0.0, 1.0] — information density
# - phase: [0, 2π] — 0=valid, π=contradicts
# - effective_amplitude: amplitude × cos(phase)
# - chunk_index: vị trí gốc
# - error_state: bool

# Matrix hỗ trợ serialization:
matrix = await entangle("some text")
json_str = matrix.to_json()  # serialize
restored = InformationSuperpositionMatrix.from_json(json_str)  # deserialize
```

## Exception Hierarchy

```
QuantumContextError (base)
├── EntanglementError    — lỗi trong entangle() (input quá lớn, total failure)
├── InterferenceError    — lỗi trong interfere()
├── CollapseError        — lỗi trong collapse() (threshold invalid, matrix rỗng)
└── DeserializationError — lỗi deserialization JSON (malformed, > 50MB)
```

Tất cả exceptions có attribute `stage` cho biết pipeline stage xảy ra lỗi.

## Type Safety

- Tất cả models dùng Pydantic strict mode (không implicit coercion)
- Amplitude phải là float trong [0.0, 1.0]
- Phase phải là float trong [0, 2π]
- Text chunk phải có min 1 character

## Testing

```bash
# Chạy test suite
python -m pytest tests/ -v

# Chạy với Hypothesis property-based testing
python -m pytest tests/ -v --hypothesis-show-statistics
```

Test suite bao gồm:
- Smoke tests (imports, API surface)
- Full pipeline end-to-end
- Edge cases (empty, whitespace, boundary conditions)
- Serialization round-trip
- Pydantic strict mode validation
- Package structure verification

## Project Structure

```
kiro_quantum/
├── __init__.py              # Public API exports
├── types.py                 # Pydantic models + exception hierarchy
└── core/
    ├── __init__.py
    ├── superposition.py     # entangle() + semantic chunking
    ├── interference.py      # interfere() + claim alignment
    └── wave_function.py     # collapse() + threshold filtering
tests/
├── conftest.py              # Hypothesis strategies
├── test_smoke.py            # Smoke tests
└── test_final_checkpoint.py # Comprehensive tests
pyproject.toml               # Package config
```

## Dependencies

| Package | Version | Mục đích |
|---------|---------|----------|
| pydantic | >= 2.9.0 | Data models, strict validation |
| hypothesis | >= 6.0.0 | Property-based testing (dev) |
| pytest | >= 7.0.0 | Test runner (dev) |
| pytest-asyncio | >= 0.23.0 | Async test support (dev) |

## License

MIT
