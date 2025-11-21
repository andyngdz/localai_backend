# Plan 007: Model Loader Modularization

## Overview

Refactor `app/cores/model_loader/model_loader.py` (≈500 lines) into a modular structure that separates progress reporting, strategy execution, and pipeline finalization. Goal: improve readability, lower cognitive complexity, and simplify future maintenance without changing runtime behavior.

## Key Objectives

1. **Encapsulation:** Extract cohesive helpers into new modules so each file has a single responsibility (progress emission, strategy loading, orchestration).
2. **Type Safety:** Preserve existing TypedDict/type checking patterns during extraction.
3. **Backward Compatibility:** Keep the public `model_loader` function API intact.
4. **Testability:** Ensure new modules are individually testable and existing tests continue to pass (ruff, pyright, pytest).

## Implementation Steps

1. **Define Target Module Layout**
   - `app/cores/model_loader/progress.py`: `map_step_to_phase`, `emit_progress` and cancellation-friendly helpers.
   - `app/cores/model_loader/strategies.py`: strategy TypedDicts, `_build_loading_strategies`, `_execute_loading_strategies`, loader utilities.
   - `app/cores/model_loader/setup.py`: device optimization, move/cleanup helpers, `_finalize_model_setup`.
   - Keep `model_loader.py` as orchestration/glue that wires modules together.

2. **Move Code Incrementally**
   - Create new modules with existing logic, updating imports.
   - Replace internal references with explicit exports to avoid circular imports.
   - Maintain logging behavior and socket emissions.

3. **Update Imports & Type Re-exports**
   - Ensure strategy types (e.g., `Strategy`) are exported for reuse.
   - Provide explicit `__all__` lists where useful.
   - Adjust relative imports within `app.cores.model_loader` package.

4. **Smoke Tests / Static Analysis**
   - Run `uv run ruff check app/cores/model_loader`.
   - Run `uv run pyright app/cores/model_loader`.
   - (Optional) targeted pytest if available for loader.

5. **Documentation**
   - Briefly note new structure in `docs/ARCHITECTURE.md` if needed (only if file now references modules explicitly).

## Risks & Mitigations

- **Circular Imports:** Keep shared constants in new strategy module; import functions via local modules inside functions if needed.
- **Behavior Regression:** Avoid touching business logic; use pure moves.
- **Large Diff:** Make atomic commits per module extraction (user handles git).

## Definition of Done

- `model_loader.py` primarily orchestrates high-level flow (<250 LOC).
- New modules exist with clear responsibilities and exported helpers.
- Ruff & Pyright succeed; no new Sonar issues.
- Manual QA (load success/cancel path) verified if feasible.
