<!--
SYNC IMPACT REPORT
==================
Version change: N/A → 1.0.0 (initial creation)

Modified principles: None (initial creation)

Added sections:
- Core Principles (5 principles)
- Code Style Standards
- Quality Gates
- Governance

Removed sections: None

Templates requiring updates:
- .specify/templates/plan-template.md ✅ No changes needed (generic Constitution Check)
- .specify/templates/spec-template.md ✅ No changes needed (uses generic requirements)
- .specify/templates/tasks-template.md ✅ No changes needed (generic task structure)

Follow-up TODOs: None
-->

# LocalAI Backend Constitution

## Core Principles

### I. Type Safety (NON-NEGOTIABLE)

Type safety is mandatory throughout the codebase:

- MUST never use `# type: ignore` comments
- MUST never use `Any` type (including in `.pyi` stubs - use unannotated `**kwargs` instead)
- MUST never use `TYPE_CHECKING` imports - use shared schemas in `app/schemas/` instead
- MUST use `Optional[T]` over `T | None` for consistency
- MUST use Pydantic `BaseModel` over `Dict`/`TypedDict` for structured data
- MUST create type stubs in `typings/` for external libraries lacking type hints

**Rationale**: Pyright enforced via pre-commit. Type safety catches bugs at development time and enables IDE support.

### II. Architecture Boundaries

The codebase follows a strict layered architecture:

- `app/features/` - Business logic feature modules (generators, downloads, models, histories)
- `app/cores/` - Domain services (model_manager, model_loader, samplers, constants)
- `app/services/` - Infrastructure utilities (device, image, storage, logger, memory)
- `app/database/` - SQLAlchemy models and CRUD operations
- `app/schemas/` - Shared Pydantic schemas (prevents circular imports)

**Rules**:
- MUST use shared schemas in `app/schemas/` instead of importing between features
- MUST NOT create circular imports
- MUST place generation utilities in `app/cores/generation/`
- MUST place infrastructure utilities in `app/services/`

**Rationale**: Clear boundaries prevent spaghetti code and enable independent testing.

### III. Code Quality Gates

All code MUST pass automated quality gates:

- **Pre-commit hooks**: `ruff format`, `ruff check`, `pyright` - commits blocked if checks fail
- **Test coverage**: Minimum 80% coverage on new code (SonarCloud enforced)
- **Duplication**: Maximum 3% code duplication (SonarCloud quality gate)
- **Testing**: Use pytest + pytest-asyncio for all tests

**Extract when**:
- Logic appears in 2+ places (especially 10+ lines)
- Create shared utilities in appropriate layer

**Rationale**: Automated enforcement ensures consistent quality without manual review overhead.

### IV. Self-Documenting Code

Code MUST be readable without extensive comments:

- MUST use descriptive names in loops (never `i`, `x`, `p`)
- MUST use `database_service` alias for `app.database.crud`
- MUST NOT add comments explaining "what" - only "why" for non-obvious business logic
- MUST add docstring examples for helper functions
- SHOULD prefer clarity over cleverness

**Rationale**: Code is read more often than written. Self-documenting code reduces maintenance burden.

### V. Modularity

Code MUST be organized into focused, manageable units:

- MUST split files exceeding 150 lines into focused modules
- MUST keep service files as thin orchestration layers
- MUST group functions by responsibility
- MUST use descriptive filenames
- MUST place all imports at file top (never mid-file)

**Rationale**: Small, focused modules are easier to understand, test, and maintain.

## Code Style Standards

**Formatting** (enforced by ruff.toml):
- Tab indentation (not spaces)
- Single quotes for strings
- 120 character line limit

**Imports**:
- All imports at top of file, after module docstring
- Group: standard library → third-party → local imports
- Use absolute imports: `from app.services import logger_service`

**Error Handling**:
- Use Pydantic models over dicts for structured responses
- Use proper exception chaining with `from error`
- No defensive code - call methods directly (e.g., `callback.reset()` not `if hasattr(callback, 'reset')`)

**Standard Library First**:
- `pathlib` for path operations
- `json`, `csv`, `configparser` for data formats
- `datetime`, `time` for time operations

## Quality Gates

| Gate | Threshold | Enforcement |
|------|-----------|-------------|
| Type checking | 0 errors | pyright (pre-commit) |
| Formatting | Compliant | ruff format (pre-commit) |
| Linting | 0 errors | ruff check (pre-commit) |
| Test coverage (new code) | ≥80% | SonarCloud |
| Code duplication | ≤3% | SonarCloud |
| Security hotspots | 100% reviewed | SonarCloud |

## Governance

This constitution supersedes all other development practices for LocalAI Backend.

**Amendment Process**:
1. Propose change with rationale
2. Document migration plan for existing code
3. Update constitution version per semantic versioning
4. Update dependent documentation (CLAUDE.md, docs/)

**Version Policy**:
- MAJOR: Backward-incompatible principle removals or redefinitions
- MINOR: New principles or materially expanded guidance
- PATCH: Clarifications, wording, non-semantic refinements

**Compliance**:
- All PRs MUST verify compliance with constitution
- Complexity violations MUST be justified in PR description
- Use CLAUDE.md for runtime development guidance

**Version**: 1.0.0 | **Ratified**: 2025-11-28 | **Last Amended**: 2025-11-28
