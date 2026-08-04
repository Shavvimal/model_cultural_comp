.PHONY: install lint format typecheck test check validate

install:
	uv sync

lint:
	uv run ruff check app scripts tests

format:
	uv run ruff check --fix app scripts tests
	uv run ruff format app scripts tests

# mypy is not yet a locked dev dependency, so it is pulled in ephemerally and
# typecheck is deliberately not part of `check` (and therefore not part of CI).
typecheck:
	uv run --with mypy mypy app

test:
	uv run pytest

check: lint test

# Reproduction gate. Requires the ~5.8GB IVS download to be present locally (see
# the README) - the WVS/GESIS data-use agreements forbid redistribution, so this
# target can never run in CI.
validate:
	uv run python scripts/validate_projection.py
	uv run python scripts/bootstrap_llms.py
