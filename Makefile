.PHONY: install lint format typecheck test check validate progress watch

install:
	uv sync

lint:
	uv run ruff check app scripts tests && uv run ruff format --check app scripts tests

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

# Reproduction gate: refit the map, assert the projection path, then the
# 2024-cohort bootstrap. Requires the ~5.8GB IVS download to be present locally
# (see the README) - the WVS/GESIS data-use agreements forbid redistribution, so
# this target can never run in CI. The 2026 cohort additionally needs the raw
# corpus under data/collection_2026/ and is run script by script:
# scripts/qc_2026.py (gate) -> analyze_2026.py -> confirmatory_2026.py,
# diagnostics_2026.py, plugin_displacement_2026.py, make_figures_2026.py.
validate:
	uv run python scripts/validate_projection.py
	uv run python scripts/bootstrap_llms.py

# Collection progress for a 2026 run in flight; needs data/collection_2026/.
progress:
	bash scripts/progress.sh

watch:
	bash scripts/progress.sh --watch
