.PHONY: install lint format typecheck test check validate validate-2026 validate-traces verify-data reproduce check-public-tree progress watch

install:
	uv sync --frozen

lint:
	uv run --frozen ruff check app scripts tests && uv run --frozen ruff format --check app scripts tests

format:
	uv run --frozen ruff check --fix app scripts tests
	uv run --frozen ruff format app scripts tests

# mypy is not yet a locked dev dependency, so it is pulled in ephemerally and
# typecheck is deliberately not part of `check` (and therefore not part of CI).
typecheck:
	uv run --with mypy mypy app

test:
	uv run --frozen pytest

check: check-public-tree lint test

# Full reproduction requires licensed IVS inputs and generated dependencies.
# Run validate before validate-2026; never collect/annotate in these targets.
# validate-traces is offline and needs retained labels, not survey microdata.
REPRO_ENV = PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
PYTHON ?= uv run --frozen python

validate: verify-data
	$(REPRO_ENV) $(PYTHON) scripts/validate_projection.py
	$(REPRO_ENV) $(PYTHON) scripts/bootstrap_llms.py

validate-2026: verify-data
	$(REPRO_ENV) $(PYTHON) scripts/qc_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/analyze_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/confirmatory_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/diagnostics_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/exploratory_correlations_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/plugin_displacement_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/language_design_sensitivity.py
	$(REPRO_ENV) $(PYTHON) scripts/coverage_calibration_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/prompt_sensitivity_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/appendix_contrasts_2026.py
	$(REPRO_ENV) $(PYTHON) scripts/family_wording_sensitivity.py
	$(REPRO_ENV) $(PYTHON) scripts/seed_sensitivity.py
	$(REPRO_ENV) $(PYTHON) scripts/reference_sensitivity.py
	$(REPRO_ENV) $(PYTHON) scripts/instrument_sensitivity.py
	$(REPRO_ENV) $(PYTHON) scripts/make_figures.py
	$(REPRO_ENV) $(PYTHON) scripts/make_figures_2026.py

validate-traces: verify-data
	$(REPRO_ENV) $(PYTHON) scripts/code_traces_2026.py --merge
	$(REPRO_ENV) $(PYTHON) scripts/trace_agreement_2026.py

# This gate checks Git's index, including forced additions of ignored files.
check-public-tree:
	$(PYTHON) scripts/check_public_tree.py

verify-data:
	$(PYTHON) scripts/reproduction_data.py verify

# Sequential even under make -j: each stage consumes the previous outputs.
reproduce: verify-data
	$(REPRO_ENV) $(PYTHON) scripts/build_country_meta.py
	$(MAKE) validate
	$(MAKE) validate-2026
	$(MAKE) validate-traces

# Collection progress for a 2026 run in flight; needs data/collection_2026/.
progress:
	bash scripts/progress.sh

watch:
	bash scripts/progress.sh --watch
