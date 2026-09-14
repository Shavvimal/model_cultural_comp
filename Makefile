.PHONY: install lint format typecheck test check validate validate-2026 validate-outputs validate-traces verify-data reproduce check-public-tree progress watch

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

# The licensed cache is prepared by hand (docs/REPRODUCING.md); never guess it.
data/ivs_df.pkl:
	@echo "Missing $@: prepare it from the licensed IVS inputs as docs/REPRODUCING.md describes." >&2
	@exit 1

# validate_projection.py reads country metadata. Built only when absent or older
# than its inputs, so `make reproduce` (which builds it first) runs it once.
data/country_codes.pkl: data/ivs_df.pkl scripts/build_country_meta.py
	$(REPRO_ENV) $(PYTHON) scripts/build_country_meta.py

validate: verify-data data/country_codes.pkl
	$(REPRO_ENV) $(PYTHON) scripts/validate_projection.py
	$(REPRO_ENV) $(PYTHON) scripts/bootstrap_llms.py

# Outputs of `make validate` that validate-2026 reads. Those written after the
# fitted instrument must not be older than it; the other two precede it by design.
VALIDATE_FITTED = data/cultural_map_model.npz
VALIDATE_AFTER_FIT = data/corrected_country_scores.csv data/validation_survey_reference.csv \
	data/validation_survey_item_baselines.csv data/llm_ellipses.csv data/llm_bootstrap_replicates.csv
VALIDATE_BEFORE_FIT = data/country_codes.pkl data/validation_rotation_sensitivity.csv

validate-outputs:
	@for f in $(VALIDATE_FITTED) $(VALIDATE_BEFORE_FIT) $(VALIDATE_AFTER_FIT); do \
		if [ ! -f "$$f" ]; then \
			echo "validate-2026: missing $$f; run make validate first." >&2; exit 1; \
		fi; \
	done
	@for f in $(VALIDATE_AFTER_FIT); do \
		if [ "$$f" -ot $(VALIDATE_FITTED) ]; then \
			echo "validate-2026: $$f is older than $(VALIDATE_FITTED); rerun make validate." >&2; exit 1; \
		fi; \
	done

validate-2026: verify-data validate-outputs
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
