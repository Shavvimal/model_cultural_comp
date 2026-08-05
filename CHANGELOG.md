# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CITATION.cff`, so GitHub renders a "Cite this repository" entry. No DOI yet —
  it carries the repository URL, the `v1.0.0` version and the write-up.
- Project URLs in `pyproject.toml` (repository, write-up, changelog, issues).

### Changed
- README rewritten around the released state: both cohorts, the two bootstrap
  estimators, the two region rules, the committed artefacts, the `v1.0.0`
  citation.
- Documentation pass over the module and script docstrings, `CONTRIBUTING.md`,
  `SECURITY.md`, the Makefile and the issue templates, so they describe the
  released pipeline (both cohorts, both estimators, both region rules) rather
  than the intermediate state they were written in.

## [1.0.0] - 2026-08-05

The release the paper's numbers were generated from, tagged `v1.0.0`. Everything
below is described in full in [Cultural Alignment of Open-Weight LLMs on the
Inglehart-Welzel Map](https://shav.dev/blog/cultural-alignment-of-open-weight-llms-on-the-inglehart-welzel-map),
which supersedes and corrects the 2024 post
[Cultural Bias in LLMs](https://shav.dev/blog/cultural-bias).

### Fixed
- **Projection correction.** LLM responses were previously projected without the
  standardization fitted on the survey data, and through a varimax rotation
  *re-fitted* on the model scores. Countries and models therefore lived in two
  different coordinate spaces and were plotted as if they lived in one. The
  rotation is now fitted exactly once on the training score matrix and stored on
  the model, and every projection — survey respondents and LLM responses alike —
  runs one path: standardize with the stored means/stds, project onto the
  principal axes, apply the stored rotation, rescale. This changes the published
  model coordinates.
- **Unit-variance rescaling.** The published WVS rescaling constants presuppose
  unit-variance factor scores, so rotated scores are now standardized by their
  fitted standard deviations before the constants are applied. Omitting it
  inflated the absolute scale — which is the scale region centroids and every
  distance statistic live on.
- **SPSS sentinel recode.** Out-of-range user-missing codes are recoded to
  missing per item, range-driven, *before* any completeness filtering, so no
  sentinel is read as data — not only the `Y003` `-3` that had zeroed that
  item's loadings on both axes.
- **Language pooling.** Chinese and English administrations of one model are
  never pooled; each is a distinct model-language cell, labelled `[zh]`.
- **SVM grid.** The region-assignment grid search now includes the regularised
  regime; the earlier grid started at `C=500` and never evaluated a smooth
  boundary at all.

### Added
- **2026 bilingual collection** (`app/cloud_survey.py`,
  `scripts/collect_cloud_2026.py`): a designed factorial over cloud-served
  frontier models × two administration languages, collected to design
  completeness and resumable per model. The harness records raw response text,
  the separated reasoning trace, the system-prompt variant id, the repeat index,
  per-attempt errors and latency — the 2024 harness stored only
  `(model, item, parsed_value)`, which is why prompt-level variance and refusals
  could not be recovered from it retrospectively. Corrected Chinese-arm
  translations ship with it: the 2024 Chinese `F118` prompt had labelled both
  scale poles "always justifiable", and several Chinese system-prompt variants
  had collapsed to duplicates in translation.
- **Two bootstrap estimators** (`app/llm_bootstrap.py`). The *item bootstrap*
  resamples each item's stored responses independently, and is the only
  estimator available for the 2024 corpus, which lacks the variant id. The
  *cluster bootstrap* resamples the ten system-prompt variants with replacement,
  carrying all items and repeats within a variant, so prompt-level correlation
  propagates into the position; it is the primary estimator for 2026.
- **Two region rules** (`app/region_svm.py`): an RBF-SVM reported with its own
  cross-validated accuracy alongside the nearest region centroid, with the
  disagreement between them reported as a result. Positional stability is
  documented as sampling uncertainty of the position under a *fixed* classifier,
  never as evidence about the classifier's own error rate.
- **2026 analysis suite** under `scripts/`: `qc_2026.py` (data-integrity gate),
  `analyze_2026.py` (positions, regions, headline statistics, language effects),
  `confirmatory_2026.py` (sign tests, origin × language permutation test,
  coherence contrast, replicate-simultaneous headline bounds),
  `diagnostics_2026.py` (variance components, prompt-variant ICCs, refusal
  rates, Manski bounds, threshold sensitivity),
  `plugin_displacement_2026.py`, `coverage_calibration_2026.py`,
  `seed_sensitivity.py` and `sample_traces_2026.py`.
- **Validation harness** (`scripts/validate_projection.py`): the exact
  path-identity regression test, the correction accounting against the 2024
  coordinates, and the full rotation-criterion sensitivity grid.
- **Figures** (`scripts/make_figures.py`, `scripts/make_figures_2026.py`) as
  vector PDF and PNG, with TrueType embedding so the PDFs are camera-ready.
- **Committed aggregate artefacts.** The small derived CSVs the write-up cites
  by filename are re-included past the `data/` ignore rule by explicit negation
  and committed, so a reader can check a quoted number against a file. The
  survey microdata remains un-redistributable and is not in the repository.
- Repository hygiene: MIT `LICENSE` plus the Apache-2.0 `NOTICE` for
  `app/ppca.py`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue
  and PR templates, CODEOWNERS, Dependabot, and a CI workflow running the
  synthetic-fixture unit suite (`make check`).

### Changed
- Migrated packaging to uv and PEP 621 (`pyproject.toml` + `uv.lock`), and made
  `app/` an installable package so intra-project imports are absolute and work
  from any working directory.
- The surveyed-model lists are deduplicated into `app/llm_meta.py` as the single
  source of truth; four divergent copies of the `chinese_llms` list previously
  lived across `culture_map.py`, `culture_map_post_hoc.py` and the notebooks, and
  disagreed about which entries were commented out.
- `app/ppca.py` stores its standardization parameters and re-applies them at
  transform time, takes an explicit seed, raises rather than silently returning
  an unconverged fit, and round-trips its full parameter set on save/load.

### Removed
- `notebooks/ppca.py`, a byte-duplicate of `app/ppca.py`. `app/ppca.py` is the
  only copy.
- The 2024 local collection harness (`app/llm_data_gen.py`,
  `app/chinese_llm_data_gen.py`), `app/culture_map_post_hoc.py`,
  `app/region_assign_llm.py` and `app/country_meta.py`, all superseded by
  `app/cloud_survey.py`, the single-path `CulturalMap.project` and
  `app/region_svm.py`. They remain in git history.

[Unreleased]: https://github.com/Shavvimal/model_cultural_comp/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Shavvimal/model_cultural_comp/releases/tag/v1.0.0
