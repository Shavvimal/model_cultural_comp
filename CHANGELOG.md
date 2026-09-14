# Changelog

All notable changes to this project are documented here.

## [1.1.0] - 2026-09-14

The corrected instrument and analyses support the revised paper. The
[v1.1.0 release](https://github.com/Shavvimal/model_cultural_comp/releases/tag/v1.1.0)
provides the source revision and separate response and results archives.

### Fixed

- Reconstruct missing Y003 from four valid constituent responses before the
  completeness filter, preserving delivered scores. Report reconstruction and
  concordance counts and the corrected fitting and mapped samples.
- Replace the imputation fixed-point loop with observed-data Gaussian PPCA
  likelihood optimisation, multiple starts, convergence diagnostics and exact
  Gaussian conditional completion. Preserve the frozen score-side varimax path.
- Resume premature optimizer objective-change stops within the original
  iteration budget using a numerically centered evaluation of the same
  likelihood, retaining the strict gradient tolerance on every start.
- Reject undefined profile correlations, malformed annotation codes and invalid
  collection execution limits before they can produce misleading results.
- Require matching recorded request/host provenance when resuming collection;
  historical records with unknown settings require a new output directory.
- Correct affine offsets, item validity and empty-cluster count pooling; project
  observed item means and use paired observed displacement estimates.
- Correct permutation denominators, tie handling, simultaneous headline counts,
  finite reference benchmarks and prompt-control completeness accounting.
- Preserve all ten items in the item sign-test BH family, assigning p=1 when
  every paired contrast is tied.
- Describe user-message prefixes, the retained no-persona system primer,
  terminal failure records and selected trace panels accurately.
- Keep the collection API key out of the harness `repr` and redact it from
  stored and printed error text. Reject hosts that embed a username, password
  or query string; records keep only the scheme, host and port.
- Persist a 4xx provider response other than 408 or 429 as a terminal
  `provider:` error, which QC counts separately from parse failures. Detect
  throttling from status 429 or a word-boundary "rate limit", not any "rate"
  substring; trace annotation now uses the same `is_throttled` check.
- Persist a trial's last parse failure even when a later attempt in the same
  invocation fails in transport, so it is no longer re-swept for another answer.
  These collection changes affect future runs only.
- Accept ASCII digits only in the response parsers. Re-parsing every retained
  answer still gives the stored parsed value.
- Reject non-finite paired deltas, empty comparison cohorts and malformed
  p-value families in the shared statistics helpers. The item sign test gives
  p=1 when every contrast is tied, and BH adjustment requires one p-value per
  declared test.
- Cast Y003 constituents to float before the signed arithmetic, so unsigned
  integer columns cannot wrap, and raise on a reconstructed value outside [-2, 2].
- Validate S017 weights as present, finite and non-negative with a positive sum
  per group; missing weights raise and are never filled.
- Raise when a PPCA fit receives a row with no observed item, and when the
  cluster bootstrap meets a cell with fewer than two prompt variants.
- Report undefined kappa and alpha as NaN instead of 1.0.
- Require the no-persona collection in `prompt_sensitivity_2026.py` instead of
  writing its outputs without the no-persona rows.
- Raise with `path:line` when trace-coding resume or merge meets a corrupt
  label line, and validate stored labels before the integer cast.
- Copy the confirmatory mean-displacement and origin-permutation CSVs byte for
  byte under their plug-in names, after checking they match the language
  effects, instead of computing them a second time.
- Correct two figure titles: Figure 0 (`fig0_countries_only`) now gives the
  survey-year range of the plotted countries (2005-2023), and Figure 6
  (`fig4_language_forest`) describes nominal 95% intervals from independently
  resampled arms.

### Added

- Mathematical and synthetic regression checks for likelihood, completion,
  scoring, resampling, inference and trace agreement.
- Reference, language/design, family/wording, instrument rotation/completion,
  prompt-control and item-profile sensitivities used by the corrected analysis.
- Frozen-code trace-panel merging, agreement and selection diagnostics.
- Prospective collection request metadata and per-attempt audit logging;
  historical missing metadata remains unknown.
- An explicit 69-file input manifest and tested data-only archive pack/install/
  verify commands, with portable JSONL replacing distributed 2024 pickles.
- A sequential offline `make reproduce` target and licensed-input provenance.
- A separate aggregate/provenance supplement preserving all appendix-named CSVs
  and the original, superseded trace-coding record without putting data in Git.
- A Git-index policy gate in `make check` and CI to reject data, generated files,
  notebooks, private state and symlinks, including forced additions.
- `reproduction_data.py verify-results`, which checks a results supplement
  against the tracked `docs/results-manifest.json` without extracting it, and
  `pack-results`, which rebuilds the supplement byte for byte from matching
  local outputs and the tracked `docs/results-supplement-README.md`.
- A golden SHA-256 test over every request prompt, so an edited prompt cannot
  pass silently.
- One shared preparation check, `check_preparation`, used by the instrument
  fit, seed sensitivity and instrument sensitivity stages.
- Makefile stage prerequisites: `make validate` builds `data/country_codes.pkl`
  when it is absent or out of date and stops with a message if the licensed
  cache is missing; `make validate-2026` fails early if the `make validate`
  outputs it reads are missing or older than the fitted instrument.
- Per-stage commands in the README, and documentation of the SPSS-free cache
  route and of which stages need the licensed survey inputs.

### Changed

- Git contains source, tests and documentation only. Stop tracking existing
  aggregates, figures and obsolete exploratory notebooks; local files remain
  ignored. Retained inputs are distributed separately and outputs regenerated.
- Keep manuscript fact checks, LaTeX table exports and blog exports with the
  paper project, not in this repository.
- Share family metadata in `app/llm_meta.py`; `family_wording_sensitivity.py`
  runs the family and wording sensitivities.
- Keep shared profile standardization in `app/appendix_contrasts.py`; scripts
  consume this application helper instead of importing it from another CLI.
- Remove notebook-only and unused direct dependencies and update the lockfile
  without upgrading retained numerical packages.
- Retain historical plan entries and older changelog sections as dated records;
  their old tracked-data policy and superseded results are not current guidance.
- Tighten the public-tree policy: more data and archive suffixes, and
  case-insensitive directory and suffix matching, with matching `.gitignore`
  patterns.
- Name the analysis-design and solver constants with their rationale, values
  unchanged: `app/study_design.py` holds the inclusion threshold, trial design,
  trial key, sensitive items and prefix families; the varimax tolerance is pinned
  at `VARIMAX_TOL = 1e-5`; PPCA optimiser, survey-year, minimum-item,
  cross-validation and tie-tolerance literals are named where they are used.
- Consolidate duplicated helpers: BH adjustment, sign and permutation tests in
  `app/stats.py`; Gaussian conditional completion in `app/ppca.py`; the 2026
  trial loader shared by the bootstrap and diagnostics stages; `.env` loading and
  throttle detection in `app/cloud_survey.py`; and the collected and excluded
  2026 model sets in `app/llm_meta.py`.
- Configure mypy to treat untyped third-party imports as `Any`.

### Known limitations

- The cluster bootstrap draws every cell from one shared random generator, so a
  cell's resamples depend on the cells drawn before it. Per-cell seeding would
  move every published interval, so it is deferred to a later version.

No published result changed in this round: every regenerated CSV is
byte-identical, and only the titles of Figures 0 and 6 differ.

## [1.0.0] - 2026-08-05

The initial paper artifact, tagged `v1.0.0`. The entries below describe that
historical release and its corrections to the 2024 post
[Cultural Bias in LLMs](https://shav.dev/blog/cultural-bias). Version 1.1.0 above
supplies the corrected results for the revised paper.

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

[1.1.0]: https://github.com/Shavvimal/model_cultural_comp/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Shavvimal/model_cultural_comp/releases/tag/v1.0.0
