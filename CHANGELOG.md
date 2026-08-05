# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Added
- Bootstrap confidence machinery for model positions: per-question resampling of
  stored responses, 95% confidence ellipses, and SVM cultural-region assignment
  with a stability fraction across replicates (`app/llm_bootstrap.py`,
  `app/region_svm.py`, `scripts/bootstrap_llms.py`).
- Figure generation (`scripts/make_figures.py`): the corrected cultural map with
  per-model CI ellipses and the SVM decision regions, as vector PDF and PNG.

### Changed
- Migrated packaging to uv and PEP 621 (`pyproject.toml` + `uv.lock`), and made
  `app/` an installable package so intra-project imports are absolute and work
  from any working directory.
- The surveyed-model lists are deduplicated into `app/llm_meta.py` as the single
  source of truth; four divergent copies of the `chinese_llms` list previously
  lived across `culture_map.py`, `culture_map_post_hoc.py` and the notebooks, and
  disagreed about which entries were commented out.

### Removed
- `notebooks/ppca.py`, a byte-duplicate of `app/ppca.py`. `app/ppca.py` is the
  only copy.
