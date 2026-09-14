# model_cultural_comp

[![ci](https://github.com/Shavvimal/model_cultural_comp/actions/workflows/ci.yml/badge.svg)](https://github.com/Shavvimal/model_cultural_comp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Research code for [Cultural Alignment of Open-Weight LLMs on the
Inglehart–Welzel Map](https://openreview.net/forum?id=xJHa9ts0mt) (ORACLE
workshop at EMNLP 2026), with an expanded
[write-up](https://shav.dev/blog/cultural-alignment-of-open-weight-llms-on-the-inglehart-welzel-map).
It fits a custom two-dimensional instrument to the Integrated Values Surveys
and projects model responses and surveyed countries through the same frozen
transformation. Coordinates describe reported responses under the specified
protocol, not held values or complete cultural profiles.

The corrected instrument uses 392,382 respondents in 112 country/entity codes;
country comparisons cover 389,341 respondents in 109 mapped entities. The
2024 cohort contains eleven model–language cells. The 2026 experiment crosses
17 models with English and Chinese, giving 34 recorded cells and 33 eligible
positions. All eligible means occupy the primary reference-relative quadrant;
Chinese administration increases projected self-expression in 15/16 paired
models. The rotation, benchmark and prompt sensitivities limit interpretation.

Version [1.1.0](https://github.com/Shavvimal/model_cultural_comp/releases/tag/v1.1.0)
contains these corrections. The source checkout contains
code, tests, documentation and input checksums. **No datasets, generated
results, fitted binaries or figures belong in Git.** The required retained
model responses are packaged separately. A second, small results/provenance
archive supplies the aggregates named in the paper and the historical trace-coding
record. Download the [response archive](https://github.com/Shavvimal/model_cultural_comp/releases/download/v1.1.0/model-cultural-comp-responses-2026-09-12.tar.gz)
and [results supplement](https://github.com/Shavvimal/model_cultural_comp/releases/download/v1.1.0/model-cultural-comp-paper-results-2026-09-12.tar.gz)
from the v1.1.0 release. See [reproduction instructions](docs/REPRODUCING.md) for
the exact inputs, result assets and checksums, and [CHANGELOG.md](CHANGELOG.md)
for repairs. `uv run python scripts/reproduction_data.py verify-results <archive>`
checks a downloaded supplement against the tracked results manifest without
extracting it.

## Quickstart

```bash
git clone https://github.com/Shavvimal/model_cultural_comp
cd model_cultural_comp
git checkout v1.1.0
uv sync --frozen
make check
```

The recorded environment uses Python 3.11, pinned in `.python-version`.
Tests generate synthetic fixtures and require neither survey inputs nor API
keys. `make check` also checks that Git's index contains no data or generated
files. The lockfile keeps the numerical dependencies fixed.

To reproduce the study, obtain the separate frozen response archive and the
specified licensed survey inputs, then run:

```bash
uv run python scripts/reproduction_data.py install /path/to/model-cultural-comp-responses-2026-09-12.tar.gz
# Prepare data/ivs_df.pkl from the licensed merged SAV as documented below.
make reproduce
```

[REPRODUCING.md](docs/REPRODUCING.md) explains download versions, merge and
cache preparation, checksums, command order, output families and limitations.
`make reproduce` refits the instrument, runs both cohorts, all reported
sensitivities including twenty fitting seeds, and merges the frozen trace
panels. It makes no collection or annotation API calls. Generated results
stay under ignored `data/` and `figures/` directories.

## Code layout

| Location | Responsibility |
|---|---|
| `app/ppca.py`, `app/culture_map.py`, `app/survey_indices.py` | Observed-data Gaussian likelihood, Y003 reconstruction, frozen projection and country aggregation |
| `app/llm_bootstrap.py`, `app/region_svm.py` | Observed cell means, item/cluster resampling, classifier-free distances and regional labels |
| `app/cloud_survey.py`, `app/qn_classes.py`, `app/llm_meta.py` | Prompts, parsers, collection and shared model/family metadata |
| `app/survey_reference.py`, `app/instrument_sensitivity.py` | Reference, rotation, completion and dated country-benchmark diagnostics |
| `app/appendix_contrasts.py`, `app/control_audit.py` | Item-profile contrasts and prompt-control trial accounting |
| `app/trace_codebook.py`, `app/trace_diagnostics.py`, `app/agreement.py` | Frozen annotation prompt, majority votes and agreement diagnostics |
| `scripts/` | Collection drivers, offline analyses, plots and reproducibility utilities |
| `tests/` | Synthetic regression and mathematical consistency checks |
| `docs/analysis-plan-2026.md` | Original plan with dated deviations; earlier entries retain superseded numbers |
| `docs/reproduction-data.json` | Filenames, sizes and hashes of the separate frozen inputs |

The research plotting scripts generate the six study figures. Manuscript
fact-checking, LaTeX/MDX formatting and interactive-blog exports live with the
paper. Historical exploratory notebooks and local collection prototypes are
superseded by the tested pipeline here and are not included.

[Collection protocol and record schemas](docs/PROTOCOL.md) explain retries,
prefixes, exclusions, known translation issues and missing historical metadata.

## Pipeline and interpretation

1. **Prepare and fit:** restrict IVS waves to 2005 onward and recode out-of-range
   missing values. Before the six-item completeness filter, reconstruct a missing
   autonomy index as `Y003 = A029 + A039 - A040 - A042` only when all four
   constituent answers are valid 0/1. Preserve valid delivered Y003 values and
   check their agreement with the scoring rule where constituents are observed.
   The EVS trend file lacks a precomputed Y003 column but usually contains these
   four answers. The full merged cache must retain the constituent columns;
   invalid or unavailable constituents leave the index missing. Preparation
   counts distinguish delivered, reconstructed and residual-missing indices.
   This follows the [official longitudinal scoring definition](https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=autonomous).
   Then fit respondent-level
   Gaussian PPCA with two factors and observed-item means/standard deviations
   fixed before fitting. Missing entries are integrated out in the observed-data
   likelihood. L-BFGS-B uses three starts; each must satisfy a maximum absolute
   gradient of at most `1e-7` for negative log likelihood per informative row.
   If an objective-change stop occurs above that bound, the same likelihood
   is centered numerically and resumed within the original iteration budget.
   The highest-likelihood converged start is retained. This is direct likelihood
   optimisation, not EM, and multiple starts do not prove a global optimum.
   Missing entries are then completed with exact Gaussian conditional means.
   Country aggregation uses S017, the original national survey weight, after the
   unweighted fit. Each country's retained survey years contribute in proportion
   to their S017 weight mass; years are not equally weighted. S018 and S019 are
   the distinct equilibrated weights. The saved NPZ
   contains numeric/string arrays only, loads with `allow_pickle=False`, and
   retains fitted loadings, noise variance and convergence diagnostics as well
   as the frozen score transform. It is regenerated locally and not distributed.
2. **Shared projection:** standardisation, score-side varimax rotation and
   rescaling parameters are frozen; countries and models use the same affine
   `CulturalMap.project` path.
3. **Reference:** `SURVEY_REFERENCE = (0.038, -0.10)` projects observed-item
   marginal means. It is not the completed respondent mean (reported in
   `data/validation_survey_reference.csv`),
   nor a world-population-weighted centre.
   `HUMAN_MEAN` remains only as a compatibility alias.
4. **Points:** displayed coordinates project observed item means; language
   displacement points are norms of observed Chinese-minus-English differences,
   not averages of bootstrap norms. Monte Carlo permutation p-values use the
   plus-one correction; exact enumerations use exact tail proportions.
5. **Uncertainty:** the item bootstrap resamples items independently; the primary
   2026 cluster bootstrap resamples ten user-message prefix variants. Empty
   variant–item groups contribute zero counts; fallback applies only when a draw
   has no retained answer on an item. Ellipses are nominal mean-position regions;
   actual ten-cluster coverage is unknown. Zero observed quadrant crossings is
   descriptive, not a simultaneous confidence guarantee.
6. **Regions and sensitivities:** SVM labels are bootstrap-modal, while centroid
   labels use observed points. Selected-grid cross-validation reuses tuning
   folds and is not unbiased accuracy. Orthogonal nested sums-of-squares shares
   describe constructed profiles, not additive causal effects. Terminal-failure
   bounds do not cover retry-induced selection.

The historical analysis projected unstandardised model answers through a separately
refitted rotation, treated the Y003 missing code −3 as data, and pooled languages.
Camera-ready repairs additionally correct rescaling offsets, zero-count cluster
pooling, displayed estimators and the missing-data fitting objective. The old
fitting loop counted imputed entries in an observed residual calculation; an
independent likelihood audit showed that its fixed point was not the claimed
observed-data likelihood optimum. The current implementation replaces that loop
and regenerates its dependent coordinates, statistics and figures. Offset
correction alone translates points and
reference together, preserving relative distances; the full revision contains
other changes and does not preserve every number. Unsupported granular historical
displacement figures are not represented as current measurements.

The historical plan and dated deviations remain in
[docs/analysis-plan-2026.md](docs/analysis-plan-2026.md). Later control, coding and
diagnostic work is distinguished from the original pre-specified analyses.

## Development

| Target | Purpose |
|---|---|
| `make check` | Git content policy, Ruff and synthetic tests |
| `make test` / `make lint` / `make format` | Tests / lint and formatting check / automatic formatting |
| `make typecheck` | mypy over `app/`, pulled in ephemerally; not part of `make check` |
| `make verify-data` | Verify the separate frozen input files against their manifest |
| `make validate` | Build country metadata when absent or older than the licensed cache (stops with a message if `data/ivs_df.pkl` is missing), refit and validate the instrument, then analyse the 2024 cohort |
| `make validate-2026` | Analyse the 2026 cohort and its sensitivities, then draw all six figures; fails early if `make validate` outputs are missing or older than the fitted instrument |
| `make validate-traces` | Offline trace-panel merge, agreement and sensitivity summaries |
| `make reproduce` | Complete sequential offline chain, including country metadata |

Each stage also runs on its own. Run them in this order from the repository
root, after `make verify-data` and with the licensed cache in place. Prefix
the commands with `OMP_NUM_THREADS=1 MPLBACKEND=Agg`, as the Makefile does,
to match the recorded outputs:

| Stage | Command |
|---|---|
| Country metadata | `uv run --frozen python scripts/build_country_meta.py` |
| Fit and validate the instrument | `uv run --frozen python scripts/validate_projection.py` |
| 2024 bootstrap | `uv run --frozen python scripts/bootstrap_llms.py` |
| 2026 analyses and sensitivities | `make validate-2026` (its scripts, in Makefile order, end with both figure scripts) |
| Trace aggregation | `make validate-traces` |
| Figures only | `uv run --frozen python scripts/make_figures.py` and `uv run --frozen python scripts/make_figures_2026.py` |
| Whole offline chain | `make reproduce` |

The figure scripts read outputs of the 2024 bootstrap, the 2026 analyses and
`instrument_sensitivity.py`, so run them after those stages.

See [CONTRIBUTING.md](CONTRIBUTING.md) for changes that could affect results.
Code is MIT licensed, with the PPCA attribution and third-party survey-text
notices retained in [NOTICE](NOTICE) and [LICENSES/](LICENSES/).
[CITATION.cff](CITATION.cff) gives the paper as the preferred citation and
also describes this software release.
