# Reproducing the corrected analysis

Run commands from the repository root with Python 3.11 and `uv sync --frozen`.
`make check` uses synthetic fixtures only. The full analysis needs the licensed
survey inputs below and a separate archive of retained model responses.

## Frozen response archive

**Delivery status: prepared locally; no immutable public download has been
published yet.** The source tree alone is not a complete reproduction deposit.
A public release must attach this archive (or link a durable research deposit)
and publish its SHA-256 alongside the source revision. Do not add its contents
to Git. The expected filename is
`model-cultural-comp-responses-2026-09-12.tar.gz`.

[reproduction-data.json](reproduction-data.json) fixes the exact 69 inputs by
relative filename, size and SHA-256:

- 11 portable 2024 response files (5,500 records);
- 34 primary 2026 files and 17 English no-persona files;
- the frozen trace sample and five annotation panels;
- the historical published 2024 country-coordinate CSV used only for correction
  accounting, not an external Inglehart–Welzel benchmark.

The archive contains these files and a copy of the manifest. It excludes survey
microdata, local caches, model binaries, bootstrap draws, generated summaries,
figures, private collection logs and the regenerable blank coding worksheet.
The [protocol](PROTOCOL.md) documents schemas and retained-information limits.
The 2024 pickle-to-JSONL conversion preserves fields and row order; tuples become
arrays. The frozen trace sample uses JSON null for absent error fields. Original
hashes for those converted inputs remain in the manifest.

Install and verify an obtained archive:

```bash
uv run python scripts/reproduction_data.py install /path/to/model-cultural-comp-responses-2026-09-12.tar.gz
make verify-data
```

Installation validates all archive members against the checkout's manifest before
writing inputs. It rejects missing, additional, duplicate, nonregular or altered
members, and refuses to overwrite different existing inputs. Verification also
rejects additional response files or human labels that would change the frozen
analysis. Keep new experiments in another checkout.

A maintainer who already holds all verified inputs can build the deterministic
archive outside the repository:

```bash
uv run python scripts/reproduction_data.py pack --output ../releases/model-cultural-comp-responses-2026-09-12.tar.gz
```

This emits the archive and its `.sha256` sidecar. The explicit manifest allowlist
prevents inclusion of other local files. Packaging neither collects new responses
nor reconstructs missing historical metadata. The software's MIT licence does
not assign a licence to model outputs or third-party survey materials; retain
applicable provenance and terms when distributing the separate archive.

## Results and historical provenance supplement

The corrected appendix promises bundled aggregate artifacts. These are supplied
separately from Git and the frozen input archive as
`model-cultural-comp-paper-results-2026-09-12.tar.gz` (207,783 bytes).
Its SHA-256 is
`744e8a7b0fb52add617d74b4e0d23a5016fe8e31ab1f449f2d394b503c32c1e4`.
**The public download remains pending.** Publish this supplement alongside the
source and response archives; source-code publication alone does not fulfil the
paper's artifact-availability statement.

The supplement contains 91 regenerated aggregate CSVs, the seed-angle log used
by the optional `seed_sensitivity.py --from-stored` summary, and the original
`data/trace_coding.json`. `RESULTS_MANIFEST.json` records per-file hashes, sizes,
roles, the analysis source tree and the frozen input-manifest hash. All seventeen
CSV filenames cited in the current appendix are included, in particular:

- `data/llm_parse_rates_2026.csv`;
- `data/llm_ellipses_2026.csv`;
- `data/llm_language_effects_plugin_2026.csv`;
- `data/diag_2026_keying_balance.csv`.

The historical `trace_coding.json` contains the original eight-coder counts and
model-output excerpts described in the appendix. It is retained verbatim for
provenance and is superseded by the five-panel analysis. The corrected pipeline
does not read it, and it cannot be recreated by rerunning the current annotators.
Current trace results are in the `trace_coding_2026.csv`,
`trace_coding_headline_2026.csv`, `trace_agreement_2026.csv` and related summaries.

Extract the supplement into a separate directory to inspect the reported outputs.
A fresh reproduction should regenerate outputs with `make reproduce` rather than
preload them. The supplement excludes licensed respondent records, fitted models,
bootstrap draws, per-trace label exports and the blank human worksheet. The frozen
response archive preserves the underlying primary records and five label panels.
No result file needs to be committed to Git to accompany the public release.

## Licensed survey input provenance

Obtain these versions from their original providers under the applicable terms:

| Input | Version and provider | Bytes | SHA-256 |
|---|---|---:|---|
| `Trends_VS_1981_2022_sav_v4_0.sav` | WVS Time Series 1981–2022, v4-0-0 (2024-06-30), [WVS download](https://www.worldvaluessurvey.org/WVSEVStrend.jsp), DOI [10.14281/18241.27](https://doi.org/10.14281/18241.27) | 534274112 | `7ad3bb018c8cbfe07ac2d0fca3b914bf444053cd3e7a7eb7493cbd727258562d` |
| `ZA7503_v3-0-0.sav` | EVS Trend File 1981–2017, v3.0.0 (2022-12-14), [GESIS ZA7503](https://search.gesis.org/research_data/ZA7503), DOI [10.4232/1.14021](https://doi.org/10.4232/1.14021) | 227393114 | `6e3cab793a87a21c00cc2ca5570e244fd4a6e152dc1394cb005b15a1b3843025` |

Apply the provider's `EVS_WVS_Merge Syntax_Spss_June2024.sps` using SPSS, adjusting
local paths as its instructions require. The retained unedited syntax SHA-256 is
`fa1e3f4314404c39308c18270ec9ee8c1df1dc9b118a71e148009a3f3bac1914`.
The source files contain 442,473 WVS and 224,434 EVS rows; the merged input contains
666,907 rows. The retained `Integrated_values_surveys_1981-2022.sav` has 858,671,834
bytes and SHA-256
`9da7dbe3921f88e4835dfe6fec00328b02773df4d945f5523f1e83c9a2906ce8`.
SAV writer metadata may differ after a fresh merge. These checksums identify the
retained source files; the repository does not distribute them or automate SPSS.

Place the merged file in ignored `data/`, then create a trusted local cache:

```bash
uv run python - <<'PYTHON'
from pathlib import Path
import pyreadstat

Path("data").mkdir(exist_ok=True)
df, _ = pyreadstat.read_sav(
    "data/Integrated_values_surveys_1981-2022.sav", encoding="latin1"
)
assert len(df) == 666907, f"Unexpected merged row count: {len(df)}"
assert {"A029", "A039", "A040", "A042"}.issubset(df.columns)
df.to_pickle("data/ivs_df.pkl")
PYTHON
```

Retain all columns. The four Y003 constituents are essential to the corrected
sample; a previously reduced cache can silently prevent reconstruction. The
pipeline reads only locally prepared pickle caches, not downloaded executable
pickle data. Allow several gigabytes of disk and memory for the full DataFrame.
No access credentials or survey files belong in Git or in the response archive.

## Offline command chain

```bash
make reproduce
```

This sequential target prepares country metadata, refits the instrument and runs:

| Stage | Outputs under ignored `data/` and `figures/` |
|---|---|
| `build_country_meta.py`, `validate_projection.py` | Country metadata, fitted Gaussian parameters and score transform, country coordinates, preparation and validation summaries |
| `bootstrap_llms.py`, `qc_2026.py`, `analyze_2026.py` | Both cohorts' observed positions, bootstrap draws, regional labels, completeness and parse diagnostics |
| Confirmatory, diagnostic, language, prompt and family/wording scripts | Paired effects, permutation/sign tests, multiplicity corrections, missingness bounds, profile and control sensitivities |
| `seed_sensitivity.py`, `reference_sensitivity.py`, `instrument_sensitivity.py` | Twenty fitting seeds, reference definitions, rotation/completion alternatives and dated country-neighbour comparisons |
| `make_figures.py`, `make_figures_2026.py` | Six static study figures as PDF and PNG |
| `code_traces_2026.py --merge`, `trace_agreement_2026.py` | Frozen-panel majority coding, agreement and selection/self-annotation diagnostics |

The targets use seed 42 where applicable, one OpenMP thread and the Agg plotting
backend. Floating-point differences can occur across BLAS/platform versions;
PDF creation timestamps also vary. The lockfile pins dependency versions, not
hardware. Reproduction should start with empty generated-output directories so
stale results cannot satisfy a missing stage.

Expected primary reconciliation: 392,382 fitting respondents across 112 entity
codes, 389,341 respondents in 109 mapped entities, 117,075 reconstructed Y003
indices, eleven 2024 and 33 eligible 2026 cells. The 2026 primary quadrant contains
33/33 observed means and 330,000/330,000 bootstrap draws; this is descriptive and
does not establish simultaneous confidence coverage. Chinese administration raises
self-expression in 15/16 paired cells, with mean displacement about 0.64.

The chain never calls collection or annotation APIs. If desired, regenerate the
blank human worksheet separately with
`uv run python scripts/code_traces_2026.py --worksheet`. Completed human labels
are absent from the frozen analysis. Do not regenerate the trace sample or rerun
annotators as part of replay: their stored texts and panel identities define the
reported agreement analysis. Live hosted model tags may change and rerunning
collection is a new experiment, not exact reproduction.

## Scope and historical paths

The append-only [analysis plan](analysis-plan-2026.md) retains superseded entries
and their dated corrections. Historical references to
`scripts/final_review_sensitivities.py` now correspond to
`scripts/family_wording_sensitivity.py`. Manuscript claim checking, literal-table
exports and blog exports live with the paper, not this source repository.
The former combined release-candidate builder has been retired in favour of the
explicit data-only archive above. Older notebooks are not part of the supported pipeline. A public source
deposit must also exclude historical commits containing data or notebook outputs;
ignoring or deleting those paths in a new commit does not remove old copies.
