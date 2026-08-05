# model_cultural_comp

[![ci](https://github.com/Shavvimal/model_cultural_comp/actions/workflows/ci.yml/badge.svg)](https://github.com/Shavvimal/model_cultural_comp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Where do LLMs sit on the world's cultural map?

This repository maps the cultural alignment of large language models onto the
Inglehart-Welzel cultural map. The map itself is refitted from scratch on the
Integrated Values Surveys (IVS) 1981-2022 — a probabilistic PCA over the ten IVS items
behind the two Inglehart-Welzel axes — and then LLMs are put through the same survey
and projected onto that map through the *same* fitted transform the countries went
through. Each model-language cell gets a position with a bootstrap confidence region
and two region-assignment rules (an SVM reported with its own cross-validated
accuracy, and the nearest region centroid), plus classifier-free headline statistics.

Two cohorts are covered. The **2024 cohort** (eleven model-language cells, served
locally at Q4 quantisation) is the corrected version of the 2024 blog post
[Cultural Bias in LLMs](https://shav.dev/blog/cultural-bias): the original projected
model responses without the survey-fitted standardization and through a re-fitted
rotation, so models and countries were not actually in the same coordinate space. The
**2026 cohort** is a designed factorial experiment over the cloud-served frontier
generation: 17 open-weight models × two administration languages (English and
Chinese), 34 cells × 500 calls, with reasoning traces and refusals recorded as data.
The full analysis is written up in
[Cultural Alignment of Open-Weight LLMs on the Inglehart-Welzel Map](https://shav.dev/blog/cultural-alignment-of-open-weight-llms-on-the-inglehart-welzel-map);
see also [CHANGELOG.md](CHANGELOG.md).

## Quickstart

```bash
git clone https://github.com/Shavvimal/model_cultural_comp
cd model_cultural_comp
uv venv --python 3.11
uv sync
make test          # unit suite: synthetic fixtures, no data needed
```

### Getting the data

**The survey data is not redistributed here.** It is licensed to you directly by the
WVS Association and GESIS under data-use agreements that forbid redistribution; `data/`
and `*.pkl` are gitignored for that reason. Download the two trend files yourself and
merge them:

| File | Source |
|---|---|
| `ZA7503_v3-0-0.sav` — EVS Trend File 1981-2017 (3.0.0) | [GESIS ZA7503](https://search.gesis.org/research_data/ZA7503) |
| `Trends_VS_1981_2022_sav_v4_0.sav` — WVS Trend File 1981-2022 (4.0.0) | [WVS/EVS trend](https://www.worldvaluessurvey.org/WVSEVStrend.jsp) |
| `EVS_WVS_Merge Syntax_Spss_June2024.sps` — the official merge syntax | shipped with the WVS trend download |

Running the merge syntax over the two `.sav` files produces the Integrated Values
Surveys 1981-2022 (~5.8GB), which goes in `data/`. This follows the procedure the WVS
Association documents for [creating the World Cultural
Map](https://www.worldvaluessurvey.org/wvs.jsp). The SPSS format is used here; the
format makes no difference to the results.

With the data present:

```bash
make validate      # reproduces the country coordinates, then the bootstrap
uv run python scripts/make_figures.py
```

## Pipeline

```
prepare  ->  fit PPCA  ->  fix ONE varimax rotation  ->  project  ->  bootstrap  ->  SVM regions
                                                        /      \
                                                 countries    models
```

1. **Prepare** (`CulturalMap.prepare_data`) — subset the IVS to the ten map items
   (`A008 A165 E018 E025 F063 F118 F120 G006 Y002 Y003`), apply the WVS index
   transforms for the two composite items, and aggregate to country level.
2. **Fit PPCA** (`app/ppca.py`) — probabilistic PCA by EM, which tolerates the missing
   entries that are unavoidable in a merged multi-wave survey. The fit is seeded, so it
   is reproducible run to run — with one documented exception, below.
3. **One stored rotation** (`CulturalMap.fit`) — a varimax rotation is fitted *exactly
   once*, on the training score matrix, oriented to the Inglehart-Welzel convention to
   resolve sign and axis-order ambiguity, and stored on the model.
4. **Project through one path** (`CulturalMap.project`) — everything, country data and
   model responses alike, goes through: standardize with the stored means/stds →
   project onto the principal axes → apply the stored rotation → rescale to published
   axis units. This single path is the correctness property the whole repository exists
   to hold; it is asserted by `scripts/validate_projection.py`.
5. **Bootstrap CIs** (`app/llm_bootstrap.py`) — because the projection is affine in the
   ten item values, a model's position depends only on its per-question mean response.
   Two estimators are reported. The *item bootstrap* (B = 1,000) resamples each item's
   stored responses independently — it omits cross-item covariance and is the only
   estimator available for 2024, where the prompt-variant id was not recorded. The
   *cluster bootstrap* (B = 10,000) resamples the ten system-prompt variants with
   replacement, carrying all items and repeats within a variant, and is the primary
   estimator for the 2026 cells (its regions are a median 2.9× larger by SD product).
6. **Region rules** (`app/region_svm.py`) — an RBF-SVM fitted on the country
   coordinates (reported with its 0.57 cross-validated accuracy) and the nearest
   region centroid, side by side; where they disagree, the disagreement is a result.
   The paper's headline statistics route through no classifier: distance to the
   pooled human respondent mean, share of countries closer, and minimum distance to
   any non-Western region centroid, computed per bootstrap replicate.

### Reproducibility of the seeded EM fit

Seeding fixes the EM initialisation, not the floating-point summation order beneath
it. Re-running `scripts/seed_sensitivity.py` reproduces every aggregate the paper
quotes — rotation spread 5.21°, country-coordinate mean across-seed SD 0.022, max SD
0.077, max range 0.262 (`data/seed_sensitivity_aggregates.csv`) — exactly at quoted
precision. The **per-country, per-seed extreme columns** in
`data/seed_sensitivity.csv` (`*_min`, `*_max`, and the ranges derived from them) are
not stable to that precision: they can differ from the committed CSV by up to 0.37
map units, because BLAS thread-order nondeterminism in the EM fit perturbs individual
seeds and the min/max columns select exactly the perturbed tails. No quoted number is
affected — the paper quotes only aggregates — and the original CSV is retained rather
than regenerated. Pin thread counts (e.g. `OMP_NUM_THREADS=1`) if you need the
per-seed extremes to match bit-for-bit.

## Make targets

| Target | What it does |
|---|---|
| `make install` | `uv sync` |
| `make lint` | `ruff check app scripts tests` |
| `make format` | `ruff` autofix + format |
| `make typecheck` | `mypy app` (not in `check`; mypy is pulled in ephemerally) |
| `make test` | `pytest` — synthetic fixtures, no data, no network, no Ollama |
| `make check` | `lint` + `test`. Byte-for-byte the CI gate. |
| `make validate` | Reproduction against the local IVS data. **Never runs in CI** — the data may not be redistributed. |

## Layout

| Path | |
|---|---|
| `app/` | the installable package: `ppca.py`, `culture_map.py`, `llm_bootstrap.py`, `region_svm.py`, `qn_classes.py` (response parsers), `llm_meta.py` (the model list — single source of truth) |
| `scripts/` | reproduction (`validate_projection.py`, `bootstrap_llms.py`), the 2026 analysis suite (`analyze_2026.py`, `confirmatory_2026.py`, `diagnostics_2026.py`, `qc_2026.py`, `plugin_displacement_2026.py`, `coverage_calibration_2026.py`, `seed_sensitivity.py`, `sample_traces_2026.py`), collection (`collect_cloud_2026.py`, `progress.sh`), and figures (`make_figures.py`, `make_figures_2026.py`) |
| `notebooks/` | exploratory work; not the test suite |
| `figures/` | generated PDFs and PNGs |
| `data/` | the IVS inputs and every large derived binary live here and are **gitignored** — they may not be redistributed. The small aggregate artefacts the paper cites by filename (`conf_2026_*.csv`, `diag_2026_*.csv`, `seed_sensitivity*.csv`, the `llm_*` parse-rate/ellipse/language-effect/region aggregates for both cohorts, `trace_coding.json`) are re-included by explicit negation and are committed |

## Models surveyed

The model lists live in `app/llm_meta.py` — the single source of truth. Models that
never produced parseable answers in the 2024 run are recorded there in
`FAILED_LLMS_2024` rather than being silently dropped.

### 2024 cohort — eleven model-language cells

Served locally through [Ollama](https://ollama.com), Q4-quantised GGUF builds, on
consumer hardware. Two models were administered *only* in Chinese and one in both
languages (analysed as two cells, marked `[zh]`/`[en]`).

- **Chinese-origin / Chinese fine-tuned:** `qwen2:7b` (both languages),
  `llama2-chinese:13b` `[zh]`, `wangshenzhi/gemma2-27b-chinese-chat` `[zh]`,
  `wangrongsheng/llama3-70b-chinese-chat`
- **Western:** `llama3:70b`, `mistral:7b`, `gemma2:27b`
- **Uncensored (Dolphin):** `dolphin-llama3:8b`, `dolphin-mistral:7b`,
  `dolphin-mixtral:8x7b`
- **Attempted, excluded for producing nothing parseable:** `yi:34b`,
  `aquilachat2:34b`, `glm4:9b`, `xuanyuan:70b`,
  `kingzeus/llama-3-chinese-8b-instruct-v3` (five model names; the tracked
  Modelfiles cannot confirm five distinct base artefacts — the `yi` and `glm`
  Modelfiles point at the AquilaChat2 GGUF — which is disclosed wherever the
  2024 coherence denominator is used)

### 2026 cohort — 17 models × two administration languages

Cloud-served (Ollama Cloud; serving precision undisclosed by the provider and
carried as a confound), 34 cells of 500 calls each, English and Chinese arms.

- **Chinese-origin (10):** `deepseek-v4-flash`, `deepseek-v4-flash:0731`,
  `deepseek-v4-pro`, `glm-5.1`, `glm-5.2`, `kimi-k2.6`, `kimi-k2.7-code`,
  `minimax-m2.7`, `minimax-m3`, `qwen3.5:397b`
- **Western (7):** `gemma4:31b`, `gpt-oss:20b`, `gpt-oss:120b`,
  `mistral-large-3:675b`, `nemotron-3-nano:30b`, `nemotron-3-super`,
  `nemotron-3-ultra`

An eighteenth model, `kimi-k3`, was excluded before any data was collected (every
call returned a billing error; zero records, no part in any denominator). One cell,
`nemotron-3-ultra [zh]`, is excluded from position estimates (F120 parsed 4/50,
under the inclusion threshold) — its refusals are analysed as data and its
worst-case Manski bound is still reported.

Each model carries its own upstream licence; check it before reuse.

## Licensing

The original work in this repository is **MIT** — see [LICENSE](LICENSE).

One file is not original: `app/ppca.py` is derived from
[pca-magic](https://github.com/allentran/pca-magic) (Copyright Allen Tran), licensed
under **Apache-2.0**. Apache-2.0 code may be redistributed inside an MIT-licensed
project provided the upstream notices are retained and the modifications are stated, so
that is what [NOTICE](NOTICE) and the header of `app/ppca.py` do. Practically, for a
downstream user: the repository is MIT, and if you redistribute `app/ppca.py` (or a
derivative of it) you must include a copy of the Apache-2.0 license (see `LICENSES/Apache-2.0.txt`), state any changes you make, and carry the `NOTICE` file and the Apache-2.0 attribution with
it. The method itself is Tipping & Bishop (1999).

The IVS/WVS/EVS data is under its own terms and is **not** covered by this licence and
**not** redistributed here — obtain it from GESIS and the WVS Association directly.

## Citation

A paper describing this work is forthcoming; this section will carry its BibTeX entry
and DOI. The release the paper's numbers were generated from is tagged
[`v1.0.0`](https://github.com/Shavvimal/model_cultural_comp/releases/tag/v1.0.0).
Until then, cite the repository and the write-up:

- Vimalendiran, S. (2026). [Cultural Alignment of Open-Weight LLMs on the
  Inglehart-Welzel Map](https://shav.dev/blog/cultural-alignment-of-open-weight-llms-on-the-inglehart-welzel-map)
  — the full analysis this repository implements.
- Vimalendiran, S. (2024). [Cultural Bias in LLMs](https://shav.dev/blog/cultural-bias)
  — the origin post, superseded and corrected by the above.

```bibtex
@software{vimalendiran_model_cultural_comp,
  author  = {Vimalendiran, Shav},
  title   = {model\_cultural\_comp: mapping LLM cultural alignment
             onto the Inglehart-Welzel map},
  year    = {2026},
  url     = {https://github.com/Shavvimal/model_cultural_comp}
}
```

Please also cite the underlying data:

- EVS (2022): *EVS Trend File 1981-2017*. GESIS Data Archive, Cologne. ZA7503 Data file
  Version 3.0.0, doi:10.4232/1.14021
- Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano, J.,
  Lagos, M., Norris, P., Ponarin, E. & Puranen, B. et al. (eds.). 2022. *World Values
  Survey Trend File (1981-2022) Cross-National Data-Set*. Madrid & Vienna: JD Systems
  Institute & WVSA Secretariat. Data File Version 4.0.0, doi:10.14281/18241.27

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and
[SECURITY.md](SECURITY.md). The short version: one logical change per PR, `make check`
green, and never commit survey data or API keys.
