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
through. Each model gets a position with a bootstrap confidence region and an SVM
cultural-region assignment.

The work started as a 2024 blog post, [Cultural Bias in
LLMs](https://shav.dev/blog/cultural-bias). This repository is the corrected and
reproducible version of that analysis: the original post projected model responses
without the survey-fitted standardization and through a re-fitted rotation, so models
and countries were not actually in the same coordinate space. See
[CHANGELOG.md](CHANGELOG.md).

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
   is reproducible run to run.
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
   Each replicate resamples that model's stored responses per question with
   replacement, projects the resulting mean respondent, and the spread over replicates
   gives a 95% confidence ellipse.
6. **SVM regions** (`app/region_svm.py`) — an SVM fitted on the country coordinates
   gives cultural-region decision boundaries; each model gets a region plus the
   fraction of bootstrap replicates that land in it (its stability).

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
| `scripts/` | `validate_projection.py`, `bootstrap_llms.py`, `make_figures.py` |
| `notebooks/` | exploratory work; not the test suite |
| `figures/` | generated PDFs and PNGs |
| `data/` | gitignored; the IVS inputs and derived artefacts live here |
| `docs/review-rubric.md` | the release rubric this repo is held to |

## Models surveyed

Models were run locally through [Ollama](https://ollama.com) at Q4 quantisation unless
noted. The list lives in `app/llm_meta.py`; models that never produced parseable
answers in the 2024 run are recorded there in `FAILED_LLMS_2024` rather than being
silently dropped.

- **Chinese-origin / Chinese fine-tuned:** `qwen2:7b`, `llama2-chinese:13b`,
  `wangshenzhi/gemma2-27b-chinese-chat`, `wangrongsheng/llama3-70b-chinese-chat`
  (plus `yi:34b`, `aquilachat2:34b`, `glm4:9b`, `xuanyuan:70b`,
  `kingzeus/llama-3-chinese-8b-instruct-v3` — all unparseable, excluded)
- **Western:** `llama3:70b`, `mistral:7b`, `gemma2:27b`
- **Uncensored (Dolphin):** `dolphin-llama3:8b`, `dolphin-mistral:7b`,
  `dolphin-mixtral:8x7b`

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
and DOI, along with the tagged release the paper cites. Until then, cite the
repository and the origin post:

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
