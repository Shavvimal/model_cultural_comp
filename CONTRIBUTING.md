# Contributing to model_cultural_comp

Thanks for your interest in improving this project. It is the code artefact behind a
paper, so the bar for anything that touches a published number is high — but
contributions of all sizes are welcome, from typo fixes to new models on the map.

## Ground rules

- Be respectful. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- Keep changes focused. One logical change per pull request.
- Discuss large changes first by opening an issue, so we agree on direction before you invest time.
- **Keep all data and generated outputs out of Git.** This includes model
  responses, small derived aggregates, figures, notebooks with outputs, fitted
  models and licensed survey inputs. The checksum manifest is metadata, not a
  dataset. Use the separate archive described in [REPRODUCING.md](docs/REPRODUCING.md).
- Keep credentials and private request logs local. `make check` enforces the
  indexed-file policy, including ignored files added with force.
- Regenerate results for changes that could affect them. Record the source
  revision, input-manifest hash, commands and relevant results in the PR;
  do not add the generated files to the commit.

## Development setup

```bash
git clone https://github.com/Shavvimal/model_cultural_comp
cd model_cultural_comp
uv sync --frozen    # or: make install
```

`app/` is installed as a package, so `from app.culture_map import CulturalMap` works
from any working directory — imports must never rely on the CWD being `app/`.

The survey data is not in the repo and is not needed for development. See the README
for how to obtain it if you want to run the reproduction.

## Before you open a PR

Run the full local gate — this mirrors CI exactly:

```bash
make check          # indexed-file policy + Ruff + pytest
```

Individual targets are available too: `make lint`, `make format`, `make test`.
`make typecheck` runs mypy over `app/` with an ephemeral mypy install; it is not
part of `make check` or CI.

`make test` runs synthetic, seeded fixtures without datasets, API credentials or
network calls. Add regression coverage when a change could affect scientific
results or input integrity; documentation changes need no artificial tests.

If a change could move a reported result, run the complete `make reproduce`
chain locally with the licensed cache and verified frozen response archive.
Paste the relevant reconciliation and command outcome in the PR. This includes
2026 QC, all sensitivity analyses and frozen-panel agreement. CI uses synthetic
tests and does not fetch the survey microdata or run live collection.

## Adding a new model

The surveyed-model list lives in `app/llm_meta.py`, and that is the single source of
truth. (There used to be four divergent copies of it across `culture_map.py`,
`culture_map_post_hoc.py` and the notebooks, which disagreed about which entries were
commented out — do not reintroduce a second copy.) Adding a model is:

1. An entry in the appropriate set in `app/llm_meta.py` — `CHINESE_LLMS_2026` /
   `WESTERN_LLMS_2026` for the cloud cohort, `CHINESE_LLMS` / `DOLPHIN_LLMS` for
   the 2024 local one — using the exact model tag it was collected under.
2. A collection run producing stored responses: `data/collection_2026/<model>[__zh].jsonl`
   via `scripts/collect_cloud_2026.py` for the current harness, in a separate experimental checkout. The frozen 2024 corpus uses
   `data/collection/*_responses_df.jsonl`; do not append new experiments to it. A new model is administered in
   **both** languages; a model surveyed in one arm only is a half cell and does
   not go on the map.
3. A protocol-document row citing the model's source (HuggingFace repo or Ollama tag),
   quantisation and licence.

A new model must satisfy the parser contract in `app/qn_classes.py`. Each of the ten
IVS items has a response class, and the model's raw output has to parse into it:

- **Bare integer** for the single-choice items (`A008`, `A165`, `E018`, `E025`,
  `F063`, `F118`, `F120`, `G006`) — one value from that item's `IntEnum`.
- **Tuple** for `Y002` — the most important and second most important goal.
- **List** for `Y003` — the chosen qualities, up to five.

A model that cannot produce parseable answers does not silently vanish from the
sample: record it in `FAILED_LLMS_2024` (or its successor) with the reason, since a
refusal or malformed-output rate is itself a result. Cells that fall below the
per-item inclusion threshold are reported as exclusions with their parse rates, not
dropped — `scripts/diagnostics_2026.py` carries the threshold sensitivity and the
Manski-style worst-case bounds for them.

## Pull request process

1. Fork and create a topic branch (`git checkout -b my-change`).
2. Make your change with tests and a green `make check`.
3. Open a PR against `main` and fill in the PR template.
4. A maintainer reviews; address review threads (they must be resolved before merge).
5. PRs are merged via **squash** to keep history linear.

## Commit messages

Keep them short and imperative ("store varimax rotation at fit time", not
"added/adds"). Reference an issue number when relevant.
