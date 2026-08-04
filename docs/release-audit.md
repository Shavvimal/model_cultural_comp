# Release Audit — `model_cultural_comp`

Branch `fix/projection-and-repo-hygiene` · audited 2026-08-04 · read-only.

Audited against `docs/review-rubric.md` (109 checkboxes), with cairn-parity items compared
directly against `~/Code/cairn`.

> **Note on the moving target.** The statistical refactor landed *during* this audit: `HEAD` moved
> from `eac29ec` to `9f1f6d5` (via `ae69a7a` "Implement statistical review", `01f43b3` "Fix remaining
> lint", `9f1f6d5` "Add EM iteration cap and multi-seed sensitivity harness"). **All findings below
> were re-verified against `9f1f6d5`** and line numbers are as of that commit. One finding (F-2) was
> fixed by `9f1f6d5` while this report was being written and is marked accordingly; everything else
> in the CRITICAL/HIGH tiers survives the refactor.

Gates were executed, not assumed, at `9f1f6d5`:

| Gate | Result |
|---|---|
| `make lint` | **PASS** — `All checks passed!` |
| `make test` | **PASS** — 48 passed in 0.12s |
| `make check` (= `lint` + `test`) | **PASS, EXIT=0** — CI is green |
| `uv run ruff format --check app scripts tests` | ❌ **FAIL — 7 of 20 files would be reformatted** |

Reproducibility of the CI claim was verified independently: `git archive HEAD` into an empty
directory, then `pytest` → green with no dataset, no network, no Ollama. Rubric §0's CI hard fail is
genuinely satisfied.

The format failure is **at `HEAD`**, is invisible to every gate, and got worse during the refactor
(1 file → 7). See F-6.

---

## 1. COMPLIANCE VERDICT

### 1.1 The `pca-magic` / Apache-2.0 chain

Overall: **CONDITIONAL FAIL — one genuine gap (§4(a)), everything else is correct and, in
places, better than required.**

| Requirement | Status | Evidence |
|---|---|---|
| **Apache-2.0 §4(a)** — "give any other recipients … a copy of this License" | ❌ **FAIL** | The Apache-2.0 licence **text** appears nowhere in the repo. `NOTICE:17-18` gives only a URL (`http://www.apache.org/licenses/LICENSE-2.0`). A URL is not a copy. `git log --all --diff-filter=A` shows only `LICENSE` and `NOTICE` were ever added — no Apache text at any point. |
| **§4(b)** — "prominent notices stating that You changed the files" | ✅ PASS | `app/ppca.py:1-12` module header names upstream, URL, licence and summarises three changes; `NOTICE:26-36` enumerates them in full under an explicit "Statement of changes (Apache-2.0 §4(b))" heading. This is a model example. |
| **§4(c)** — retain copyright/attribution notices from the Source | ⚠️ VERIFY | `NOTICE:14` carries `Copyright Allen Tran`; `app/ppca.py:4` repeats it. Upstream `pca-magic` ships no `NOTICE` and no per-file copyright header, so there is nothing further to retain. **Action:** confirm against the upstream repo at the commit you derived from and record that commit SHA in `NOTICE` — right now the derivation names no version, which makes the "what changed" claim unauditable. |
| **§4(d)** — carry forward upstream `NOTICE` contents | ✅ PASS (vacuously + voluntarily) | §4(d) only binds if the upstream Work ships a `NOTICE`; `pca-magic` does not. This repo's `NOTICE` is therefore voluntary good practice. Verified it actually ships: a built wheel contains `model_cultural_comp-0.1.0.dist-info/licenses/{LICENSE,NOTICE}` (hatchling auto-detects both). |
| **MIT `LICENSE` at root** | ✅ PASS | Byte-identical to cairn's. |
| **MIT copyright year** | ⚠️ MINOR | `LICENSE:3` says `Copyright (c) 2026`. First commit is 2024-07-10 and half the history is 2024. Should read `Copyright (c) 2024-2026 Shav Vimalendiran` (and the same in `NOTICE:2`). |
| **`pyproject.toml` licence declaration** | ❌ **FAIL — rubric §0 hard fail, verbatim** | `pyproject.toml` has **no `license` and no `license-files` key**. Confirmed downstream: the built wheel's `METADATA` has `License-File: LICENSE` / `License-File: NOTICE` but **no `License-Expression:` field at all** — the published package declares no licence. |
| **README "Licensing" section** | ⚠️ **INCOMPLETE — understates the obligation** | `README:131-145` is otherwise excellent (states MIT/Apache-2.0 compatibility deliberately, per rubric §7). But `README:138-140` says redistribution is permitted "provided the upstream notices are retained and the modifications are stated" — it **omits §4(a)**, the copy-of-the-licence requirement. It then tells downstream users they "must carry the `NOTICE` file and the Apache-2.0 attribution", which is the same omission repeated. As written the README instructs downstream users to under-comply. |
| **`app/ppca.py` header** | ✅ PASS | `app/ppca.py:1-12`. Names project, URL, copyright holder, licence, and changes. |

**Exact fixes required:**

1. **Add the Apache-2.0 licence text.** Create `LICENSES/Apache-2.0.txt` containing the verbatim
   Apache License 2.0. Then:
   - `NOTICE:17-18` — replace *"You may obtain a copy of the Apache License, Version 2.0 at: <url>"*
     with *"A copy of the Apache License, Version 2.0 is included at `LICENSES/Apache-2.0.txt`,
     and is also available at http://www.apache.org/licenses/LICENSE-2.0"*.
   - `README:139-142` — replace *"provided the upstream notices are retained and the modifications
     are stated"* with *"provided a copy of the Apache-2.0 licence is supplied (§4(a)), the
     modifications are stated (§4(b)), and the attribution notices are retained (§4(c)-(d))"*, and
     change the downstream instruction to *"you must carry `NOTICE`, `LICENSES/Apache-2.0.txt`, and
     the file header with it"*.
   - `pyproject.toml` — the new file must ship in the wheel. Hatchling's default licence-file globs
     are `LICEN[CS]E*`/`COPYING*`/`NOTICE*`/`AUTHORS*`, which will **not** match
     `LICENSES/Apache-2.0.txt`. Set it explicitly (see fix 2).

2. **Declare the licence in `pyproject.toml`** (rubric §0 hard fail). Add to `[project]`:
   ```toml
   license = "MIT"
   license-files = ["LICENSE", "NOTICE", "LICENSES/*"]
   urls = { Repository = "https://github.com/Shavvimal/model_cultural_comp" }
   ```
   (`urls` also closes the rubric §7 metadata item; cairn has it, this repo does not.)

3. **Pin the upstream derivation.** Add to `NOTICE:13` the upstream commit SHA / release the file
   was derived from.

4. **Fix the copyright years** in `LICENSE:3` and `NOTICE:2` to `2024-2026`.

### 1.2 Secrets

**PASS — clean, and this is the strongest result in the audit.**

- `git log --all -p -S 'f625433d'` → **zero commits**. The live key
  (`OLLAMA_API_KEY=f625433d…`) never entered git in any form.
- `git log --all -S 'OLLAMA_API_KEY'` → one commit (`1166517`), and the added lines are
  dereferences only: `os.environ["OLLAMA_API_KEY"]` and a docstring. No value.
- `.env` was **never tracked**. `.gitignore:141-142` covers `**.env` and `.env`; `git check-ignore`
  confirms.
- Full-history sweep for `sk-…`, `ghp_…`, `AKIA…`, `hf_…`, `glpat-`, `AIza`, `xox*`, PEM private
  keys, and a generic `(key|token|secret|passw|auth)\s*[=:]\s*[A-Za-z0-9._-]{24,}` entropy pattern
  → **zero hits** across all 26 commits. Historical GraphRAG configs contain only
  `${OPENAI_API_KEY}` placeholders.

**Residual actions (hygiene, not blockers):** rotate the live Ollama key as routine practice before
going public; enable GitHub secret scanning + push protection (rubric §6 requires this and it cannot
be verified from the local clone).

### 1.3 Data redistribution (WVS / GESIS)

**PASS on the blocker; one narrow gap in the ignore rules.**

- `git ls-files data/` → **empty**. Nothing under `data/` is tracked.
- Every path ever added to history matching `data/|*.pkl|sav|sps|npy|npz|csv|jsonl|parquet|log|dta|
  xlsx|h5|zip|gz` across all 26 commits is exactly one file: `examples/single_verb/input/data.csv`,
  a 3-line GraphRAG fixture (`col1,col2 / 2,4 / 5,10`), since deleted. **No IVS/WVS/EVS microdata
  ever entered history.**
- `.gitignore:12` `data/` covers every format present. Verified individually with `git check-ignore`:
  `data/x.{csv,npy,npz,sav,sps}` all ignored.
- Largest blob ever: 2.5 MB (`notebooks/3-UMAP.ipynb`, an old outputs-laden revision). `.git` total
  7.9 MB. No history rewrite needed on size or data grounds.

⚠️ **Gap:** `*.sav`, `*.sps`, `*.npy`, `*.npz` are ignored *only* by virtue of living under `data/`.
`git check-ignore` confirms a root-level `model.npy` or `out.csv` is **NOT IGNORED**. Any script run
from a different CWD (several default to relative paths — see F-3) can drop a licensed artefact
outside `data/` where `git add -A` will take it. **Fix:** add global `*.sav`, `*.sps`, `*.npz`,
`*.npy` rules to `.gitignore`.

⚠️ **Also unignored right now:** `.DS_Store` (present, untracked, no ignore rule — `grep -c DS_Store
.gitignore` → 0) and `docs/wiki-math-excerpts.md`. See 1.6.

❗ **The one live data landmine: `notebooks/.ipynb_checkpoints/1-ivs-checkpoint.ipynb`** (209,644 bytes).
It is correctly gitignored (`.gitignore:97`, confirmed by `git check-ignore`), but its contents are
exactly what may not be redistributed:

- cell 3 (`ivs_df.head()`) renders **raw respondent rows** (`studyno 4001.0, version 4-0-0
  (2024-06-30), doi doi.org/10.14281/18241.27 …`);
- cell 8 renders respondent-level `S020 S003 A008 A165 E018 E025 F063 F118 F120 G006 Y002` values;
- cell 9 is a `.describe()` over **666,907 respondents**;
- cell 18 embeds an **`image/png` of 148,336 base64 chars (~109 KB)**.

An ignore rule is the only thing between this file and a WVS/GESIS agreement breach. It survives
`git add -f`, any `.gitignore` edit, and — critically — **any zip/tarball of the working tree**,
which is how code artefacts are usually sent to reviewers and camera-ready submission systems.
**Fix: delete `notebooks/.ipynb_checkpoints/` from disk**, not just from git's view.

### 1.4 Vendored third-party code in git history — **the "caught out later" risk**

⚠️ **FLAG — the largest unaddressed provenance item in the repo.**

History contains **393 files of Microsoft GraphRAG** plus an unrelated news/RSS pipeline, added in
`162f0fd "updated graphrag"` (2024-08-08) and removed in `dd33fad "cultural rollbakc"` (2024-08-11).
501 paths in total were added-then-deleted: 393 `graphrag/`, 43 `examples/`, 24 `docs/`, 14 `bin/`,
10 `pipeline/`, 9 `app/`, 6 `notebooks/`.

- The vendored files **do** retain their per-file headers
  (`# Copyright (c) 2024 Microsoft Corporation. / # Licensed under the MIT License`), which satisfies
  the attribution half of MIT.
- But **GraphRAG's own `LICENSE` file was never added** — `git log --all --diff-filter=A` shows only
  this repo's `LICENSE` and `NOTICE` were ever committed. MIT requires *"The above copyright notice
  **and this permission notice** shall be included"*; the permission notice is absent.
- Git history is distribution. Flipping the repo public publishes 393 files of someone else's code
  under an incomplete licence grant, in a repo whose root `LICENSE` asserts
  `Copyright (c) 2026 Shav Vimalendiran` over the whole tree.

Risk is **low-severity but real and permanent**, and it is exactly the class of thing that surfaces
later. **Recommended fix — squash.** The rubric's own escape hatch (§6, "Git history is
presentable") already applies for an independent reason (see 1.7), and a squash to a single initial
public commit resolves the GraphRAG provenance, the 2024 commit-message typos, and the historical
outputs-laden notebook blobs in one move. The alternative — leaving it and adding a `NOTICE` stanza
for GraphRAG — is more work for a worse result.

### 1.5 `modelfiles/` — **delete before release**

❌ **FAIL on three independent counts.** All five files, one line each:

```
modelfiles/aquilachat2:1  FROM C:\Users\shavh\.ollama\models\huggingface-hub\aquilachat2-34b-16k.Q4_K_M.gguf
modelfiles/deepseek:1     FROM C:\Users\shavh\.ollama\models\huggingface-hub\deepseek-llm-67b-chat.Q4_K_M.gguf
modelfiles/glm:1          FROM C:\Users\shavh\.ollama\models\huggingface-hub\aquilachat2-34b-16k.Q4_K_M.gguf
modelfiles/xuan:1         FROM C:\Users\shavh\.ollama\models\huggingface-hub\XuanYuan-70B.Q4_0.gguf
modelfiles/yi:1           FROM C:\Users\shavh\.ollama\models\huggingface-hub\aquilachat2-34b-16k.Q4_K_M.gguf
```

1. **Personal machine leakage** — hardcoded absolute Windows paths and the username `shavh`, in
   tracked files. Rubric §0 hard fail #5, GOTCHAS Gotcha 4.
2. **Scientific misattribution** — `modelfiles/glm` and `modelfiles/yi` both point at the
   **AquilaChat2** GGUF. Neither references a GLM or Yi weight file at all. `README:123-124` and
   `app/llm_meta.py:36-40` both claim `yi:34b`, `glm4:9b` and `aquilachat2:34b` were surveyed and
   failed to parse; the tracked artefacts do not support which model produced which failure. (This
   is moot for the published numbers — all three are in `FAILED_LLMS_2024` and contribute no rows —
   but it is a claim in the paper's model table with no reproducible basis.)
3. **Unreproducible** — the paths resolve on exactly one machine, so the files serve no reader.

**Fix:** `git rm -r modelfiles/`. If the provenance matters, replace with a README table row naming
the HuggingFace repo + quantisation per model (which rubric §7 asks for anyway), and note in
`app/llm_meta.py:36-40` that the GLM/Yi/AquilaChat2 modelfile provenance could not be reconstructed.

### 1.6 Untracked-but-unignored files

⚠️ Three files sit one `git add -A` from being committed:

| File | Risk |
|---|---|
| `.DS_Store` | Noise. No `.gitignore` rule exists. |
| `docs/statistical-review.md` | Intentional (in-flight work) — fine to commit, but confirm it should be public. |
| `docs/wiki-math-excerpts.md` (626 lines) | **Should never be committed.** Contents are Shav's *own* wiki, so **no third-party copyright issue** — but it embeds personal absolute paths on nearly every "Source:" line (`/Users/shav/Code/ShavWiki/pages/…`, `/Users/shav/Code/next-mdx-blog/content/blog/cultural-bias-2026.mdx`) and names an unpublished venue ("ORACLE @ EMNLP 2026"). Rubric §0 hard fail #5. |

**Fix:** add `.DS_Store` and `docs/wiki-math-excerpts.md` to `.gitignore` (or move the latter out of
the repo entirely).

### 1.7 Notebooks and figures

- **Notebook outputs:** of 10 tracked notebooks, 8 are fully cleared (`7-pca-db.ipynb` is fully
  clean, execution counts included). Two retain outputs:
  - `1-ivs.ipynb` — 1 output (cell 5): the SPSS "Variable View" dataframe, `[838 rows x 7 columns]`
    truncated to 10 visible rows. **Variable metadata, not microdata** (`Y023C … Welzel choice-3`).
    The `ivs_df.head()` cells were correctly stripped. Low risk.
  - `10-results-view.ipynb` — **11 of 12 cells have outputs**, 83,499 bytes, mostly output not code.
    Contents: five `text/html` dataframe tables of **country-level aggregates** (derived, not raw);
    an **11,823-char `stdout` dump of the entire result set** at full precision (cell 10, e.g.
    `{'x': 2.9175488829472203, …, 'label': 'dolphin-llama3:8b'}`); a **12,658-char sklearn
    estimator-repr HTML/CSS blob** plus an 80-line `GridSearchCV` verbose log (cell 5); and the leak
    below. No survey microdata, no `image/png`.
  - **The one real leak:** `10-results-view.ipynb` cell 7 stderr (JSON line 1645) —
    `C:\Users\shavh\AppData\Local\Temp\ipykernel_33196\569078938.py:11: SettingWithCopyWarning`.
    Leaks OS, username, and kernel PID, and cross-links to `shavhugan@gmail.com` in
    `pyproject.toml:6`. Rubric §0 hard fails #5 and #6. Clearing that notebook's outputs removes it.
- ❗ **Four notebooks are non-executable as committed.** `notebooks/2-pca.ipynb:20`,
  `5-decison-boundary.ipynb:266`, `7-pca-db.ipynb:14` and `9-pca-llm.ipynb:17` all do
  `from ppca import PPCA` — a bare CWD-dependent import of a module that no longer exists
  (`notebooks/ppca.py` was consolidated into `app/ppca.py` per `CHANGELOG.md:38-40`, but the
  notebooks were not updated). A reviewer opening any of them hits `ModuleNotFoundError` at cell 0.
  This is rubric §1's bare-import checkbox, surviving in the notebooks after being fixed in `app/`.
  Fix: `from app.ppca import PPCA`.
- **Figures:** all six are generated by `scripts/make_figures.py` from the fitted model — **no WVS-
  published cultural-map image is reproduced**, so no Inglehart-Welzel figure copyright is engaged
  (confirmed by rendering `fig1_cultural_map.png`: a matplotlib scatter with legend and CI ellipses,
  not a scan or crop). Embedded fonts are `DejaVuSans` / `DejaVuSans-Bold`, subsetted and fully
  embedded — matplotlib's default, under the Bitstream Vera/DejaVu licence, freely redistributable
  and embeddable. PDF metadata carries only `Matplotlib v3.11.1`; no author or host strings.
  **No compliance issue.**
  ⚠️ **One publication-practical caveat:** the fonts are embedded as **Type 3** (matplotlib's
  `pdf.fonttype` default). IEEE and several ACM/ACL tracks reject Type 3 outright.
  `scripts/make_figures.py` sets no rcParam; add `matplotlib.rcParams["pdf.fonttype"] = 42` before
  the figures are regenerated for camera-ready.
- **Commit hygiene:** 13 of 26 commit subjects fail the rubric, all already on `origin/main`,
  including the two the rubric names by shape: `dd33fad "cultural rollbakc"` and
  `3399651 "decsion boundaries"`, plus six contentless nouns (`README` ×2, `map`, `chinese LLMs`,
  `output parsers`, `refusal script`). The feature branch is clean bar `709804a` (a 215-character
  machine-generated subject) and `eac29ec "Update dependencies and refactor code for improved
  readability"`. Reinforces the squash recommendation in 1.4.

### 1.8 Survey item texts (IVS prompts) — is reproducing them a risk?

**Low legal risk — but more WVS text is reproduced verbatim than the repo acknowledges, and right
now it is the one category of third-party content that is entirely undeclared.** Worth fixing with
citations, not redactions.

The ten prompts live at `app/cloud_survey.py:111-120` (English, 623 words / 3,398 chars) and
`:144-153` (Chinese, 1,029 chars), with a near-identical earlier copy in
`notebooks/6-model_answers.ipynb` cell 7. Reading them against the WVS master questionnaire, the
mix is **not** uniformly "adapted":

- **Near-verbatim WVS stems:** `A165` ("Generally speaking, would you say that most people can be
  trusted or that you need to be very careful in dealing with people?" — exact), `F063` ("How
  important is God in your life?" plus the 1-10 anchor — exact), `Y002`'s stem (near-exact).
- **Fully verbatim response battery:** `Y003` (`:120`) reproduces **all eleven WVS A027-A042 child-
  quality labels word for word** — "Good manners / Independence / Hard work / Feeling of
  responsibility / Imagination / Tolerance and respect for other people / Thrift, saving money and
  things / Determination, perseverance / Religious faith / Not being selfish (unselfishness) /
  Obedience". This is the largest single verbatim lift in the repo.
- **Genuinely paraphrased:** `A008`, `E018`, `E025`, `F118`, `F120`, `G006` — all restructured from
  WVS card/battery format into single self-contained sentences, with the authors' own format
  instruction appended.
- **Also reproduced, and uncited:** the WVS **codebook recoding syntax** for the two derived indices,
  quoted as SPSS in `notebooks/6-model_answers.ipynb` cells 10/14 and `8-llm-collate.ipynb` cells
  4/6 (`IF ((Q154=1 and Q155=3) or (Q154=3 and Q155=1)) then 1`, `Y003=(Q15 + Q17)-(Q8+Q14)`),
  attributed only as "the World Values Survey cookbook". And the WVS **value labels** in
  `app/qn_classes.py:24-25, 35-37, 48-50, 113-116` (`1: Most people can be trusted / 2: Can´t be too
  careful`), carried over from the `.sav` file complete with its latin1 acute accents.

**Legal verdict: still low risk, and not a release blocker.** Reproducing survey instrument text in
the methods artefact of a paper is necessary for reproducibility and is squarely within quotation
for research and criticism; the WVS/GESIS agreements you are bound by govern the **microdata**, a
separate work, correctly not redistributed. But the *asymmetry* is the thing that would embarrass
you later: this repo declares its third-party **code** meticulously (`NOTICE`, `app/ppca.py`) and its
third-party **data** repeatedly (README, SECURITY, CONTRIBUTING) — and says nothing at all about the
third-party **text** it reproduces. Three fixes, all one-liners, and all of them are scientific as
much as legal:

1. **Cite the questionnaire in place** (rubric §7: "Any … procedure taken from the Inglehart-Welzel /
   WVS publications is cited in place"). Add a docstring above `IV_QN_PROMPTS`
   (`app/cloud_survey.py:110`): item stems and the Y003 battery reproduced/adapted from the WVS-7
   Master Questionnaire, with citation, noting the format instructions and personas are the authors'.
   Same for the cookbook syntax in notebooks 6 and 8, and the value labels in `app/qn_classes.py`.
2. **Add a "Survey instrument" stanza to `NOTICE`** alongside the pca-magic stanza. This costs
   nothing and closes the asymmetry completely.
3. **Disclose that the Chinese prompts are the authors' own translations.** `IV_QN_PROMPTS_ZH`
   (`:143-153`) carries no such statement — the header at `:136-141` says only that the 2024
   translations were "corrected", without saying by whom or against what. The WVS publishes
   *official* Chinese questionnaires; choosing unofficial translations is a methodological decision
   a reviewer will press on, and the language-effect result (`analyze_2026.py:79-112`) rests on it.
   **This is the single highest-value disclosure in this section.**

⚠️ **One substantive methodological finding surfaced by the same read:** `G006` (`:118`) asks "How
proud are you to be **your nationality**?" — the WVS item is `How proud are you to be [Nationality]?`
with a respondent-anchored substitution. The literal string "your nationality" leaves the model no
nationality to anchor on, which is a real construct-validity issue for an item that loads on the
map's traditional-values axis. Worth a line in the paper's limitations regardless of the licensing.

### 1.9 Other third-party derivation sweep

| Surface | Verdict |
|---|---|
| `app/ppca.py` | Apache-2.0, handled (1.1). |
| Varimax rotation | **Not vendored** — `factor_analyzer.Rotator` is imported as a dependency (`app/culture_map.py:20`). Correct. |
| `notebooks/ppca.py` duplicate | **Removed** — deleted on this branch, recorded in `CHANGELOG.md:38-40`. Confirmed absent from the tree. |
| `app/country_meta.py:3-508` | ~500 lines of country name + numeric code. The parenthetical-`(the)` style (`"British Indian Ocean Territory (the)"`) is the **ISO 3166-1 English short-name convention**, so this is copied from the ISO/Wikipedia listing, uncited. ISO 3166 name lists are near-universally treated as non-copyrightable facts, so this is **not a legal issue** — but a one-line provenance comment closes the question for free. |
| `app/country_meta.py:530+` | `cultural_regions = {...}`, ~110 countries → the eight Inglehart-Welzel zones. This is substantively **Inglehart & Welzel's published classification** and is the one genuinely borrowed *intellectual* artefact in the codebase with no inline citation — the only comment is `# Adding cultural regions for the regions in our dataset`. The README cites WVS/EVS for the *data* but nothing points this mapping at its source. Rubric §7. |
| Plotting idioms | `scripts/make_figures.py:167-176` is the canonical sklearn `meshgrid → predict → contourf` decision-boundary pattern — idiomatic, not a verbatim gallery paste (no `make_meshgrid`/`plot_contours` helpers). `llm_bootstrap.confidence_ellipses` (`:209-233`) is **not** the matplotlib gallery `confidence_ellipse()` recipe (no Pearson/`Affine2D().rotate_deg(45)` construction). Both original enough; no attribution needed. |
| Notebooks | Swept for `stackoverflow`, "adapted from", "copied", blog URLs → **zero hits** across `app/`, `scripts/`, `tests/` and all 10 notebooks. |
| `uv.lock` dependencies | Spot-checked the three the rubric names: `factor_analyzer` (GPLv2 → **see F-2**), `pyreadstat` (Apache-2.0), `ollama` (MIT). |
| `data/collection_2026_run.log`, `docs.md` | Untracked and gitignored (`.gitignore:13`, `*.log`). No issue. |

❗ **`factor_analyzer` is GPLv2.** `app/culture_map.py:20` imports `Rotator` from it, and it sits in
`[project.dependencies]` as a hard runtime requirement. This does **not** taint the repo's source
(you are not distributing `factor_analyzer`, and mere-aggregation/dynamic-import arguments apply),
so the MIT licence on your own code stands. But rubric §7's last checkbox is *"Third-party dependency
licences are compatible with MIT redistribution — spot-check the non-obvious ones
(`factor_analyzer`, …)"*, and the honest answer is "GPLv2, and anyone who redistributes a combined
binary/container of this project inherits GPL obligations". **Fix:** state it in the README's
Licensing section in one sentence. Varimax is ~15 lines; if you would rather not carry the
dependency at all, implementing it directly (and citing Kaiser 1958) removes the question entirely
and removes a dependency from the reproduction path.

---

## 2. RUBRIC SCORECARD

**109 checkboxes: 53 ✅ · 25 ⚠️ · 31 ❌ — 49% clean pass, 72% pass-or-partial.**

Seven of the 20 ⚠️ are "not verifiable from a local clone" (branch protection, secret scanning,
repo About/topics, GHSA enablement, community profile) rather than known failures.

### §0 Hard fails (9) — 4 ✅ · 2 ⚠️ · 3 ❌ · **RELEASE BLOCKED**

| | Item | Evidence |
|---|---|---|
| ❌ | LICENSE declared in `pyproject.toml` | LICENSE exists; `license`/`license-files` keys **absent**. Wheel METADATA has no `License-Expression`. |
| ⚠️ | NOTICE + `ppca.py` header | Both excellent for §4(b)-(d); **§4(a) copy-of-licence missing**. |
| ✅ | No secrets in tree or history | `-S 'f625433d'` → 0 commits; `.env` never tracked; 0 credential-pattern hits in 26 commits. |
| ✅ | No survey microdata in history | Only ever-added data file is a 3-line GraphRAG fixture. `git ls-files data/` empty. |
| ❌ | No personal paths/machine names | 5× `modelfiles/*:1` `C:\Users\shavh\…`; `notebooks/10-results-view.ipynb:1645`. |
| ❌ | Notebooks have cleared outputs | `10-results-view.ipynb` 11 of 12 cells incl. the Windows temp path + an 11.8 KB full-precision result dump; `1-ivs.ipynb` 1 output. Plus the ignored-but-on-disk `1-ivs-checkpoint.ipynb` holding real microdata (1.3). |
| ✅ | CI green on clean checkout, no dataset | Verified: `git archive HEAD` → clean dir → `pytest` = 46 passed. |
| ✅ | Numbers reproducible from tagged commit | `make validate` documented, `scripts/validate_projection.py` is a real gate. (Tag itself: §6.) |
| ⚠️ | No silent fallback on a published-number path | **Three found** — F-1, F-4, F-7. |

### §1 Python craft (15) — 6 ✅ · 2 ⚠️ · 7 ❌

| | Item | Evidence |
|---|---|---|
| ✅ | PEP 604/585 typing, no `Optional`/`List` | `grep 'from typing\|Optional\[\|List\['` across `app/ scripts/ tests/` → **zero hits**. Rubric's noted violations are fixed. |
| ❌ | Every public function/method fully typed | `app/ppca.py` has **zero annotations** on all 6 methods; `CulturalMap.__init__`, `prepare_data`, `fit`, `_rescale`, `save_model`, `load_model`, `visualize_cultural_map`, `y002_transform` all partly/wholly unannotated. |
| ✅ | Import grouping stdlib→3rd→1st | Correct in every module. |
| ❌ | Absolute package-qualified imports | Fixed in `app/` ✅ — but **four notebooks still do the exact bare import the rubric names**: `from ppca import PPCA` at `2-pca.ipynb:20`, `5-decison-boundary.ipynb:266`, `7-pca-db.ipynb:14`, `9-pca-llm.ipynb:17`, targeting a module that no longer exists. Also `llm_bootstrap.py:296` is a function-local import of an already-module-imported symbol. |
| ✅ | No duplicated module | `notebooks/ppca.py` removed; verified absent. |
| ⚠️ | Private single underscore | Mostly correct; **`app/cloud_survey.py:237` and `scripts/collect_cloud_2026.py:35` reach into another object's privates** (`parser._valid_values`, `survey._client`). |
| ✅ | `snake_case.py` | All files. |
| ⚠️ | Comments explain *why* | Mostly exemplary. But `app/qn_classes.py:12,22,34,46,61,78,95,112` are pasted numpy reprs (`[ 1.  2.  3.  4. nan]`) masquerading as docstrings. |
| ✅ | Section-marker comments | `app/culture_map.py:92,194,223,302,328`, `app/cloud_survey.py:46,106`. |
| ❌ | No `print()` in library code | **7 sites in `app/`**: `ppca.py:102`, `llm_bootstrap.py:72,108,200`, `cloud_survey.py:322,335,338`. |
| ❌ | Log messages carry context | No logging exists at all — `grep 'logging\|getLogger' app/ scripts/` → **zero hits**. |
| ❌ | Error logs use `exc_info=True` | No logging exists. |
| ❌ | Log levels used correctly | No logging exists. |
| ✅ | No log spam | Vacuously — the `print`s that exist are milestone-shaped, not per-item. |
| ❌ | `ruff check` **and** `ruff format --check` pass | `ruff check` ✅. `ruff format --check` **FAILS at `HEAD`** on `scripts/analyze_2026.py`; 3 files in the working tree. |

### §2 API & config design (11) — 3 ✅ · 3 ⚠️ · 5 ❌

| | Item | Evidence |
|---|---|---|
| ❌ | Tunables in a Pydantic config class | **No configuration class exists.** Every value the rubric names is a bare literal: `tol=1e-4`, `min_obs=10` (`ppca.py:35`), `d` (`culture_map.py:134`), `PC_RESCALE_PARAMS` (`culture_map.py:47`), `MAX_ATTEMPTS=3`/`N_REPEATS=5` (`cloud_survey.py:193-194`), `concurrency=6` (`cloud_survey.py:204`). |
| ❌ | Field `description` says *why* | No config class. |
| ❌ | Range constraints on fields | No config class. `n_boot`, `concurrency`, `min_obs` accept 0 or negative silently. |
| ✅ | Mutable defaults use `default_factory` | `cloud_survey.py:202-203` correct. |
| ❌ | Config embedded via `default_factory=XxxConfiguration` | No config class. |
| ⚠️ | Published defaults == paper's values | `SEED=42` consistent across all four scripts. But `n_boot` defaults **disagree with callers**: `llm_bootstrap.py:122` `n_boot=1000` / `:154` `n_boot=10_000` vs `analyze_2026.py:42-43`. A reader passing nothing does not get the paper's numbers. |
| ⚠️ | Item codes / colours / mappings are named constants | `IV_QNS`, `PC_RESCALE_PARAMS`, `CULTURAL_REGION_COLORS`, `ITEM_VALID_RANGES` all module-level ✅ — but **none is `Final[...]`**, and `country_meta.py:516` **re-types `iv_qns` as a fifth copy**. |
| ✅ | Cosmetic constants stay hardcoded | `make_figures.py` fontsizes/DPI correctly not promoted. |
| ✅ | No speculative config class | Vacuously true. |
| ❌ | No `os.getenv()` buried in classes | `cloud_survey.py:202-203` reads `OLLAMA_HOST` **and** `OLLAMA_API_KEY` inside the dataclass. Rubric names this case explicitly. |
| ⚠️ | File paths are parameters, never hardcoded | `CulturalMap(ivs_df_path=…)` correct ✅. But `culture_map.py:68` defaults `data_dir="../data"`; `culture_map.py:382-383` and **`country_meta.py:509,762` hardcode `../data/…` at module scope** (F-3). |

### §3 Composition, structure & error handling (13) — 7 ✅ · 3 ⚠️ · 3 ❌

| | Item | Evidence |
|---|---|---|
| ❌ | Collaborators injected for testability | `cloud_survey.py:211` constructs `AsyncClient` internally in `__post_init__`. No seam — hence **zero tests for `CloudSurvey` or any parser**. |
| ❌ | Each module does one thing | `app/country_meta.py` is a **script inside the package**: executes at import, reads `../data/ivs_df.pkl` (`:509`) and **writes `../data/country_codes.pkl` (`:762`)**. Verified: `import app.country_meta` → `FileNotFoundError`. It ships in the wheel. |
| ✅ | No class wraps a single dependency | `CloudSurvey` adds resumption, concurrency, retry, parsing. |
| ✅ | No over-engineering | Modules 87-339 lines (`country_meta.py` excepted); no manager/strategy layers. |
| ✅ | Stateful pipeline validates prerequisites | `culture_map.py:129-130, 187-188, 206-207`; `ppca.py:136-137`; `region_svm.py:59,64`. Tested (`test_culture_map.py:39-48`). |
| ✅ | Pydantic for parsed data contracts | `app/qn_classes.py` — `Y002`/`Y003` are `BaseModel` with `Field(description=…)`. |
| ⚠️ | Invariant violations raise loudly with context | Mostly good — `culture_map.py:158-163` (merge row-count) and `llm_bootstrap.py:134-136, 165-169` are model examples. **But** `culture_map.py:231-232` returns the sentinel `-5` instead of raising (F-4), and `_run_task` records dicts to disk that the 2026 record contract does not type. |
| ❌ | Semantic exception types | **None exist.** `grep 'class.*Error'` → zero. Bare `RuntimeError` at `ppca.py:137`, `culture_map.py:130,188,207,159`, `region_svm.py:59,64`; bare `ValueError` for every parse failure. No `ResponseParseError`, `ConvergenceError`, `InsufficientDataError`. |
| ✅ | No bare `except Exception:` that swallows | One broad catch, `cloud_survey.py:310`, `# noqa: BLE001`, records the error, does not swallow. The `llm_data_gen.py`/`chinese_llm_data_gen.py` violations the rubric names are gone (files deleted). |
| ✅ | Fallbacks carry the 3-part justification | `cloud_survey.py:306-309` and `llm_bootstrap.py:192-194` both do all three parts. Exemplary. |
| ⚠️ | Retry catches specific parse exception, caps, raises after exhaustion | Caps at `MAX_ATTEMPTS` ✅, records ✅. But `cloud_survey.py:302` catches `KeyError` **alongside** `ValueError` — a `KeyError` from `PARSERS[qn]` or `response["message"]` is mislabelled `parse:` (F-5). And exhaustion returns a record rather than raising — defensible given the record is persisted. |
| ✅ | Dropped responses counted and reported | `llm_bootstrap.py:69-76, 105-108`; `analyze_2026.py:52-76` writes `llm_parse_rates_2026.csv`. Genuinely well done. |
| ⚠️ | 5+ arm branching uses dispatch dict | `PARSERS` dict ✅. But `_to_value` (`llm_bootstrap.py:41-46`) is a 3-arm if-chain over question type — under the threshold, so passing, but it will grow. |

### §4 Numerics & reproducibility (12) — 7 ✅ · 3 ⚠️ · 2 ❌

| | Item | Evidence |
|---|---|---|
| ✅ | Randomness seeded, seed documented | `ppca.py:65` `default_rng(seed)`; `SEED=42` in all four scripts; `StratifiedKFold(random_state=0)`. |
| ✅ | Explicit `Generator`, not global state | `default_rng` everywhere; no `np.random.randn` remains. |
| ⚠️ | Non-convergence raises | **Fixed mid-audit by `9f1f6d5`**: `max_iter=1000` (`ppca.py:35`) with a raise at `:107-110` carrying `diff`, iters and `tol`. Remaining gap: bare `RuntimeError`, not a semantic `ConvergenceError` (F-8). |
| ✅ | Sign/rotation indeterminacy resolved deterministically | `culture_map.py:165-184` `_orient_rotation` pins order and sign via F118/F063; `ppca.py:117-120` pins loading signs. Documented and tested (`test_culture_map.py:51-57`). Best-in-repo. |
| ⚠️ | `pc_rescale_params` provenance documented | `culture_map.py:46` says "Published WVS rescaling constants" — names no publication, page, or derivation. The rubric calls this "the highest-risk number in the repo". |
| ❌ | **Survey weights `S017` applied consistently** | **`S017` is selected and renamed to `weight` at `culture_map.py:100-101` and then never used again anywhere.** `calculate_mean_scores` (`:211-217`) takes an unweighted `.mean()`. Country coordinates are unweighted (F-1). |
| ✅ | Missing-data handling explicit and stated | `thresh=6` (`:117`), `min_obs` (`ppca.py:48`), `ITEM_VALID_RANGES` sentinel recode with counts (`:110-114`), reported by `validate_projection.py:110-113`. |
| ⚠️ | Bootstrap states B, unit, and interval method | Code is excellent — `llm_bootstrap.py:1-21` names both estimators, resampling unit, and the Cameron/Gelbach/Miller caveat. **But `README.md:82-86` describes only the item bootstrap** and omits the cluster bootstrap, B, and the method. README is stale against the code. |
| ✅ | FP comparisons use `assert_allclose` with tolerance | Throughout `tests/`; `assert_array_equal` only for exact-equality intent. |
| ✅ | One lockfile, committed | `uv.lock` tracked; `poetry.lock` deleted. Rubric's violation is fixed. |
| ✅ | Validation script reproduces coords against reference within tolerance | `validate_projection.py:42-54`, `TOL_EXACT=1e-8`, non-zero exit on failure. |
| ❌ | Data provenance documented but not shipped | Provenance is documented well (`README:34-51`) ✅ — but **the README's pipeline description is now wrong**: it omits the `ITEM_VALID_RANGES` sentinel recode and the unit-variance `score_stds` step, both of which change published numbers. Rubric says "verify it is still accurate". |

### §5 Testing (11) — 7 ✅ · 1 ⚠️ · 3 ❌

| | Item | Evidence |
|---|---|---|
| ✅ | Green with no network/ollama/dataset | Verified on a clean `git archive` checkout: 46 passed in 0.11s. |
| ⚠️ | All external deps mocked | Vacuously true — **nothing that touches ollama is tested at all**, so nothing needs mocking. Not the spirit of the rule. |
| ✅ | Synthetic fixtures seeded, as code | `tests/conftest.py:14-53`, `default_rng(0)`, no binary blobs. |
| ✅ | PPCA tested against known closed-form answer | `test_ppca.py:22-24` recovers planted 2-factor structure, `var_exp > 0.9`. |
| ✅ | PPCA tested with missing entries | `test_ppca.py:40-46`, 10% NaN. |
| ❌ | **Every response-parser class tested** for valid / refusal / out-of-range / malformed / boundary | **The parsers are entirely untested.** `EnumOutputParser`, `Y002OutputParser`, `Y003OutputParser` and `clean_content` (`app/cloud_survey.py:39-103`) have **zero tests**. `tests/test_qn_classes.py` tests the *pydantic enums*, not the parsers — no refusal string, no malformed string, no `<think>` stripping, no code-fence stripping. Only 3 of the 8 scale enums are covered (`A008`, `A165`, `F118`; `E018`, `E025`, `F063`, `F120`, `G006` untested). |
| ❌ | Error paths tested | Partially: `test_ppca.py:61-70` and `test_culture_map.py:39-48` cover 4 raises. **Untested:** the merge-row-count `RuntimeError` (`culture_map.py:159`), both `llm_bootstrap` `ValueError`s (`:136`, `:166`), `cohort_2026`'s `ValueError` (`llm_meta.py:88`), and the non-existent convergence raise. |
| ✅ | Tests assert on values | Throughout — coefficients, orthogonality, sign conventions, exact index values. |
| ❌ | Dataset-requiring tests marked and deselected | No `@pytest.mark.integration`/`slow` anywhere; no `markers` in `pyproject.toml`. Currently moot (no such test exists), but the marker infrastructure the rubric asks for is absent. |
| ✅ | `testpaths` matches an existing `tests/` | `pyproject.toml:37`; `tests/` exists with 5 files, 46 tests. Rubric's violation is fixed. |
| ✅ | Notebooks are not the test suite | All logic in `app/`; notebooks excluded from `testpaths`. |

### §6 Open-source readiness (27) — 14 ✅ · 8 ⚠️ · 5 ❌

| | Item | Evidence |
|---|---|---|
| ✅ | README covers claim→install→data→reproduce→layout→citation→licence, in order | `README.md` follows exactly that order. |
| ✅ | Badge row under title | `README:3-5` — ci, MIT, Python 3.11+. |
| ❌ | `CITATION.cff` + BibTeX | BibTeX present (`README:153-160`) ✅; **`CITATION.cff` absent**. This repo is a paper artefact — it needs one more than cairn does. |
| ✅ | `CONTRIBUTING.md` | 1:1 heading parity with cairn; documents setup, `make check`, PR flow, squash/linear. **Exceeds cairn** with the `make test` vs `make validate` two-gate split. |
| ✅ | `CODE_OF_CONDUCT.md` Covenant 2.1, GitHub private reporting | **Byte-identical to cairn's** (5624 bytes). Enforcement routes to GitHub private reporting; **no personal email**. |
| ⚠️ | `SECURITY.md` via GHSA, private reporting enabled | File is excellent — routes to GHSA, and its "Sensitive surface" section is substantively stronger than cairn's. **No supported-versions table** (cairn lacks one too). Whether private reporting is *enabled* on the repo cannot be verified locally. |
| ✅ | Both issue templates + `config.yml` | All three present; `bug_report.yml` adds an IVS-data dropdown, `feature_request.yml` adds a published-results-impact checkbox. Above parity. |
| ✅ | `PULL_REQUEST_TEMPLATE.md` with checklist | `make check`, tests, docs, linked issue — plus a no-data/no-keys line and a `make validate` line. Above parity. |
| ✅ | `.github/CODEOWNERS` | Byte-identical to cairn's. |
| ✅ | `.github/dependabot.yml` | Byte-identical: `github-actions` + `uv`, both weekly. |
| ⚠️ | `CHANGELOG.md` Keep-a-Changelog, seeded with the paper's release | Header byte-identical to cairn's, correct Added/Changed/Fixed/Removed. **But everything sits under `## [Unreleased]`** — no released section, no dates, no link-reference definitions, and `pyproject`'s `version = "0.1.0"` appears nowhere in it. |
| ⚠️ | `Makefile` with all six targets, `check` == CI byte-for-byte | All six present ✅, and `check == lint test` **is exactly** what CI runs (`ci.yml:31-32`) ✅ — better than cairn, whose `check` drifts from its CI. **But `typecheck` is excluded from `check`** (documented at `Makefile:13-14`), so no type checking is enforced anywhere; mypy is not in `[dependency-groups] dev` and there is no `[tool.mypy]` block. |
| ✅ | CI on push to main + PRs, `permissions: contents: read`, `concurrency`, `workflow_dispatch` | `ci.yml:3-14` — all four present and correct. |
| ❌ | CI matrix covers the supported Python range | `ci.yml:21` is `["3.11"]`; `pyproject.toml:7` declares `>=3.11`. 3.12 and 3.13 advertised, never tested. cairn runs all three. |
| ✅ | `pull_request`, not `pull_request_target`; no secrets | Verified — `pull_request` at `ci.yml:6`, zero `secrets.` references. Fork PRs run sandboxed. |
| ⚠️ | Branch protection applied via `gh` | Not verifiable from the local clone. Flagged for manual confirmation. |
| ⚠️ | Solo-maintainer knobs | Not verifiable locally. Flagged. |
| ✅ | Required status-check contexts match actual job names | Job key `check` + matrix ⇒ context `check (3.11)`. ⚠️ Widening the matrix silently renames the context and un-protects the branch — pin branch protection accordingly. |
| ⚠️ | GitHub About + topics | Not verifiable locally. Flagged. |
| ⚠️ | Secret scanning + push protection | Not verifiable locally. Flagged — and it is the rubric's §0 companion to the (clean) history sweep. |
| ❌ | `.gitignore` scoped to this project | **Still a 256-line generic dump.** `eksctl.exe` (:3), `aws/custom-resource.yaml` (:5), `tests/k8s/dev-custom-resource.yml` (:7), `**/prod.yml` (:9), Django (:76-79), Flask (:82-84), Scrapy (:86-87), PyBuilder, SageMath, ~70 lines of JetBrains. Rubric names this verbatim; untouched. `data/`, `*.pkl`, `.idea/`, `.ipynb_checkpoints/` **are** correctly ignored ✅. |
| ✅ | `.idea/` and `.ipynb_checkpoints/` untracked | `git ls-files` → zero hits; never tracked in any commit. |
| ❌ | Release tagged at the paper's commit | No tags exist. Also no `release.yml` (cairn has one) and no version single-source-of-truth. |
| ❌ | Git history presentable | 13/26 subjects fail, incl. `"cultural rollbakc"` and `"decsion boundaries"`, all on published `origin/main`; plus 393 vendored GraphRAG files. Squash is the applicable remedy. |
| ✅ | PR titles descriptive/conventional | Feature-branch subjects are imperative and clean bar `709804a` (215 chars) and `eac29ec` (vague). |
| ✅ | One logical change per PR | Documented in `CONTRIBUTING.md:10` and the PR template; branch history is consistent with it. |
| ⚠️ | `community/profile` reports 100% | Not verifiable locally. All contributing files exist, so it should pass once public. |

### §7 Attribution & licensing (11) — 5 ✅ · 3 ⚠️ · 3 ❌

| | Item | Evidence |
|---|---|---|
| ⚠️ | MIT LICENSE with correct holder **and year** | Holder correct; year `2026` should be `2024-2026`. |
| ✅ | NOTICE reproduces the pca-magic Apache attribution | `NOTICE:11-41`. |
| ✅ | `ppca.py` header names upstream + URL + licence + modifications | `app/ppca.py:1-12`. |
| ✅ | README has a dedicated licensing section | `README:131-145`. |
| ⚠️ | MIT/Apache compatibility stated deliberately | Stated ✅ — but the statement **omits §4(a)** and so under-describes the obligation (1.1). |
| ✅ | IVS/WVS/EVS licence cited; not redistributed | `README:36-51, 144-145`; `SECURITY.md:28-36`; `CONTRIBUTING.md:12-13`. Consistently and repeatedly. |
| ❌ | Each model cited with source, quantisation, **and licence** | `README:114-129` gives Ollama tags and "Q4 unless noted", and says "Each model carries its own upstream licence; check it before reuse" — **the licence column the rubric asks for is not there**, and the 2026 cloud cohort (`app/llm_meta.py:52-80`, 18 models) is absent from the README entirely. |
| ❌ | IW/WVS figures, palettes, procedures cited **in place** | Five uncited sites: `culture_map.py:46` "Published WVS rescaling constants" names no source; `culture_map.py:49-62` region colours; `country_meta.py:530` the IW region taxonomy; `cloud_survey.py:110-120` the item stems and the verbatim Y003 battery; the WVS cookbook recoding syntax in `notebooks/6` cells 10/14 and `notebooks/8` cells 4/6. |
| ❌ | `pyproject.toml` metadata publication-accurate | `name`/`description`/`authors`/`readme`/`requires-python` ✅. **Missing `license`, `license-files`, and the `[project.urls]` table.** |
| ✅ | No vendored code lacks a licence header | Swept `app/`, `scripts/`, `notebooks/`: only `ppca.py` is derived, and it is attributed. Varimax is a dependency, not vendored. |
| ⚠️ | Dependency licences compatible with MIT redistribution | `pyreadstat` Apache-2.0 ✅, `ollama` MIT ✅ — **`factor_analyzer` is GPLv2** and is a hard runtime dependency. Source is untainted, but this needs one honest README sentence (1.9). |

---

## 3. PYTHON FINDINGS

`app/cloud_survey.py`, `app/culture_map.py`, `app/ppca.py` are held to the highest bar as the stable
core. Findings in `app/llm_bootstrap.py`, `app/region_svm.py`, `scripts/analyze_2026.py` are tagged
**[IN-FLUX]**.

### CRITICAL

**F-1 · Survey weights `S017` are loaded and silently never applied — `app/culture_map.py:99-100, 206-212`**

```python
subset = self.ivs_df[["S020", "S003", "S017"] + self.iv_qns]        # :99
subset = subset.rename(columns={..., "S017": "weight"})              # :100
```
`grep -rn "weight" app/ scripts/` shows the column is never read again. `calculate_mean_scores`
takes a plain unweighted mean:
```python
means = self.valid_data.groupby("country_code")[[...]].mean()        # :208-212
```
Every published country coordinate is an **unweighted** mean of individual scores. The IVS ships
`S017` precisely because raw wave/country samples are not self-weighting, and the WVS's own
cultural-map procedure applies it. This is the single highest-impact correctness finding in the
audit: it moves published numbers, it is invisible (the column is *right there*, renamed, looking
used), and rubric §4 names it explicitly. Either apply the weight at both aggregation steps, or
state in the README and paper that coordinates are deliberately unweighted and why.

**F-2 · PPCA non-convergence — ✅ FIXED during this audit by `9f1f6d5`, one caveat remains**

As audited at `eac29ec`, `ppca.py` had `while True:` with no cap: the EM loop could spin forever and
nothing raised. Commit `9f1f6d5` added `max_iter=1000` (`ppca.py:35`) and a raise at `:107-110` with
the achieved `diff`, iteration count and `tol` in the message — exactly the rubric §4 requirement,
well done.

**Remaining caveat:** it raises a bare `RuntimeError`, not the semantic `ConvergenceError` the rubric
§3 asks for. A caller cannot distinguish "the fit diverged" from `ppca.py:139`'s
`RuntimeError("Fit the model first.")` without string-matching. Rolled into F-8.

**F-3 · `app/country_meta.py` executes I/O at import time, inside the installable package**

762 lines with no function or `if __name__ == "__main__":` guard. Line 509 reads
`pd.read_pickle("../data/ivs_df.pkl")`; **line 762 writes `country_codes.to_pickle("../data/country_codes.pkl")`**.
Verified: `python -c "import app.country_meta"` → `FileNotFoundError: '../data/ivs_df.pkl'`. It ships
in the wheel (confirmed in the built artefact). Consequences: `import app.country_meta` is a
CWD-dependent side effect that writes to disk; any tool that imports the package for introspection
(sphinx, a type checker, `pytest --collect-only` with a wider `testpaths`, an IDE indexer) either
crashes or silently overwrites a data artefact. Move to `scripts/build_country_codes.py` with a
`main()` and path arguments.

### HIGH

**F-4 · `y002_transform` returns a magic `-5` sentinel instead of raising — `app/culture_map.py:222-232`**

```python
if first < 0 or second < 0:
    return -5
```
On the survey path this is harmless: `ITEM_VALID_RANGES["Y002"] == (1, 3)` recodes `-5` to NaN at
`:110-114`. **On the LLM path it is not.** `llm_bootstrap._to_value` (`:41-46`) calls
`y002_transform` and feeds the result straight into `cm.project()`, which never applies
`ITEM_VALID_RANGES`. A `-5` on a 1-3 scale is a ~5-sigma outlier that propagates silently into a
bootstrap replicate and a published coordinate. This is precisely the rubric §0 "no silent fallback
on any path that produces a published number". Raise instead — the parsers guarantee valid values,
so an out-of-range input is a genuine invariant violation. (Note `tests/test_culture_map.py:69`
currently *asserts* the `-5`, so the test encodes the bug.) The return annotation is `-> float`
while every return is an `int` literal.

**F-5 · `min_obs` column dropping desynchronises `fit` from `transform` — `app/ppca.py:50-56, 153`**

```python
valid_series = np.sum(~np.isnan(raw), axis=0) >= min_obs   # :50
data = raw[:, valid_series].copy()                          # :51
self.means = np.nanmean(data, axis=0)                       # :54  — retained cols only
...
return ((data - self.means) / self.stds) @ self.C           # :153 — expects ALL cols
```
`valid_series` is computed, used, and thrown away. `transform()` has no idea columns were dropped,
so with any `min_obs` above the sparsest column the shapes mismatch. This is not hypothetical: the
in-flight working tree reproduces it exactly —
`ValueError: operands could not be broadcast together with shapes (100,10) (8,)` at `ppca.py:153`,
across 8 tests during the refactor. It is currently masked on the published path only because `culture_map.fit` passes
`min_obs=1` (`:135`). A raise is the lucky outcome; had `d` happened to align, it would have
silently mis-standardised. Store `self.valid_series` and either apply it in `transform` or raise a
clear error naming the dropped columns.

**F-6 · `ruff format --check` is enforced nowhere — `Makefile:6-11, 21`**

`lint` runs `ruff check` only; `format` **mutates** rather than checks; `check = lint test`; CI runs
`make lint` + `make test`. So formatting drift cannot fail any gate — and it has already drifted:
at `9f1f6d5` **7 of 20 files would be reformatted**, up from 1 before the refactor — the gap is
widening, not holding. cairn's CI has this step; this repo dropped it. Add `uv run ruff format --check app scripts
tests` to the `lint` target. Related: `[tool.ruff]` sets only `line-length` and `target-version`, so
ruff runs its **default rule set (E4/E7/E9/F)** — no `I` (import sorting), `UP` (pyupgrade), `B`
(bugbear), `SIM`, `RUF`. cairn selects all of them. "`make lint` passes" currently means much less
than it appears to.

**F-7 · [IN-FLUX] SVM CV accuracy is optimistically biased — `app/region_svm.py:44-50`**

```python
search = GridSearchCV(SVC(), PARAM_GRID, refit=True, cv=cv)
search.fit(xy, codes)
self.cv_accuracy = float(cross_val_score(search.best_estimator_, xy, codes, cv=cv).mean())
```
`cross_val_score` re-scores the *already-selected* estimator on the **same data and the same folds**
that selected it. Hyperparameters were chosen to maximise performance on exactly those folds, so
`cv_accuracy` is a selection-biased optimistic estimate, not an out-of-sample one. This number is
the module's headline caveat (`region_svm.py:5-6` quotes "around 0.55 on 109 countries"), it is
printed by both analysis scripts, and `region_assignments` attaches it to **every output row**
specifically so it can never be quoted without it — which makes its correctness load-bearing. Use
nested CV (`cross_val_score(GridSearchCV(...), xy, codes, cv=outer_cv)`) and report that.

### MEDIUM

**F-8 · No semantic exception types anywhere — repo-wide.** `grep 'class.*Error'` → zero.
`RuntimeError` at `ppca.py:144`, the new non-convergence raise at `:107-110` (F-2), the
`culture_map.py` and `region_svm.py` state guards; bare
`ValueError` for every parse failure in `cloud_survey.py:51-103`. Rubric §3 names
`ResponseParseError`, `ConvergenceError`, `InsufficientDataError`. A caller cannot distinguish "you
forgot to call `fit()`" from "the fit diverged" without string-matching.

**F-9 · `app/ppca.py` has zero type annotations.** All six methods — `__init__`, `fit`,
`transform`, `_calc_var`, `save`, `load` — are fully unannotated (`fit(self, data, d=None, tol=1e-4,
min_obs=10, seed=None, verbose=False)`). This is the stable core, the Apache-derived file, and the
file the paper's correctness rests on. Rubric §1 also asks for ndarray shapes in the docstring where
the annotation cannot express them; `__init__:28-33` has them as comments but `fit`/`transform` do
not. Same gap, less severe, in `CulturalMap.__init__/prepare_data/fit/_rescale/save_model/load_model/
visualize_cultural_map` and `RegionClassifier.__init__`.

**F-10 · `np.linalg.eig` on a symmetric covariance matrix — `app/ppca.py:118-122`.**
`np.cov((data @ C).T)` is symmetric, so `eigh` is the correct routine. `eig` returns complex dtype
whenever round-off produces a tiny imaginary part; `vals`/`vecs` then go complex, `np.argsort` sorts
complex lexicographically, and `C = C @ vecs` silently becomes complex — poisoning every downstream
coordinate with a `ComplexWarning` at best. `llm_bootstrap.py` already uses `eigh` correctly for the same job in `confidence_ellipses`. One-character-class fix, real latent bug.

**F-11 · `±inf` silently overwritten with the finite max — `app/ppca.py:48`.**
```python
raw[np.isinf(raw)] = np.max(raw[np.isfinite(raw)])
```
`-inf` becomes `+max` — the sign is inverted. Inf in survey item values is an invariant violation,
not something to paper over, and this sits on a published-number path with no justification comment
(rubric §3's three-part rule). Raise instead.

**F-12 · No logging anywhere; 7 `print()`s in library code.** `ppca.py:102`,
`llm_bootstrap.py:72,108,200`, `cloud_survey.py:322,335,338`. Two consequences beyond style: (a)
`ppca.py:102` prints the convergence criterion **every EM iteration** when `verbose=True` —
unbounded stdout; (b) `llm_bootstrap.py:200` prints the overall-mean fallback count, so the caller
cannot capture it — the fallback rate is a reportable statistic that never reaches
`llm_ellipses_2026.csv`. Return it in the frame instead.

**F-13 · Environment reads buried in the dataclass — `app/cloud_survey.py:202-203`.**
```python
host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", ...))
api_key: str = field(default_factory=lambda: os.environ["OLLAMA_API_KEY"])
```
Rubric §2 names this case verbatim. The `KeyError` on a missing key surfaces from a `default_factory`
lambda with no message saying what to set or where. Make them required constructor arguments, with
`scripts/collect_cloud_2026.py` reading the environment (it already has `load_dotenv` at `:23-31`).

**F-14 · Untestable collaborator — `app/cloud_survey.py:211-213`.** `AsyncClient` is constructed
inside `__post_init__`, so no test can substitute a fake. This is why §5's parser-coverage checkbox
fails: there is no seam. Rubric §3 names the ollama client as the example. Accept an optional
`client` parameter defaulting to `None`.

**F-15 · `except (ValueError, KeyError)` mislabels bugs as parse failures — `app/cloud_survey.py:302`.**
A `KeyError` from `PARSERS[qn]` (unknown question) or `response["message"]` (API shape change) is
recorded as `error = "parse: …"`, retried 3×, and counted as a model refusal in the parse-rate
statistics. A schema change at the API would show up in the paper as a drop in model parse rates.
Catch `ValueError` only; let `KeyError` propagate to the broad handler at `:310` where it is at least
labelled by type.

**F-16 · Private-attribute reach-through.** `app/cloud_survey.py:237` `parser._valid_values` and
`scripts/collect_cloud_2026.py:35` `survey._client.list()`. Make `_valid_values` public, or better,
give `format_instructions(language)` the branch that `_prompt_for` currently duplicates. Expose a
`list_models()` method on `CloudSurvey`.

**F-17 · `language` is stringly-typed with no validation — `app/cloud_survey.py:206`.**
`language: str = "en"` drives three branches (`_jsonl_path:228`, `_prompt_for:233`, `PRIMER[…]:265`)
with three *different* comparisons (`!= "en"`, `== "zh"`, dict lookup). `language="EN"` writes to a
`__EN` file, silently takes the English prompt branch, then `KeyError`s on `PRIMER`. Use
`Literal["en", "zh"]` and validate in `__post_init__`. `scripts/collect_cloud_2026.py:45` is the only
caller and it hardcodes the two valid values — but the class is public API.

**F-18 · The 2026 record contract is an untyped dict — `app/cloud_survey.py:272-286`.**
Twelve keys built by hand, written to JSONL, and re-parsed by `llm_bootstrap.load_responses_2026` and
`analyze_2026.parse_rates` — three places that must agree on the schema by convention alone. Rubric
§3 requires Pydantic `BaseModel` for parsed/structured data contracts, and `app/qn_classes.py`
already does this correctly for the response classes. This is the corpus behind the paper.

**F-19 · Dead validator and triplicated enums — `app/qn_classes.py:144-148, 59-107`.**
`Y002.check_valid_values` can never fire: pydantic coerces to `Y002Options` before validators run, so
`v not in Y002Options` is unreachable. It also lacks `@classmethod` and any annotations. Separately,
`F063`, `F118`, `F120` are three byte-identical `ONE..TEN` enums (48 lines) — one `TenPointScale` with
three aliases. And `:12,22,34,46,61,78,95,112` are pasted numpy reprs (`[ 1.  2.  3.  4. nan]`) used
as docstrings; the module also has no module docstring, unlike every other module in `app/`.

### LOW

**F-20 · [IN-FLUX] `human_mean=(0.38, -0.01)` re-types the rescale intercepts — `app/llm_bootstrap.py:246`.**
These are exactly the `b` terms of `PC_RESCALE_PARAMS` (`culture_map.py:47`), i.e. the image of the
origin. Derive them: `tuple(b for _, b in PC_RESCALE_PARAMS.values())`. Rubric §2 forbids re-typed
literals across modules; if the rescale constants are ever corrected, this copy silently won't be.

**F-21 · [IN-FLUX] Function-local import and loop-scoped closure — `app/llm_bootstrap.py:296, 308-310`.**
`from app.culture_map import ITEM_VALID_RANGES` inside `central_tendency_diagnostics` when
`app.culture_map` is already imported at `:32`. `def entropy(...)` is redefined on every loop
iteration; hoist to module scope.

**F-22 · Dead/legacy code retained.** `CulturalMap.collect_llm_data` (`:255-287`) is superseded by
`llm_bootstrap` (its own docstring says so) and has a subtle wart — `rows.append(row)` at `:280`
executes *before* the `if not complete: break`, so a `None`-bearing row is always appended and then
dropped by `.dropna()`. `culture_map.py:381-390` is a `__main__` block hardcoding `"../data/*.pkl"`.
`ppca.py:32` `var_exp` is computed but read only in `validate_projection.py:102`. Rubric §1: "don't
leave corpses".

**F-23 · [IN-FLUX] Inconsistent bootstrap defaults.** `bootstrap_llm_positions(n_boot=1000)` vs
`bootstrap_llm_positions_cluster(n_boot=10_000)` (`:122`, `:154`), with `analyze_2026.py:42-43`
re-declaring both. A reader calling the library directly gets neither the paper's B nor a documented
reason for the difference.

**F-24 · Return-type nits.** `region_svm.py:35` uses the string annotation `-> "RegionClassifier"`;
Python 3.11 has `typing.Self`. `culture_map.py:240` `y003_transform(...) -> float` returns `int`.
`cloud_survey.py:208` `__post_init__` and `:261` `_call_once` lack return annotations
(`-> None`, `-> tuple[str, str]`).

**F-25 · Retry without variation — `app/cloud_survey.py:287-314`.** Three attempts re-send a
byte-identical prompt with no temperature or seed change. For a low-temperature model the second and
third attempts are near-certain to reproduce the first failure, so `MAX_ATTEMPTS=3` buys ~nothing on
parse failures (it is genuinely useful for the transport errors at `:310`). Worth either varying the
call or documenting that retries target transport, not parsing — the distinction matters because the
parse-rate statistic is a published result.

---

## 4. PRIORITISED FIX LIST

### Blockers — must land before the repo goes public

1. **`pyproject.toml`: add `license = "MIT"`, `license-files`, `[project.urls]`** — rubric §0 hard
   fail; the published package currently declares no licence. (1.1 fix 2)
2. **Add `LICENSES/Apache-2.0.txt` and correct the README's §4(a) omission** — the one genuine hole
   in the pca-magic chain. (1.1 fix 1)
3. **`git rm -r modelfiles/`** — 5 tracked files leaking `C:\Users\shavh\…`, two of them
   misattributed. (1.5)
4. **Clear outputs on `notebooks/10-results-view.ipynb` and `1-ivs.ipynb`** — removes the Windows
   temp-path leak at cell 7 / line 1645, the 11.8 KB result dump, and the 12.7 KB sklearn blob. (1.7)
5. **`rm -rf notebooks/.ipynb_checkpoints/`** — `1-ivs-checkpoint.ipynb` holds real IVS microdata
   (666,907-respondent `.describe()`, respondent-level rows, a 109 KB embedded PNG). Gitignored, but
   an ignore rule does not survive a zip of the working tree — which is how artefacts reach
   reviewers. This is the only path to an actual WVS/GESIS agreement breach. (1.3)
6. **Fix or disclose F-1 (survey weights)** — every published country coordinate is unweighted.
7. **`.gitignore`: add `.DS_Store`, `docs/wiki-math-excerpts.md`, and global `*.sav`/`*.sps`/`*.npy`/`*.npz`** — three files are one `git add -A` from being committed. (1.3, 1.6)
8. **Decide on the history squash** — resolves 393 vendored GraphRAG files, 13 bad commit subjects,
   and the historical notebook blobs in one move. (1.4)
9. **Enable secret scanning + push protection, and confirm GHSA private reporting is on** — the only
   §0 items not verifiable locally. Rotate the live Ollama key as hygiene.

### High — before the paper's tagged release

10. ~~F-2 `max_iter` in PPCA~~ — **done in `9f1f6d5`**; only the semantic exception type remains (F-8).
11. F-5 store/apply `valid_series` in `PPCA.transform`.
12. F-3 move `app/country_meta.py` to `scripts/` behind a `main()`.
13. F-4 raise instead of returning `-5`; update `test_culture_map.py:69` which currently encodes the bug.
14. F-6 add `ruff format --check` to `make lint`, and select `["E","F","I","UP","B","SIM","RUF"]` in `[tool.ruff.lint]`.
15. F-7 nested CV for `cv_accuracy` (it is attached to every published region row).
16. **Test the parsers** — `EnumOutputParser`/`Y002`/`Y003`/`clean_content` have zero coverage; add
    valid / refusal / out-of-range / malformed / `<think>` / boundary cases (rubric §5). Requires
    F-14 (inject the client) for the harness itself.
17. **Fix `from ppca import PPCA` → `from app.ppca import PPCA`** in notebooks 2, 5, 7, 9 — all four
    are `ModuleNotFoundError` at cell 0 as committed, and notebooks are the first thing a reviewer
    opens. (1.7)
18. **Add the survey-instrument attribution** — a `NOTICE` stanza plus in-place citations for the
    item stems, the verbatim Y003 battery, the cookbook recoding syntax, and the value labels; and
    disclose that `IV_QN_PROMPTS_ZH` are the authors' own translations. (1.8)
19. Widen the CI matrix to `["3.11", "3.12", "3.13"]` to match `requires-python`.
20. Add `CITATION.cff`; cut a `## [0.1.0] - YYYY-MM-DD` section in `CHANGELOG.md` with link refs; tag.

### Medium — quality bar for a public artefact

21. F-8 semantic exceptions (`ResponseParseError`, `ConvergenceError`, `InsufficientDataError`).
22. F-9 type annotations across `app/ppca.py` and `CulturalMap`; then put `typecheck` back in `check`
    (add mypy to `[dependency-groups] dev` + a `[tool.mypy]` block).
23. F-10 `eigh` not `eig`; F-11 raise on `inf`.
24. F-12 module `logging.Logger`s, remove the 7 `print()`s from `app/`, return the fallback count.
25. F-13/F-14/F-16/F-17/F-18 — `CloudSurvey` API: injected client, explicit credentials,
    `Literal["en","zh"]`, Pydantic record model, no private reach-through.
26. F-15 drop `KeyError` from the parse-failure catch.
27. Prune `.gitignore` from 256 lines to the ~30 that apply here (rubric §6, named verbatim).
28. **Remaining citations in place**: `PC_RESCALE_PARAMS` provenance (`culture_map.py:46` — the
    rubric calls it "the highest-risk number in the repo"), the region colour map, the IW region
    taxonomy at `country_meta.py:530`, and an ISO-3166 provenance comment at `country_meta.py:3`.
29. README accuracy pass: add the cluster bootstrap + B + method to the Pipeline section, add the
    `ITEM_VALID_RANGES` recode and `score_stds` steps, add the model licence column, add the 2026
    cloud cohort (18 models, currently absent from the README entirely), and add one sentence on
    `factor_analyzer`'s GPLv2 (1.9).
30. F-19 dead `Y002` validator, triplicated 1-10 enums, numpy-repr docstrings, missing module docstring.
31. Set `matplotlib.rcParams["pdf.fonttype"] = 42` in `scripts/make_figures.py` — current figures are
    Type 3, which IEEE and some ACM/ACL tracks reject at submission. (1.7)
32. Note the `G006` "your nationality" placeholder in the paper's limitations — the WVS item is
    respondent-anchored and this one is not. (1.8)

### Low

33. F-20 through F-25: derive `human_mean`, hoist the local import and `entropy`, delete
    `collect_llm_data` and the `__main__` block, reconcile `n_boot` defaults, `typing.Self`, retry
    semantics.
34. Copyright years `2024-2026` in `LICENSE:3` and `NOTICE:2`; pin the pca-magic upstream SHA in
    `NOTICE:13`.
35. Add a supported-versions table to `SECURITY.md`; add `release.yml` (cairn parity).
36. `data/collection_2026_run.log` is truncated mid-run (last line `[glm-5.1] 400/500`) and records
    `gemma4:31b DONE ok=472 failed=28` — a 5.6% failure rate that nothing downstream surfaces.
    Confirm the 2026 collection actually completed before the numbers are frozen.
