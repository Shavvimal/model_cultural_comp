# Review Rubric — `model_cultural_comp`

A checkable review rubric for this repo as it moves from private research code to the
public code artefact behind a peer-reviewed NLP paper.

**Sources.** Rules are distilled from two places and cited inline:

| Cite | Source document |
|---|---|
| `CORE_PRINCIPLES` | `~/Code/sammy/api/docs/pr/services/CORE_PRINCIPLES.md` |
| `GOTCHAS` | `~/Code/sammy/api/docs/pr/services/GOTCHAS.md` |
| `CONFIG_PATTERN` | `~/Code/sammy/api/docs/pr/services/CONFIGURATION_DEFAULTS_PATTERN.md` |
| `COMPOSITION` | `~/Code/sammy/api/docs/pr/services/COMPOSITION_PATTERN.md` |
| `ADVANCED` | `~/Code/sammy/api/docs/pr/services/ADVANCED_PATTERNS.md` |
| `SERVICES_README` | `~/Code/sammy/api/docs/pr/services/README.md` |
| `LEGIT` | `~/Code/cairn/scratchpad/legit.md` |
| `PUBLIC_CHECKLIST` | `~/Code/cairn/scratchpad/public-ready-checklist.md` |
| `MAKING_A_PR` | `~/Code/cairn/scratchpad/making-a-pr.md` |

**Adaptation note.** The Sammy docs govern a multi-tenant FastAPI/daemon service. Service-only
concerns (FastAPI DI, request scoping, `organisation_id`/RLS, `SharedSettings`, `DANGEROUS_`
clients, task schemas, thread-safety across concurrent requests) do **not** transfer and are
omitted. Everything else — typing, config-over-magic-numbers, composition, explicit failure,
naming, logging, comment hygiene, mocking in tests — transfers directly. Where a rule needed
reshaping for a research pipeline, the item says **[adapted]** and why.

---

## 0. Hard fails — these block the public release

Any unchecked box here is a release blocker. No exceptions, no "fix it in a follow-up".

- [ ] **`LICENSE` (MIT) exists at repo root and is declared in `pyproject.toml`** (`license = "MIT"`, `license-files = ["LICENSE"]`) — without it nobody may legally use the code (PUBLIC_CHECKLIST §0; LEGIT).
- [ ] **`NOTICE` exists and attributes the Apache-2.0 `pca-magic` derivation**, and `app/ppca.py` carries a header naming the upstream project, its URL, its Apache-2.0 licence, and what was changed — Apache-2.0 §4 requires retained notices and a statement of modification (PUBLIC_CHECKLIST §0 "the single most important file"; extended for a derived work).
- [ ] **No secrets, tokens, API keys, or credentials anywhere in the working tree or in git history** — `git log -p | grep`-style sweep plus GitHub secret scanning + push protection enabled before the repo flips public (PUBLIC_CHECKLIST §6).
- [ ] **No survey microdata, licensed WVS/EVS `.sav` files, or any part of the 5.8 GB dataset is committed or reachable in git history** — `git log --all --diff-filter=A --name-only` shows zero `data/` payloads; redistribution of IVS data is not ours to grant (PUBLIC_CHECKLIST §0 `.gitignore` discipline).
- [ ] **No personal absolute paths, machine names, private URLs, or personal email beyond the intended author contact** appear in tracked files or notebooks (PUBLIC_CHECKLIST §5; GOTCHAS Gotcha 4 "no environment/context leakage").
- [ ] **Every committed notebook has cleared outputs** or is deliberately kept with outputs that contain no data-subject rows and no credentials — stray outputs are the commonest accidental data leak in research repos **[adapted: notebooks have no analogue in Sammy]**.
- [ ] **CI is green on a clean checkout with no dataset present** — the full `make check` gate passes using only synthetic fixtures (PUBLIC_CHECKLIST §3; MAKING_A_PR "`make check` EXIT=0").
- [ ] **The numbers in the paper are reproducible from the tagged commit** — a documented command reproduces the reported cultural-map coordinates from the raw IVS inputs, and the tag is the artefact cited in the paper (PUBLIC_CHECKLIST §4 tagged releases; extended for a paper artefact).
- [ ] **No silent fallback sits on any path that produces a published number** — no `except Exception: continue/return []` around PPCA fitting, response parsing, or score computation (CORE_PRINCIPLES §8).

---

## 1. Python craft

- [ ] **PEP 604 / PEP 585 typing throughout**: `str | None`, `list[int]`, `dict[str, float]` — no `Optional[...]`, `List[...]`, `Dict[...]` imports (GOTCHAS Gotcha 1). *Currently violated in `app/qn_classes.py`, `app/llm_data_gen.py`, `app/chinese_llm_data_gen.py`, `app/culture_map.py`.*
- [ ] **Every public function and method has full type hints on parameters and return** — including `ndarray` shapes documented in the docstring where the annotation cannot express them (GOTCHAS Gotcha 2).
- [ ] **Imports are grouped stdlib → third-party → first-party → `TYPE_CHECKING`**, alphabetised within groups (GOTCHAS Gotcha 5; CORE_PRINCIPLES §4).
- [ ] **All intra-package imports are absolute and package-qualified** (`from model_cultural_comp.ppca import PPCA`), never bare `from ppca import PPCA` which only works when CWD happens to be `app/` (CORE_PRINCIPLES §4 import boundaries **[adapted: the failure mode here is "breaks when installed / breaks in CI", not "breaks in daemon"]**). *Currently violated at `app/culture_map.py:4`.*
- [ ] **No duplicated module lives in two places** — `notebooks/ppca.py` and `app/ppca.py` are the same code in two files; one canonical copy, imported everywhere else (CORE_PRINCIPLES §7 do one thing well; GOTCHAS Gotcha 16 "don't leave corpses").
- [ ] **Private helpers use a single leading underscore; public API has none; double underscores are reserved for magic methods** (GOTCHAS Gotcha 13).
- [ ] **All Python files are `snake_case.py`** (GOTCHAS Gotcha 14).
- [ ] **Comments explain *why*, never *what*** — no restating the code, no changelog comments (git is the changelog), no commented-out code blocks (GOTCHAS Gotcha 16).
- [ ] **Long multi-step methods use section-marker comments** (`# ---- Step 3: varimax rotation ----`) rather than being an undivided 100-line block (GOTCHAS Gotcha 16 "section markers").
- [ ] **No `print()` in library code** — use a module `logging.Logger`; `print` is acceptable only in scripts' top-level user-facing output (GOTCHAS Gotcha 17 **[adapted: Sammy's `@logger` decorator is service infrastructure; a plain `logging.getLogger(__name__)` is the research-repo equivalent]**). *Currently violated across `app/*.py` (~15 sites).*
- [ ] **Log messages carry context** (model name, question ID, country, iteration number) — never bare `"Error occurred"` (GOTCHAS Gotcha 17).
- [ ] **Error logs use `exc_info=True`** so the stack trace survives (GOTCHAS Gotcha 17).
- [ ] **Log levels are used correctly**: `info` for pipeline milestones, `debug` for per-item detail, `warning` for recoverable degradation, `error` for failures (ADVANCED "Logging Best Practices").
- [ ] **No log spam** — no `"starting step N" / "finished step N"` pairs; one line per meaningful milestone, with metrics (`"PPCA converged in 41 iters, ΔC=8.7e-5"`) (GOTCHAS Gotcha 17).
- [ ] **`ruff check` and `ruff format --check` pass with zero findings** at the configured `line-length = 100`, `target-version = "py311"` (PUBLIC_CHECKLIST §0 Makefile gate).

## 2. API & config design

- [ ] **Every tunable numeric or model-identifier value lives in a Pydantic configuration class with `Field(default=..., description=...)`, not as a bare literal in logic** (CONFIG_PATTERN Layer 1; GOTCHAS Gotcha 10). Concretely, this repo's magic numbers to promote: PPCA `tol=1e-4`, `min_obs=10`, latent dimension `d`, the `pc_rescale_params = {'PC1': (1.81, 0.38), 'PC2': (1.61, -0.01)}` rescaling constants, retry counts / `max_retries`, worker counts, and the ollama model list.
- [ ] **Every config field's `description` says *why* the value exists, not just what it is** — "prevents runaway EM iterations" beats "tolerance" (CONFIG_PATTERN "Rules for fields").
- [ ] **Every config field carries range constraints where the valid range is known** (`ge=0.0, le=1.0` for similarity/variance ratios, `ge=1` for counts) so a bad value fails at construction, not 40 minutes into a fit (CONFIG_PATTERN "Rules for fields").
- [ ] **Mutable defaults use `default_factory`**, never a shared list/dict literal (CONFIG_PATTERN "Rules for fields").
- [ ] **Config is embedded in the caller-facing entry point via `default_factory=XxxConfiguration`** so a reader who wants the paper's defaults passes nothing, and a reader who wants to tune one knob overrides exactly that knob (CONFIG_PATTERN Layer 2).
- [ ] **The published defaults are exactly the values used for the paper's reported results** — if any default diverges from what was run, it is documented in the README **[adapted: for a paper artefact, defaults are a scientific claim, not just ergonomics]**.
- [ ] **The ten IVS question codes, cultural-region colour map, and country-code mappings are named module-level constants (`Final[...]`), not re-typed literals** across `culture_map.py` and `culture_map_post_hoc.py` (CONFIG_PATTERN "Anti-Pattern: What This Replaces").
- [ ] **Purely cosmetic constants stay hardcoded** — plot DPI tweaks, log truncation widths, and axis label padding do *not* need a config class (CONFIG_PATTERN "What Belongs in a Configuration Class"). Do not over-promote.
- [ ] **No configuration class is created speculatively** for a module with no tunable values (CONFIG_PATTERN "don't create one speculatively").
- [ ] **No `os.getenv()` / environment reads buried inside classes** — the ollama host, model list, and output paths are constructor or CLI arguments (GOTCHAS Gotcha 4; CORE_PRINCIPLES §2 **[adapted: injection here means "explicit function argument", not a DI container]**).
- [ ] **File paths are parameters, never hardcoded absolutes** — `CulturalMap(ivs_df_path=..., country_codes_path=...)` is already right; keep it, and never reintroduce a literal `/Users/...` (GOTCHAS Gotcha 4).

## 3. Composition, structure & error handling

- [ ] **Classes receive their collaborators as constructor arguments rather than constructing them internally where a test would want to substitute one** — e.g. the ollama client in the data-collection harness is injected so tests never hit a live model (CORE_PRINCIPLES §2 **[adapted: full DI is overkill here; the testable seam is what matters]**).
- [ ] **Each module does one thing** — data collection, PPCA, map construction, and plotting are separate modules with no cross-contamination (CORE_PRINCIPLES §7).
- [ ] **No class wraps a single dependency without adding orchestration** — if a class only forwards to `ollama`, it should be a function or a method on the client wrapper (CORE_PRINCIPLES §7 "Service vs Client Method").
- [ ] **No over-engineering: no manager/protocol/strategy layers introduced for modules under ~300 lines** — this repo's modules are 50–315 lines, which is minimal-template territory (GOTCHAS Gotcha 15; COMPOSITION "Minimal Service Template").
- [ ] **A class does not accumulate mutable per-run state that later methods silently depend on** — or, where a pipeline class legitimately holds fitted state (`self.ppca_df`, `self.country_scores_pca`), each stage validates its prerequisite is populated and raises a clear error if `prepare_data()` was never called (CORE_PRINCIPLES §5 **[adapted: research pipeline objects are inherently stateful; the transferable rule is "no implicit ordering dependency", enforced by explicit precondition checks]**).
- [ ] **Pydantic `BaseModel` is used for all parsed/structured data contracts, not `@dataclass` or raw dicts** — the LLM survey-response parsers already do this; keep it, with `Field(description=...)` on each field (CORE_PRINCIPLES §6).
- [ ] **Invariant violations raise, loudly, with context** — missing country codes, a PPCA fit that fails to converge, a response set with zero parseable answers, or a mismatch between expected and actual question counts must raise, never return `None`/`[]`/`{}` (CORE_PRINCIPLES §8 "Invariant Violations (Must Error)").
- [ ] **Semantic exception types exist** (e.g. `ResponseParseError`, `ConvergenceError`, `InsufficientDataError`) instead of bare `RuntimeError`/`ValueError` everywhere (CORE_PRINCIPLES §8 "Custom Exception Types"; ADVANCED "Error Handling Patterns").
- [ ] **No bare `except Exception:`** that swallows and continues. *Currently violated at `app/llm_data_gen.py:192,212,226` and `app/chinese_llm_data_gen.py:224,244,258`.* (CORE_PRINCIPLES §8 "AI-Assisted Programming: Mandatory Review Rule").
- [ ] **Every surviving fallback carries the three-part written justification**: (1) why this can fail, (2) why continuing is still correct, (3) what is intentionally degraded (CORE_PRINCIPLES §8 "The Fallback Justification Rule").
- [ ] **Retry-on-parse-failure catches a specific parse exception, caps retries via config, and raises after exhaustion** — an LLM that never returns a parseable answer must not silently vanish from the sample (CORE_PRINCIPLES §8; ADVANCED "Graceful Degradation" — collection is exactly the "optional, log-and-continue" case *only if* the dropped item is recorded).
- [ ] **Dropped/refused/unparseable model responses are counted and reported, not discarded** — a refusal rate is a result, and silently dropping rows biases the map **[adapted: this is CORE_PRINCIPLES §8 "corrupt audit trails" translated into "corrupt sample"]**.
- [ ] **Type-based branching with 5+ arms uses a module-level dispatch dict, not an if/elif chain** — relevant to per-question response parsing and per-region handling (ADVANCED Pattern 5).

## 4. Numerics & reproducibility

*This section has no direct Sammy analogue; it is the research-code translation of CORE_PRINCIPLES §8
(correctness over convenience) and CONFIG_PATTERN (no unexplained constants). **[adapted throughout]***

- [ ] **Every source of randomness is seeded and the seed is a documented config field** — `np.random.randn` in the PPCA initialiser (`app/ppca.py:73`), any train/test split, any bootstrap resampling. A reader running the code twice must get identical numbers.
- [ ] **Randomness uses an explicit `np.random.Generator` / `default_rng(seed)` passed in, not global `np.random.*` state**, so one component cannot perturb another's stream.
- [ ] **PPCA convergence is checked and non-convergence raises** rather than returning whatever the last iterate happened to be (CORE_PRINCIPLES §8).
- [ ] **Sign and rotation indeterminacy of the PCA/varimax solution is explicitly resolved** by a documented, deterministic convention — otherwise the map can flip between runs and the published figure is unreproducible.
- [ ] **The `pc_rescale_params` affine constants are documented with their provenance** (which published source or fitting procedure produced 1.81/0.38 and 1.61/−0.01) — an undocumented magic constant on the output axis is the highest-risk number in the repo (CONFIG_PATTERN "the description says *why*").
- [ ] **Survey weights (`S017`) are applied consistently and the weighting scheme is documented** at every aggregation step.
- [ ] **Missing-data handling is explicit and stated**: which questions/rows are dropped, what `min_obs` means, and how PPCA imputation interacts with the weighting.
- [ ] **Bootstrap confidence intervals state B (number of resamples), the resampling unit (respondent? country? response?), and the interval method** (percentile vs BCa) in both code and README.
- [ ] **Floating-point comparisons in tests use `np.testing.assert_allclose` with an explicit, justified tolerance**, never `==`.
- [ ] **Dependency versions are pinned by a committed lockfile, and only one lockfile is tracked** — the repo currently tracks `poetry.lock` while `uv.lock` is untracked; pick the tool the README documents and delete the other (PUBLIC_CHECKLIST §0 reproducible environment).
- [ ] **The validation script reproduces the published country coordinates against a reference and asserts agreement within a stated tolerance**, and is runnable by a reviewer who has obtained the IVS data themselves.
- [ ] **Data provenance is documented but data is not shipped**: exact source files, versions (WVS 4.0.0, EVS 3.0.0), download URLs, and the merge-syntax step, so a third party can rebuild the 5.8 GB input (README already does much of this — verify it is still accurate).

## 5. Testing

- [ ] **`pytest` runs to green with zero network access, zero ollama, and zero dataset present** (GOTCHAS Gotcha 18 "tests run in isolation - no external dependencies").
- [ ] **All external dependencies are mocked** — the ollama client is a fake returning canned strings; no test constructs a live client (GOTCHAS Gotcha 18).
- [ ] **Synthetic fixtures are generated from a fixed seed inside the test suite**, small enough to run in seconds, and committed as code (a generator function) rather than as binary blobs.
- [ ] **PPCA is tested against a case with a known closed-form answer** — e.g. data generated from a known low-rank factor model recovers the subspace within tolerance — not merely "it runs".
- [ ] **PPCA is tested with missing entries present**, since imputation is the whole point of using PPCA over PCA.
- [ ] **Every response-parser class has tests for: a valid response, a refusal, an out-of-range value, a malformed/unparseable string, and a boundary value** — parser edge cases are where survey data silently corrupts.
- [ ] **Error paths are tested**: the invariant-violation raises from §3 each have a test asserting the exception type and message (CORE_PRINCIPLES §8 — an untested raise is an unproven raise).
- [ ] **Tests assert on values, not just absence of exceptions** — `assert result.status == "completed"`-style concrete assertions (GOTCHAS Gotcha 18 example).
- [ ] **Any test needing the real 5.8 GB dataset is marked (`@pytest.mark.integration` / `slow`) and deselected by default in CI** (COMPOSITION "Testing Strategy" unit-vs-integration split).
- [ ] **`testpaths = ["tests"]` in `pyproject.toml` matches a `tests/` directory that actually exists** — currently configured but absent.
- [ ] **Notebooks are not the test suite** — any logic a notebook demonstrates lives in an importable module that tests exercise **[adapted: no Sammy analogue; the research-repo failure mode]**.

## 6. Open-source readiness

- [ ] **`README.md` covers, in order: what the paper claims, install, how to obtain the data, how to reproduce the figures, repo layout, citation, licence** (PUBLIC_CHECKLIST §0 README; LEGIT "a clear README").
- [ ] **README badge row directly under the title**: CI status, License MIT, Python 3.11+ (PUBLIC_CHECKLIST §2).
- [ ] **A `CITATION.cff` and a BibTeX block in the README** point at the peer-reviewed paper, with the archival DOI if one exists **[adapted: PUBLIC_CHECKLIST §4's "tagged release" for a paper artefact means a citable, archived version]**.
- [ ] **`CONTRIBUTING.md`** — environment setup, the `make check` gate, branch-and-PR flow, squash/linear-history expectation (PUBLIC_CHECKLIST §1; LEGIT "high value").
- [ ] **`CODE_OF_CONDUCT.md`** — Contributor Covenant v2.1 verbatim, enforcement routed through GitHub private reporting rather than a personal email exposed on a public repo (PUBLIC_CHECKLIST §1).
- [ ] **`SECURITY.md`** — private vulnerability reporting via GitHub Security Advisories, not public issues; GitHub private vulnerability reporting actually enabled on the repo so the channel exists (PUBLIC_CHECKLIST §1, §5).
- [ ] **`.github/ISSUE_TEMPLATE/bug_report.yml` + `feature_request.yml` + `config.yml`** (`blank_issues_enabled: false`, contact links) (PUBLIC_CHECKLIST §1).
- [ ] **`.github/PULL_REQUEST_TEMPLATE.md`** with a checklist: `make check` passes · tests added/updated · docs updated · linked issue (PUBLIC_CHECKLIST §1; MAKING_A_PR "The PR body = the template, filled").
- [ ] **`.github/CODEOWNERS`** — required for `require_code_owner_review` to be anything other than a no-op (LEGIT; PUBLIC_CHECKLIST §0).
- [ ] **`.github/dependabot.yml`** — weekly `github-actions` + the Python ecosystem in use (PUBLIC_CHECKLIST §1).
- [ ] **`CHANGELOG.md`** in Keep-a-Changelog format, seeded with the release that accompanies the paper (PUBLIC_CHECKLIST §1).
- [ ] **`Makefile`** with `install / lint / format / typecheck / test / check`, where `make check` is byte-for-byte the CI gate — so a contributor can reproduce CI locally (PUBLIC_CHECKLIST §0; MAKING_A_PR "`make check` mirrors CI exactly").
- [ ] **CI workflow runs on push to `main` and on PRs to `main`**, with `permissions: { contents: read }` at top level, a `concurrency` block cancelling superseded runs, and `workflow_dispatch: {}` (PUBLIC_CHECKLIST §3; LEGIT §2 "CI scope fix").
- [ ] **CI matrix covers the supported Python range** declared in `pyproject.toml` (`>=3.11`) (PUBLIC_CHECKLIST §0).
- [ ] **CI uses `pull_request`, not `pull_request_target`, and needs no secrets** — so fork PRs run sandboxed with a read-only token (PUBLIC_CHECKLIST §6).
- [ ] **Branch protection on `main` is applied via `gh`, not clickops**: PR required, required status checks strict/up-to-date, linear history, force-push and deletion blocked, conversation resolution required, squash-only merge, `delete_branch_on_merge` (PUBLIC_CHECKLIST §5; LEGIT §1).
- [ ] **Solo-maintainer knobs are set so you cannot lock yourself out**: `required_approving_review_count: 0`, `enforce_admins: false` — raise to 1 + code-owner review once collaborators exist (PUBLIC_CHECKLIST §5; LEGIT §1 "you can't approve your own PR").
- [ ] **Required status-check contexts match the *actual* job context names** produced by the matrix (e.g. `check (3.11)`), or a single stable aggregator job is added and required instead — a required context that never reports can never pass (LEGIT §1 "there is no `ci-all` context").
- [ ] **GitHub About section set via `gh repo edit`**: description mirroring `pyproject.toml`, plus topics (`nlp`, `llm`, `cultural-alignment`, `world-values-survey`, `pca`, `reproducible-research`, …) (PUBLIC_CHECKLIST §5).
- [ ] **Secret scanning and push protection enabled** on the repo (PUBLIC_CHECKLIST §6).
- [ ] **`.gitignore` is scoped to this project** — the current file is a 200-line generic dump containing AWS/Django/Scrapy/k8s stanzas irrelevant here; prune it, and make sure `data/`, `*.pkl`, `.idea/`, and `.ipynb_checkpoints/` stay ignored (PUBLIC_CHECKLIST §0).
- [ ] **`.idea/` and `.ipynb_checkpoints/` are untracked** — verify with `git ls-files` after pruning, since ignore rules do not retroactively untrack (PUBLIC_CHECKLIST §0).
- [ ] **A release is tagged (`v1.0.0`) at the exact commit the paper cites**, with generated release notes (PUBLIC_CHECKLIST §4).
- [ ] **Git history is presentable**: commit subjects are short and imperative, no "cultural rollbakc"-style typo'd or contentless subjects on the release branch, and history contains no large binaries or secrets (MAKING_A_PR "Commit message rules"; PUBLIC_CHECKLIST §6). If history cannot be cleaned in place, squash to a single well-described initial public commit and say so in the README.
- [ ] **PR titles are descriptive and conventional** — squash-merge uses the PR title and auto-generated release notes are built from merged PR titles (MAKING_A_PR "The PR title matters").
- [ ] **One logical change per PR**, referencing an issue where one exists (MAKING_A_PR "Commit message rules").
- [ ] **`gh api repos/<owner>/<repo>/community/profile` reports `health_percentage: 100`** as the objective sign-off on §6 (PUBLIC_CHECKLIST §1 "Result verified").

## 7. Attribution & licensing

- [ ] **MIT `LICENSE` at root, with correct copyright holder and year** (PUBLIC_CHECKLIST §0).
- [ ] **`NOTICE` file reproduces the upstream `pca-magic` Apache-2.0 attribution** — Apache-2.0 §4(d) requires carrying forward any NOTICE content from the derived work.
- [ ] **`app/ppca.py` header states: derived from `<upstream project + URL>`, licensed Apache-2.0, and enumerates the modifications made** — Apache-2.0 §4(b) requires "prominent notices stating that You changed the files".
- [ ] **README has a dedicated "Licensing and attribution" section** explaining that the repo is MIT except for the Apache-2.0-derived PPCA implementation, and what that means for a downstream user.
- [ ] **MIT/Apache-2.0 compatibility is stated deliberately, not assumed** — Apache-2.0 code may be redistributed inside an MIT-licensed project provided the Apache notices and change statement are retained; confirm the combined-licence claim in the README says exactly this.
- [ ] **The IVS/WVS/EVS data licence and terms of use are cited, and the README states that the data is *not* redistributed here** and must be obtained from GESIS / the WVS Association directly.
- [ ] **Each model evaluated is cited with its source (HuggingFace repo / Ollama tag), quantisation, and its own licence** — several of the listed models (Llama-2/3 derivatives, Yi, GLM, Qwen, DeepSeek) carry bespoke licences with usage restrictions; the README's model table already lists sources, add the licence column.
- [ ] **Any figure, colour palette, or methodological procedure taken from the Inglehart–Welzel / WVS publications is cited in place**, not just in the bibliography.
- [ ] **`pyproject.toml` metadata is publication-accurate**: `name`, `description`, `authors`, `license`, `readme`, `requires-python`, and a `urls` table pointing at the repo and the paper.
- [ ] **No vendored third-party code lacks a licence header** — sweep every file for copy-pasted snippets from Stack Overflow / blogs / other repos and attribute or rewrite them.
- [ ] **Third-party dependency licences are compatible with MIT redistribution** — spot-check the non-obvious ones (`factor_analyzer`, `pyreadstat`, `ollama`) (LEGIT "what makes it legit: a license").

---

## How to use this rubric

1. Work §0 first. Nothing else matters until the hard fails are clear.
2. Then §7 (attribution), because it constrains what can be published at all.
3. Then §1–§5 as the code review proper, one section per PR — one logical change per PR
   (MAKING_A_PR), each landing green through `make check`.
4. §6 last, as the release gate: it is largely mechanical (`gh` commands + template files),
   and the community-profile check gives a single objective pass/fail.
5. Re-run the §0 sweep immediately before flipping the repo public. Secrets and data leaks are
   the only failures that cannot be undone by a follow-up commit.
