# Pre-specified analysis plan — 2026 cloud collection

**Status at time of writing (2026-08-04):** the English arm is mid-collection
(~9 of 18 models complete); the Chinese arm has produced no records; no 2026
response has been projected onto the map. Every analysis below is specified
before the data it consumes exists in analysable form. Deviations get logged
at the bottom of this file, with reasons.

Estimator and threshold choices follow `docs/statistical-review.md`; the
frozen instrument is `data/cultural_map_model.npz` (post Y003-recode,
unit-variance rescale, weighted country means).

---

## 0. Data integrity & QC — before any statistic

Gate: no downstream analysis runs until every check below is reported.

- [ ] **Design completeness.** Per model × language cell: records = 500
      (10 items × 10 variants × 5 repeats), each (item, variant, repeat) key
      present exactly once after dedup. Report any cell short of design
      (deferred transport failures never retried, throttle-window losses).
- [ ] **Dedup audit.** Resumed runs: count duplicate keys removed;
      confirm keep-last is order-safe (append order = chronology).
- [ ] **Parse-rate table.** Per model × language × item: parsed / attempted.
      Flag any cell < 95% (the plan's original target) and any item below
      `MIN_PER_QUESTION = 10`.
- [ ] **Failure taxonomy.** Classify every error record by inspection of
      `raw_content`: (a) refusal ("as an AI…", "I can't…"), (b) format
      non-compliance (prose answer, wrong delimiter), (c) out-of-range
      value, (d) transport (should be zero in final data — deferred rows are
      unwritten). Report counts per model × language × item.
- [ ] **Retry-intensity audit.** Distribution of `attempts` per cell —
      the rejection-sampling selection pressure (Limitations claims depend
      on it). Compare en vs zh.
- [ ] **Latency sanity.** `duration_ms` distributions per model (now
      measures latency, not queue-wait, post-fix — only trust records
      written after the fix for latency analysis; flag the boundary).
- [ ] **Determinism census.** Per model × item: response entropy; count
      items answered identically in all repeats (2026 frontier models are
      near-deterministic on some items — feeds §3.4's "sampling variance
      stops being informative" argument).
- [ ] **Thinking-trace inventory.** Which models emit thinking; trace
      length distributions; spot-check 20 traces per model for content
      leakage into `raw_content` (think-block stripping worked?).
- [ ] **Index validity.** Y002 pairs and Y003 choice-lists: all parsed
      values within valid ranges (parser guarantees it — verify anyway;
      the Y003 sentinel taught us why).

## 1. Primary confirmatory analyses

Pre-specified estimators; no alternatives tried first.

- [ ] **Positions.** Cluster bootstrap (B = 10,000, seed 42, resampling the
      ten prompt variants) per model × language cell → mean position, 95%
      confidence ellipse for the mean, per-model empirical Mahalanobis-q95
      vs χ²₀.₉₅,₂ (normality check). Item bootstrap (B = 1,000) alongside as
      the lower bound; report SD ratio and ellipse-area ratio per cell.
- [ ] **Classifier-free headline statistics** per cell, per replicate:
      distance to pooled human mean (0.38, −0.01); share of 109 countries
      closer to that mean; minimum distance to any non-Western region
      centroid. Report whether the 2024 headline ("farther than ≥95% of
      countries; never within 1.6 of a non-Western centroid, in every
      replicate") replicates on 2026 cells — state the thresholds *before*
      looking: replicate-simultaneous claims use whatever bound holds for
      every replicate; no cherry-picked rounding.
- [ ] **Region assignment**, both rules (SVM with CV accuracy attached;
      nearest centroid), positional stability, full SVM share vector,
      rule-disagreement flags.
- [ ] **Language effect δₘ = position(zh) − position(en)** per model:
      replicate-paired CI on δ components and on ‖δ‖; per-item language
      effects (predicted largest from 2024: G006, F063, E018 — state as a
      directional hypothesis: zh administration → traditional pole).
- [ ] **Direction test across models:** sign test on δ_PC2 across the 18
      models + mean displacement vector with across-model bootstrap CI.
- [ ] **Origin × language interaction:** Chinese-origin vs Western mean δ
      (two-sample permutation test on the δ vectors, 10⁴ permutations,
      report both components and ‖δ‖). This is the honest Confucian test:
      *do Chinese-origin models move toward the Confucian cluster under
      native-language administration more than Western models do?*
- [ ] **Confucian hypothesis, both arms separately:** per-cohort mean
      distance to the Confucian centroid under en and under zh; report
      with CIs, no pooled cross-arm test.
- [ ] **The one longitudinal statistic:** coherence rate of attempted
      Chinese-origin models, 2024 (4/9) vs 2026 (n/11), with exact
      binomial CIs. No other cross-year inference.

## 2. Secondary & diagnostic analyses

- [ ] **Variance-components (G-theory) decomposition** — a methodological
      first for values instruments: crossed components model × language ×
      prompt-variant × repeat, reported in map units per axis. The harness
      records every needed facet. (Added from the personas/ephemerality
      literature review: no G-theory decomposition exists for any LLM
      values instrument.)
- [ ] **Item keying balance / response-directionality diagnostic:** report
      the ten items' keying orientation and each cell's directional
      response bias (acquiescence-style statistics), since directional
      bias can masquerade as value placement.

- [ ] **Prompt-variant ICC** per item × model × language (the cluster
      bootstrap's justification — report the max-ICC table; 2026-en early
      values reached 0.56).
- [ ] **Central-tendency diagnostics** per cell: distance of the item-mean
      vector from the all-midpoint vector; mean per-item entropy; share of
      items where the modal response = scale midpoint. Correlate
      midpoint-distance with PC2′ across cells (the "modal responding reads
      as secularity" mechanism — if cells with low midpoint-distance have
      high PC2′, say so).
- [ ] **Refusal analysis** (its own subsection in the paper):
      - per item × model × language refusal rates; which pole the refused
        items load on (F118/F120 are the top self-expression loaders);
      - **does administration language change refusal patterns?** (zh
        political-sensitivity hypothesis: petition/E025, free-speech
        Y002 options, national pride/G006 — directional check on
        Chinese-origin models under zh);
      - refusal phrasing taxonomy by model family (verbatim examples);
      - Manski-style worst-case bound: place every refused item at its
        most traditional/survival admissible value and report which cells'
        quadrant membership survives — the honest bound behind the
        "conditional on answering" claim.
- [ ] **Reasoning-trace analysis** (thinking models): sample ≥30 traces per
      model, code for (a) explicit modal-response targeting ("a common
      moderate response"), (b) explicit persona reasoning, (c) explicit
      guideline/safety citations, (d) native-language reasoning under zh
      prompts vs English reasoning about a Chinese prompt — (d) bears
      directly on the mechanism of the language effect. Report coder +
      counts; qualitative, clearly labelled as such.
- [ ] **Within-family contrasts** (protocol-matched, same origin — the
      cleanest scale/version comparisons in the data):
      - deepseek-v4-flash vs flash:0731 vs pro (version/tier);
      - gpt-oss:20b vs 120b, nemotron nano/super/ultra (scale ladders);
      - kimi-k2.6 vs k2.7-code vs k3 (version + code-specialisation — is
        the code model an outlier on a values instrument?);
      - glm-5.1 vs 5.2, minimax-m2.7 vs m3.
      Report positions + δₘ per family member; descriptive, no scale
      regression (n too small).
- [ ] **Item-level profiles:** per-cell item means vs the human item means;
      leave-one-item-out positions (which items carry each model's
      placement); item-level en−zh deltas for all ten items.
- [ ] **2024/2026 on one map** (descriptive figure only): frozen instrument,
      distinct markers per cohort AND per bootstrap estimator, the full
      confound list in the caption, no pooled statistics, no per-model
      cross-year displacement.

## 3. Sensitivity analyses — decision rules fixed now

- [ ] `MIN_PER_QUESTION` at 10 (primary) and {5, 25} (sensitivity); report
      which models' inclusion flips.
- [ ] Ellipses under χ² vs empirical Mahalanobis quantile (report both;
      χ² is primary).
- [ ] Headline statistics excluding the two most-refused items (F118,
      F120) entirely — does the quadrant placement survive an instrument
      without the items models most often refuse?
- [ ] SVM vs centroid rule disagreement rate on 2026 cells.
- [ ] Multiple comparisons: replicate-simultaneous claims need no
      correction (by construction); any pairwise model comparison we
      choose to report uses Benjamini–Hochberg over the comparisons
      actually made, stated in the caption.

## 4. Figures & tables to produce

- Fig 3 — 2026 map, both arms: en cells as diamonds, zh as triangles,
  cluster-bootstrap ellipses, en→zh displacement arrows per model
  (colour by origin cohort).
- Fig 4 — language-effect forest plot: δ_PC1 and δ_PC2 with CIs per model,
  grouped by cohort (the origin × language interaction, visually).
- Table: parse/refusal rates per model × language (the coherence-rate
  table; feeds Appendix A).
- Table: 2026 positions + headline statistics (mirrors the 2024 table).
- Table: δₘ with CIs + sign-test summary.
- Appendix tables: ICC; entropy/midpoint diagnostics; within-family
  contrasts; failure taxonomy with verbatim refusal examples.

## 5. Paper wiring

| Analysis | Paper section |
|---|---|
| Parse/coherence rates | §4.x lead + Appendix A |
| Positions + headline stats | §4.x main result |
| δₘ + interaction | §4.x language subsection (the new contribution) |
| Refusal analysis + Manski bound | §3.5 + Limitations |
| Central tendency + traces | §5 Discussion |
| ICC + bootstrap ratios | §3.4 (replaces the "early signal" placeholders) |
| Within-family contrasts | Appendix (one Discussion sentence if striking) |
| Sensitivity battery | Appendix C |

## Deviations log

*(append-only; date + what changed + why)*

- 2026-08-04: Added G-theory variance decomposition and keying-balance
  diagnostic (from docs review of stability/psychometrics literature);
  noted interlocutor-crossed control as a future robustness arm. Added
  before any 2026 projection was computed.
- 2026-08-04: kimi-k3 excluded from both arms — it requires metered
  "extra usage" billing outside the Ollama Pro subscription (every call
  returned a billing error; zero records collected). Cohort is 17 models
  (10 Chinese-origin) unless Shav funds extra usage, in which case one
  resumable command per arm adds it back. Report in Appendix A.
- 2026-08-04 (implementation, pre-completion): QC gate (scripts/qc_2026.py)
  dedups on a TYPE-NORMALISED key — the JSONL serialises system_prompt_id
  and repeat as both str and int across original vs resumed runs, which
  would defeat a naive drop_duplicates; `load_responses_2026` hardened
  identically, with a regression test. No estimator change; the audit
  reports how many duplicates a naive key would have missed.
- 2026-08-04 (implementation): §3's "headline statistics excluding F118 and
  F120 entirely" is implemented as "F118/F120 neutralised at the human item
  means" (diag_2026_sensitivity_neutralised.csv). Dropping items is
  undefined under the frozen 10-item projection without refitting, which
  the frozen-instrument rule forbids; neutralisation isolates the two
  items' contribution while preserving the instrument. Same intent,
  different mechanics, labelled as such wherever reported.
- 2026-08-04 (implementation): the Manski bound is computed PER AXIS
  (separate worst-case vectors minimising PC1' and PC2'), with item keying
  derived empirically from the frozen projection — a single joint
  worst-case response vector does not exist when items key in opposite
  directions across the two axes.
- 2026-08-04 (implementation): the G-theory decomposition is a balanced
  NESTED sums-of-squares approximation (model / language-within-model /
  variant-within-cell / repeat-within-variant), not a fully crossed EMS
  solution — with a 2-level language facet and 10 variants, crossed EMS
  variance components are not stably estimable; the artefact and any paper
  text label the approximation.
- 2026-08-04 (observation, no plan change): QC on the completed en arm
  identified nemotron-3-ultra as a second major refuser (F118 0.68 / F120
  0.58 parse, refusals also on F063 and G006) with premise-rejecting
  rather than "As an AI" phrasing — flows into the pre-specified refusal
  taxonomy, which now distinguishes boilerplate-refusal from
  premise-rejection phrasing.
- 2026-08-04 (exploration, post-confirmatory): fan-out lanes ran the
  pre-specified m=7 central-tendency correlation family (null on PC2';
  reported), one post-hoc determinism-extremity correlation (labelled post
  hoc; significant under a sensitivity m=8 BH family), one planned E025
  contrast (m=1), a m=6 mechanism-probe family (all BH-null), and single
  paired sign tests for refusal direction and zh determinism. All labelled
  exploratory in exploration-2026.md; only the defensible subset promoted.
- 2026-08-04 (measurement note): the refusal marker classifier undercounts
  prose declinations in nemotron-3-ultra [zh] (all 34 residual
  format-classified failures are first-person declinations, verified by
  reading each); refusal-classified counts are reported as floors wherever
  quoted.
- 2026-08-05 (verification observation): re-running scripts/seed_sensitivity.py
  reproduces every aggregate the paper quotes (rotation spread 5.21 deg; country
  coordinate mean across-seed SD 0.022, max SD 0.077, max range 0.262) exactly at
  quoted precision, but per-country per-seed extreme columns differ from the
  stored CSV by up to 0.37 — BLAS/thread-order nondeterminism in the EM fit,
  amplified in the min/max columns. No paper number is affected (only aggregates
  are quoted); original CSV retained; worth one sentence in the repo README's
  reproducibility section.
- 2026-08-05 (post-submission-analysis Discussion addition, author-requested):
  investigated the "values as post-training design choice / distillation
  lineage" conjecture. One new inferential test on existing artefacts
  (exact permutation on item-profile correlations, within- vs cross-cohort,
  m=1, p=0.0063); everything else descriptive/exploratory (fleet pairwise
  dispersion, cohort tightness, gpt-oss anchor geometry, nearest-neighbour
  census, deviation sign-agreement). Results changed nothing in §4; a
  hedged provenance passage was added to §5 with 14 newly verified
  citations; allegations cited as allegations only. Full record:
  design-choice-investigation.md.
- 2026-08-05 (estimator artefact): the plug-in ‖δm‖ permutation row is now a
  committed artefact (data/conf_2026_origin_permutation_plugin.csv, displacement
  p = 0.8866 → the paper's 0.89; component rows bit-identical to the committed
  folded-norm artefact, whose displacement row reads 0.9347). The §1 interaction
  test's displacement estimand changed from the folded paired-norm mean to the
  plug-in norm mid-analysis; both estimators and both p-values are reported.
- 2026-08-05 (relabel): the E025 five-item contrast previously logged as
  "planned" was formulated post-confirmatory and selected on E025's outlier
  status in the m = 10 family; the paper now labels its p-value descriptive,
  not confirmatory.
- 2026-08-05 (coverage calibration): a simulation calibrated to the K = 10
  cluster design puts the χ²(0.95, 2) ellipse's true coverage near 84%
  (nominal 95%); the Hotelling-style radius 2(K−1)/(K−2)·F(2, K−2) = 10.03 is
  the conservative alternative. Disclosed in §3.4 and Limitations; the
  Mahalanobis-quantile check is relabelled a shape (ellipticity) check, since
  a replicate-based quantile is near the χ² value for any elliptical cloud.
- 2026-08-05 (statistical re-review, four adversarial lanes + verification):
  corrected a false replicate-simultaneous claim (beyond-the-median-country
  holds for 32 of 33 cells; the simultaneous floor is the 41.3rd percentile,
  per conf_2026_simultaneous_headline.csv); disclosed per-item sign-test
  denominators after ties (A165 15/16, Y003 13/15, F120 13/16, E018 13/17);
  restated the variance-decomposition headline under the orthogonal partition
  (PC1′ model 20.9% vs language 11.6%; the mean-square convention had inflated
  the two-level language factor ×2); item-bootstrap "lower bound" reworded
  (understates the cluster bootstrap on 27 of 33 cells, overstates on ≥1 axis
  for 6); Fisher tests added to the coherence contrast (p = 0.011 vs 4/9,
  p = 0.051 vs 4/7); trace-coding counts + verbatims copied into the release
  as data/trace_coding.json.
- 2026-08-05 (artefact reconciliation, paper-review pass): three re-review
  statistics that existed only as in-session computations are now committed
  code + artefacts in the research repo (Fisher coherence tests, orthogonal
  SS variance partition, K=10 coverage calibration). Reconciliation against
  the committed artefacts corrected the paper in four places: orthogonal
  partition digits restated from the artefact (PC1' model 21.0% / language
  11.8% / variant 34.8% / repeat 32.4%; PC2' 20.9/5.5/33.6/40.0 — the
  earlier 20.9/11.6 and 20.4/5.5/34.2/39.9 were in-session values, max
  deviation 0.56pp, substance unchanged); ellipse coverage restated "near
  85%" (simulated 84.7–84.9% under the divisor-K plug-in covariance the
  bootstrap actually converges to; the earlier "near 84%" was imprecise);
  the zero-failure cell count corrected 21→19 of 34 (audited against
  llm_parse_rates_2026.csv); keying-balance 0.02–0.09 scoped as cohort×arm
  group means (per-cell values span up to 0.19).
- 2026-08-05 (labelling): the m=7 correlation family is described in the
  paper by composition (five of seven members are trace-length correlates,
  plus midpoint-distance and entropy central-tendency members); the
  2026-08-04 ledger entry's "central-tendency correlation family" label was
  shorthand for the same pre-specified family, not a different one. E018's
  per-item trend direction now stated (toward less respect, i.e.
  secular-ward), which verifies the "only survival-ward mover is E025"
  claim; the pre-specified G006/F063/E018 traditional-pole predictions
  failed item by item (two flat, one opposite) and the paper now says so.
- 2026-09-08 (post-acceptance camera-ready, reviewer-requested; all descriptive, post-confirmatory): (a) rescale offsets corrected to the WVS syntax (0.38→0.038, −0.01→−0.10; GitHub issue #12) — a rigid translation of the map by (−0.342, −0.090); every artefact regenerated and diffed against v1.0.0, no distance/rank/quadrant/bootstrap statement changes, all absolute coordinates re-quoted. (b) Prompt-cue sensitivity: cells re-projected from persona sub-families already in the corpus (six averaging-cue prefixes / three bare / one world-citizen) and from a new no-persona English arm (`data/collection_2026_nosys/`, 50 repeats × 10 items = 500 calls per cell; own directory because the 2026 scripts key arms on `language`). The cluster bootstrap is undefined below ten clusters, so both sides of every contrast use the item bootstrap, stated wherever reported. This is the interlocutor-crossed control noted as a future arm on 2026-08-04, in weak form. (c) Trace coding re-done: the original 900-trace counts came from eight parallel agentic coders with per-group rules and no trace-level labels, so no agreement statistic was computable; every trace re-coded under one frozen codebook (`app/trace_codebook.py`) by independent LLM annotators plus a blind 150-trace human subsample; Cohen's/Fleiss' kappa and Krippendorff's alpha reported; adjudicated label = strict majority of the LLM annotators. The §5 headline shares are re-reported as adjudicated values with the per-annotator range; the original single-pass values are retained in the appendix for provenance. None of (a)–(c) joins the m = 7 BH family or changes any confirmatory result.
- 2026-09-11 (post-confirmatory, descriptive; completes the 2026-09-08 entry):
  (a) Trace re-coding complete. Annotators: gpt-oss:120b, nemotron-3-super,
  glm-5.3, mistral-large-3:675b, gemma4:31b (five developers; glm-5.3 outside
  the cohort; gemma4:31b and mistral-large-3:675b emit no traces), 900/900
  traces each, temperature 0, one trace per call, original coding withheld.
  Adjudicated (strict-majority) shares: modal targeting 55% (per-annotator
  range 50–65%, Fleiss κ 0.78, Krippendorff α 0.78, pairwise Cohen κ
  0.70–0.86); persona reasoning 5% (2–17%, κ 0.35, α 0.35, pairwise
  0.17–0.63); guideline citation 42% (35–48%, κ 0.75, α 0.75, pairwise
  0.71–0.79); no unadjudicated ties. Reasoning-language counts 554 English /
  177 mixed / 169 Chinese, identical to the submitted coding. Headline moved
  from 57/10/48 to 55/5/42 in §5 and the abstract. Human 150-trace worksheet
  drawn and released (data/trace_coding_human_worksheet.csv) but not coded at
  camera-ready; the paper says so. Artefacts: data/trace_labels_2026__*.jsonl,
  trace_labels_2026.csv, trace_agreement_2026.csv, trace_coding_2026.csv,
  trace_coding_headline_2026.csv.
  (b) No-persona arm complete for all 17 models (data/collection_2026_nosys/,
  English, 50 repeats × 10 items; 8,442 unique calls: qwen3.5:397b 450,
  nemotron-3-ultra 492, the rest 500). 15 cells estimable; 15/15 in the
  quadrant; 14/15 farther from the human mean than under the persona protocol
  (median 3.41 vs 2.22 map units on the same 15 cells); the move is almost
  entirely up the secular axis (15/15 higher, median δPC2′ +1.90,
  item-bootstrap interval excluding zero for 13; median δPC1′ −0.09); no cell
  changes quadrant. Two cells not estimable: gemma4:31b refuses every call on
  six of the ten items (301/500 parse failures, all "as an AI" declinations);
  qwen3.5:397b never returns on F120 within the budget (450 records, nine
  items) and parses 7/50 on G006. Refusal rises without the persona
  (gpt-oss:120b 167/500, nemotron-3-ultra 82/492, vs ≤40/500 in the English
  persona arm). Bare deepseek-v4-flash / -pro tags were served in September
  and may be a later snapshot than the August persona arm (disclosed). Item
  bootstrap on both sides (single prefix, nothing to cluster). Artefacts:
  data/prompt_sensitivity_2026.csv (nosys-all_persona contrasts),
  data/prompt_sensitivity_variant_family_2026.csv (family = nosys).
  Neither (a) nor (b) joins the m = 7 BH family or alters any confirmatory
  result; both are reported in Appendices G/H and §5 as descriptive.

## 2026-09-11 camera-ready repair ledger — superseding implementation notes

This entry was appended after the camera-ready adversarial review and
regeneration on 11 September. The dated plan and earlier log above remain
unchanged as a historical record. The following corrections supersede the
incompatible implementation/status statements above; they are not additions
retroactively claimed to have been pre-specified on 4 August.

- **Observed means and the survey reference.** Reported model positions now
  project the observed per-item means directly; bootstrap draws estimate
  uncertainty, rather than their Monte Carlo average defining the point.
  The fixed reference `(0.038, -0.10)` is the projection of the observed
  per-item survey marginal-mean vector. It is not necessarily the mean of
  completed respondent coordinates after PPCA imputation, nor an
  equal-country mean. The earlier “pooled human mean” description was
  incorrect. `app/survey_reference.py` and the
  `data/validation_survey_*.csv` / `validation_reference_sensitivity.csv`
  artifacts disclose the reference and missing-data sensitivities.
  The offset-only correction remains an affine translation; the additional
  estimator and pooling repairs in this entry are separate changes.

- **Cluster pooling and uncertainty.** Empty variant/item groups contribute
  zero to pooled sums and counts, not NaN. The overall observed item mean is
  used only when the complete resampled set has zero parsed responses for
  that item, and such fallback occurrences are reported. A cluster bootstrap
  is mathematically defined with fewer than ten clusters; with a single
  cluster it cannot estimate between-prefix variability. The September
  statement that it is “undefined below ten clusters” was false.
  The item bootstrap is an independence-assumption sensitivity, not a
  guaranteed lower bound. Nominal per-cell ellipses and finite simulated
  no-crossing counts do not establish joint confidence coverage across
  cells or replicates. Bounds holding in every generated replicate are
  descriptive extrema; the original “no correction ... by construction”
  statement must not be read as a familywise-coverage guarantee.

- **Language displacement estimands and revised numbers.** The primary
  per-model norm is the norm of that model's observed zh-minus-en vector;
  the cohort headline averages those 16 norms, rather than taking the norm
  of the cohort-mean vector or averaging folded replicate norms.
  Regenerated `data/conf_2026_mean_displacement_plugin.csv` gives
  `0.655292995993` map units, reported as **0.66**; the plug-in norm
  origin-interaction permutation value in
  `data/conf_2026_origin_permutation_plugin.csv` is `0.884011598840`,
  reported as **0.88**. Earlier 0.65/0.89 summaries are superseded.
  Signed component inference remains distinct from positive norm ranges;
  a norm range excluding zero is not itself evidence of a directional
  effect. These repairs change numerical outputs and invalidate the
  blanket earlier assurance that no confirmatory result changed.

- **No-persona inclusion, denominators and estimators.** There were 8,500
  planned English control trials, but 8,442 unique terminal records, 58
  absent records and 13,523 recorded attempts attached to retained trials.
  Attempt counts omit unrecorded transport deferrals and are lower bounds
  on API traffic. The 724 terminal parse failures are not all verified
  refusals; in particular, Gemma's 301 include substantive misformatted
  answers as well as declinations. The corpus is not a complete 8,500-trial
  collection. The primary control now requires at least ten parsed answers
  on every item, matching the persona-arm rule: **13 pairs, all 13 in the
  quadrant, 12 farther from the survey reference**. The explicitly labelled
  one-per-item sensitivity has **15 pairs, all 15 in the quadrant, 14
  farther**. The persona side retains the main cluster replicates; the
  single-prefix control uses item resampling, with independent random
  streams. In 2,000 paired draws, positive secular-component intervals
  occur in 11/13 primary and 12/15 relaxed pairs, replacing the earlier
  13/15 item-on-both-sides count.
  Sources: `data/prompt_control_{summary,coverage,item_coverage}_2026.csv`
  and `data/prompt_sensitivity_2026.csv`.

- **What the prompt contrasts establish.** Prefix sub-family positions
  condition on the recorded prefixes and require at least one parsed
  answer per item; this is a labelled descriptive sensitivity, not a change
  to the full-protocol primary threshold. The no-persona condition removes
  only the persona prefix: the format instruction and refusal-mitigation
  system primer remain. Persona records are from 4 August and no-persona
  records from 8–11 September; hosted tags do not certify identical resolved
  weights across those dates. Neither these later-arm contrasts nor the
  existing-prefix comparisons isolate a causal effect of framing on the
  reasoning process. Existing-label prefix rates describe selected
  successful excerpts, not a randomised or independently coded no-persona
  reasoning experiment.

- **Trace coding and human validation.** Five complete LLM panels cover the
  same 900 successful sampled excerpts. The 150-excerpt human worksheet was
  prepared but **not human-coded**; no human validation or human–LLM
  reliability result exists in this release. Aggregation is an LLM
  strict-majority vote, not human adjudication. The frozen machine key
  `modal_targeting` denotes **typicality-or-moderation targeting**, including
  middle-of-the-road targeting, not a pure population-mode estimation code.
  No frozen codebook or judge label was changed by this repair. Counts are
  491/900, 41/900 and 380/900; Fleiss kappas round to 0.78, 0.35 and 0.75.
  Of the excerpts, 164 reach the 2,000-character storage cap. Per-trace
  exclusion of an own-model judge leaves 487/891, 41/899 and 379/887
  classified positives, with 9, 1 and 13 ties respectively; those ties are
  explicitly retained in all-sample prevalence bounds. Separate calls and
  developer diversity do not establish independent errors, and agreement
  on excerpts does not establish the faithfulness of reported reasoning.
  Sources: `data/trace_self_rater_sensitivity_2026.csv`,
  `trace_sample_coverage_2026.csv` and `trace_prefix_code_rates_2026.csv`.

- **Scope of the sums-of-squares partition.** The primary
  `pct_of_total_ss` field is an orthogonal descriptive partition of
  synthetic ten-item profiles assembled from separately elicited item
  calls, with the documented variant/cell-mean fills. It is not a fully
  crossed causal variance-components or generalisability analysis.
  Language-within-model includes language-by-model variation; item
  responses sharing a repeat index do not constitute a jointly elicited
  respondent. The legacy `pct_of_total` field normalises nested
  level-mean variances and is not an additive partition of total profile
  variation. Numerical comparisons must state which field and profile
  construction they use. Source: `scripts/diagnostics_2026.py` and
  `data/diag_2026_variance_components.csv`.

- **Reproduction scope.** No new collection or annotation API call is
  needed to replay the released trace labels, stored seed aggregates or
  aggregate-only appendix contrasts. Full projection fitting and
  regeneration of bootstrap artifacts still require the licensed IVS
  inputs described in the README; the survey microdata, fitted-model
  binaries and bootstrap-replicate CSVs are not redistributed by this
  correction. `scripts/verify_paper_claims.py` checks an explicit finite
  claim ledger and raw-to-artifact invariants; its full run requires those
  locally regenerated artifacts and must fail if they are absent. It is
  not a claim that every paper statement is automatically verified.

- **Recovered exploratory correlations (11 September integration).** The
  seven historical tests compare raw-item midpoint distance with each map
  coordinate; median stored excerpt length with midpoint distance and each
  coordinate; and each language arm's median excerpt length with the
  observed paired-model displacement norm. They are recomputed from current
  observed points, not old bootstrap-mean coordinates, with BH over exactly
  seven tests. Midpoint versus secularity gives rho=0.045120, raw p=0.803106,
  BH p=0.994044 (33 cells); all five length tests also have BH p=0.994044.
  Valid pairs are 29 cells or 14 paired models for the length tests.
  Censoring and same-model dependence limit interpretation; these are not
  causal mechanism tests.

  The historical post-hoc entropy/midpoint test used raw-answer-string
  entropy (rho=-0.653075, p=0.00003791). The planned entropy/item-bootstrap
  diagnostic used transformed-index entropy and the geometric mean of
  the two marginal coordinate SDs (rho=0.647727, p=0.00004603).
  Transformed-index entropy versus midpoint distance is a separately
  labelled definition sensitivity (rho=-0.582219, p=0.00037861).
  These are not interchangeable definitions or one pooled testing family.
  The ambiguous six-proxy mechanism-null claim is removed because its
  historical CJK-share denominator was not recovered unambiguously.

  Sources: `scripts/exploratory_correlations_2026.py`,
  `data/diag_2026_exploratory_correlations.csv` and
  `data/diag_2026_exploratory_features.csv`. This is a dated repair of
  recoverable exploratory definitions, not retrospective pre-registration.


### 11 September 2026 — final independent review corrections (author-authorized)

This entry is appended after the independent final review, before regenerating
its replacement analyses. It does not rewrite the original plan or present these
sensitivity analyses as pre-specified. The review evidence is preserved in
`notes/oracle-camera-ready/final-review-2026-09-11/` and the original PDF hash is
`c8a569752a0eb25f1417aa25ac8c146f1a8a4566b4543ffe60d2f181c26e24f0`.

- **Estimator correction.** Independent synthetic and actual-input checks found
  that the inherited missing-data loop is not exact PPCA EM: it includes imputed
  signal in its residual numerator and omits required conditional second moments.
  Replace it with direct optimization of the observed-data Gaussian PPCA
  likelihood, with observed-item means/SDs, two latent dimensions, multiple seeded
  starts, explicit convergence diagnostics and exact conditional-mean completion.
  Retain the downstream projection/rotation convention so this corrects the
  estimator rather than silently changing the measurement target. Validate against
  independent likelihood gradients, missing-data conditional moments and the
  complete-data analytic PPCA solution. Regenerate every dependent aggregate,
  bootstrap result, figure and manuscript quantity; previous figures in this
  append-only ledger remain historical records of their corresponding versions.
- **Family-unit sensitivity.** Add exploratory item sign tests after averaging
  within the existing nine developer families, with two-sided exact sign tests,
  ties removed and BH adjustment across the same ten items. Add family-mean raw
  and survey-standardized profile permutation comparisons with exhaustive origin
  label enumeration over those nine families. Report changed weighting and the
  selected-family exchangeability limitation; these p-values are not uniquely
  correct superpopulation tests or evidence of causal origin effects.
- **Known autonomy-wording sensitivity.** Add an exploratory exclusion of only
  GLM-5.1, GLM-5.2, Kimi-K2.6 and Qwen3.5's Y003 language comparisons, retaining
  other model/item comparisons and the same ten-item BH family. Quantify projected
  language differences with the Y003 contribution held constant as a separate
  diagnostic. These conservative exclusions do not assert that every response of
  each model was wrong, and do not reconstruct corrected counterfactual answers.
- **Exact claims and provenance.** Remove or narrow unverifiable historical fit,
  serving and failure-text claims. Distinguish retained recorded attempts from all
  network requests; document unknown hosted generation settings, requested-only
  model tags, and English-before-Chinese collection order. Correct verbatim
  quotations, subsets/denominators, full-precision rounding and literature
  attributions against their recorded sources. Expand bibliography names and align
  bylines to the cited source version. Do not infer decoder determinism from
  observed identical repeats or call item-bootstrap intervals lower bounds.
- **Reproducibility and release.** Make the QC gate reject absent expected cells,
  duplicate/missing trial keys and invalid stored responses while preserving
  genuine terminal failures as reportable data. Preserve fitting diagnostics in
  the frozen local model file and provide nonidentifying aggregate provenance.
  Prepare an immutable complete release candidate, excluding licensed respondent
  data, and verify it in isolation. Publishing/submitting remains a separate final
  action on the reviewed result. No new respondent or annotation API calls are
  needed for these corrections.

### 11 September 2026 — regenerated descriptive floors and final mathematical precision

The replacement fit changes the finite 2024 replicate minima to 89/109 mapped
countries (81.6514%) and 1.1235523 map units from a non-Western centroid. The
manuscript uses conservative descriptive floors of 81% and 1.12 units; 13 of 33
2026 cells breach at least one of these in a generated replicate. Using a rounded
81.7% as a comparison threshold would wrongly count cells tied at 89/109. The
original 95%/1.6 point-threshold comparison is retained, with 27 of 33 cells
breaching in some replicate. These are corrected descriptive summaries after an
estimator repair, not new pre-specified thresholds or simultaneous confidence
bounds. Historical 82%/1.2 statements refer to the superseded fit.

The explained-variance fraction now uses a consistent sample-variance denominator
for completed profiles (0.4196863). The projection-basis covariance identity is
C^T Sigma C = Lambda within the fitted loading subspace; the manuscript does not
claim that C is a full-covariance eigenbasis. An independent likelihood/moment/EM
check of the replacement real-data fit passed before manuscript synchronization.
The 20-seed refit is rerun from source; its new tiny numerical spread replaces
the earlier multi-degree seed sensitivity claim.


### 11 September 2026 — final-paper reviewer Y003 reconstruction and instrument reporting

This entry follows the independent review of the PDF with SHA-256
`11e8bcbe655e3c278153a0fbd7f2eb0843291b12b50492697ab5e1c10b8da217`.
The author authorized implementation after the review and its isolated refits.
Earlier entries and numerical summaries above remain historical records; this
appendix does not retrospectively preregister the added diagnostics.

- Recover missing longitudinal Y003 from observed independence (A029),
  determination/perseverance (A039), religious faith (A040), and obedience (A042)
  as A029 + A039 - A040 - A042 only when every constituent is valid 0/1. Preserve
  valid delivered indices and check concordance where both forms are available.
  Apply the existing six-item eligibility rule after recovery. The EVS source
  lacks the derived column; this does not imply absence of the constituent
  responses. Invalid or unavailable constituents leave the index missing.
- Export direct, reconstructed, residual-missing and eligibility counts, then
  refit the same likelihood/rotation/scaling protocol and regenerate dependent
  analyses, controls, figures and prose. Recovered observations are deterministic
  survey scores, not an additional statistical imputation model. Retain the
  official harmonization exclusions and all remaining measurement limitations.
- Add headline outcome columns to the existing six-criterion rotation comparison,
  transporting points/draws through the same fitted subspace with each anchored
  orientation and its own score standard deviations. These alternatives alter
  axis meaning and are not equally validated official IW measurements.
- Add bounded latest-available-year versus pooled-country comparisons and a
  seeded valid-choice baseline; display the scale-midpoint reference. These are
  exploratory instrument/benchmark diagnostics, not contemporary human ground
  truth or a calibrated null for the selected LLMs. Any masked-item or range
  diagnostic is explicitly conditional on the fit and is not held-out validation.
- Correct S017 to original national weight, state pooled-year contribution, and
  distinguish rotated projection coefficients from Gaussian loadings and
  correlations. Correct trace-module prose while preserving the exact annotation
  prompt and retained labels. No new model collection or annotation is involved.

The official longitudinal scoring definition is available at
https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=autonomous.
The new remediation report records the final regenerated quantities, validation
scope, exact artifacts and public-delivery status. Earlier successful numerical
replay established computation conditional on the prepared input; it did not
validate the omitted derived-index reconstruction.
