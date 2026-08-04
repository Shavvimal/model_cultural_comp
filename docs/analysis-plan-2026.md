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
