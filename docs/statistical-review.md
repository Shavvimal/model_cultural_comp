# Statistical Review — `model_cultural_comp`

**Scope.** Pre-submission methods review for the ORACLE @ EMNLP 2026 paper draft
(`cultural-bias-2026.mdx`), written from the position of a hostile-but-fair quantitative
reviewer. Every claim below that is marked *(verified)* was computed against the artefacts
in `data/` using the repo's own code; the reproduction scripts are described in §5.

**Headline.** The projection pipeline is now internally sound and the correction narrative is
real. But three things will not survive an adversarial reviewer as currently written:

1. **`Y003` is not a weakly-loading item — it is a corrupted column.** 31.9% of retained rows
   carry the SPSS user-missing sentinel `-3` as if it were data *(verified)*. The paper's
   Limitations attributes the near-zero loading to missingness; the true cause is a
   missing-value code read as a value. Fixing it moves country coordinates by up to 0.65 map
   units and model coordinates by 0.24–0.74 *(verified)* — larger than the projection defect
   the paper is built around correcting.
2. **The two "Confucian" model assignments are perfectly confounded with language of
   administration.** `llama2-chinese:13b` and `wangshenzhi/gemma2-27b-chinese-chat` are the
   *only* two models whose entire 2024 corpus was collected under Chinese-language prompts, and
   they are exactly the two models the SVM assigns to Confucian *(verified)*. On the one model
   run in both languages (`qwen2:7b`), switching prompt language moves the map position by
   2.72 units (95% CI [2.23, 3.23]) — 20× the reported bootstrap SD and larger than the
   `llama3:70b` correction the paper highlights *(verified)*.
3. **The bootstrap ellipses are too small by a factor of ~2.5–3.6 in area**, because resampling
   items independently sets every cross-item covariance to zero. On 2026 data where the design
   factor is recorded, a cluster bootstrap over the ten system-prompt variants gives SDs 1.4–2.2×
   larger *(verified)*, with per-item ICCs up to 0.56.

None of these overturn the qualitative headline (models cluster in the secular/self-expression
quadrant, far from every non-Western centroid). All three change what the paper is entitled to
say, and two of them change published numbers.

---

## 1. Verdicts table

| # | Decision | Verdict | One line |
|---|---|---|---|
| 1a | Varimax fitted on the N×2 **score** matrix rather than the 10×2 loadings | **DEFENSIBLE WITH CAVEAT** | It is a real estimator (Rohe & Zeng 2023) and lands 5° from the Procrustes rotation onto the published IW target, but must be *named* correctly, not justified as "it looked right". |
| 1b | Kaiser row-normalisation over 392k respondent rows (`Rotator` default) | **DEFENSIBLE WITH CAVEAT** | Load-bearing and currently unreported: without it the same fit gives −2.1° instead of −41.8° *(verified)*. Must be stated as part of the criterion. |
| 1c | Rotating **unwhitened** scores | **CHANGE RECOMMENDED** | Orthogonal rotation of unequal-variance scores leaves the two axes correlated at r = 0.34 *(verified)*; the paper implies orthogonal dimensions. Either whiten first or report the correlation. |
| 1d | Sign/order pinning by F118 / F063 anchors | **DEFENSIBLE** | Deterministic, documented, reproducible; the right way to resolve rotational indeterminacy. Keep. |
| 1e | Reported rotation-sensitivity range ("7–18°") | **CHANGE RECOMMENDED** | The repo's own diagnostic gives −25.7° for Kaiser-normalised loadings varimax *(verified)*. Publish the full grid, not a range that does not reproduce. |
| 2 | Borrowing the published WVS constants (1.81/0.38, 1.61/−0.01) | **CHANGE RECOMMENDED** | The constants presuppose unit-variance factor scores; ours have SD 1.39 and 1.33 *(verified)*, so absolute placement is inflated ~35%. Harmless for geometry, not for SVM regions, centroid distances, or comparisons to published country positions. |
| 3a | PPCA-EM imputation over listwise deletion | **DEFENSIBLE** | Correct call, standard, and the right citation chain (Tipping & Bishop 1999; Little & Rubin 2019). |
| 3b | Countries EM-imputed vs LLMs required complete | **DEFENSIBLE WITH CAVEAT** | The asymmetry is benign because the LLM path is affine in complete inputs, but it must be stated that model positions carry no imputation uncertainty while country positions do. |
| 3c | Y003 loading ≈ 0 explained as "high missingness" | **CHANGE RECOMMENDED — CORRECTNESS BUG** | 31.9% of rows carry the sentinel `-3` as data; Y003 is the *least* missing item (0.42% NaN) *(verified)*. Fix the column and refit. |
| 4a | Independent per-item bootstrap; "pairing never mattered" | **CHANGE RECOMMENDED** | The affineness argument is correct for the **mean** and silent about the **variance**. Cluster bootstrap over prompt variants gives 2.5–3.6× the ellipse area *(verified)*. |
| 4b | Bivariate normality of replicate means | **DEFENSIBLE** | Empirical 95th percentile of the replicate Mahalanobis² is 5.55–6.33 against χ²₀.₉₅,₂ = 5.99 *(verified)*. CLT holds; say so with the number. |
| 4c | χ²(0.95, df = 2) quantile for the ellipse | **DEFENSIBLE** | Correct quantile for a confidence region *for the mean*. Only the label needs tightening — it is not a region for the response distribution. |
| 5a | RBF-SVM region assignment on 109 countries, 8 classes | **CHANGE RECOMMENDED** | 5-fold CV accuracy is 0.552 (train 0.679) *(verified)*. Region labels from a coin-flip classifier cannot carry the Confucian argument. |
| 5b | Grid `C ∈ {500…2000}` | **CHANGE RECOMMENDED** | The grid excludes the regularised regime entirely; the selected C = 1000 is near-hard-margin interpolation of 6–21 points per class. |
| 5c | "Stability" as reported | **CHANGE RECOMMENDED** | It is the sampling variability of the model's position conditional on a fixed classifier; it does not include the classifier's own ~45% error rate. Rename and report both. |
| 5d | Prefer nearest-centroid? | **CHANGE RECOMMENDED — report both** | Nearest-centroid disagrees on exactly the two Confucian models, putting both in Protestant Europe *(verified)*. That disagreement *is* the finding. |
| 6a | Plot 2024 and 2026 on one map | **DEFENSIBLE WITH CAVEAT** | Legitimate: one frozen instrument, one coordinate space. Caveat the four confounds (precision, language, scale, survivorship) in the figure caption itself. |
| 6b | Pooled statistical test across years | **CHANGE RECOMMENDED — do not** | Two non-random, non-overlapping convenience samples of models under differing protocols. Report two cross-sections; no year effect, no trend test. |
| 6c | Freezing the 2024-fitted projection for 2026 | **DEFENSIBLE** | A fixed measurement instrument is the feature, not the bug — provided the instrument is described as a 2005–2022 human-value basis, which is a design choice, not an estimate of 2026 human values. |
| 6d | 2024 bilingual vs 2026 English-only | **CHANGE RECOMMENDED** | Worse than "bilingual": two models are Chinese-**only**, one is a 50/50 pool, and those are precisely the models carrying the Confucian claim *(verified)*. |
| 7 | Multiple comparisons / headline inference | **CHANGE RECOMMENDED** | "All models land in the quadrant" needs a simultaneous statement, and the paper needs an SVM-free headline statistic. One is proposed in §2.7. |
| 8a | `validate_projection.py` check A (self-consistency) | **DEFENSIBLE WITH CAVEAT** | It is a regression test against a specific past bug, not validation of the map. It cannot detect an error common to both paths — it did not detect the Y003 sentinel. Reword in the paper. |
| 8b | Seed handling | **DEFENSIBLE WITH CAVEAT** | Exactly reproducible, but the estimate carries EM-initialisation noise: across four seeds the rotation spans 40.60°–41.76° and country coordinates move up to 0.28 units *(verified)*. Report the across-seed dispersion. |
| 8c | `MIN_PER_QUESTION = 10` exclusion | **CHANGE RECOMMENDED** | An outcome-dependent exclusion: models that refuse on F118/F120 (the two highest-PC1 items) are dropped whole, biasing the surviving sample toward the self-expression pole. |
| 8d | Retry-until-parse survivorship | **CHANGE RECOMMENDED** | It is rejection sampling conditioned on terse compliance, and the intensity differs across runs: `max_retries = 15` in 2024 vs `MAX_ATTEMPTS = 3` in 2026 *(verified)*. |
| 8e | "Sure thing!" primer | **DEFENSIBLE WITH CAVEAT** | It is sent as a **trailing system message**, not an assistant prefill; its effect is chat-template-dependent and therefore not guaranteed constant across 2024 local GGUF and 2026 cloud serving. |
| 8f | Mid-scale response artefact (new finding) | **CHANGE RECOMMENDED — add analysis** | A respondent answering the midpoint of every scale lands at PC2′ = 3.00, above Sweden (2.50), the most secular country on the map *(verified)*. Central-tendency responding is a live alternative explanation for the headline. |
| 8g | Model provenance in `modelfiles/` | **CHANGE RECOMMENDED — verify** | `modelfiles/yi` and `modelfiles/glm` both point at the *aquilachat2* GGUF *(verified)*. If those files produced the 2024 runs, three of the five "failed Chinese models" were one model. |
| 8h | `analyze_2026.py` reads `data/collection_2026/pickles` | **CHANGE RECOMMENDED — broken** | That directory does not exist; only `.jsonl` files are written. The JSONL→pickle conversion step is absent from the repo. |

---

## 2. Detailed analysis, with quotable text

### 2.1 The rotation (decision 1)

**What is actually happening.** `factor_analyzer.Rotator(method="varimax")` is applied to the
392,513 × 2 training score matrix with its default `normalize=True`, i.e. Kaiser normalisation
applied to the *rows*, which here are respondents, not variables. Each respondent's score vector
is scaled to unit length, so the criterion becomes a function of the angular distribution of the
score cloud alone. This is not a footnote — it is the whole result:

| Rotation criterion | Angle *(verified)* | IW item agreement | Congruence with IW target |
|---|---|---|---|
| Scores, Kaiser on (**used**) | −41.76° | 8/10 | 0.807 |
| Scores, Kaiser off | −2.10° | — | — |
| Whitened scores, Kaiser on | −36.51° | — | — |
| Whitened scores, Kaiser off | −6.95° | — | — |
| Loadings `C`, Kaiser on | −24.87° | 6/10 | 0.754 |
| Loadings `C`, Kaiser off | −7.69° | 6/10 | 0.627 |
| Loadings `C·√λ`, Kaiser on | −25.68° | 6/10 | 0.754 |
| Loadings `C·√λ`, Kaiser off | −7.72° | 6/10 | 0.627 |
| **Procrustes onto the published IW target pattern** | **−47.02°** | **9/10** | **0.810** |
| No rotation | 0° | 7/10 | 0.552 |

(Angles from the fitted model; the score-varimax angle also moves 4.5° between the EM-imputed
and complete-case row sets, so it is not a sharply identified quantity.)

**Is it defensible as "the rotation criterion"?** Yes, but not on the grounds currently given.
"Only this variant reproduces the recognised map" is a reviewer magnet: it reads as choosing the
analysis by its output. Two better stories exist, and you should use both.

*Story A — name the estimator.* Varimax applied to the sample principal-component score matrix
is a studied estimator with asymptotic theory, not an improvisation: Rohe & Zeng (2023, *JRSS-B*
85(4), "Vintage factor analysis with varimax performs statistical inference") show that varimax
rotation of the PCA scores consistently estimates the factor rotation under a semi-parametric
factor model with leptokurtic (near-sparse) factor scores. With Kaiser row-normalisation the
criterion is a fourth-moment projection-pursuit index over the directional distribution of the
score cloud — the same family as kurtosis-based ICA. That is a defensible criterion with a
citation, and it is *not* the same object as Kaiser's (1958) varimax on a loading matrix. Two
deviations from Rohe & Zeng must be acknowledged: they rotate the orthonormal score basis
(whitened), the repo rotates `U·Λ^{1/2}` (unwhitened), and their guarantees assume leptokurtic
scores, which a 10-item ordinal survey cloud only approximately satisfies.

*Story B — validate the choice against the published target.* The Inglehart–Welzel instrument
specifies which five items define each axis. Scoring each candidate rotation against that
hypothesis (the "IW item agreement" column above) is an external criterion that does not depend
on eyeballing the figure, and the orthogonal Procrustes rotation onto that target (Hurley &
Cattell 1962) is the principled limit of the same idea. The score-varimax solution sits 5.3°
from the Procrustes solution and recovers 8/10 of the published item-to-axis assignment; every
loadings-based variant sits 21–39° away and recovers 6/10. That is the argument to make.

**Method text (quotable):**

> The two map axes are identified only up to an orthogonal rotation. We fit a single varimax
> rotation, once, to the 392,513 × 2 matrix of training scores, with Kaiser row normalisation
> (the `factor_analyzer` default), and store it; every subsequent projection reuses the stored
> rotation. Varimax applied to the score matrix rather than the loading matrix is the estimator
> analysed by Rohe and Zeng (2023), who show it consistently recovers the factor rotation under
> a leptokurtic factor model; with row normalisation the criterion is a fourth-moment
> projection-pursuit index over the directional distribution of the score cloud, and is
> therefore a different object from Kaiser's (1958) varimax on loadings. We validate the choice
> against an external target rather than against the appearance of the figure: scored against
> the published Inglehart–Welzel assignment of five items to each axis, the score-varimax
> solution recovers 8 of 10 item placements (Tucker congruence 0.81 with the target pattern)
> and sits 5.3° from the orthogonal Procrustes rotation onto that target (Hurley and Cattell
> 1962), whereas varimax on the loading matrix recovers 6 of 10 and sits 21–39° away. Axis
> order and sign are then pinned deterministically by two anchor items (F118 positive on
> self-expression, F063 negative on secular-rational). We report the full rotation grid and the
> Procrustes solution as a sensitivity analysis in Appendix C.

**Limitations text (quotable):**

> The rotation is the least sharply identified step in the pipeline. Kaiser row normalisation is
> load-bearing: without it the same criterion returns a 2° rotation rather than 42°. The angle
> also moves 4.5° between the EM-imputed and complete-case row sets and 1.2° across EM
> initialisation seeds. Because the rotation is applied to unwhitened scores, the two rotated
> axes are not uncorrelated: across respondents they correlate at r = 0.34. Readers should treat
> the axes as an interpretable, externally validated frame rather than as statistically
> orthogonal dimensions, and all reported coordinates carry a rotational uncertainty of order
> 0.1 map units from these sources combined.

**On 1c specifically.** `cov(SR) = RᵀΛR` is not diagonal when Λ is not proportional to the
identity. With λ = (2.484, 1.219) and R at 41.76°, the rotated axes have variances (1.923,
1.780) and covariance 0.628, i.e. r = 0.34 *(verified)*. If you want genuinely orthogonal axes,
whiten before rotating (which also brings you to the exact Rohe–Zeng estimator, at the cost of
moving the angle to −36.5°). If you keep the current pipeline, report r = 0.34; the published
IW dimensions are themselves correlated at the country level, so this is a disclosure issue, not
a fatal one.

**One arithmetic detail.** `PPCA._calc_var` divides by the variance of the *EM-imputed*
standardised matrix, whose total is 9.64 rather than the nominal 10 *(verified)*, because
conditional-mean imputation shrinks variance. The reported 38.4% is 3.703/9.64; against the
nominal total variance of ten standardised items it is 37.0%. Either denominator is arguable;
state which one you used.

### 2.2 Borrowed rescaling constants (decision 2)

The affine map `PC′ = aPC + b` cannot change any relative geometry: distances scale by
(1.81, 1.61) per axis, angles change, ranks do not. If the paper only made statements of the
form "model X is closer to region A than to region B", borrowing would be cosmetic and I would
pass it. It does not: the SVM boundaries, the region centroids, the reported distance of
`wangshenzhi` from the Confucian cluster, and the comparison against China's published position
all live on the absolute scale.

The specific problem is that the WVS constants were calibrated against factor scores on a
particular scale — conventionally standardised to unit variance at the individual level. Our
rotated scores have SD 1.387 and 1.334 *(verified)*. Applying constants designed for unit-variance
inputs to inputs with 33–39% more spread inflates the map by that factor, which is why our
country coordinates span PC1′ ∈ [−2.05, 4.36] against a published map that is conventionally
drawn on roughly [−2.5, 2.5].

**Recommendation, in order of preference.** (a) Calibrate `a` and `b` yourself by regressing
your country means on the published WVS country coordinates, and report the R² — this converts
an unverifiable borrowing into an external validity check, which is the single cheapest
credibility gain available in this paper. (b) Failing that, standardise the rotated scores to
unit variance over the training respondents before applying the published constants, so the
constants are applied to the kind of input they were derived for. (c) Failing both, drop the
constants and report on the native scale, stating that absolute comparison with published
country coordinates is not supported.

**Method/Limitations text (quotable):**

> We apply the published WVS rescaling `PC1′ = 1.81·PC1 + 0.38`, `PC2′ = 1.61·PC2 − 0.01` so
> that our coordinates are readable against the familiar map. These constants were derived by
> WVS for their own factor scores, not ours. Because the map is affine, the choice cannot affect
> any relative statement — distances, orderings and region memberships are unchanged up to a
> known scaling — but it does affect absolute placement, and our rotated scores have standard
> deviations 1.39 and 1.33 rather than the unit variance the constants presuppose, so absolute
> coordinates should be read as approximately 35% wider than the published map. All conclusions
> in this paper are stated relationally for this reason. [If (a) is adopted:] We additionally
> report constants calibrated directly against the published country coordinates (R² = …), and
> confirm that no conclusion changes under either scaling.

### 2.3 Missing data and Y003 (decision 3)

**PPCA vs listwise: keep it.** Discarding 28.4% of rows that are missing non-randomly (missingness
is concentrated by country-wave, i.e. by survey design) would induce exactly the selection the
paper is trying to avoid. EM under a Gaussian latent model is the standard answer and the
missing-by-design structure makes MAR unusually plausible here (Little & Rubin 2019, ch. 1–2;
Graham 2009 on planned missingness). Say that explicitly — "missing by design is the benign
case" is a stronger sentence than "imputation would bias the components".

**The asymmetry is fine, and worth one sentence.** Country coordinates come from EM-imputed
scores; LLM coordinates come from complete per-item means. Because the projection is affine, a
model's position is exactly a linear functional of its ten item means and involves no imputation
at all. The asymmetry to disclose is that country positions inherit imputation uncertainty that
is nowhere propagated into their plotted position, while model positions carry only sampling
uncertainty — so the two point types on Figure 1 do not have commensurable error bars, and only
the models have any drawn.

**Y003 is a bug, not a limitation.** *(verified, and this is the most consequential finding in
this review.)* The retained analysis rows contain:

- `Y003 = −3` in 125,718 of 394,524 rows (31.9%), outside the valid range [−2, 2];
- `Y003` NaN in only 0.42% of rows — the **lowest** missingness of all ten items;
- no out-of-range values in any of the other nine items.

`-3` is the SPSS user-missing code "Not applicable"; the merge syntax shipped in `data/` declares
it as such twice (`missing values Y003 (-3).`, `missing values Y003 (-5).`). It has been read as
data. The consequences:

- Y003's loading is (−0.013, 0.004), i.e. the item contributes nothing to a map on which it is
  one of the five items that officially define the traditional/secular axis.
- The "at least 6 of 10 answered" filter counted a non-answer as an answer for 31.9% of rows.
- Recoding `-3` to NaN and refitting changes: explained variance 0.384 → 0.417; rotation
  −41.76° → −39.57°; Y003 loading (−0.013, 0.004) → (0.188, 0.241); country coordinates move by
  mean 0.192, median 0.172, max 0.650 (r = 0.994 / 0.997 per axis); the ten 2024 model positions
  move by mean 0.47, max 0.74.
- With `-3` treated as missing, Y003's individual-level correlations become exactly the published
  pattern: −0.390 with F063 (importance of God), +0.272 with F120, +0.265 with F118, +0.191 with
  E018, +0.172 with G006. At the country level it loads 0.844 on the first component — one of the
  strongest loaders, alongside F120 (0.914), F063 (−0.912) and F118 (0.877).

For calibration: the paper's own robustness claim is that the projection correction moved nine of
ten models by less than 0.18 units. The Y003 sentinel moves them by 0.24–0.74. It is a larger
perturbation than the defect the paper exists to correct, and a reviewer who runs
`df.Y003.value_counts()` will find it in thirty seconds.

**Do not ship the current Limitations sentence.** Replace it with a refit. If a refit is
impossible before the deadline, the sentence must be:

> The autonomy index Y003 contributes essentially nothing to our fitted map (loadings −0.01 and
> 0.00). This is a defect of our reconstruction, not a property of the item: 31.9% of retained
> rows carry the SPSS user-missing code `-3` for Y003, which our pipeline read as a datum rather
> than as missing. Recoding it to missing and refitting raises explained variance from 38.4% to
> 41.7%, restores Y003 to a substantive loading (0.19, 0.24) and to its expected correlational
> profile (r = −0.39 with importance of God), and moves country coordinates by a mean of 0.19
> map units (maximum 0.65, per-axis r > 0.99) and model coordinates by a mean of 0.47 (maximum
> 0.74). No qualitative conclusion in this paper changes under the refit; all reported
> coordinates should nonetheless be read with that displacement in mind.

**One further caveat once fixed.** The sentinel is structural, not sporadic: 15 of 109 countries
have it in 100% of rows and 65 have it in none; by year it is 94.8% in 2008 and 0% in 2005–2007
*(verified)*. After recoding, EM imputes Y003 for entire countries from the other nine items —
legitimate under MAR-by-design but pure model-based extrapolation for those countries. Report
which countries those are.

### 2.4 The bootstrap (decision 4)

**(a) The independence assumption is the weak point, and the affineness argument does not rescue
it.** The repo's claim — "because the projection is affine in the item values, a model's mean
position depends only on per-item means, so pairing was irrelevant" — is *true for the point
estimate and silent about the variance*. Writing the position as `p = w₀ + Σⱼ wⱼ x̄ⱼ`,

  Var(p) = Σⱼ Σₖ wⱼ wₖ Cov(x̄ⱼ, x̄ₖ),

and resampling items independently forces every off-diagonal `Cov(x̄ⱼ, x̄ₖ)` to zero. The design
guarantees those covariances are not zero: all ten items were elicited under the same ten
system-prompt variants, and a variant that pushes the model toward, say, more agreeable answers
moves several items together.

On the 2026 data, where `system_prompt_id` is recorded, this is measurable *(verified)*:

| Model | Per-item ICC by prompt variant (max) | SD ratio cluster/naive, PC1 | PC2 | Ellipse area ratio |
|---|---|---|---|---|
| `deepseek-v4-flash` | 0.56 (F118), 0.41 (Y002), 0.36 (F063), 0.35 (G006) | 1.86 | 1.39 | 2.52 |
| `deepseek-v4-flash:0731` | 0.48 (F063), 0.39 (G006), 0.35 (Y002), 0.24 (A165) | 2.20 | 1.60 | 3.56 |

The naive ellipse understates the confidence region by a factor of 2.5–3.6 in area. That is not
a rounding issue; it is the difference between "these two models are significantly apart" and
"they are not".

**Recommendation.** For 2026, make the cluster bootstrap over the ten system-prompt variants the
primary estimator (resample variants with replacement, carrying all items and repeats within a
variant; Field & Welsh 2007; Davison & Hinkley 1997 §3.8). Report the naive item bootstrap
alongside as an explicit lower bound. With only K = 10 clusters the nonparametric cluster
bootstrap is itself known to under-cover (Cameron, Gelbach & Miller 2008); state that, use
B ≥ 10,000, and treat the resulting interval as approximate. For 2024, where the variant id was
not recorded, the cluster bootstrap is unavailable and the existing ellipses must be relabelled
as lower bounds — with the 2026 SD ratios cited as the empirical scale of the understatement.

**(b) Normality: defensible, and now with evidence.** Across the ten 2024 models the empirical
95th percentile of the replicate Mahalanobis² ranges 5.55–6.33 against χ²₀.₉₅,₂ = 5.99
*(verified)*. The normal-theory ellipse is well calibrated to the replicate cloud. Report that
number rather than appealing to the CLT in the abstract — with n = 50 per item and bounded
discrete supports, Berry–Esseen is comfortable, but the check is free.

**(c) χ²(0.95, df = 2) is the right quantile** for a 95% confidence region *for the mean
position*, which is what a bootstrap over replicate means estimates. It is not a 95% region for
the model's response distribution, and the current caption ("95% bootstrap confidence regions")
is correct but easy to misread. Say "confidence region for the model's mean position". If you
want to drop the normality assumption entirely, replace `chi2.ppf(0.95, 2)` with the empirical
95th percentile of the replicates' Mahalanobis² — a one-line change that here moves the ellipse
by less than 6% in linear extent.

**Method text (quotable):**

> A model's map position is an affine functional of its ten per-item mean responses, so the
> point estimate does not depend on how individual responses are paired into pseudo-respondents.
> The variance does. We therefore report two bootstraps. The *item bootstrap* (B = 1,000)
> resamples each item's stored responses independently with replacement; because it forces all
> cross-item covariances to zero, it is a lower bound on the true uncertainty. The *cluster
> bootstrap* (B = 10,000) resamples the ten system-prompt variants with replacement, carrying
> all items and repeats within a variant, and so propagates prompt-level correlation into the
> position; it is our primary estimator wherever the variant identifier was recorded. On the
> 2026 corpus, per-item intraclass correlations by prompt variant reach 0.56 and the cluster
> bootstrap yields confidence ellipses 2.5–3.6 times larger in area than the item bootstrap. The
> 2024 corpus does not record the variant identifier, so only the item bootstrap is available
> there and its ellipses must be read as lower bounds. Ellipses are normal-theory regions from
> the replicate covariance at χ²₀.₉₅,₂; the replicate clouds are well approximated by a bivariate
> normal (empirical 95th percentile of Mahalanobis² = 5.55–6.33 against 5.99). With only ten
> clusters the cluster bootstrap is itself approximate (Cameron et al. 2008).

**One more source of variance the ellipses do not contain.** For `deepseek-v4-flash`, items A008
and A165 returned an identical value in all 500 calls *(verified)*. Frontier models are far more
deterministic than the 2024 cohort, so sampling-only ellipses will shrink toward zero and become
progressively less informative about anything a reader cares about. This is an additional
argument for making prompt-design variance, not token sampling variance, the reported quantity.

### 2.5 SVM regions (decision 5)

The classifier is the weakest inferential object in the paper and it carries one of the two
substantive claims (the Confucian argument). *(All verified.)*

- n = 109 countries, 8 classes, sizes: African-Islamic 21, Orthodox Europe 19, Catholic Europe
  17, Latin America 15, West & South Asia 13, Confucian 10, **Protestant Europe 8**,
  **English-Speaking 6**. Every model in the paper is assigned to a region defined by 6–8 points.
- Selected hyperparameters C = 1000, γ = 0.05. Training accuracy 0.679; **5-fold CV accuracy
  0.552**. The grid `C ∈ {500, 1000, 1500, 2000}` never evaluates a regularised model, so
  "grid-searched" conveys a rigour the search does not have.
- Nearest-centroid disagrees with the SVM on exactly two models — and they are exactly the two
  Confucian assignments. Nearest-centroid places both in Protestant Europe.
- `wangshenzhi/gemma2-27b-chinese-chat` is 2.74 from the Protestant Europe centroid and 2.75
  from the Confucian centroid. The SVM calls it Confucian with "stability 1.00".
- Both Confucian models' nearest country is Japan (0.76 and 1.18 away). Japan is the most secular
  country on the map; proximity to Japan is proximity to an outlier within its class, not
  evidence of Confucian value alignment.
- Extrapolation is real but not uniform: model-to-nearest-country distances run 0.14–1.19, against
  a median country-to-country nearest-neighbour distance of 0.20 (p90 = 0.50). Three models sit
  within normal inter-country spacing; `dolphin-llama3:8b` and `wangshenzhi` sit ~6× the median
  spacing from any country.

**"Stability" is conflating two things, and the paper should say so in those words.** The reported
share is Pr(region | fixed classifier) over the bootstrap distribution of the model's position.
It contains no information about Pr(correct region | position), which the CV estimate puts near
0.55. A model can be at stability 1.00 in a region the classifier would get wrong 45% of the time.

**Recommendation.** Report both assignments in the main table and treat their disagreement as a
result. Rename "stability" to "positional stability" and add the CV accuracy adjacent to it.
Extend the grid downward (C ∈ {0.1 … 2000}, γ ∈ {0.01 … 1}) with stratified CV, and report the
resulting accuracy honestly. Consider dropping the SVM from the main narrative entirely in favour
of §2.7's centroid-distance statistic, and keeping the decision-boundary figure as illustration
only.

**Results/Limitations text (quotable):**

> Region assignments come from an RBF-SVM fitted to 109 country coordinates across eight regions,
> with as few as six countries in a class. Its 5-fold cross-validated accuracy is 0.55 (training
> accuracy 0.68), so a single region label is weak evidence about any point, and weaker still
> about points lying off the country manifold. We therefore report two assignments per model —
> the SVM's modal region with its positional stability across bootstrap replicates, and the
> nearest region centroid — and we report the positional stability as what it is: the probability
> that the model's position falls in a fixed decision region under resampling, which does not
> include the classifier's own error rate. The two rules agree on eight of ten models and
> disagree on exactly the two models assigned to the Confucian region, both of which nearest
> centroid places in Protestant Europe; one of them (`wangshenzhi/gemma2-27b-chinese-chat`) is
> 2.74 units from the Protestant Europe centroid and 2.75 from the Confucian centroid, i.e. the
> assignment is a coin flip that the SVM reports at stability 1.00. We conclude that no model in
> either cohort is meaningfully placed in a non-Western cultural region.

### 2.6 Joining 2024 and 2026 (decision 6) — see also §3

**(a) One map: yes.** The instrument is frozen, the coordinate space is shared by construction,
and check A proves the arithmetic path is identical. Plotting them together is legitimate and is
the point of freezing the projection. The caption must carry the confounds; a reader who sees two
cohorts on one map will read a trend whether or not you licence one.

**(b) A pooled test: no.** Neither cohort is a random sample from any population of models. They
do not overlap in membership (the 2026 list contains no 2024 model), so there is no paired
design; they differ in serving precision, prompt language, model scale, and the survivorship
filter simultaneously. A year effect in this design is not identified, and any p-value attached
to one would be indefensible. Report two cross-sections and describe the difference
descriptively. If you want *any* inferential statement about change, the only defensible target
is a within-family comparison where the same lab's model appears in both cohorts under the same
protocol — and even there the estimand is "this family's expressed values changed", not "models
changed".

**(c) Freezing the 2024 fit: a feature, argued explicitly.** The instrument is a basis derived
from 2005–2022 human survey responses. It is a *ruler*, and rulers do not get re-cut between
measurements — refitting per cohort is the exact defect the paper's whole correction narrative is
about. The caveat to state is that the basis represents the value structure of the 2005–2022
human population and is not claimed to be the value structure of 2026 humans; if human values
have shifted, both cohorts are measured against a common historical reference, which is what makes
them comparable and also what makes "distance from humanity today" an over-claim.

> **Quotable:** The projection is fitted once, on human survey data from 2005–2022, and frozen.
> Both model cohorts are measured against that fixed basis. This is deliberate: refitting the
> instrument per cohort would place each cohort in its own coordinate space and make the
> comparison meaningless. The corresponding limitation is that the axes encode the value
> structure of the 2005–2022 surveyed population, so all statements are of the form "relative to
> the human value structure measured in 2005–2022", not "relative to human values in 2026".

**(d) The language confound is the serious one, and it is larger than the paper states.**
*(verified.)* The 2024 corpus is eleven files for ten model labels:

| Model | 2024 elicitation language | 2024 region assignment |
|---|---|---|
| `llama2-chinese:13b` | **Chinese only** | **Confucian (0.70)** |
| `wangshenzhi/gemma2-27b-chinese-chat` | **Chinese only** | **Confucian (1.00)** |
| `qwen2:7b` | English + Chinese, pooled 50/50 under one label | Protestant Europe (1.00) |
| `wangrongsheng/llama3-70b-chinese-chat` | English only | Protestant Europe (1.00) |
| all six others | English only | Protestant Europe |

The two Confucian assignments are perfectly confounded with Chinese-language administration.
The paper currently says Chinese models were "additionally prompted in Chinese"; for these two
there is no English run at all — the English run of `llama2-chinese:13b` was abandoned as "too
error prone" and never produced stored data.

The magnitude of the language effect is measurable on `qwen2:7b`, the one model run both ways:

- English-only position (1.966, 3.277); Chinese-only position (3.385, 0.961).
- Distance 2.72 map units; bootstrap 95% CI [2.23, 3.23]; per-axis differences PC1 [−1.86, −0.99]
  and PC2 [+1.90, +2.75], both excluding zero by a wide margin.
- Per-model bootstrap SDs are 0.11–0.19. The language effect is ~15–25 SDs.
- The largest item shifts are G006 national pride (2.72 → 1.35, much prouder in Chinese), F063
  importance of God (7.32 → 9.12) and E018 respect for authority (1.98 → 1.20) — all three are
  traditional-pole items, and all three move toward the traditional pole under Chinese prompting.
- The published `qwen2:7b` row (2.66 ± 0.12, 2.13 ± 0.15) is the midpoint of a 2.72-unit gap,
  with an error bar 20× smaller than the gap. That is not a defensible summary of that model.

**This must be fixed before submission, and the fix is cheap:** split `qwen2:7b` into two rows
(`qwen2:7b [en]`, `qwen2:7b [zh]`), label the two Chinese-only models as Chinese-only in the
main table, and rewrite the Confucian paragraph. The honest reading strengthens the paper: the
only two models that look Confucian are the two asked in Chinese, and the one model measured
both ways moves 2.7 units when the language changes — which makes "language of elicitation, not
model origin, is what moves position on this map" a finding rather than an embarrassment.

> **Quotable (Results):** Language of elicitation, not developer origin, is the largest
> single factor moving a model on this map. `qwen2:7b` is the only 2024 model administered in
> both English and Chinese; its two positions lie 2.72 map units apart (bootstrap 95% CI
> [2.23, 3.23]), against per-position sampling standard deviations of 0.11–0.19 and against the
> 2.05-unit shift we report for the projection defect. Under Chinese administration it becomes
> markedly more traditional on the secular axis, driven by national pride, importance of God and
> respect for authority. Both models assigned to the Confucian region in 2024 were administered
> in Chinese only, so origin and elicitation language are perfectly confounded for exactly the
> assignments that carry the Confucian claim. We therefore report those two models as
> Chinese-administered and do not treat their region assignment as evidence about Chinese
> foundation models.

> **Quotable (Limitations):** The 2024 protocol was not uniform. Two models were administered in
> Chinese only, one in both languages with the responses pooled, and the remaining seven in
> English only; the 2026 protocol is English only throughout. Because the language effect we
> measure is an order of magnitude larger than sampling uncertainty, no 2024→2026 comparison is
> valid for the Chinese-administered models, and the 2026 cohort tests the Confucian hypothesis
> under English administration only. A bilingual 2026 arm is the obvious next experiment and we
> did not run it.

**On the "Confucian hypothesis now testable" claim.** With 2026 English-only, what becomes
testable is: *do frontier Chinese-developed models, asked in English, express values closer to
the Confucian cluster than Western models asked in English?* That is a clean, well-powered,
worth-stating question — it is simply not the same question as "do Chinese models hold Confucian
values". Say the narrower thing. The 2024 run cannot be pooled in to widen it, because in 2024
the models nearest that hypothesis were the ones asked in a different language.

### 2.7 Multiple comparisons and a better headline (decision 7)

**Is a correction needed?** Not in the usual sense — the paper is not testing eighteen null
hypotheses and reporting the survivors. But three specific claims need tightening:

1. *"All models land in the secular/self-expression quadrant"* is a **simultaneous** claim over
   models, currently supported by per-model marginal ellipses. State it as such: with per-model
   95% regions, the probability that all m regions simultaneously cover is not 0.95. Either
   Bonferroni the per-model level to 1 − 0.05/m for the joint statement, or — cleaner — note that
   every model's *entire* bootstrap replicate cloud lies in the quadrant, which is a statement
   about the replicates and needs no correction at all. The latter is both true here and free.
2. *Per-model region assignments with stability* are eighteen (or ten) selective statements; the
   modal region is a max over eight categories, which is upward-biased as a confidence
   statement. Report the full share vector, not just modal + runner-up, and never describe
   stability as a confidence level.
3. Any *pairwise* model comparison ("model A is more secular than model B") over 18 models is 153
   comparisons. If the paper makes any such claim, apply Benjamini–Hochberg (1995) over the
   comparisons actually made and say so.

**A better headline statistic — recommended.** The current headline routes through an SVM with
0.55 CV accuracy. It does not need to. The pooled IVS respondent grand mean is at (0.38, −0.01)
by construction (scores are mean-zero before rescaling), and the country distribution around it
is known. Two distribution-free statistics, both bootstrappable through exactly the existing
machinery *(verified for the 2024 cohort)*:

| Model | Distance to pooled human respondent mean | % of 109 countries closer to it than the model is | Min distance to any non-Western region centroid |
|---|---|---|---|
| `dolphin-llama3:8b` | 3.83 | 96.3% | 2.91 |
| `dolphin-mistral:7b` | 3.30 | 95.4% | 2.31 |
| `dolphin-mixtral:8x7b` | 3.02 | 92.7% | 2.02 |
| `gemma2:27b` | 3.44 | 96.3% | 2.51 |
| `llama2-chinese:13b` | 3.29 | 95.4% | 2.48 |
| `llama3:70b` | 2.94 | 90.8% | 2.02 |
| `mistral:7b` | 3.93 | 97.2% | 2.93 |
| `qwen2:7b` (pooled) | 3.13 | 94.5% | 2.17 |
| `wangrongsheng/…-chinese-chat` | 3.20 | 95.4% | 2.20 |
| `wangshenzhi/…-chinese-chat` | 3.69 | 96.3% | 2.75 |

(Country median distance from the grand mean is 1.78; p90 is 2.91; the maximum is Sweden at 4.71.)

> **Quotable:** Every model in the 2024 cohort lies further from the pooled human respondent mean
> than 91–97% of the 109 surveyed countries, and no model comes within 2.0 map units of any
> non-Western region centroid. Both statements hold for every bootstrap replicate of every model,
> so they require no multiplicity correction; both are independent of the region classifier.

Two refinements worth adding: (i) weight the human reference by population rather than by survey
sample size — WVS sample sizes are not proportional to population, so the current grand mean
over-weights small, heavily-sampled countries; merge UN WPP population weights and report both;
(ii) report the minimum distance to a non-Western centroid with a bootstrap CI, which the
existing replicate matrix supplies directly.

### 2.8 Everything else

**Check A is a regression test, not validation.** `check_a_self_consistency` pushes complete
training rows through `project()` and compares with the fitted coordinates. Both paths execute
the same standardisation, the same `C`, the same `R`, and the same constants, so the check can
only fail if those objects differ between paths — which is precisely the 2024 bug and precisely
nothing else. It cannot detect a wrong rotation, wrong constants, a wrong item transform, or the
Y003 sentinel, and in fact did not. Keep it, in CI, forever; do not describe it in the paper as
evidence that the map is right. Suggested wording: "a regression test that the model and country
projection paths are byte-identical — the defect present in the 2024 code — not a validation of
the map itself."

Two fragilities in the same function: it aligns `valid_data` to `subset_ivs_df` positionally
after a `merge(..., how="left")` on `country_code`; a single duplicated `Numeric` in
`country_codes` would silently duplicate rows and shift every subsequent index without failing
the check. Assert `len(valid_data) == len(subset_ivs_df)` immediately after the merge.

**Seed handling.** Reproducible in the strict sense, but the estimate carries real Monte Carlo
noise from the EM initialisation *(verified)*: across seeds {42, 0, 7, 2024} the rotation spans
40.60°–41.76° and country coordinates move by mean 0.05–0.10, max 0.23–0.28. That is the same
order as the model displacements the paper reports as evidence of robustness. Fit K = 20 seeds,
report the across-seed SD of each country and model coordinate, and add it to the reported
uncertainty budget. The sign convention in `PPCA.fit` (largest-magnitude loading positive) plus
the F118/F063 anchor pinning does fully determine orientation, so there is no flip risk — good.

**`MIN_PER_QUESTION = 10` is an outcome-dependent exclusion.** A model is dropped entirely if any
single item falls below ten parsed responses. The items most likely to trigger it are F118
(justifiability of homosexuality) and F120 (abortion) — the two items with the highest positive
PC1 loadings. A model that refuses on those items is exactly a model that would have been placed
toward the survival/traditional pole, and it is silently removed. Combined with the 2024
survivorship (five Chinese models produced nothing parseable), the surviving sample is selected
on a variable correlated with the outcome.

> **Quotable (Limitations):** Our sample is selected on the outcome. Models are included only if
> they produce parseable answers to all ten items, and the items most often refused (justifiability
> of homosexuality and of abortion) are the two items with the largest positive loadings on the
> self-expression axis. A model that declines those questions is disproportionately a model that
> would have been placed toward the survival pole, and is instead excluded. Our claim is therefore
> strictly conditional: *among models that answer the instrument*, all express values in the
> secular/self-expression quadrant. The five 2024 models and any 2026 models excluded on this
> criterion are reported with their per-item parse rates so that readers can bound the selection;
> we make no claim about where they would have landed.

If you want a defensible bound rather than a caveat, the response space is bounded, so the
achievable map region is computable: the extreme admissible respondent positions are PC1′ ∈
[−7.07, 8.36] and PC2′ ∈ [−3.69, 9.70] *(verified)*. A Manski-style worst-case bound (Manski
2003) — place every excluded model at the most traditional/survival admissible response vector
and report whether the "all models" claim survives — is honest and takes one hour. It will not
survive, which is exactly why the claim must be stated conditionally.

**Retry-until-parse is rejection sampling, and its intensity is not constant across cohorts.**
*(verified)*: the 2024 harness used `max_retries = 15`; `cloud_survey.py` uses
`MAX_ATTEMPTS = 3`. Retrying until an answer parses conditions the retained response distribution
on "the model emitted a bare number this time". If the propensity to emit prose rather than a
bare number depends on the answer the model would have given — a model that wants to hedge on
F118 writes a paragraph; a model that is comfortable answering writes "8" — then the retained
sample is biased toward whatever answers the model states tersely, and the bias is stronger with
15 attempts than with 3. The 2024 records do not preserve the attempt count, so the intensity of
selection is unrecoverable there; 2026 stores `attempts` but overwrites `raw_content` with the
last attempt, discarding the rejected drafts.

> **Quotable (Limitations):** Responses were elicited by retrying until the output parsed (up to
> 15 attempts in 2024, 3 in 2026). This is rejection sampling conditioned on terse compliance: if
> a model's willingness to answer with a bare number depends on the answer it would give, the
> retained distribution is shifted toward answers the model states without hedging, and the shift
> is larger where more retries were permitted. The 2024 corpus does not record attempt counts, so
> we cannot quantify this; the 2026 corpus records the attempt count but not the rejected drafts.
> Future runs should record every attempt.

**The "Sure thing!" primer.** It is sent as `{"role": "system"}` *after* the user turn, in both
harnesses. That is not an assistant prefill; how a trailing system block is rendered is entirely
a function of each model's chat template. Two consequences: the primer's effect is not guaranteed
constant across models within a cohort, and it is certainly not guaranteed constant between 2024
local GGUF templates and 2026 cloud serving — which undercuts the claim that the protocol is
"deliberately identical to 2024 so comparisons change one variable at a time". The honest
statement is that the *prompt strings* are identical and the *rendering* is not controlled.
A cheap fix for 2026 is to send it as `{"role": "assistant"}` and document the change, or to run
a small primer-ablation arm on two or three models and report the displacement.

**A new and serious alternative explanation: central-tendency responding.** *(verified.)* A
respondent who answers the exact midpoint of every scale projects to (0.65, **3.00**). The most
secular country on the map is Sweden at PC2′ = 2.50. Half the 2024 models sit above 2.50. The
mechanism is that the world's respondents are far from the scale midpoints on the traditional-pole
items — the fitted means are F063 = 7.19 (midpoint 5.5), G006 = 1.57 (midpoint 2.5), E018 = 1.54
(midpoint 2.0) — so mid-scale answering reads on this instrument as strongly secular. The 2026
reasoning traces make the mechanism explicit; `deepseek-v4-flash` reasons "As an average human,
many people might say quite happy or very happy… I'll go with 2, as it's a common moderate
response." That is a model targeting the modal response, not expressing a value.

This is the strongest available attack on the paper's headline and it should be pre-empted, not
left for a reviewer. Two cheap analyses defuse it: (i) report the distance of each model's item
vector from the all-midpoint vector alongside its map position, so readers can see that models are
not simply mid-scale responders (several are not — `mistral:7b` at PC1′ = 3.84 is far from the
midpoint respondent's 0.65); (ii) report each model's per-item response *entropy* — a model that
always returns the same value on an item is not answering a survey in the sense the instrument
assumes.

> **Quotable (Limitations):** The map is not centred on the response scale. A respondent choosing
> the midpoint of every item projects to (0.65, 3.00), which is more secular than Sweden, the most
> secular surveyed country. This is a property of the human distribution — the world's respondents
> are far more religious, more nationally proud and more deferential to authority than the scale
> midpoints — but it means that any tendency toward central-tendency or modal responding registers
> on this instrument as secularity. Reasoning traces from the 2026 cohort show models explicitly
> selecting "a common moderate response". We report each model's distance from the all-midpoint
> response vector and its per-item response entropy so that readers can distinguish expressed
> values from modal-response behaviour; we cannot fully separate the two, and this is a limit of
> survey-style elicitation rather than of this particular map.

**Model provenance.** `modelfiles/yi` and `modelfiles/glm` both contain
`FROM …\aquilachat2-34b-16k.Q4_K_M.gguf` *(verified)* — the same GGUF as `modelfiles/aquilachat2`.
If those Modelfiles are what produced the 2024 `yi:34b` and `glm4:9b` runs, then three of the
five reported Chinese-model failures were one model tested three times, and Appendix A's failure
list is wrong. It is entirely possible the runs used the official Ollama registry tags instead and
the Modelfiles are stale scratch. Either way this cannot be adjudicated from the artefact, and a
reviewer who opens the repo will ask. Determine which weights were actually served, correct
Appendix A if needed, and delete or fix the stale Modelfiles before the repo goes public.

**A number in the draft does not match the artefact.** The paper reports China at (−0.03, 0.90);
`data/corrected_country_scores.csv` has (−0.12, 0.79) *(verified)*. Sweep every numeral in the
draft against the shipped CSVs before submission.

---

## 3. The 2024/2026 joining policy

### Do

- **Do** plot both cohorts on the same axes. One frozen instrument, one coordinate space, proven
  identical by check A. This is the payoff of the correction and you should take it.
- **Do** describe the design in the paper as *two independent cross-sections measured with a
  frozen instrument*, in exactly those words.
- **Do** carry the confound list in the Figure caption itself, not only in Limitations: serving
  precision (Q4 local vs unknown cloud), elicitation language (mixed vs English-only), model
  scale (7B–70B vs 20B–675B), retry intensity (15 vs 3), and survivorship (5 of 15 attempted
  Chinese models produced nothing in 2024).
- **Do** report both cohorts' positions with cluster-bootstrap uncertainty where possible and
  item-bootstrap lower bounds where not, and say which is which per point.
- **Do** make within-cohort comparisons freely — they are protocol-matched and that is where the
  paper's real inferential content lives (Chinese-origin vs Western-origin in 2026 is a clean,
  well-powered comparison).
- **Do** use one descriptive cross-cohort statistic if you want a longitudinal sentence: the
  *coherence rate* (fraction of attempted Chinese-origin models producing a parseable corpus),
  which went from 4/9 to whatever 2026 gives. That is a protocol-robust count, not a position
  comparison, and it is the 2024 limitation the re-run was designed to resolve.
- **Do** state that any 2026-only conclusion about Chinese models is conditional on English
  administration.

### Don't

- **Don't** run any pooled test, year effect, difference-in-means across cohorts, regression with
  a year dummy, or trend line. The cohorts share no models, no serving stack, and no language
  protocol; a year coefficient is not identified.
- **Don't** compute a "displacement" for any model between cohorts. No model appears in both.
- **Don't** compare `qwen2:7b` (2024, half Chinese-administered) with `qwen3.5:397b` (2026,
  English) as a within-family trend. That comparison confounds language, scale, precision and
  time in one number.
- **Don't** describe the 2024 protocol as "English plus Chinese for some Chinese models". Two
  models are Chinese-only and one is a pooled mixture.
- **Don't** let a 2026 ellipse and a 2024 ellipse appear in the same figure with the same legend
  entry when one is a cluster bootstrap and the other is an item bootstrap. Different estimators,
  different symbols, stated in the caption.
- **Don't** claim the frozen instrument measures distance from *contemporary* human values. It
  measures distance from the 2005–2022 surveyed human value structure.
- **Don't** describe either cohort as a sample of "open-weight models". Both are convenience
  samples of what was locally runnable / cloud-served at one moment; every population-level
  statement should be scoped to the tested set.

---

## 4. Recommended code changes, ranked

Ranked by effect on published numbers first, then on defensibility. **I have changed nothing.**

| # | Change | Where | Why it ranks here |
|---|---|---|---|
| **1** | Recode `Y003 ∈ {−3, −5}` (and sweep every item for out-of-range sentinels) to `NaN` *before* the ≥6-of-10 filter, then refit and regenerate every artefact and every number in the paper. | `culture_map.prepare_data` | Fixes a correctness bug that moves country coordinates up to 0.65 and model coordinates up to 0.74 units, restores an item that officially defines one of the two axes, and raises explained variance to 41.7%. Everything else in this list is smaller than this. |
| **2** | Split the 2024 corpus by elicitation language: separate `llm` labels (or a `language` column) for the `c-*` files; stop pooling English and Chinese `qwen2:7b`; propagate the label into the results table and every figure. | `llm_bootstrap.load_transformed_responses`, `llm_meta` | The two Confucian assignments are perfectly confounded with Chinese-only administration, and the pooled `qwen2:7b` row averages across a 2.72-unit gap while reporting a ±0.12 error bar. Directly changes what §4 of the paper is allowed to say. |
| **3** | Add a cluster bootstrap resampling `system_prompt_id` (all items and repeats within a variant), make it primary for 2026, and report the item bootstrap alongside as a lower bound. | `llm_bootstrap.bootstrap_llm_positions` | Naive ellipses understate the confidence region by 2.5–3.6× in area. Cheap: the variant id is already recorded. |
| **4** | Fix the 2026 analysis path: `analyze_2026.py` reads `data/collection_2026/pickles`, which does not exist. Add the JSONL→values loader (with the same `(question, system_prompt_id, repeat)` dedup that `parse_rates` already applies) and drop the pickle round-trip. | `scripts/analyze_2026.py`, `llm_bootstrap` | Phase 3 cannot currently run to completion, and without the dedup a resumed run double-counts retried rows. |
| **5** | Report both region rules (SVM modal + nearest centroid), attach the SVM's cross-validated accuracy to every stability figure, rename `stability` → `positional_stability`, and extend the grid to include regularised models (`C` from 0.1, `gamma` from 0.01) with `StratifiedKFold`. | `region_svm.RegionClassifier` | A 0.55-accuracy classifier is currently reporting "1.00" next to the paper's most contested claim. |
| **6** | Calibrate the rescaling constants against published WVS country coordinates (report `a`, `b`, R²) or standardise rotated scores to unit variance before applying the published constants; store the choice in the npz. | `culture_map._rescale` | Converts an unverifiable borrowing into an external validity check, and fixes a ~35% inflation of the absolute scale that the SVM and all centroid distances depend on. |
| **7** | Add a rotation-sensitivity artefact: emit the full grid (scores/whitened/loadings × Kaiser on/off), the Procrustes-to-IW-target rotation, the IW item-agreement score, and the post-rotation axis correlation. Replace the current single "sensitivity" line. | `scripts/validate_projection.py` | Turns the rotation from the paper's softest claim into a documented, externally validated choice; also surfaces `r = 0.34` between axes, which must be disclosed. |
| **8** | Add a multi-seed harness: refit over K = 20 seeds, write the across-seed SD of every country and model coordinate, and include it in the reported uncertainty budget. | new `scripts/seed_sensitivity.py` | The across-seed dispersion (max 0.28) is the same order as displacements the paper cites as evidence of robustness; reporting it pre-empts the obvious question. |
| **9** | Replace the SVM-based headline with the centroid-distance statistics (distance to the human grand mean, percentile against the country distribution, minimum distance to any non-Western centroid), each with a bootstrap CI; add UN WPP population weights for the human reference. | new function in `llm_bootstrap`, `scripts/analyze_2026.py` | Gives the abstract a classifier-free, multiplicity-free headline that is already true in the data. |
| **10** | Record every attempt in the 2026 harness (append each attempt's raw content rather than overwriting), and expose `MAX_ATTEMPTS` as a documented config field alongside a per-model, per-item refusal-rate report. | `cloud_survey._run_task` | Makes the rejection-sampling selection auditable in the one cohort where it still can be. |
| **11** | Add the central-tendency diagnostics: distance of each model's item vector from the all-midpoint vector, and per-item response entropy. | new function, `scripts/analyze_2026.py` | Pre-empts the strongest alternative explanation for the headline result. |
| **12** | Add an assertion that the `country_codes` merge in `fit()` preserves row count, and cap PPCA EM iterations with an explicit `ConvergenceError`. | `culture_map.fit`, `ppca.PPCA.fit` | Two silent-corruption paths on the critical route to a published number; the rubric's §0 "no silent fallback" already requires the second. |
| **13** | Resolve the `modelfiles/yi` and `modelfiles/glm` provenance (both point at the aquilachat2 GGUF), correct Appendix A if the served weights were wrong, and delete stale Modelfiles. | `modelfiles/` | Provenance of a reported result; also a public-release embarrassment (they contain a Windows user path, which the rubric's §0 already flags). |
| **14** | Replace `chi2.ppf(0.95, 2)` with the empirical 95th percentile of replicate Mahalanobis² (keep χ² as a reported comparison). | `llm_bootstrap.confidence_ellipses` | Removes a distributional assumption for one line of code. Low priority precisely because the assumption checks out (5.55–6.33 vs 5.99). |

---

## 5. Reproducing the numbers in this review

Every *(verified)* figure above was computed against the committed artefacts with the repo's own
`.venv`. The essential ones, in decreasing order of importance:

```python
# 1. The Y003 sentinel (the finding that matters most)
s = pd.read_pickle("data/subset_ivs_df.pkl")
s["Y003"].value_counts().sort_index()        # -3.0 appears 125,718 times
s[IV_QNS].isna().mean()                       # Y003 is the LEAST missing item (0.42%)

# 2. Impact: refit with Y003 sentinel -> NaN, compare country and model coordinates
#    var_exp 0.384 -> 0.417; angle -41.76 -> -39.57; country shift mean 0.192 max 0.650;
#    model shift mean 0.47 max 0.74

# 3. Rotation grid: Rotator(method="varimax", normalize=True/False) on
#    scores / whitened scores / C / C*sqrt(eig); Procrustes via SVD of (C*sqrt(eig)).T @ T
#    where T is the published IW 5+5 target pattern

# 4. Axis correlation after rotation: cov = R.T @ diag(eig_vals) @ R -> r = 0.3396

# 5. Language effect: project data/collection/qwen2-7b_responses_df.pkl and
#    data/collection/c-qwen2-7b_responses_df.pkl separately -> 2.716 units apart

# 6. Cluster bootstrap: resample system_prompt_id in data/collection_2026/*.jsonl
#    -> SD ratios 1.39-2.20, ellipse area ratios 2.52 and 3.56

# 7. SVM: cross_val_score(SVC(C=1000, gamma=0.05), xy, labels, cv=StratifiedKFold(5)) -> 0.552

# 8. Mid-scale artefact: cm.project(midpoint_vector) -> (0.65, 3.00) vs Sweden at 2.50
```

## 6. References worth citing in the Method

- Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal component analysis. *JRSS-B*,
  61(3), 611–622. — already cited; the EM-with-missing-data justification.
- Rohe, K., & Zeng, M. (2023). Vintage factor analysis with varimax performs statistical
  inference. *JRSS-B*, 85(4), 1037–1060. — **the citation that legitimises varimax on the score
  matrix**; currently absent and the single most valuable addition to §3.2.
- Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis.
  *Psychometrika*, 23(3), 187–200. — for the criterion the repo is *not* using on loadings, and
  for the row-normalisation whose role must be disclosed.
- Hurley, J. R., & Cattell, R. B. (1962). The Procrustes program: producing direct rotation to
  test a hypothesized factor structure. *Behavioral Science*, 7(2), 258–262. — the target-rotation
  validation of the rotation choice.
- Lorenzo-Seva, U., & ten Berge, J. M. F. (2006). Tucker's congruence coefficient as a meaningful
  index of factor similarity. *Methodology*, 2(2), 57–64. — for the 0.807/0.810 congruence values.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. —
  B, resampling unit, percentile vs normal-theory intervals.
- Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*, §3.8. —
  bootstrapping hierarchical/clustered data.
- Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *JRSS-B*, 69(3), 369–390. —
  the cluster bootstrap over prompt variants.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for
  inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427. — the
  few-clusters caveat at K = 10.
- Kish, L. (1965). *Survey Sampling*. Wiley. — design effect `1 + (m−1)ρ`, for reporting the ICCs.
- Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data*, 3rd ed. —
  MAR, missing-by-design, EM.
- Graham, J. W. (2009). Missing data analysis: making it work in the real world. *Annual Review of
  Psychology*, 60, 549–576. — planned missingness as the benign case.
- Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American
  Sociological Review*, 15(3), 351–357. — individual- vs country-level loading differences, which
  is the right frame for why our individual-level fit is not the published country-level one.
- Cattell, R. B. (1966). The scree test for the number of factors. *Multivariate Behavioral
  Research*, 1(2), 245–276. — cite when stating that d = 2 is fixed by the instrument rather than
  chosen from the data.
- Davidov, E. (2009). Measurement equivalence of nationalism and constructive patriotism in the
  ISSP. *Political Analysis*, 17(1), 64–82. — cross-national measurement invariance, the standard
  caveat on treating one factor solution as comparable across all countries (and, a fortiori,
  across languages — directly relevant to §2.6d).
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1),
  289–300. — for any pairwise model comparisons.
- Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer. — the
  worst-case bound for models excluded by the parse-rate filter.
