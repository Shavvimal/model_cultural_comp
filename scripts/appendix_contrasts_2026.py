"""Recompute the post-hoc petition and origin-profile contrasts.

Both are computed from locally generated aggregates only - no survey data, no raw
corpus, no API - and the statistics live in app/appendix_contrasts.py.

  (a) Petition-signing contrast within the refusal-sensitive set. For each
      of the 17 models, E025's keyed-PC1 zh-en shift (delta on the raw
      item scale x that item's PC1 loading per raw unit, i.e. its map-unit
      contribution to the model's PC1 displacement; negative = toward the
      survival pole) minus the mean keyed-PC1 shift of the other four
      sensitive items (F118, F120, F063, G006). Reported: the number of
      models in which E025 is the more survival-ward, the median contrast
      and a two-sided exact binomial sign test. Reads
      data/conf_2026_item_language_effects.csv (delta) and
      data/diag_2026_item_keying.csv (d_pc1_per_unit).
      The paper labels this p descriptive, not confirmatory: the contrast
      was formulated after E025 stood out in the m = 10 item sign tests.

  (b) Origin similarity of item profiles. Each model's English-arm profile
      is its ten item means (data/diag_2026_item_profiles.csv,
      language == "en", cell_mean); the statistic is the mean pairwise
      Pearson correlation over within-cohort pairs (Chinese-origin and
      Western pooled, 45 + 21 pairs) minus the mean over the 70
      cross-cohort pairs, with an exact null over all C(17, 10) = 19,448
      origin labellings. The within-Chinese, within-Western and cross means
      are written as descriptives, plus the Chinese-minus-cross contrast
      under the same enumeration. Profiles are the raw item means, not the
      human-mean deviations: subtracting a per-item constant changes the
      across-item correlation, and the raw means are what the draft's
      r = .989 / .976 / .970 triple was computed on.
      The standardised sensitivity uses the ten frozen fit means/SDs in
      data/validation_survey_item_baselines.csv; it needs no fitted NPZ.

Everything here is deterministic (exact enumeration, no sampling); SEED is
kept for the shared convention and the row order of the outputs.

Writes:
    data/conf_2026_petition_contrast.csv             (a) per-model rows + a
                                                     "== SUMMARY ==" row
    data/conf_2026_origin_similarity_permutation.csv (b) two statistic rows
    data/conf_2026_origin_similarity_standardised.csv survey-scale sensitivity
Each file carries a ``definition`` column stating the statistic.

Run from the repo root:  uv run python scripts/appendix_contrasts_2026.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from app.appendix_contrasts import origin_profile_permutation, petition_contrast
from app.culture_map import IV_QNS
from app.llm_meta import cohort_2026

SEED = 42  # no randomness is consumed; kept for the repo-wide convention
# Refusal-sensitive items, diagnostics_2026.py:62 (SENSITIVE_QNS).
SENSITIVE_QNS = ["F118", "F120", "F063", "G006", "E025"]
TARGET_ITEM = "E025"
PROFILE_ARM = "en"
PROFILE_COLUMN = "cell_mean"

ITEM_FX_CSV = "data/conf_2026_item_language_effects.csv"
KEYING_CSV = "data/diag_2026_item_keying.csv"
PROFILES_CSV = "data/diag_2026_item_profiles.csv"
ITEM_BASELINES_CSV = "data/validation_survey_item_baselines.csv"
OUT_PETITION = "data/conf_2026_petition_contrast.csv"
OUT_ORIGIN = "data/conf_2026_origin_similarity_permutation.csv"
OUT_ORIGIN_STANDARDISED = "data/conf_2026_origin_similarity_standardised.csv"

PETITION_DEFINITION = (
    "per model: keyed_pc1(E025) - mean keyed_pc1(F118,F120,F063,G006); "
    "keyed_pc1 = (mean_zh - mean_en) * d_pc1_per_unit; negative = E025 more "
    "survival-ward; two-sided exact binomial sign test over models with "
    "contrast != 0"
)
ORIGIN_DEFINITION = (
    "profiles = 10 English-arm item means (cell_mean) per model; Pearson r "
    "over all model pairs; statistic = mean r within-cohort (Chinese and "
    "Western pooled) - mean r cross-cohort; exact enumeration of all "
    "C(17,10) origin labellings; p_exact_ge = share of labellings with "
    "statistic >= observed"
)


def standardise_profiles(profiles: pd.DataFrame, item_baselines: pd.DataFrame) -> pd.DataFrame:
    """Apply the released frozen fit moments, aligned by item identifier.

    These are observed-item standardisation parameters from the fitted
    instrument, not moments recalculated from model profiles or completed
    human scores. Reject partial or invalid aggregates instead of silently
    producing a different transform.
    """
    required = {"question", "fit_standardisation_mean", "fit_standardisation_sd"}
    if not required.issubset(item_baselines.columns):
        raise ValueError(f"item baselines require columns {sorted(required)}")
    if item_baselines["question"].duplicated().any():
        raise ValueError("item baselines must contain one row per question")
    if set(item_baselines["question"]) != set(IV_QNS):
        raise ValueError("item baselines must contain exactly the ten instrument items")
    if profiles.columns.duplicated().any() or set(profiles.columns) != set(IV_QNS):
        raise ValueError("profiles must contain exactly the ten instrument items")
    fitted = item_baselines.set_index("question").reindex(profiles.columns)
    means = fitted["fit_standardisation_mean"].astype(float)
    stds = fitted["fit_standardisation_sd"].astype(float)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("fitted standardisation means must be finite and SDs positive")
    return (profiles - means) / stds


def main() -> int:
    np.random.default_rng(SEED)  # no draws; see module docstring

    item_fx = pd.read_csv(ITEM_FX_CSV)
    keying = pd.read_csv(KEYING_CSV)
    profiles_long = pd.read_csv(PROFILES_CSV)

    # (a) petition-signing contrast
    reference = [q for q in SENSITIVE_QNS if q != TARGET_ITEM]
    per_model, summary = petition_contrast(item_fx, keying, TARGET_ITEM, reference)
    per_model = per_model.sort_values("llm").reset_index(drop=True)
    s = summary.iloc[0]
    summary_row = pd.DataFrame(
        [
            {
                "llm": "== SUMMARY ==",
                f"keyed_pc1_{TARGET_ITEM}": np.nan,
                "keyed_pc1_reference_mean": np.nan,
                "contrast": s["median_contrast"],
                "target_more_survival_ward": np.nan,
                "n_models": int(s["n_models"]),
                "n_effective": int(s["n_effective"]),
                "n_target_more_survival_ward": int(s["n_target_more_survival_ward"]),
                "median_contrast": s["median_contrast"],
                "p_sign_two_sided": s["p_sign_two_sided"],
            }
        ]
    )
    petition = pd.concat([per_model, summary_row], ignore_index=True)
    for col in ["n_models", "n_effective", "n_target_more_survival_ward"]:
        petition[col] = petition[col].astype("Int64")  # counts stay integers beside NaN
    petition["definition"] = PETITION_DEFINITION

    # (b) origin similarity of item profiles
    wide = (
        profiles_long[profiles_long["language"] == PROFILE_ARM]
        .pivot(index="llm", columns="question", values=PROFILE_COLUMN)
        .sort_index()
    )
    if wide.isna().any().any():
        raise ValueError("incomplete item profiles in the English arm")
    is_cn = np.array([cohort_2026(m) == "Chinese" for m in wide.index])
    origin = origin_profile_permutation(wide, is_cn)
    origin["profile_arm"] = PROFILE_ARM
    origin["profile_column"] = PROFILE_COLUMN
    origin["definition"] = ORIGIN_DEFINITION
    origin.loc[origin["statistic"] == "chinese_minus_cross", "definition"] = (
        ORIGIN_DEFINITION.replace("within-cohort (Chinese and Western pooled)", "within-Chinese")
    )

    # Standardise each item on the frozen survey scale: raw-scale Pearson
    # correlations can be dominated by between-item location/range differences.
    item_baselines = pd.read_csv(ITEM_BASELINES_CSV, float_precision="round_trip")
    standardised = origin_profile_permutation(standardise_profiles(wide, item_baselines), is_cn)
    standardised["profile_arm"] = PROFILE_ARM
    standardised["profile_column"] = "survey_standardised_cell_mean"
    standardised["definition"] = ORIGIN_DEFINITION.replace(
        "10 English-arm item means (cell_mean)", "10 survey-standardised English-arm item means"
    )
    standardised.loc[standardised["statistic"] == "chinese_minus_cross", "definition"] = (
        standardised.loc[
            standardised["statistic"] == "chinese_minus_cross", "definition"
        ].str.replace("within-cohort (Chinese and Western pooled)", "within-Chinese", regex=False)
    )
    standardised.to_csv(OUT_ORIGIN_STANDARDISED, index=False)

    # Check mathematical invariants; do not force regenerated data to agree
    # with historical paper numerals after an estimator correction.
    assert 0 <= int(s["n_target_more_survival_ward"]) <= int(s["n_effective"])
    assert 0 <= s["p_sign_two_sided"] <= 1
    wc = origin[origin["statistic"] == "within_minus_cross"].iloc[0]
    assert (origin["p_exact_ge"].between(0, 1)).all()

    petition.to_csv(OUT_PETITION, index=False)
    origin.to_csv(OUT_ORIGIN, index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("=== (a) petition-signing contrast within the refusal-sensitive set ===")
        print(per_model.round(4).to_string(index=False))
        print(
            f"\nE025 more survival-ward than mean({', '.join(reference)}) in "
            f"{int(s['n_target_more_survival_ward'])} of {int(s['n_effective'])} models; "
            f"median contrast {s['median_contrast']:+.3f} map units; "
            f"exact sign test p = {s['p_sign_two_sided']:.4f}"
        )
        print("\n=== (b) origin similarity of English-arm item profiles ===")
        cols = [
            "statistic",
            "observed",
            "n_labellings",
            "n_ge_observed",
            "p_exact_ge",
            "p_exact_abs",
        ]
        print(origin[cols].round(5).to_string(index=False))
        print(
            f"\nwithin-Chinese r = {wc['mean_r_within_chinese']:.3f}, "
            f"cross r = {wc['mean_r_cross']:.3f}, "
            f"within-Western r = {wc['mean_r_within_western']:.3f}; "
            f"exact permutation p = {wc['p_exact_ge']:.4f} "
            f"({int(wc['n_ge_observed'])}/{int(wc['n_labellings'])})"
        )
    print(f"\nWrote {OUT_PETITION}, {OUT_ORIGIN} and {OUT_ORIGIN_STANDARDISED}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
