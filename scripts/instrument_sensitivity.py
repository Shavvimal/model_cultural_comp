"""Offline instrument diagnostics after corrected fit and 2026 bootstrap.

Writes aggregate CSVs only; no respondent records or new API calls. All
diagnostics use the corrected preparation, then hold the saved fit fixed.
"""

import numpy as np
import pandas as pd

from app.culture_map import ITEM_VALID_RANGES, IV_QNS, CulturalMap
from app.instrument_sensitivity import (
    SLOPES,
    XY,
    conditional_complete,
    country_year_benchmarks,
    rotation_outcomes,
    uniform_choice_baseline,
    weighted_coordinates,
)


def imputation_diagnostics(cm, standardized, completed, positions):
    """Range and frozen-fit Y003 masking diagnostics, explicitly in sample."""
    observed = cm.subset_ivs_df
    raw = observed[IV_QNS].to_numpy()
    decoded = completed * cm.ppca.stds + cm.ppca.means
    rows = []
    for j, question in enumerate(IV_QNS):
        values = decoded[~np.isfinite(raw[:, j]), j]
        lo, hi = ITEM_VALID_RANGES[question]
        rows.append(
            {
                "question": question,
                "n_imputed": len(values),
                "imputed_min": values.min() if len(values) else np.nan,
                "imputed_max": values.max() if len(values) else np.nan,
                "n_below_range": int((values < lo).sum()),
                "n_above_range": int((values > hi).sum()),
            }
        )
    ranges = pd.DataFrame(rows)
    ranges.to_csv("data/validation_imputation_ranges.csv", index=False)
    # Flag which country ingredients still rely on imputation. Reconstructed
    # Y003 is counted as observed here; its separate provenance is exported
    # by validate_projection. These are country/item totals, never rows.
    grouped = observed.groupby("country_code")
    coverage = (
        grouped[IV_QNS]
        .count()
        .rename_axis(columns="question")
        .stack()
        .rename("n_observed")
        .reset_index()
    )
    coverage["n_respondents"] = coverage.country_code.map(grouped.size())
    coverage["n_missing"] = coverage.n_respondents - coverage.n_observed
    coverage["missing_fraction"] = coverage.n_missing / coverage.n_respondents
    coverage["wholly_missing"] = coverage.n_observed.eq(0)
    coverage = coverage.merge(
        cm.country_codes[["Numeric", "Country"]],
        left_on="country_code",
        right_on="Numeric",
        how="left",
    )
    coverage["is_mapped"] = coverage.Numeric.notna()
    coverage = coverage.drop(columns="Numeric")
    coverage["scope"] = (
        "after Y003 recovery and eligibility; observed includes reconstructed indices"
    )
    np.testing.assert_array_equal(
        coverage.groupby("question").n_missing.sum().reindex(IV_QNS).to_numpy(),
        ranges.set_index("question").n_imputed.reindex(IV_QNS).to_numpy(),
    )
    coverage.to_csv("data/validation_country_item_coverage.csv", index=False)
    # Range clipping is a frozen-coordinate perturbation, not a replacement
    # estimator. Observed values are valid already and remain unchanged.
    clipped = decoded.copy()
    for j, question in enumerate(IV_QNS):
        clipped[:, j] = np.clip(clipped[:, j], *ITEM_VALID_RANGES[question])
    clipped_positions = positions.copy()
    clipped_positions[XY] = cm.project(pd.DataFrame(clipped, columns=IV_QNS))[XY].to_numpy()
    pooled = weighted_coordinates(positions, ["country_code"])
    clip_pooled = weighted_coordinates(clipped_positions, ["country_code"])
    np.testing.assert_array_equal(pooled.country_code, clip_pooled.country_code)
    clipping_shift = np.linalg.norm(pooled[XY].to_numpy() - clip_pooled[XY].to_numpy(), axis=1)

    y003 = IV_QNS.index("Y003")
    masked = standardized.copy()
    masked[:, y003] = np.nan
    predicted = conditional_complete(masked, cm.ppca.loadings_, cm.ppca.noise_variance_)
    predicted = predicted[:, y003] * cm.ppca.stds[y003] + cm.ppca.means[y003]
    present = np.isfinite(raw[:, y003])
    # Reuse weighted aggregation with temporary axis names for actual/predicted
    # scalar values. Outcomes were used in fitting: this is not held-out testing.
    masking = positions.loc[present, ["country_code", "weight"]].copy()
    masking[XY[0]] = raw[present, y003]
    masking[XY[1]] = predicted[present]
    masking = weighted_coordinates(masking, ["country_code"]).rename(
        columns={XY[0]: "observed_y003_mean", XY[1]: "predicted_y003_mean"}
    )
    masking["prediction_error"] = masking.predicted_y003_mean - masking.observed_y003_mean
    coefficient = cm.ppca.C[y003] @ cm.rotation / cm.score_stds * SLOPES / cm.ppca.stds[y003]
    masking["single_item_coordinate_shift"] = masking.prediction_error.abs() * np.linalg.norm(
        coefficient
    )
    masking["scope"] = (
        "frozen fit; observed Y003 used in training; in-sample diagnostic, not held out"
    )
    masking.to_csv("data/validation_y003_masking_country.csv", index=False)
    y003_observed_counts = observed.groupby("country_code").Y003.count()
    mapped = y003_observed_counts.index.isin(cm.country_codes.Numeric)
    return {
        "remaining_imputed_entries": int(ranges.n_imputed.sum()),
        "out_of_range_imputed_entries": int(
            ranges.n_below_range.sum() + ranges.n_above_range.sum()
        ),
        "fit_n_country_codes": len(y003_observed_counts),
        "whole_fit_entity_y003_missing": int(y003_observed_counts.eq(0).sum()),
        "whole_mapped_country_y003_missing": int(y003_observed_counts.loc[mapped].eq(0).sum()),
        "y003_remaining_missing_rows": int(observed.Y003.isna().sum()),
        "frozen_clipping_country_shift_median": float(np.median(clipping_shift)),
        "frozen_clipping_country_shift_max": float(clipping_shift.max()),
        "y003_masked_observed_rows": int(present.sum()),
        "y003_masked_country_mean_absolute_error_median": float(
            masking.prediction_error.abs().median()
        ),
        "y003_masked_country_mean_absolute_error_max": float(masking.prediction_error.abs().max()),
        "y003_masked_single_item_coordinate_shift_median": float(
            masking.single_item_coordinate_shift.median()
        ),
        "y003_masked_single_item_coordinate_shift_max": float(
            masking.single_item_coordinate_shift.max()
        ),
        "masking_scope": "in-sample frozen-fit diagnostic, not held-out predictive validation",
        "clipping_scope": "completed-value perturbation, not a refitted estimator or prescribed correction",
    }


def main():
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
    cm.prepare_data()
    if cm.survey_preparation_report["y003"]["missing_constituent_columns"]:
        raise ValueError("Y003 constituent columns are missing from the harmonized inputs")
    if cm.survey_preparation_report["y003"]["discordant_direct"]:
        raise ValueError("delivered Y003 disagrees with valid constituents")
    cm.load_model("data/cultural_map_model.npz")
    observed = cm.subset_ivs_df
    if len(observed) != cm.ppca.n_informative_rows_:
        raise ValueError("prepared respondents do not match saved fit count; rerun validate")
    raw = observed[IV_QNS].to_numpy()
    np.testing.assert_allclose(np.nanmean(raw, axis=0), cm.ppca.means, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.nanstd(raw, axis=0), cm.ppca.stds, rtol=0, atol=1e-12)
    standardized = (raw - cm.ppca.means) / cm.ppca.stds
    completed = conditional_complete(standardized, cm.ppca.loadings_, cm.ppca.noise_variance_)
    xy = cm._rescale(completed @ cm.ppca.C @ cm.rotation)[XY].to_numpy()
    positions = observed[["country_code", "year", "weight"]].reset_index(drop=True)
    positions[XY] = xy
    points = pd.read_csv("data/llm_ellipses_2026.csv", float_precision="round_trip")
    replicates = pd.read_csv("data/llm_bootstrap_replicates_2026.csv", float_precision="round_trip")
    grid = pd.read_csv("data/validation_rotation_sensitivity.csv", float_precision="round_trip")
    if not grid.fit_n_rows.eq(len(observed)).all():
        raise ValueError("rotation metadata do not match prepared respondents")
    rotations = rotation_outcomes(grid, cm, points, replicates)
    rotations.to_csv("data/validation_rotation_outcomes_2026.csv", index=False)
    comparison, nearest, yearly = country_year_benchmarks(positions, cm.country_codes, points)
    # A reconstruction check guards against row alignment or weight drift.
    canonical = pd.read_csv(
        "data/corrected_country_scores.csv", float_precision="round_trip"
    ).set_index("country_code")
    np.testing.assert_allclose(
        comparison[[f"{q}_pooled" for q in XY]].to_numpy(),
        canonical.loc[comparison.country_code, XY].to_numpy(),
        atol=1e-10,
    )
    comparison.to_csv("data/validation_country_latest_year.csv", index=False)
    nearest.to_csv("data/validation_country_latest_nearest_2026.csv", index=False)
    yearly.to_csv("data/validation_country_year_aggregates.csv", index=False)
    baseline = uniform_choice_baseline(cm)
    baseline.to_csv("data/validation_uniform_choice_baseline.csv", index=False)
    summary = {
        "fit_n_rows": len(observed),
        "mapped_countries": len(comparison),
        "country_year_groups": len(yearly),
        "latest_year_country_shift_median": float(comparison.coordinate_shift.median()),
        "latest_year_country_shift_max": float(comparison.coordinate_shift.max()),
        "latest_year_nearest_labels_changed": int(nearest.label_changed.sum()),
        "n_model_cells": len(points),
        "latest_year_scope": "alternative dated benchmark; fixed fit and model points; S017 preserved",
    }
    summary.update(imputation_diagnostics(cm, standardized, completed, positions))
    pd.DataFrame({"quantity": summary.keys(), "value": summary.values()}).to_csv(
        "data/validation_instrument_summary.csv", index=False
    )
    print(rotations.to_string(index=False))
    print(baseline.to_string(index=False))
    print(pd.Series(summary).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
