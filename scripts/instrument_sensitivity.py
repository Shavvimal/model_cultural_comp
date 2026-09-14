"""Offline instrument diagnostics after corrected fit and 2026 bootstrap.

Writes aggregate CSVs only; no respondent records or new API calls. All
diagnostics use the corrected preparation, then hold the saved fit fixed.
"""

import numpy as np
import pandas as pd

from app.culture_map import IV_QNS, CulturalMap, check_preparation
from app.instrument_sensitivity import (
    XY,
    conditional_complete,
    country_year_benchmarks,
    imputation_diagnostics,
    rotation_outcomes,
    uniform_choice_baseline,
)


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
    cm.prepare_data()
    check_preparation(cm.survey_preparation_report)
    cm.load_model("data/cultural_map_model.npz")
    observed = cm.subset_ivs_df
    if len(observed) != cm.ppca.n_informative_rows_:
        raise ValueError("prepared respondents do not match saved fit count; rerun validate")
    raw = observed[IV_QNS].to_numpy()
    np.testing.assert_allclose(np.nanmean(raw, axis=0), cm.ppca.means, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.nanstd(raw, axis=0), cm.ppca.stds, rtol=0, atol=1e-12)
    standardized = (raw - cm.ppca.means) / cm.ppca.stds
    completed = conditional_complete(standardized, cm.ppca.loadings_, cm.ppca.noise_variance_)
    xy = cm.rescale(completed @ cm.ppca.C @ cm.rotation)[XY].to_numpy()
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
    ranges, coverage, masking, imputation_summary = imputation_diagnostics(
        cm, standardized, completed, positions
    )
    ranges.to_csv("data/validation_imputation_ranges.csv", index=False)
    coverage.to_csv("data/validation_country_item_coverage.csv", index=False)
    masking.to_csv("data/validation_y003_masking_country.csv", index=False)
    summary.update(imputation_summary)
    pd.DataFrame({"quantity": summary.keys(), "value": summary.values()}).to_csv(
        "data/validation_instrument_summary.csv", index=False
    )
    print(rotations.to_string(index=False))
    print(baseline.to_string(index=False))
    print(pd.Series(summary).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
