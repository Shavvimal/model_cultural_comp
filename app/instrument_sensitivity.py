"""Bounded diagnostics of a frozen cultural-map instrument.

Rotations change only the orientation within the fitted two-dimensional
subspace. Country-year comparisons change only the dated descriptive benchmark.
Neither supplies an externally calibrated alternative IW map.
"""

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from factor_analyzer import Rotator
from numpy.typing import ArrayLike

from app.culture_map import (
    ITEM_VALID_RANGES,
    IV_QNS,
    PC_RESCALE_PARAMS,
    SURVEY_REFERENCE,
    VARIMAX_TOL,
    CulturalMap,
    validate_weights,
)
from app.ppca import PPCA, conditional_complete

__all__ = [
    "SLOPES",
    "XY",
    "conditional_complete",
    "country_year_benchmarks",
    "imputation_diagnostics",
    "matrix_from_row",
    "oriented_rotation",
    "rotation_grid",
    "rotation_outcomes",
    "transport_coordinates",
    "uniform_choice_baseline",
    "weighted_coordinates",
]

XY = ["PC1_rescaled", "PC2_rescaled"]
SLOPES = np.array([PC_RESCALE_PARAMS[pc][0] for pc in ("PC1", "PC2")])
# Illustrative uniform-choice baseline: fixed seed (the 11 Sep 2026 run date),
# simulated cells, and responses per item matching the 2026 design's 50 calls.
UNIFORM_BASELINE_SEED = 11092026
UNIFORM_BASELINE_CELLS = 100_000
UNIFORM_BASELINE_RESPONSES = 50


def _fitted_state(cm: CulturalMap) -> tuple[PPCA, np.ndarray, np.ndarray]:
    """Return the frozen PPCA, rotation and score SDs, or raise if not fitted."""
    ppca = cm.ppca
    if (
        cm.rotation is None
        or cm.score_stds is None
        or ppca.C is None
        or ppca.means is None
        or ppca.stds is None
    ):
        raise RuntimeError("fit() or load_model() the cultural map first")
    return ppca, cm.rotation, cm.score_stds


def oriented_rotation(basis: np.ndarray, rotation: ArrayLike) -> np.ndarray:
    """Apply the primary F118-positive/F063-negative coefficient anchors."""
    result = np.array(rotation, dtype=float, copy=True)
    coefficients = basis @ result
    if abs(coefficients[IV_QNS.index("F118"), 0]) < abs(coefficients[IV_QNS.index("F118"), 1]):
        result = result[:, ::-1]
        coefficients = coefficients[:, ::-1]
    result *= np.array(
        [
            1 if coefficients[IV_QNS.index("F118"), 0] > 0 else -1,
            1 if coefficients[IV_QNS.index("F063"), 1] < 0 else -1,
        ]
    )
    return result


def rotation_grid(cm: CulturalMap) -> pd.DataFrame:
    """Persist full oriented matrices, avoiding angle-only reconstruction.

    Every orientation scores the original (unwhitened) fitted subspace and
    uses its own population score SDs. Whitening changes the selection
    criterion only, not the scoring operator. The angle is pre-anchoring,
    retained for compatibility with the previous reporting table. Historical
    CSV keys saying "loadings C" name score-axis bases, not Gaussian loadings
    or item/score correlations; manuscript tables label these as bases.
    """
    ppca, primary_rotation, primary_stds = _fitted_state(cm)
    if ppca.C is None or ppca.eig_vals is None:
        raise RuntimeError("fit() or load_model() the cultural map first")
    basis = ppca.C
    scores = ppca.transform()
    eig = ppca.eig_vals
    criteria = {
        "scores (Kaiser)": (scores, True),
        "scores (no Kaiser)": (scores, False),
        "whitened scores (Kaiser)": (scores / np.sqrt(eig), True),
        "loadings C (Kaiser)": (basis, True),
        "loadings C*sqrt(eig) (Kaiser)": (basis * np.sqrt(eig), True),
        "loadings C*sqrt(eig) (no Kaiser)": (basis * np.sqrt(eig), False),
    }
    rows = []
    for criterion, (values, kaiser) in criteria.items():
        # Same pinned tolerance as the primary fit; see VARIMAX_TOL.
        rotator = Rotator(method="varimax", normalize=kaiser, tol=VARIMAX_TOL)
        rotator.fit_transform(values)
        rotation = oriented_rotation(basis, rotator.rotation_)
        stds = (scores @ rotation).std(axis=0, ddof=0)
        row = {
            "criterion": criterion,
            "angle_deg": float(
                np.degrees(np.arctan2(rotator.rotation_[1, 0], rotator.rotation_[0, 0]))
            ),
            "fit_n_rows": len(scores),
            "score_sd_pc1": stds[0],
            "score_sd_pc2": stds[1],
            "anchors": "F118 positive PC1; F063 negative PC2",
            "operator": "unwhitened fitted scores; orientation-specific population SDs",
        }
        row.update({f"r_{i}{j}": rotation[i, j] for i in range(2) for j in range(2)})
        rows.append(row)
    result = pd.DataFrame(rows)
    primary = result.iloc[0]
    np.testing.assert_allclose(matrix_from_row(primary), primary_rotation, atol=1e-10)
    np.testing.assert_allclose(
        primary[["score_sd_pc1", "score_sd_pc2"]].astype(float), primary_stds
    )
    return result


def matrix_from_row(row: pd.Series | dict[str, Any]) -> np.ndarray:
    """Read an orthogonal matrix, including possible reflection, from a row."""
    matrix = np.array([[row[f"r_{i}{j}"] for j in range(2)] for i in range(2)], dtype=float)
    if not np.allclose(matrix.T @ matrix, np.eye(2), atol=1e-10):
        raise ValueError("rotation matrix must be orthogonal")
    return matrix


def transport_coordinates(
    xy: ArrayLike,
    primary_rotation: np.ndarray,
    primary_stds: ArrayLike,
    rotation: np.ndarray,
    stds: ArrayLike,
) -> np.ndarray:
    """Undo the primary affine map and apply the alternative frozen orientation."""
    stds = np.asarray(stds, dtype=float)
    if stds.shape != (2,) or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("two positive finite score SDs are required")
    scores = (
        (np.asarray(xy) - SURVEY_REFERENCE) / SLOPES * np.asarray(primary_stds) @ primary_rotation.T
    )
    return scores @ rotation / stds * SLOPES + SURVEY_REFERENCE


def rotation_outcomes(
    grid: pd.DataFrame, cm: CulturalMap, points: pd.DataFrame, replicates: pd.DataFrame
) -> pd.DataFrame:
    """Point and finite-bootstrap quadrant outcomes for all six orientations."""
    if points.llm.duplicated().any() or set(replicates.llm) != set(points.llm):
        raise ValueError("replicate and unique eligible point labels must match exactly")
    if (
        not np.isfinite(points[XY].to_numpy()).all()
        or not np.isfinite(replicates[XY].to_numpy()).all()
    ):
        raise ValueError("all point and replicate coordinates must be finite")
    _, primary_rotation, primary_stds = _fitted_state(cm)
    rows = []
    for _, row in grid.iterrows():
        rotation = matrix_from_row(row)
        stds = row[["score_sd_pc1", "score_sd_pc2"]].to_numpy(dtype=float)
        xy = transport_coordinates(points[XY], primary_rotation, primary_stds, rotation, stds)
        cloud = transport_coordinates(
            replicates[XY], primary_rotation, primary_stds, rotation, stds
        )
        if row.criterion == "scores (Kaiser)":
            np.testing.assert_allclose(xy, points[XY].to_numpy(), rtol=0, atol=1e-10)
            np.testing.assert_allclose(cloud, replicates[XY].to_numpy(), rtol=0, atol=1e-10)
            # Avoid round-trip rounding at a strict boundary in the primary map.
            xy, cloud = points[XY].to_numpy(), replicates[XY].to_numpy()
        inside = (xy > SURVEY_REFERENCE).all(axis=1)
        draw_inside = (cloud > SURVEY_REFERENCE).all(axis=1)
        rows.append(
            {
                "criterion": row.criterion,
                "angle_deg": row.angle_deg,
                "n_point_means": len(points),
                "n_finite_point_means": len(points),
                "n_point_means_in_quadrant": int(inside.sum()),
                "outside_point_labels": "|".join(points.loc[~inside, "llm"]),
                "n_draws": len(replicates),
                "n_finite_draws": len(replicates),
                "min_draws_per_cell": int(replicates.groupby("llm").size().min()),
                "max_draws_per_cell": int(replicates.groupby("llm").size().max()),
                "n_draws_in_quadrant": int(draw_inside.sum()),
                "n_cells_with_any_draw_outside": replicates.loc[~draw_inside, "llm"].nunique(),
                "min_point_margin_pc1": (xy[:, 0] - SURVEY_REFERENCE[0]).min(),
                "min_point_margin_pc2": (xy[:, 1] - SURVEY_REFERENCE[1]).min(),
                "scope": "fixed fitted subspace; same anchors; not equally validated IW constructs",
            }
        )
    return pd.DataFrame(rows)


def weighted_coordinates(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """S017-weighted aggregates under the same weight contract as the map.

    Weights must be present, finite and non-negative (see
    :func:`app.culture_map.validate_weights`); each group needs a positive sum.
    ``conditional_complete`` is re-exported here from :mod:`app.ppca`, where
    the fit itself uses it: a diagnostic application of the fitted PPCA
    covariance, not a second estimation procedure.
    """
    values = frame.copy()
    validate_weights(values["weight"], context="weighted coordinates")
    for name in XY:
        values[f"weighted_{name}"] = values[name] * values.weight
    summed = values.groupby(groups)[["weight", *[f"weighted_{q}" for q in XY]]].sum()
    if (summed.weight <= 0).any():
        raise ValueError("country aggregate weights must have positive sums")
    for name in XY:
        summed[name] = summed.pop(f"weighted_{name}") / summed.weight
    summed = summed.rename(columns={"weight": "weight_sum"})
    summed["n_respondents"] = values.groupby(groups).size()
    return summed.reset_index()


def country_year_benchmarks(
    positions: pd.DataFrame, country_codes: pd.DataFrame, points: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare pooled respondents to each country's latest available year.

    Both benchmarks preserve S017; latest-year is a different dated reference,
    not an estimate of contemporaneous 2026 human coordinates. The frozen
    instrument and all model coordinates remain unchanged.
    """
    pooled = weighted_coordinates(positions, ["country_code"])
    last_year = positions.groupby("country_code").year.transform("max")
    latest_rows = positions.loc[positions.year.eq(last_year)]
    latest = weighted_coordinates(latest_rows, ["country_code", "year"])
    comparison = pooled.merge(latest, on="country_code", suffixes=("_pooled", "_latest"))
    comparison = comparison.merge(
        country_codes[["Numeric", "Country"]], left_on="country_code", right_on="Numeric"
    ).drop(columns="Numeric")
    before = comparison[[f"{q}_pooled" for q in XY]].to_numpy()
    after = comparison[[f"{q}_latest" for q in XY]].to_numpy()
    comparison["coordinate_shift"] = np.linalg.norm(after - before, axis=1)
    if comparison.empty:
        raise ValueError("at least one mapped country is required")
    old_nearest = np.argmin(np.linalg.norm(points[XY].to_numpy()[:, None] - before, axis=2), axis=1)
    new_nearest = np.argmin(np.linalg.norm(points[XY].to_numpy()[:, None] - after, axis=2), axis=1)
    nearest = points[["llm"]].copy()
    nearest["nearest_pooled"] = comparison.Country.to_numpy()[old_nearest]
    nearest["nearest_latest_year"] = comparison.Country.to_numpy()[new_nearest]
    nearest["label_changed"] = nearest.nearest_pooled.ne(nearest.nearest_latest_year)
    nearest["scope"] = "fixed map/model points; alternative dated descriptive country benchmark"
    yearly = weighted_coordinates(positions, ["country_code", "year"])
    return comparison, nearest, yearly


def uniform_choice_baseline(
    cm: CulturalMap,
    *,
    cells: int = UNIFORM_BASELINE_CELLS,
    responses: int = UNIFORM_BASELINE_RESPONSES,
    seed: int = UNIFORM_BASELINE_SEED,
) -> pd.DataFrame:
    """Illustrative content-free choices, independent over responses and items.

    Y002 samples distinct pairs uniformly; Y003 samples exactly five of eleven
    qualities uniformly. This is one explicit valid-choice mechanism, not a
    calibrated null model, a human population, or a test of alignment.
    """
    if cells < 2 or responses < 1:
        raise ValueError("at least two cells and one response are required")
    rng = np.random.default_rng(seed)
    means = np.empty((cells, len(IV_QNS)))
    expectations = []
    for j, question in enumerate(IV_QNS):
        options: list[float]
        if question == "Y002":
            options = [cm.y002_transform(pair) for pair in combinations(range(1, 5), 2)]
        elif question == "Y003":
            options = [cm.y003_transform(list(five)) for five in combinations(range(1, 12), 5)]
        else:
            lo, hi = ITEM_VALID_RANGES[question]
            options = list(range(lo, hi + 1))
        means[:, j] = rng.choice(options, size=(cells, responses)).mean(axis=1)
        expectations.append(np.mean(options))
    xy = cm.project(pd.DataFrame(means, columns=IV_QNS))[XY].to_numpy()
    expected = cm.project(pd.DataFrame([expectations], columns=IV_QNS))[XY].iloc[0]
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "simulation_cells": cells,
                "responses_per_item": responses,
                "fraction_in_quadrant": float((xy > SURVEY_REFERENCE).all(axis=1).mean()),
                "expected_pc1": expected.iloc[0],
                "expected_pc2": expected.iloc[1],
                "simulation_mean_pc1": xy[:, 0].mean(),
                "simulation_mean_pc2": xy[:, 1].mean(),
                "simulation_sd_pc1": xy[:, 0].std(ddof=1),
                "simulation_sd_pc2": xy[:, 1].std(ddof=1),
                "scope": "illustrative independent uniform valid choices; not a calibrated null",
                "y003_choice_rule": "exactly five of eleven distinct qualities, uniformly sampled",
            }
        ]
    )


def imputation_diagnostics(
    cm: CulturalMap,
    standardized: np.ndarray,
    completed: np.ndarray,
    positions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    """Range and frozen-fit Y003 masking diagnostics, explicitly in sample.

    Returns ``(ranges, coverage, masking, summary)``: per-item imputed ranges,
    per-country/item coverage, per-country Y003 masking errors, and the scalar
    summary rows. Writes nothing; the caller owns every output file.
    """
    ppca, rotation, score_stds = _fitted_state(cm)
    if (
        cm.subset_ivs_df is None
        or ppca.C is None
        or ppca.means is None
        or ppca.stds is None
        or ppca.loadings_ is None
        or ppca.noise_variance_ is None
    ):
        raise RuntimeError("prepare_data() and a fitted Gaussian model are required")
    observed = cm.subset_ivs_df
    raw = observed[IV_QNS].to_numpy()
    decoded = completed * ppca.stds + ppca.means
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
    predicted = conditional_complete(masked, ppca.loadings_, ppca.noise_variance_)
    predicted = predicted[:, y003] * ppca.stds[y003] + ppca.means[y003]
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
    coefficient = ppca.C[y003] @ rotation / score_stds * SLOPES / ppca.stds[y003]
    masking["single_item_coordinate_shift"] = masking.prediction_error.abs() * np.linalg.norm(
        coefficient
    )
    masking["scope"] = (
        "frozen fit; observed Y003 used in training; in-sample diagnostic, not held out"
    )
    y003_observed_counts = observed.groupby("country_code").Y003.count()
    mapped = y003_observed_counts.index.isin(cm.country_codes.Numeric)
    summary: dict[str, int | float | str] = {
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
    return ranges, coverage, masking, summary
