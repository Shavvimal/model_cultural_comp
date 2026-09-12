"""Bounded diagnostics of a frozen cultural-map instrument.

Rotations change only the orientation within the fitted two-dimensional
subspace. Country-year comparisons change only the dated descriptive benchmark.
Neither supplies an externally calibrated alternative IW map.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from factor_analyzer import Rotator

from app.culture_map import ITEM_VALID_RANGES, IV_QNS, PC_RESCALE_PARAMS, SURVEY_REFERENCE

XY = ["PC1_rescaled", "PC2_rescaled"]
SLOPES = np.array([PC_RESCALE_PARAMS[pc][0] for pc in ("PC1", "PC2")])


def oriented_rotation(basis, rotation):
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


def rotation_grid(cm):
    """Persist full oriented matrices, avoiding angle-only reconstruction.

    Every orientation scores the original (unwhitened) fitted subspace and
    uses its own population score SDs. Whitening changes the selection
    criterion only, not the scoring operator. The angle is pre-anchoring,
    retained for compatibility with the previous reporting table. Historical
    CSV keys saying "loadings C" name score-axis bases, not Gaussian loadings
    or item/score correlations; manuscript tables label these as bases.
    """
    scores = cm.ppca.transform()
    eig = cm.ppca.eig_vals
    criteria = {
        "scores (Kaiser)": (scores, True),
        "scores (no Kaiser)": (scores, False),
        "whitened scores (Kaiser)": (scores / np.sqrt(eig), True),
        "loadings C (Kaiser)": (cm.ppca.C, True),
        "loadings C*sqrt(eig) (Kaiser)": (cm.ppca.C * np.sqrt(eig), True),
        "loadings C*sqrt(eig) (no Kaiser)": (cm.ppca.C * np.sqrt(eig), False),
    }
    rows = []
    for criterion, (values, kaiser) in criteria.items():
        rotator = Rotator(method="varimax", normalize=kaiser)
        rotator.fit_transform(values)
        rotation = oriented_rotation(cm.ppca.C, rotator.rotation_)
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
    np.testing.assert_allclose(matrix_from_row(primary), cm.rotation, atol=1e-10)
    np.testing.assert_allclose(
        primary[["score_sd_pc1", "score_sd_pc2"]].astype(float), cm.score_stds
    )
    return result


def matrix_from_row(row):
    """Read an orthogonal matrix, including possible reflection, from a row."""
    matrix = np.array([[row[f"r_{i}{j}"] for j in range(2)] for i in range(2)], dtype=float)
    if not np.allclose(matrix.T @ matrix, np.eye(2), atol=1e-10):
        raise ValueError("rotation matrix must be orthogonal")
    return matrix


def transport_coordinates(xy, primary_rotation, primary_stds, rotation, stds):
    """Undo the primary affine map and apply the alternative frozen orientation."""
    stds = np.asarray(stds, dtype=float)
    if stds.shape != (2,) or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("two positive finite score SDs are required")
    scores = (np.asarray(xy) - SURVEY_REFERENCE) / SLOPES * primary_stds @ primary_rotation.T
    return scores @ rotation / stds * SLOPES + SURVEY_REFERENCE


def rotation_outcomes(grid, cm, points, replicates):
    """Point and finite-bootstrap quadrant outcomes for all six orientations."""
    if points.llm.duplicated().any() or set(replicates.llm) != set(points.llm):
        raise ValueError("replicate and unique eligible point labels must match exactly")
    if (
        not np.isfinite(points[XY].to_numpy()).all()
        or not np.isfinite(replicates[XY].to_numpy()).all()
    ):
        raise ValueError("all point and replicate coordinates must be finite")
    rows = []
    for _, row in grid.iterrows():
        rotation = matrix_from_row(row)
        stds = row[["score_sd_pc1", "score_sd_pc2"]].to_numpy(dtype=float)
        xy = transport_coordinates(points[XY], cm.rotation, cm.score_stds, rotation, stds)
        cloud = transport_coordinates(replicates[XY], cm.rotation, cm.score_stds, rotation, stds)
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


def conditional_complete(standardized, loadings, noise):
    """Frozen Gaussian conditional means; observed entries are never replaced.

    This is a diagnostic application of the fitted PPCA covariance, not a
    second estimation procedure. Entirely unobserved rows predict zero.
    """
    values = np.array(standardized, dtype=float, copy=True)
    if np.isinf(values).any() or not np.isfinite(noise) or noise <= 0:
        raise ValueError("finite values or NaN and positive Gaussian noise are required")
    patterns, membership = np.unique(np.isfinite(values), axis=0, return_inverse=True)
    for i, observed in enumerate(patterns):
        if observed.all():
            continue
        rows = np.flatnonzero(membership == i)
        if not observed.any():
            values[rows] = 0.0
            continue
        w = loadings[observed]
        gain = np.linalg.solve(w @ w.T + noise * np.eye(len(w)), w @ loadings[~observed].T)
        values[np.ix_(rows, ~observed)] = values[np.ix_(rows, observed)] @ gain
    return values


def weighted_coordinates(frame, groups):
    """S017-weighted aggregates; same missing-weight=1 convention as the map."""
    values = frame.copy()
    values["weight"] = values["weight"].fillna(1.0)
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


def country_year_benchmarks(positions, country_codes, points):
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


def uniform_choice_baseline(cm, *, cells=100_000, responses=50, seed=11092026):
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
