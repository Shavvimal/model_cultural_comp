"""Scientific invariants for the frozen-instrument sensitivity operators."""

from copy import copy
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from app.culture_map import IV_QNS, SURVEY_REFERENCE
from app.instrument_sensitivity import (
    SLOPES,
    XY,
    conditional_complete,
    country_year_benchmarks,
    matrix_from_row,
    oriented_rotation,
    rotation_grid,
    rotation_outcomes,
    transport_coordinates,
    uniform_choice_baseline,
)
from scripts.instrument_sensitivity import imputation_diagnostics
from scripts.validate_projection import write_validation_summary


def test_transport_matches_direct_projection_and_population_sd(fitted_map):
    cm = fitted_map
    grid = rotation_grid(cm)
    scores = cm.ppca.transform()
    xy = cm._rescale(scores @ cm.rotation)[XY].to_numpy()
    for _, row in grid.iterrows():
        rotation = matrix_from_row(row)
        stds = row[["score_sd_pc1", "score_sd_pc2"]].to_numpy(dtype=float)
        direct = scores @ rotation
        np.testing.assert_allclose(stds, direct.std(axis=0, ddof=0), atol=1e-12)
        transported = transport_coordinates(xy, cm.rotation, cm.score_stds, rotation, stds)
        np.testing.assert_allclose(
            transported, direct / stds * SLOPES + SURVEY_REFERENCE, atol=1e-12
        )
        coefficients = cm.ppca.C @ rotation
        assert coefficients[IV_QNS.index("F118"), 0] > 0
        assert coefficients[IV_QNS.index("F063"), 1] < 0


def test_anchor_orientation_handles_swaps_and_reflections(fitted_map):
    cm = fitted_map
    for swap in (False, True):
        for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            rotation = cm.rotation[:, ::-1] if swap else cm.rotation
            np.testing.assert_allclose(oriented_rotation(cm.ppca.C, rotation * signs), cm.rotation)


def test_conditional_completion_matches_fit_and_scalar_gaussian_formula(fitted_map):
    cm = fitted_map
    raw = cm.subset_ivs_df[IV_QNS].to_numpy()
    z = (raw - cm.ppca.means) / cm.ppca.stds
    actual = conditional_complete(z, cm.ppca.loadings_, cm.ppca.noise_variance_)
    np.testing.assert_allclose(actual, cm.ppca.data, atol=1e-12)
    # C12/C11 = (2*3)/(2**2+5) = 2/3; a zero-observation row predicts zero.
    got = conditional_complete([[6, np.nan], [np.nan, np.nan], [2, 7]], np.array([[2.0], [3.0]]), 5)
    np.testing.assert_allclose(got, [[6, 4], [0, 0], [2, 7]])


def test_latest_year_preserves_weights_and_changes_only_country_benchmark():
    positions = pd.DataFrame(
        {
            "country_code": [1, 1, 1, 2, 2],
            "year": [2005, 2020, 2020, 2005, 2019],
            "weight": [6.0, 1.0, 3.0, np.nan, 1.0],
            "PC1_rescaled": [0.0, 3.0, 7.0, 4.0, 4.0],
            "PC2_rescaled": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    countries = pd.DataFrame({"Numeric": [1, 2], "Country": ["A", "B"]})
    points = pd.DataFrame({"llm": ["m"], "PC1_rescaled": [2.5], "PC2_rescaled": [0.0]})
    snapshot = points.copy(deep=True)
    comparison, nearest, yearly = country_year_benchmarks(positions, countries, points)
    a = comparison.set_index("country_code").loc[1]
    assert a.PC1_rescaled_pooled == 2.4
    assert a.PC1_rescaled_latest == 6
    assert a.year == 2020
    assert nearest.nearest_pooled.iloc[0] == "A"
    assert nearest.nearest_latest_year.iloc[0] == "B"
    assert nearest.label_changed.iloc[0]
    assert len(yearly) == 4
    pd.testing.assert_frame_equal(snapshot, points)


def test_rotation_outcomes_count_strict_boundaries_and_reject_unmatched_cells(fitted_map):
    cm = fitted_map
    grid = rotation_grid(cm).iloc[:1]
    points = pd.DataFrame(
        {
            "llm": ["inside", "outside", "boundary"],
            "PC1_rescaled": [1.0, -1.0, SURVEY_REFERENCE[0]],
            "PC2_rescaled": [1.0, -1.0, SURVEY_REFERENCE[1]],
        }
    )
    result = rotation_outcomes(grid, cm, points, points).iloc[0]
    assert result.n_point_means_in_quadrant == 1
    assert result.n_draws_in_quadrant == 1
    assert result.outside_point_labels == "outside|boundary"
    with pytest.raises(ValueError, match="eligible"):
        rotation_outcomes(grid, cm, points.iloc[:1], points)


def test_country_item_coverage_reconciles_remaining_missingness_and_mapping(
    fitted_map, tmp_path, monkeypatch
):
    cm = copy(fitted_map)
    cm.country_codes = cm.country_codes.iloc[1:].copy()
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    observed = cm.subset_ivs_df
    standardized = (observed[IV_QNS].to_numpy() - cm.ppca.means) / cm.ppca.stds
    positions = observed[["country_code", "year", "weight"]].reset_index(drop=True)
    positions[XY] = cm._rescale(cm.ppca.transform() @ cm.rotation)[XY]
    summary = imputation_diagnostics(cm, standardized, cm.ppca.data, positions)
    coverage = pd.read_csv("data/validation_country_item_coverage.csv")
    assert len(coverage) == observed.country_code.nunique() * len(IV_QNS)
    assert coverage.n_missing.sum() == observed[IV_QNS].isna().sum().sum()
    assert summary["remaining_imputed_entries"] == coverage.n_missing.sum()
    assert len(coverage.loc[~coverage.is_mapped]) == len(IV_QNS)
    assert (coverage.n_observed + coverage.n_missing == coverage.n_respondents).all()
    np.testing.assert_allclose(
        coverage.missing_fraction, coverage.n_missing / coverage.n_respondents
    )
    assert coverage.scope.str.contains("observed includes reconstructed").all()


def test_uniform_valid_choice_baseline_has_exact_midpoint_expectation(fitted_map):
    cm = fitted_map
    assert np.mean([cm.y002_transform(pair) for pair in combinations(range(1, 5), 2)]) == 2
    assert np.mean([cm.y003_transform(list(v)) for v in combinations(range(1, 12), 5)]) == 0
    result = uniform_choice_baseline(cm, cells=1000, responses=5, seed=4)
    again = uniform_choice_baseline(cm, cells=1000, responses=5, seed=4)
    pd.testing.assert_frame_equal(result, again)
    midpoint = pd.DataFrame([[2.5, 1.5, 2, 2, 5.5, 5.5, 5.5, 2.5, 2, 0]], columns=IV_QNS)
    expected = cm.project(midpoint)[XY].iloc[0].to_numpy()
    np.testing.assert_allclose(result[["expected_pc1", "expected_pc2"]].iloc[0], expected)
    assert 0 <= result.fraction_in_quadrant.iloc[0] <= 1


def test_numeric_validation_summary_serializes_boolean_metadata_as_integer_flags(tmp_path):
    path = tmp_path / "summary.csv"
    write_validation_summary(
        {
            "preparation_y003_input_index_column_present": True,
            "other_flag": np.bool_(False),
            "rows": 392382,
            "ppca_max_abs_mean_nll_gradient": 4.2e-8,
        },
        str(path),
    )
    summary = pd.read_csv(path).set_index("quantity").value
    assert pd.api.types.is_numeric_dtype(summary)
    assert summary["preparation_y003_input_index_column_present"] == 1
    assert summary["other_flag"] == 0
    assert summary["ppca_max_abs_mean_nll_gradient"] <= 1e-7
    assert "rows,392382\n" in path.read_text()
    with pytest.raises(ValueError, match="export text separately"):
        write_validation_summary({"provenance": "text belongs in its own file"}, str(path))
