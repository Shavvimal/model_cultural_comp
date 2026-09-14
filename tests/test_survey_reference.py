from copy import copy

import numpy as np
import pandas as pd
import pytest

from app.culture_map import SURVEY_REFERENCE
from app.survey_reference import XY, reference_sensitivity, survey_reference_tables


def test_reference_diagnostics_distinguish_fit_from_mapped_entities(fitted_map):
    cm = copy(fitted_map)
    cm.valid_data = fitted_map.valid_data.copy()
    first_code = cm.valid_data.country_code.iloc[0]
    missing = cm.valid_data.country_code.eq(first_code)
    cm.valid_data.loc[missing, "Numeric"] = np.nan
    # Non-unit S017 so that a weighted/unweighted label swap cannot pass.
    weights = np.random.default_rng(11).uniform(0.1, 4.0, len(cm.valid_data))
    cm.valid_data["weight"] = weights
    summary, refs, items, unmapped = survey_reference_tables(cm)
    assert summary["fit_n_respondents"] == len(cm.valid_data)
    assert summary["mapped_n_respondents"] == int((~missing).sum())
    assert summary["fit_n_country_codes"] == 8
    assert summary["mapped_n_country_codes"] == 7
    assert unmapped.n_respondents.tolist() == [int(missing.sum())]
    refs = refs.set_index("reference")
    # Independent identity: the frozen standardisation means project to the reference.
    projected_means = cm.project(pd.DataFrame([cm.ppca.means], columns=cm.iv_qns))[XY]
    np.testing.assert_allclose(
        refs.loc["survey_reference", XY].to_numpy(dtype=float),
        projected_means.to_numpy()[0],
        atol=1e-12,
    )
    xy = cm.valid_data[XY].to_numpy()
    mapped = (~missing).to_numpy()
    expected = {
        "completed_all_unweighted": xy.mean(axis=0),
        "completed_mapped_unweighted": xy[mapped].mean(axis=0),
        "completed_all_weighted": np.average(xy, axis=0, weights=weights),
        "completed_mapped_weighted": np.average(xy[mapped], axis=0, weights=weights[mapped]),
    }
    for name, point in expected.items():
        np.testing.assert_allclose(refs.loc[name, XY].to_numpy(dtype=float), point)
    assert (
        np.abs(expected["completed_all_weighted"] - expected["completed_all_unweighted"]).min()
        > 1e-4
    )
    assert not refs.loc["scale_midpoint", "is_comparison_reference"]
    assert "not a joint mode" in refs.loc["mode_unweighted", "definition"]
    assert len(items) == 10
    assert items.set_index("question").loc["A165", "scale_midpoint"] == 1.5
    np.testing.assert_array_equal(items["fit_standardisation_mean"], cm.ppca.means)
    np.testing.assert_array_equal(items["fit_standardisation_sd"], cm.ppca.stds)


@pytest.mark.parametrize("bad", [np.nan, -0.5, np.inf])
def test_reference_tables_reject_invalid_weights(fitted_map, bad):
    cm = copy(fitted_map)
    cm.valid_data = fitted_map.valid_data.copy()
    cm.valid_data.loc[cm.valid_data.index[0], "weight"] = bad
    with pytest.raises(ValueError, match="S017 weights must be present"):
        survey_reference_tables(cm)


def test_reference_tables_reject_zero_weight_sum_for_an_item(fitted_map):
    cm = copy(fitted_map)
    cm.subset_ivs_df = fitted_map.subset_ivs_df.copy()
    cm.subset_ivs_df["weight"] = 0.0
    with pytest.raises(ValueError, match="A008 weighted mode: S017 weights sum to zero"):
        survey_reference_tables(cm)


def test_empirical_modes_have_explicit_weighting_and_tie_rule(fitted_map):
    cm = copy(fitted_map)
    cm.subset_ivs_df = fitted_map.subset_ivs_df.copy()
    # Exercise only the aggregation diagnostic: deliberately vary weighting
    # while leaving the already-fitted map fixed, as the helper promises.
    cm.subset_ivs_df["A008"] = 1.0
    cm.subset_ivs_df.iloc[-1, cm.subset_ivs_df.columns.get_loc("A008")] = 4.0
    cm.subset_ivs_df["weight"] = 1.0
    cm.subset_ivs_df.iloc[-1, cm.subset_ivs_df.columns.get_loc("weight")] = len(cm.subset_ivs_df)
    _, _, items, _ = survey_reference_tables(cm)
    happiness = items.set_index("question").loc["A008"]
    assert happiness.mode_unweighted == 1.0
    assert happiness.mode_weighted == 4.0
    assert happiness.mode_unweighted_ties == "1.0"
    # Aggregation modifications must not replace the frozen fit parameters.
    assert happiness.fit_standardisation_mean == fitted_map.ppca.means[0]
    assert happiness.fit_standardisation_sd == fitted_map.ppca.stds[0]


def test_sensitivity_changes_reference_without_moving_points():
    points = pd.DataFrame({"llm": ["m"], "PC1_rescaled": [1.0], "PC2_rescaled": [1.0]})
    before = points.copy(deep=True)
    countries = pd.DataFrame({"PC1_rescaled": [0.0, 2.0], "PC2_rescaled": [0.0, 2.0]})
    refs = pd.DataFrame(
        {
            "reference": ["survey_reference", "alternative"],
            "PC1_rescaled": [SURVEY_REFERENCE[0], 1.1],
            "PC2_rescaled": [SURVEY_REFERENCE[1], 0.0],
            "is_comparison_reference": [True, True],
        }
    )
    result = reference_sensitivity(points, countries, refs, replicates=points).set_index(
        "reference"
    )
    assert result.loc["survey_reference", "distance_change_from_fixed_reference"] == 0
    assert not result.loc["survey_reference", "quadrant_changed_from_fixed_reference"]
    assert result.loc["alternative", "quadrant_changed_from_fixed_reference"]
    assert result.loc["alternative", "replicates_outside_quadrant"] == 1
    assert result.loc["alternative", "point_distance"] == pytest.approx(np.sqrt(1.01))
    pd.testing.assert_frame_equal(points, before)


@pytest.mark.parametrize("frame", ["points", "replicates"])
def test_sensitivity_rejects_non_finite_coordinates(frame):
    points = pd.DataFrame(
        {"llm": ["m", "n"], "PC1_rescaled": [1.0, 0.5], "PC2_rescaled": [1.0, 0.5]}
    )
    replicates = points.copy()
    bad = points if frame == "points" else replicates
    bad.loc[1, "PC1_rescaled"] = np.nan
    countries = pd.DataFrame({"PC1_rescaled": [0.0, 2.0], "PC2_rescaled": [0.0, 2.0]})
    refs = pd.DataFrame(
        {
            "reference": ["survey_reference"],
            "PC1_rescaled": [SURVEY_REFERENCE[0]],
            "PC2_rescaled": [SURVEY_REFERENCE[1]],
            "is_comparison_reference": [True],
        }
    )
    with pytest.raises(ValueError, match=r"must be finite; non-finite for \['n'\]"):
        reference_sensitivity(points, countries, refs, replicates)
