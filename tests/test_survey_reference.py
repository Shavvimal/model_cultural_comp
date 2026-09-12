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
    summary, refs, items, unmapped = survey_reference_tables(cm)
    assert summary["fit_n_respondents"] == len(cm.valid_data)
    assert summary["mapped_n_respondents"] == int((~missing).sum())
    assert summary["fit_n_country_codes"] == 8
    assert summary["mapped_n_country_codes"] == 7
    assert unmapped.n_respondents.tolist() == [int(missing.sum())]
    refs = refs.set_index("reference")
    np.testing.assert_allclose(
        refs.loc["survey_reference", XY].to_numpy(dtype=float), SURVEY_REFERENCE
    )
    np.testing.assert_allclose(
        refs.loc["completed_mapped_unweighted", XY].to_numpy(dtype=float),
        cm.valid_data.loc[~missing, XY].mean().to_numpy(),
    )
    np.testing.assert_allclose(
        refs.loc["completed_all_weighted", XY].to_numpy(dtype=float),
        np.average(cm.valid_data[XY], axis=0, weights=cm.valid_data.weight.fillna(1)),
    )
    assert not refs.loc["scale_midpoint", "is_comparison_reference"]
    assert "not a joint mode" in refs.loc["mode_unweighted", "definition"]
    assert len(items) == 10
    assert items.set_index("question").loc["A165", "scale_midpoint"] == 1.5
    np.testing.assert_array_equal(items["fit_standardisation_mean"], cm.ppca.means)
    np.testing.assert_array_equal(items["fit_standardisation_sd"], cm.ppca.stds)


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
