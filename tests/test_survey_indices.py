"""The survey-side autonomy index must use observed binary constituents."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from app.culture_map import IV_QNS, CulturalMap
from app.survey_indices import Y003_CONSTITUENTS, recover_y003


def test_all_binary_states_match_questionnaire_index_convention():
    """Cross-check longitudinal 0/1 coding against the model-side 1/2 path."""
    states = list(itertools.product([0, 1], repeat=4))
    frame = pd.DataFrame(states, columns=Y003_CONSTITUENTS)
    recovered = recover_y003(frame)
    expected = [
        CulturalMap.y003_transform(
            [choice for choice, mentioned in zip([2, 8, 9, 11], state, strict=True) if mentioned]
        )
        for state in states
    ]
    np.testing.assert_array_equal(recovered.values, expected)
    assert recovered.values.iloc[0] == 0  # all zero is fully observed
    assert recovered.values.min() == -2
    assert recovered.values.max() == 2
    assert recovered.report["reconstructed"] == 16
    assert not recovered.report["input_index_column_present"]


@pytest.mark.parametrize("invalid", [np.nan, -5, -4, -3, -2, -1, 2, 0.5, np.inf])
def test_every_constituent_must_be_binary(invalid):
    frame = pd.DataFrame(0.0, index=range(4), columns=Y003_CONSTITUENTS)
    for position, column in enumerate(Y003_CONSTITUENTS):
        frame.loc[position, column] = invalid
    frame["Y003"] = -3.0
    recovered = recover_y003(frame)
    assert recovered.values.isna().all()
    assert recovered.report["reconstructed"] == 0
    assert recovered.report["still_missing"] == 4


def test_delivered_values_are_preserved_and_concordance_is_reported():
    frame = pd.DataFrame(
        {
            "Y003": [-2, -1, 0, 1, 2, np.nan, -3],
            "A029": [0, 0, 0, 1, 1, 0, 1],
            "A039": [0, 0, 0, 0, 1, 0, 1],
            "A040": [1, 1, 0, 0, 0, 1, 0],
            "A042": [1, 0, 0, 0, 1, 1, 0],
        }
    )
    # Row 4's delivered +2 disagrees with its constituent +1: expose it;
    # choosing silently between competing observed sources would hide it.
    recovered = recover_y003(frame)
    np.testing.assert_array_equal(recovered.values, [-2, -1, 0, 1, 2, -2, 2])
    assert recovered.report["direct"] == 5
    assert recovered.report["comparable_direct"] == 5
    assert recovered.report["concordant_direct"] == 4
    assert recovered.report["discordant_direct"] == 1
    assert recovered.report["reconstructed"] == 2
    assert recovered.report["still_missing"] == 0
    json.dumps(recovered.report, allow_nan=False)


@pytest.mark.parametrize("provided", [[], ["A029", "A039", "A040"]])
def test_legacy_input_without_all_constituent_columns_retains_missingness(provided):
    frame = pd.DataFrame({"Y003": [-2, -1, 0.25, -3, np.nan]})
    for column in provided:
        frame[column] = 1
    recovered = recover_y003(frame)
    np.testing.assert_array_equal(recovered.values[:3], [-2, -1, 0.25])
    assert recovered.values.iloc[3:].isna().all()
    assert recovered.report["reconstructed"] == 0
    assert recovered.report["missing_constituent_columns"] == [
        column for column in Y003_CONSTITUENTS if column not in provided
    ]


def test_recovery_preserves_input_and_nonunique_nonmonotonic_index():
    frame = pd.DataFrame(
        [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]],
        columns=Y003_CONSTITUENTS,
        index=[90, 3, 90],
    )
    frame["Y003"] = pd.array([-3, pd.NA, -3], dtype="Float64")
    before = frame.copy(deep=True)
    recovered = recover_y003(frame)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_index_equal(recovered.values.index, frame.index)
    pd.testing.assert_index_equal(recovered.provenance.index, frame.index)
    np.testing.assert_array_equal(recovered.values, [2, -2, 1])


def test_preparation_recovers_before_eligibility_and_does_not_mutate_input():
    frame = pd.DataFrame(np.nan, index=[40, 5, 18, 2, 97], columns=IV_QNS)
    frame[IV_QNS[:5]] = 1.0
    frame["Y003"] = -3.0
    frame["S020"] = [2005, 2010, 2010, 2004, 2010]
    frame["S003"] = 1
    frame["S017"] = 1.0
    for column, value in zip(Y003_CONSTITUENTS, [1, 1, 0, 0], strict=True):
        frame[column] = value
    frame.loc[5, "A040"] = -5  # incomplete constituent set cannot rescue this row
    frame.loc[18, "Y003"] = -1  # valid delivered negative is preserved
    frame.loc[18, "A040"] = -5  # cannot compare this delivered value
    frame.loc[97, "A008"] = -5  # recovery still leaves fewer than six items
    before = frame.copy(deep=True)
    cm = CulturalMap(frame, pd.DataFrame())
    cm.prepare_data()
    pd.testing.assert_frame_equal(frame, before)
    assert list(cm.subset_ivs_df.index) == [40, 18]
    np.testing.assert_array_equal(cm.subset_ivs_df["Y003"], [2, -1])
    assert list(cm.subset_ivs_df.columns) == ["year", "country_code", "weight", *IV_QNS]
    assert cm.sentinel_counts["Y003"] == 3
    report = cm.survey_preparation_report
    assert report["post_2005_rows"] == 4
    assert report["eligible_before_y003_recovery"] == 1
    assert report["eligible_after_y003_recovery"] == 2
    assert report["added_eligible_rows"] == 1
    assert report["y003"]["direct"] == 1
    assert report["y003"]["reconstructed"] == 2
    assert report["y003"]["still_missing"] == 1
    assert report["retained_y003"] == {"direct": 1, "reconstructed": 1, "still_missing": 0}
    json.dumps(report, allow_nan=False)
    first_preparation = cm.subset_ivs_df.copy()
    cm.prepare_data()
    pd.testing.assert_frame_equal(cm.subset_ivs_df, first_preparation)
    assert cm.survey_preparation_report == report


def test_preparation_supports_absent_delivered_index_column():
    frame = pd.DataFrame(1.0, index=[8], columns=[qn for qn in IV_QNS if qn != "Y003"])
    frame["S020"] = 2005
    frame["S003"] = 1
    frame["S017"] = 1.0
    for column in Y003_CONSTITUENTS:
        frame[column] = 0
    cm = CulturalMap(frame, pd.DataFrame())
    cm.prepare_data()
    assert cm.subset_ivs_df.loc[8, "Y003"] == 0
    assert not cm.survey_preparation_report["y003"]["input_index_column_present"]
    assert cm.survey_preparation_report["retained_y003"]["reconstructed"] == 1


@pytest.mark.parametrize("dtype", ["uint8", "int8", "int64", "float64"])
def test_reconstruction_is_signed_for_every_constituent_dtype(dtype):
    """Unsigned subtraction would wrap -2 and -1 to 254 and 255."""
    frame = pd.DataFrame(
        [[0, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
        columns=Y003_CONSTITUENTS,
    ).astype(dtype)
    recovered = recover_y003(frame)
    np.testing.assert_array_equal(recovered.values, [-2, -1, 2, 0])
    assert recovered.values.dtype == np.float64
    assert recovered.report["reconstructed"] == 4
