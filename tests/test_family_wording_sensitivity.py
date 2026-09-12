"""Inference-unit and multiplicity contracts of the final-review sensitivities."""

import numpy as np
import pandas as pd
import pytest

from app.culture_map import IV_QNS
from scripts.family_wording_sensitivity import family_means, item_signs


def test_repeated_releases_get_one_family_vote():
    frame = pd.DataFrame(
        {"A165": [3.0, 3.0, 3.0, -1.0]},
        index=["deepseek-v4-flash", "deepseek-v4-flash:0731", "deepseek-v4-pro", "glm-5.1"],
    )
    grouped = family_means(frame)
    assert len(grouped) == 2
    assert grouped["A165"].mean() == 1.0
    assert frame["A165"].mean() == 2.0
    with pytest.raises(ValueError, match="explicit family"):
        family_means(frame.rename(index={"glm-5.1": "unclassified-model"}))


def test_tied_items_stay_in_the_ten_test_adjustment():
    frame = pd.DataFrame(0.0, index=range(8), columns=IV_QNS)
    frame["A165"] = 1.0
    result = item_signs(frame, "synthetic").set_index("question")
    assert result.loc["A165", "p_sign_two_sided"] == 2 / 256
    assert result.loc["A165", "p_bh"] == 20 / 256
    assert result.loc["Y003", "n_effective"] == 0
    assert result.loc["Y003", "p_bh"] == 1


def test_targeted_exclusion_does_not_remove_other_item_comparisons():
    frame = pd.DataFrame(1.0, index=range(8), columns=IV_QNS)
    frame.loc[range(4), "Y003"] = np.nan
    result = item_signs(frame, "synthetic").set_index("question")
    assert result.loc["Y003", "n_units"] == 4
    assert result.loc["A165", "n_units"] == 8
    with pytest.raises(ValueError, match="exactly the ten"):
        item_signs(frame.drop(columns="A008"), "synthetic")
