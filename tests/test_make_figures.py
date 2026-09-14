"""Figure 0's title range comes from the retained country-year aggregates."""

import numpy as np
import pandas as pd
import pytest

from scripts.make_figures import survey_year_range


def _countries(codes):
    return pd.DataFrame({"country_code": codes, "PC1_rescaled": 0.0, "PC2_rescaled": 0.0})


def test_year_range_spans_plotted_countries_only():
    yearly = pd.DataFrame(
        {
            "country_code": [8.0, 8.0, 356.0, 356.0, 999.0],
            "year": [2008.0, 2018.0, 2005.0, 2023.0, 2030.0],
        }
    )
    # 999 is an unmapped entity: it is not drawn, so its year must not widen the title.
    assert survey_year_range(yearly, _countries([8.0, 356.0])) == (2005, 2023)


@pytest.mark.parametrize(
    ("yearly", "match"),
    [
        (pd.DataFrame({"country_code": [8.0], "wave": [2008.0]}), "year columns"),
        (pd.DataFrame({"country_code": [1.0], "year": [2008.0]}), "no country-year aggregate"),
        (pd.DataFrame({"country_code": [8.0, 8.0], "year": [2008.0, np.nan]}), "finite whole"),
        (pd.DataFrame({"country_code": [8.0], "year": [2008.5]}), "finite whole"),
    ],
)
def test_year_range_rejects_unusable_aggregates(yearly, match):
    with pytest.raises(ValueError, match=match):
        survey_year_range(yearly, _countries([8.0]))


def test_main_fails_loudly_without_year_aggregates(tmp_path, monkeypatch):
    from scripts import make_figures

    (tmp_path / "data").mkdir()
    _countries([8.0]).to_csv(tmp_path / "data/corrected_country_scores.csv", index=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="instrument_sensitivity"):
        make_figures.main()
    assert not (tmp_path / "figures").exists()
