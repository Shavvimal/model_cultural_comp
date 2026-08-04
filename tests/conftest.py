"""Synthetic fixtures: no real IVS data is available in CI (the WVS/GESIS
data-use agreements forbid redistribution), so everything here is generated."""

import numpy as np
import pandas as pd
import pytest

from app.culture_map import IV_QNS

N_COUNTRIES = 8
ROWS_PER_COUNTRY = 250


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def synthetic_ivs(rng):
    """An IVS-shaped frame with planted 2-factor structure and some NaNs.

    Values are squashed into each item's valid range so the sentinel recode
    in prepare_data() leaves the planted structure intact.
    """
    from app.culture_map import ITEM_VALID_RANGES

    n = N_COUNTRIES * ROWS_PER_COUNTRY
    latent = rng.standard_normal((n, 2))
    weights = rng.uniform(-1, 1, size=(2, len(IV_QNS)))
    x = latent @ weights + 0.3 * rng.standard_normal((n, len(IV_QNS)))

    df = pd.DataFrame(x, columns=IV_QNS)
    for qn in IV_QNS:
        lo, hi = ITEM_VALID_RANGES[qn]
        col = df[qn]
        df[qn] = lo + (hi - lo) * (col - col.min()) / (col.max() - col.min())
    df["S020"] = 2010
    df["S003"] = np.repeat(np.arange(1, N_COUNTRIES + 1), ROWS_PER_COUNTRY)
    df["S017"] = 1.0

    # knock out ~5% of item values, but never more than 4 of 10 per row,
    # so every row survives the thresh=6 filter
    mask = rng.random(df[IV_QNS].shape) < 0.05
    vals = df[IV_QNS].to_numpy().copy()
    vals[mask] = np.nan
    df[IV_QNS] = vals
    return df


@pytest.fixture(scope="session")
def synthetic_country_codes():
    return pd.DataFrame(
        {
            "Numeric": np.arange(1, N_COUNTRIES + 1),
            "Country": [f"Country {i}" for i in range(1, N_COUNTRIES + 1)],
            "Cultural Region": (
                ["Protestant Europe", "Confucian", "African-Islamic", "Latin America"] * 2
            ),
            "Islamic": [False, False, True, False] * 2,
        }
    )


@pytest.fixture(scope="session")
def fitted_map(synthetic_ivs, synthetic_country_codes):
    from app.culture_map import CulturalMap

    cm = CulturalMap(synthetic_ivs, synthetic_country_codes)
    cm.prepare_data()
    cm.fit(seed=7)
    cm.calculate_mean_scores()
    return cm
