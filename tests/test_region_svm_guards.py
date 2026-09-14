"""Guards on the region classifier's reported score and point lookup."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from app.region_svm import CV_FOLDS, CV_RANDOM_STATE, RegionClassifier


@pytest.fixture(scope="module")
def countries():
    rng = np.random.default_rng(12)
    centres = {
        "Confucian": (-1.0, 1.0),
        "Protestant Europe": (1.5, 1.0),
        "Latin America": (0.0, -1.0),
    }
    rows = []
    for region, (x, y) in centres.items():
        for _ in range(12):
            rows.append(
                {
                    "PC1_rescaled": x + rng.normal(scale=0.6),
                    "PC2_rescaled": y + rng.normal(scale=0.6),
                    "Cultural Region": region,
                }
            )
    return pd.DataFrame(rows)


def test_reported_score_equals_refit_cross_validation(countries):
    clf = RegionClassifier().fit(countries)
    xy = countries[["PC1_rescaled", "PC2_rescaled"]].to_numpy(dtype=float)
    codes = pd.Categorical(countries["Cultural Region"]).codes.astype(int)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    assert clf.cv_accuracy == float(cross_val_score(clf.svm, xy, codes, cv=cv).mean())


def test_duplicate_point_estimates_raise_a_named_error(countries):
    clf = RegionClassifier().fit(countries)
    boot = pd.DataFrame({"llm": ["m", "m"], "PC1_rescaled": [0.0, 0.1], "PC2_rescaled": [0.0, 0.1]})
    points = pd.DataFrame(
        {"llm": ["m", "m"], "PC1_rescaled": [0.0, 0.2], "PC2_rescaled": [0.0, 0.2]}
    )
    with pytest.raises(ValueError, match=r"one row per llm; duplicated \['m'\]"):
        clf.region_assignments(boot, point_estimates=points)
    assert len(clf.region_assignments(boot, point_estimates=points.iloc[:1])) == 1
