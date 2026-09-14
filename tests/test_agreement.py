"""Textbook checks for the agreement statistics."""

import numpy as np
import pytest

from app.agreement import (
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    percent_agreement,
)


class TestCohen:
    def test_perfect_agreement_is_one(self):
        assert cohen_kappa([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0

    def test_chance_level_is_zero(self):
        # Marginals independent: po == pe exactly.
        assert cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)

    def test_matches_sklearn(self):
        pytest.importorskip("sklearn")
        from sklearn.metrics import cohen_kappa_score

        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 200)
        b = np.where(rng.random(200) < 0.8, a, 1 - a)
        assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


class TestFleiss:
    def test_wikipedia_worked_example(self):
        # Fleiss' kappa, Wikipedia: 10 subjects, 14 raters, 5 categories -> 0.210
        counts = np.array(
            [
                [0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7],
            ]
        )
        assert fleiss_kappa(counts) == pytest.approx(0.210, abs=5e-4)

    def test_unequal_rater_counts_rejected(self):
        with pytest.raises(ValueError):
            fleiss_kappa([[2, 1], [1, 1]])


class TestKrippendorff:
    def test_wikipedia_worked_example_nominal(self):
        # Krippendorff's alpha, Wikipedia: three coders, fifteen units,
        # nominal metric -> 0.691
        nan = np.nan
        data = np.array(
            [
                [nan, nan, nan, nan, nan, 3, 4, 1, 2, 1, 1, 3, 3, nan, 3],
                [1, nan, 2, 1, 3, 3, 4, 3, nan, nan, nan, nan, nan, nan, nan],
                [nan, nan, 2, 1, 3, 4, 4, nan, 2, 1, 1, 3, 3, nan, 4],
            ]
        )
        assert krippendorff_alpha_nominal(data) == pytest.approx(0.691, abs=5e-4)

    def test_two_raters_binary_matches_cohen_at_scale(self):
        # For large n, nominal alpha and kappa coincide to O(1/n).
        rng = np.random.default_rng(1)
        a = rng.integers(0, 2, 5000)
        b = np.where(rng.random(5000) < 0.85, a, 1 - a)
        assert krippendorff_alpha_nominal(np.vstack([a, b])) == pytest.approx(
            cohen_kappa(a, b), abs=2e-3
        )


def test_percent_agreement():
    assert percent_agreement([1, 1, 0, 0], [1, 0, 0, 0]) == 0.75


class TestUndefinedAgreement:
    """Kappa and alpha are 0/0 when chance agreement is certain: NaN, not 1.0."""

    def test_constant_raters_give_nan_cohen_kappa(self):
        assert np.isnan(cohen_kappa([0, 0, 0], [0, 0, 0]))

    def test_single_category_fleiss_kappa_is_nan(self):
        assert np.isnan(fleiss_kappa([[3, 0], [3, 0]]))

    def test_single_category_krippendorff_alpha_is_nan(self):
        assert np.isnan(krippendorff_alpha_nominal(np.zeros((3, 4))))
