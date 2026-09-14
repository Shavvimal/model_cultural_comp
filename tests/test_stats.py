"""Contracts of the shared exact-inference helpers in app.stats."""

import numpy as np
import pytest
from scipy.stats import binomtest

from app.stats import bh_adjust, permutation_mean_difference, sign_test


def _inline_bh(sorted_p: np.ndarray) -> np.ndarray:
    """The pre-refactor confirmatory_2026 arithmetic, kept as the reference."""
    m = len(sorted_p)
    bh = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(bh, 1.0)


class TestBhAdjust:
    def test_bit_identical_to_replaced_inline_arithmetic_with_ties(self):
        rng = np.random.default_rng(7)
        for _ in range(500):
            m = int(rng.integers(1, 15))
            p = np.sort(rng.choice(np.round(rng.random(6), 3), size=m))
            np.testing.assert_array_equal(bh_adjust(p, m), _inline_bh(p))

    def test_restores_input_order(self):
        p = np.array([0.04, 0.01, 0.03])
        expected = np.empty(3)
        expected[[1, 2, 0]] = _inline_bh(np.sort(p))
        np.testing.assert_array_equal(bh_adjust(p, 3), expected)

    def test_undefined_test_does_not_shrink_declared_family(self):
        corrected = bh_adjust(np.array([0.01, 0.04, np.nan]), family_size=3)
        assert corrected[:2] == pytest.approx([0.03, 0.06])
        assert np.isnan(corrected[2])

    def test_family_size_must_match_supplied_tests(self):
        with pytest.raises(ValueError, match="declared test"):
            bh_adjust(np.full(9, 0.01), family_size=10)

    @pytest.mark.parametrize("family_size", [0, -1, 2.0, True])
    def test_invalid_family_size_rejected(self, family_size):
        with pytest.raises(ValueError, match="family_size"):
            bh_adjust(np.array([0.5, 0.5]), family_size=family_size)

    def test_out_of_range_p_named(self):
        with pytest.raises(ValueError, match=r"p-values must be in \[0, 1\].*1\.5"):
            bh_adjust(np.array([0.1, 1.5]), family_size=2)

    def test_capped_at_one_when_undefined_tests_inflate_the_family(self):
        corrected = bh_adjust(np.array([0.9, np.nan]), family_size=2)
        assert corrected[0] == 1.0


class TestSignTest:
    def test_counts_and_p_match_binomtest(self):
        result = sign_test([1.0, 2.0, -1.0, 0.0, 3.0])
        assert (result.n_positive, result.n_negative) == (3, 1)
        assert (result.n_effective, result.n_ties) == (4, 1)
        assert result.p_value == binomtest(3, 4, 0.5).pvalue

    def test_negated_greater_is_directional_test_for_negatives(self):
        d = np.array([-1.0, -2.0, -0.5, 0.3, -4.0])
        assert sign_test(-d, "greater").p_value == binomtest(4, 5, 0.5, "greater").pvalue

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_all_tied_is_uninformative_not_an_error(self, alternative):
        result = sign_test(np.zeros(6), alternative)
        assert (result.n_effective, result.n_ties, result.p_value) == (0, 6, 1.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_delta_is_not_counted_as_a_sign(self, bad):
        with pytest.raises(ValueError, match="non-finite"):
            sign_test([1.0, bad, -1.0])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            sign_test([])

    def test_unknown_alternative_rejected(self):
        with pytest.raises(ValueError, match="alternative"):
            sign_test([1.0], "two_sided")


def _inline_permutation(v, is_cn, seed, n):
    """The pre-refactor confirmatory_2026 loop, kept as the reference."""
    rng = np.random.default_rng(seed)
    obs = v[is_cn].mean() - v[~is_cn].mean()
    perm = np.empty(n)
    for b in range(n):
        lab = rng.permutation(is_cn)
        perm[b] = v[lab].mean() - v[~lab].mean()
    return obs, float((1 + (np.abs(perm) >= abs(obs)).sum()) / (n + 1))


class TestPermutationMeanDifference:
    def test_identical_to_replaced_loop_and_stream(self):
        rng = np.random.default_rng(3)
        v = rng.normal(size=16)
        is_cn = np.array([True] * 10 + [False] * 6)
        expected = _inline_permutation(v, is_cn, 42, 999)
        assert permutation_mean_difference(v, is_cn, np.random.default_rng(42), 999) == expected

    def test_p_has_plus_one_floor(self):
        v = np.array([10.0, 10.0, 0.0, 0.0])
        _, p = permutation_mean_difference(
            v, np.array([True, True, False, False]), np.random.default_rng(0), 50
        )
        assert p >= 1 / 51

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_values_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            permutation_mean_difference(
                [1.0, bad, 2.0, 3.0], [True, True, False, False], np.random.default_rng(0), 10
            )

    @pytest.mark.parametrize("labels", [[True] * 4, [False] * 4])
    def test_empty_cohort_rejected(self, labels):
        with pytest.raises(ValueError, match="non-empty"):
            permutation_mean_difference([1.0, 2.0, 3.0, 4.0], labels, np.random.default_rng(0), 10)

    @pytest.mark.parametrize("labels", [[1, 1, 0, 0], ["True", "True", "False", "False"]])
    def test_non_boolean_labels_rejected(self, labels):
        with pytest.raises(ValueError, match="boolean"):
            permutation_mean_difference([1.0, 2.0, 3.0, 4.0], labels, np.random.default_rng(0), 10)

    def test_misaligned_labels_rejected(self):
        with pytest.raises(ValueError, match="aligned"):
            permutation_mean_difference(
                [1.0, 2.0, 3.0], [True, False], np.random.default_rng(0), 10
            )

    def test_invalid_permutation_count_rejected(self):
        with pytest.raises(ValueError, match="n_permutations"):
            permutation_mean_difference([1.0, 2.0], [True, False], np.random.default_rng(0), 0)
