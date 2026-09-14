from copy import copy

import numpy as np
import pandas as pd
import pytest

from app.culture_map import (
    HUMAN_MEAN,
    IV_QNS,
    MIN_OBSERVED_ITEMS,
    MIN_SURVEY_YEAR,
    PC_RESCALE_PARAMS,
    SURVEY_REFERENCE,
    VARIMAX_TOL,
    CulturalMap,
    check_preparation,
)


class TestRotationIsFittedOnce:
    def test_projection_does_not_refit_rotation(self, fitted_map, rng):
        """The 2024 defect: varimax was re-fitted on the projected data."""
        before = fitted_map.rotation.copy()
        data = pd.DataFrame(5 + rng.standard_normal((20, len(IV_QNS))), columns=IV_QNS)
        fitted_map.project(data)
        np.testing.assert_array_equal(fitted_map.rotation, before)

    def test_rotation_is_orthogonal(self, fitted_map):
        np.testing.assert_allclose(
            fitted_map.rotation @ fitted_map.rotation.T, np.eye(2), atol=1e-10
        )

    def test_projection_is_deterministic(self, fitted_map, rng):
        data = pd.DataFrame(5 + rng.standard_normal((20, len(IV_QNS))), columns=IV_QNS)
        pd.testing.assert_frame_equal(fitted_map.project(data), fitted_map.project(data))


class TestSelfConsistency:
    def test_complete_rows_reproduce_fitted_coordinates(self, fitted_map):
        """The Phase-1 gate in miniature: one coordinate space for everyone."""
        complete = fitted_map.subset_ivs_df.dropna(subset=IV_QNS)
        projected = fitted_map.project(complete)

        positions = fitted_map.subset_ivs_df.index.get_indexer(complete.index)
        fitted = fitted_map.valid_data.iloc[positions]
        np.testing.assert_allclose(
            projected[["PC1_rescaled", "PC2_rescaled"]].to_numpy(),
            fitted[["PC1_rescaled", "PC2_rescaled"]].to_numpy(),
            atol=1e-8,
        )

    def test_saved_map_retains_fitting_evidence_without_respondents(self, fitted_map, tmp_path):
        path = tmp_path / "map.npz"
        fitted_map.save_model(path)
        with np.load(path, allow_pickle=False) as archive:
            assert "data" not in archive
            assert archive["converged_"].item()
            assert archive["gradient_norm_"].item() <= archive["tolerance_"].item()
            np.testing.assert_array_equal(archive["loadings_"], fitted_map.ppca.loadings_)
        loaded = CulturalMap(pd.DataFrame(), pd.DataFrame())
        loaded.load_model(path)
        assert loaded.ppca.noise_variance_ == fitted_map.ppca.noise_variance_
        assert loaded.ppca.method_ == fitted_map.ppca.method_
        assert loaded.ppca.data is None
        complete = fitted_map.subset_ivs_df.dropna(subset=IV_QNS)
        pd.testing.assert_frame_equal(loaded.project(complete), fitted_map.project(complete))

    def test_legacy_map_does_not_claim_validated_likelihood(self, fitted_map, tmp_path):
        path = tmp_path / "legacy.npz"
        np.savez(
            path,
            C=fitted_map.ppca.C,
            means=fitted_map.ppca.means,
            stds=fitted_map.ppca.stds,
            eig_vals=fitted_map.ppca.eig_vals,
            rotation=fitted_map.rotation,
            score_stds=fitted_map.score_stds,
        )
        loaded = CulturalMap(pd.DataFrame(), pd.DataFrame())
        loaded.load_model(path)
        assert not loaded.ppca.converged_
        assert loaded.ppca.log_likelihood_ is None
        assert loaded.ppca.loadings_ is None


class TestRescaling:
    def test_constants_match_wvs_syntax(self):
        """Pin the affine constants to the WVS Association's published SPSS
        syntax (SurvSAgg = 1.81*SurvSelf + .038; TradAgg = 1.61*TradRat5 - .1).
        Guards against the decimal slip (0.38 / -0.01) inherited from Tao et
        al. (2024) and carried until v1.1.0 (GitHub issue #12)."""
        assert PC_RESCALE_PARAMS["PC1"] == (1.81, 0.038)
        assert PC_RESCALE_PARAMS["PC2"] == (1.61, -0.10)
        assert HUMAN_MEAN == (0.038, -0.10)
        assert HUMAN_MEAN is SURVEY_REFERENCE

    def test_reference_is_projected_observed_item_marginal_means(self, fitted_map):
        means = pd.DataFrame([fitted_map.ppca.means], columns=IV_QNS)
        point = fitted_map.project(means)[["PC1_rescaled", "PC2_rescaled"]].to_numpy()[0]
        np.testing.assert_allclose(point, SURVEY_REFERENCE, atol=1e-14)

    def test_published_wvs_constants(self, fitted_map, rng):
        data = pd.DataFrame(5 + rng.standard_normal((10, len(IV_QNS))), columns=IV_QNS)
        out = fitted_map.project(data)
        a1, b1 = PC_RESCALE_PARAMS["PC1"]
        a2, b2 = PC_RESCALE_PARAMS["PC2"]
        np.testing.assert_allclose(out["PC1_rescaled"], a1 * out["PC1"] + b1)
        np.testing.assert_allclose(out["PC2_rescaled"], a2 * out["PC2"] + b2)


class TestOrientation:
    def test_iw_sign_convention(self, fitted_map):
        """F118 marks self-expression (+PC1); F063 marks traditional (-PC2)."""
        loadings = fitted_map.ppca.C @ fitted_map.rotation
        assert loadings[IV_QNS.index("F118"), 0] > 0
        assert loadings[IV_QNS.index("F063"), 1] < 0


class TestWvsIndexTransforms:
    @pytest.mark.parametrize(
        "ans,expected",
        [
            ((1, 3), 1),
            ((3, 1), 1),  # materialist
            ((2, 4), 3),
            ((4, 2), 3),  # post-materialist
            ((1, 2), 2),
            ((3, 4), 2),  # mixed
        ],
    )
    def test_y002(self, ans, expected):
        assert CulturalMap.y002_transform(ans) == expected

    @pytest.mark.parametrize("ans", [(-1, 3), (0, 2), (1, 5)])
    def test_y002_rejects_out_of_range(self, ans):
        """Out-of-range choices raise; a sentinel would flow into a
        published coordinate because the model path applies no recode."""
        with pytest.raises(ValueError, match="out of range"):
            CulturalMap.y002_transform(ans)

    @pytest.mark.parametrize(
        "choices,expected",
        [
            ([9, 11], 2 - 4),  # faith+obedience mentioned, autonomy not
            ([2, 8], 4 - 2),  # independence+determination mentioned
            ([2, 8, 9, 11], 2 - 2),  # all four mentioned
            ([1, 3, 5], 0),  # none of the four mentioned
        ],
    )
    def test_y003(self, choices, expected):
        assert CulturalMap.y003_transform(choices) == expected


class TestGuards:
    def test_project_requires_fit(self, synthetic_ivs, synthetic_country_codes):
        cm = CulturalMap(synthetic_ivs, synthetic_country_codes)
        with pytest.raises(RuntimeError):
            cm.project(pd.DataFrame(np.zeros((1, len(IV_QNS))), columns=IV_QNS))

    def test_fit_requires_prepare(self, synthetic_ivs, synthetic_country_codes):
        cm = CulturalMap(synthetic_ivs, synthetic_country_codes)
        with pytest.raises(RuntimeError):
            cm.fit()


class TestSentinelRecode:
    def test_out_of_range_values_become_nan(self, synthetic_ivs, synthetic_country_codes):
        """The Y003 bug: SPSS user-missing codes must never count as data."""
        poisoned = synthetic_ivs.copy()
        poisoned.iloc[:50, poisoned.columns.get_loc("Y003")] = -3.0
        cm = CulturalMap(poisoned, synthetic_country_codes)
        cm.prepare_data()
        assert cm.sentinel_counts["Y003"] == 50
        assert not ((cm.subset_ivs_df["Y003"] < -2) | (cm.subset_ivs_df["Y003"] > 2)).any()

    def test_sentinels_do_not_count_toward_completeness(
        self, synthetic_ivs, synthetic_country_codes
    ):
        poisoned = synthetic_ivs.copy()
        # five items set to sentinels: with the recode the row has at most
        # five answered items and must be dropped by the >=6 filter
        for qn in ["A008", "A165", "E018", "E025", "F063"]:
            poisoned.iloc[0, poisoned.columns.get_loc(qn)] = -5.0
        cm = CulturalMap(poisoned, synthetic_country_codes)
        cm.prepare_data()
        assert poisoned.index[0] not in cm.subset_ivs_df.index


class TestPreparationContract:
    @pytest.mark.parametrize(
        "y003,message",
        [
            ({"missing_constituent_columns": ["A042"], "discordant_direct": 0}, "A042"),
            ({"missing_constituent_columns": [], "discordant_direct": 3}, "in 3 rows"),
        ],
    )
    def test_check_preparation_rejects_unusable_y003(self, y003, message):
        with pytest.raises(ValueError, match=message):
            check_preparation({"y003": y003})

    def test_check_preparation_accepts_complete_concordant_inputs(self):
        check_preparation({"y003": {"missing_constituent_columns": [], "discordant_direct": 0}})

    def test_report_states_the_rules_the_filter_used(self, fitted_map):
        report = fitted_map.survey_preparation_report
        assert report["years_min"] == MIN_SURVEY_YEAR == 2005
        assert report["minimum_observed_items"] == MIN_OBSERVED_ITEMS == 6

    def test_varimax_tolerance_is_the_released_default(self):
        """Tightening it moves published coordinates; see VARIMAX_TOL."""
        assert VARIMAX_TOL == 1e-5


class TestWeights:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
    def test_country_means_reject_missing_nonfinite_or_negative_weight(self, fitted_map, bad):
        cm = copy(fitted_map)
        cm.valid_data = fitted_map.valid_data.copy()
        cm.valid_data.loc[cm.valid_data.index[3], "weight"] = bad
        with pytest.raises(ValueError, match="S017 weights must be present"):
            cm.calculate_mean_scores()

    def test_country_means_reject_zero_weight_sum(self, fitted_map):
        cm = copy(fitted_map)
        cm.valid_data = fitted_map.valid_data.copy()
        cm.valid_data.loc[cm.valid_data.country_code.eq(2), "weight"] = 0.0
        with pytest.raises(ValueError, match=r"sum to zero for groups \[2\]"):
            cm.calculate_mean_scores()

    def test_zero_weights_are_allowed_and_country_means_are_weighted(self, fitted_map):
        cm = copy(fitted_map)
        cm.valid_data = fitted_map.valid_data.copy()
        weights = np.random.default_rng(5).uniform(0.2, 3.0, len(cm.valid_data))
        weights[0] = 0.0
        cm.valid_data["weight"] = weights
        cm.calculate_mean_scores()
        first = cm.valid_data.country_code.eq(1)
        expected = np.average(
            cm.valid_data.loc[first, "PC1_rescaled"], weights=weights[first.to_numpy()]
        )
        got = cm.country_scores_pca.set_index("country_code").loc[1, "PC1_rescaled"]
        assert got == pytest.approx(expected, abs=1e-12)
        unweighted = cm.valid_data.loc[first, "PC1_rescaled"].mean()
        assert abs(got - unweighted) > 1e-6

    def test_public_rescale_is_the_projection_step(self, fitted_map):
        scores = fitted_map.ppca.transform() @ fitted_map.rotation
        pd.testing.assert_frame_equal(
            fitted_map.rescale(scores)[["PC1_rescaled", "PC2_rescaled"]],
            fitted_map.valid_data[["PC1_rescaled", "PC2_rescaled"]],
        )
