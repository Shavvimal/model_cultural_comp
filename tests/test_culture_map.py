import numpy as np
import pandas as pd
import pytest

from app.culture_map import IV_QNS, PC_RESCALE_PARAMS, CulturalMap


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


class TestRescaling:
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
            ((-1, 3), -5),  # missing
        ],
    )
    def test_y002(self, ans, expected):
        assert CulturalMap.y002_transform(ans) == expected

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
        assert not (
            (cm.subset_ivs_df["Y003"] < -2) | (cm.subset_ivs_df["Y003"] > 2)
        ).any()

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
