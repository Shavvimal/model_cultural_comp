"""Synthetic checks for the two appendix contrasts."""

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from app.appendix_contrasts import (
    keyed_pc1_shift,
    origin_profile_permutation,
    petition_contrast,
)
from app.culture_map import IV_QNS
from app.survey_reference import survey_reference_tables
from scripts.appendix_contrasts_2026 import standardise_profiles

ITEMS = ["E025", "F063", "F118", "F120", "G006", "A008"]


@pytest.fixture(scope="module")
def keying():
    return pd.DataFrame(
        {
            "question": ITEMS,
            "d_pc1_per_unit": [-0.5, -0.1, 0.2, 0.15, -0.3, -0.8],
        }
    )


def _item_fx(deltas: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for qn, per_model in deltas.items():
        for i, d in enumerate(per_model):
            rows.append({"llm": f"m{i}", "question": qn, "delta": d})
    return pd.DataFrame(rows)


class TestPetitionContrast:
    def test_keyed_shift_is_delta_times_loading(self, keying):
        fx = _item_fx({"E025": [0.4, -0.2], "F118": [1.0, 2.0]})
        out = keyed_pc1_shift(fx, keying)
        expected = fx["delta"] * fx["question"].map({"E025": -0.5, "F118": 0.2})
        assert np.allclose(out["keyed_pc1"], expected)

    def test_missing_keying_row_raises(self, keying):
        fx = _item_fx({"Z999": [0.1]})
        with pytest.raises(ValueError, match="Z999"):
            keyed_pc1_shift(fx, keying)

    def test_target_in_reference_raises(self, keying):
        fx = _item_fx({q: [0.1, 0.2] for q in ITEMS})
        with pytest.raises(ValueError):
            petition_contrast(fx, keying, "E025", ["E025", "F063"])

    def test_planted_unanimous_contrast(self, keying):
        # Target moves +0.5 on the raw scale (keyed -0.25) in every model;
        # references stay at zero, so every contrast is -0.25.
        n = 9
        fx = _item_fx(
            {
                "E025": [0.5] * n,
                "F063": [0.0] * n,
                "F118": [0.0] * n,
                "F120": [0.0] * n,
                "G006": [0.0] * n,
            }
        )
        per_model, summary = petition_contrast(fx, keying, "E025", ["F063", "F118", "F120", "G006"])
        assert len(per_model) == n
        assert np.allclose(per_model["contrast"], -0.25)
        assert per_model["target_more_survival_ward"].all()
        s = summary.iloc[0]
        assert s["n_target_more_survival_ward"] == n
        assert s["n_effective"] == n
        assert s["median_contrast"] == pytest.approx(-0.25)
        assert s["p_sign_two_sided"] == pytest.approx(binomtest(n, n, 0.5).pvalue)

    def test_ties_drop_out_of_the_sign_test(self, keying):
        fx = _item_fx(
            {
                "E025": [0.5, 0.0, -0.5],
                "F063": [0.0, 0.0, 0.0],
                "F118": [0.0, 0.0, 0.0],
            }
        )
        _, summary = petition_contrast(fx, keying, "E025", ["F063", "F118"])
        s = summary.iloc[0]
        assert s["n_models"] == 3
        assert s["n_effective"] == 2
        assert s["n_target_more_survival_ward"] == 1
        assert s["p_sign_two_sided"] == pytest.approx(1.0)

    def test_reference_mean_uses_only_reference_items(self, keying):
        # A008 is in the frame but not in the reference set; it must not leak in.
        fx = _item_fx({"E025": [0.0], "F063": [1.0], "F118": [1.0], "A008": [100.0]})
        per_model, _ = petition_contrast(fx, keying, "E025", ["F063", "F118"])
        assert per_model["keyed_pc1_reference_mean"].iloc[0] == pytest.approx((-0.1 + 0.2) / 2)


def _brute_force(profiles: np.ndarray, is_cn: np.ndarray) -> tuple[float, float]:
    """Independent re-implementation: pooled within minus cross, and its p."""
    corr = np.corrcoef(profiles)
    n, k = len(profiles), int(is_cn.sum())

    def stat(lab):
        within, cross = [], []
        for i, j in combinations(range(n), 2):
            (within if lab[i] == lab[j] else cross).append(corr[i, j])
        return float(np.mean(within) - np.mean(cross))

    obs = stat(is_cn)
    count = 0
    for idx in combinations(range(n), k):
        lab = np.zeros(n, dtype=bool)
        lab[list(idx)] = True
        count += stat(lab) >= obs - 1e-12
    return obs, count / comb(n, k)


class TestOriginProfilePermutation:
    def test_matches_brute_force_on_random_profiles(self):
        rng = np.random.default_rng(3)
        n, k, items = 8, 4, 6
        profiles = pd.DataFrame(rng.normal(size=(n, items)))
        is_cn = np.array([True] * k + [False] * (n - k))
        out = origin_profile_permutation(profiles, is_cn)
        row = out[out["statistic"] == "within_minus_cross"].iloc[0]
        obs, p = _brute_force(profiles.to_numpy(), is_cn)
        assert row["observed"] == pytest.approx(obs)
        assert row["p_exact_ge"] == pytest.approx(p)
        assert row["n_labellings"] == comb(n, k)
        assert row["n_within_pairs"] == comb(k, 2) + comb(n - k, 2)
        assert row["n_cross_pairs"] == k * (n - k)

    def test_planted_cohort_structure_is_extreme(self):
        # Two tight clusters of profiles: the observed labelling is the
        # unique maximiser, so p is exactly 1 / C(n, k).
        rng = np.random.default_rng(0)
        base_a = rng.normal(size=6)
        base_b = -base_a
        profiles = pd.DataFrame(
            np.vstack(
                [base_a + 0.01 * rng.normal(size=6) for _ in range(4)]
                + [base_b + 0.01 * rng.normal(size=6) for _ in range(4)]
            )
        )
        is_cn = np.array([True] * 4 + [False] * 4)
        out = origin_profile_permutation(profiles, is_cn)
        row = out[out["statistic"] == "within_minus_cross"].iloc[0]
        # The complementary labelling gives the same statistic, so two
        # labellings attain the maximum.
        assert row["n_ge_observed"] == 2
        assert row["p_exact_ge"] == pytest.approx(2 / comb(8, 4))
        assert row["mean_r_within_chinese"] > row["mean_r_cross"]
        assert row["mean_r_within_western"] > row["mean_r_cross"]

    def test_label_permutation_invariance(self):
        # Relabelling which cohort is "Chinese" cannot change the pooled statistic.
        rng = np.random.default_rng(1)
        profiles = pd.DataFrame(rng.normal(size=(7, 5)))
        is_cn = np.array([True, False, True, True, False, False, False])
        a = origin_profile_permutation(profiles, is_cn)
        b = origin_profile_permutation(profiles, ~is_cn)
        ra = a[a["statistic"] == "within_minus_cross"].iloc[0]
        rb = b[b["statistic"] == "within_minus_cross"].iloc[0]
        assert ra["observed"] == pytest.approx(rb["observed"])
        assert ra["p_exact_ge"] == pytest.approx(rb["p_exact_ge"])

    def test_degenerate_cohorts_rejected(self):
        profiles = pd.DataFrame(np.eye(4))
        with pytest.raises(ValueError):
            origin_profile_permutation(profiles, np.array([True] * 4))
        with pytest.raises(ValueError):
            origin_profile_permutation(profiles, np.array([True, False]))


class TestReleasedStandardisation:
    def test_csv_route_matches_frozen_npz_transform_and_statistics(self, fitted_map, tmp_path):
        _, _, items, _ = survey_reference_tables(fitted_map)
        csv_path = tmp_path / "item_baselines.csv"
        items.to_csv(csv_path, index=False)
        npz_path = tmp_path / "instrument.npz"
        fitted_map.save_model(npz_path)
        # Neither CSV row order nor profile column order defines alignment.
        profiles = pd.DataFrame(
            np.random.default_rng(13).normal(size=(8, len(IV_QNS))), columns=IV_QNS
        )[IV_QNS[::-1]]
        released = pd.read_csv(csv_path, float_precision="round_trip").iloc[::-1]
        actual = standardise_profiles(profiles, released)
        with np.load(npz_path) as instrument:
            means = pd.Series(instrument["means"], index=IV_QNS).reindex(profiles.columns)
            stds = pd.Series(instrument["stds"], index=IV_QNS).reindex(profiles.columns)
        expected = (profiles - means) / stds
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        labels = np.array([True] * 4 + [False] * 4)
        pd.testing.assert_frame_equal(
            origin_profile_permutation(actual, labels),
            origin_profile_permutation(expected, labels),
            check_exact=True,
        )

    @pytest.mark.parametrize("invalid", ["missing", "duplicate", "zero_sd", "nan_mean"])
    def test_rejects_incomplete_or_invalid_fit_aggregate(self, fitted_map, invalid):
        _, _, items, _ = survey_reference_tables(fitted_map)
        if invalid == "missing":
            items = items.iloc[:-1]
        elif invalid == "duplicate":
            items = pd.concat([items, items.iloc[:1]])
        elif invalid == "zero_sd":
            items.loc[0, "fit_standardisation_sd"] = 0
        else:
            items.loc[0, "fit_standardisation_mean"] = np.nan
        with pytest.raises(ValueError):
            standardise_profiles(pd.DataFrame([np.ones(10)], columns=IV_QNS), items)

    def test_full_appendix_replay_does_not_load_a_fitted_npz(
        self, fitted_map, tmp_path, monkeypatch
    ):
        import scripts.appendix_contrasts_2026 as script

        rng = np.random.default_rng(19)
        models = ["deepseek-v4-flash", "glm-5.1", "gemma4:31b", "gpt-oss:20b"]
        profiles = pd.DataFrame(rng.normal(size=(4, 10)), index=models, columns=IV_QNS)
        long = (
            profiles.rename_axis("llm")
            .reset_index()
            .melt(id_vars="llm", var_name="question", value_name="cell_mean")
        )
        long["language"] = "en"
        item_fx = long[["llm", "question"]].assign(delta=rng.normal(size=40))
        keying = pd.DataFrame({"question": IV_QNS, "d_pc1_per_unit": rng.normal(size=10)})
        _, _, items, _ = survey_reference_tables(fitted_map)
        for constant, frame in [
            ("ITEM_FX_CSV", item_fx),
            ("KEYING_CSV", keying),
            ("PROFILES_CSV", long),
            ("ITEM_BASELINES_CSV", items),
        ]:
            path = tmp_path / f"{constant}.csv"
            frame.to_csv(path, index=False)
            monkeypatch.setattr(script, constant, str(path))
        for constant in ["OUT_PETITION", "OUT_ORIGIN", "OUT_ORIGIN_STANDARDISED"]:
            monkeypatch.setattr(script, constant, str(tmp_path / f"{constant}.csv"))

        def reject_npz(*args, **kwargs):
            raise AssertionError("aggregate-only replay must not load a fitted NPZ")

        monkeypatch.setattr(np, "load", reject_npz)
        assert script.main() == 0
        for constant in ["OUT_PETITION", "OUT_ORIGIN", "OUT_ORIGIN_STANDARDISED"]:
            assert not pd.read_csv(getattr(script, constant)).empty
