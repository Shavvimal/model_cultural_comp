import json

import numpy as np
import pandas as pd
import pytest

from app.culture_map import IV_QNS
from app.llm_bootstrap import (
    bootstrap_llm_positions,
    bootstrap_llm_positions_cluster,
    centroid_statistics,
    confidence_ellipses,
    load_transformed_responses,
    project_cell_means,
)


def test_portable_2024_answers_preserve_language_and_index_transforms(tmp_path, fitted_map):
    records = [
        {"llm": "same-model", "question": "A008", "response": 2},
        {"llm": "same-model", "question": "Y002", "response": [2, 4]},
        {"llm": "same-model", "question": "Y003", "response": [2, 8, 6]},
        {"llm": "same-model", "question": "F120", "response": None},
    ]
    for stem in ["model", "c-model"]:
        (tmp_path / f"{stem}_responses_df.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n"
        )
    # Old local pickles must not be loaded alongside the portable archive.
    (tmp_path / "model_responses_df.pkl").write_bytes(b"not a pickle")
    result = load_transformed_responses(fitted_map, str(tmp_path))
    assert set(result.llm) == {"same-model", "same-model [zh]"}
    assert set(result.language) == {"en", "zh"}
    assert result.system_prompt_id.isna().all()
    for _, group in result.groupby("language"):
        assert group.set_index("question").value.to_dict() == {
            "A008": 2.0,
            "Y002": 3.0,
            "Y003": 2.0,
        }


def test_2024_loader_rejects_missing_response_fields(tmp_path, fitted_map):
    (tmp_path / "model_responses_df.jsonl").write_text('{"llm":"m","question":"A008"}\n')
    with pytest.raises(ValueError, match="every record"):
        load_transformed_responses(fitted_map, str(tmp_path))


@pytest.fixture(scope="module")
def responses(fitted_map):
    rng = np.random.default_rng(3)
    rows = []
    for llm, offset in [("model-a", 0.0), ("model-b", 1.5)]:
        for q in IV_QNS:
            for value in 5 + offset + rng.standard_normal(30):
                rows.append({"llm": llm, "question": q, "value": value})
    return pd.DataFrame(rows)


class TestBootstrap:
    def test_replicate_count(self, fitted_map, responses):
        boot = bootstrap_llm_positions(fitted_map, responses, n_boot=50, seed=1)
        assert len(boot) == 2 * 50
        assert set(boot["llm"]) == {"model-a", "model-b"}

    def test_seed_determinism(self, fitted_map, responses):
        a = bootstrap_llm_positions(fitted_map, responses, n_boot=25, seed=1)
        b = bootstrap_llm_positions(fitted_map, responses, n_boot=25, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_replicates_center_on_projected_mean(self, fitted_map, responses):
        """Projection is affine, so replicate positions must scatter around
        the projection of the per-question means."""
        boot = bootstrap_llm_positions(fitted_map, responses, n_boot=400, seed=2)
        for llm, group in responses.groupby("llm"):
            means = group.groupby("question")["value"].mean()
            center = fitted_map.project(
                pd.DataFrame([means[list(IV_QNS)].to_numpy()], columns=IV_QNS)
            )
            got = boot[boot["llm"] == llm][["PC1_rescaled", "PC2_rescaled"]].mean()
            np.testing.assert_allclose(
                got.to_numpy(),
                center[["PC1_rescaled", "PC2_rescaled"]].to_numpy()[0],
                atol=0.05,
            )

    def test_missing_question_raises(self, fitted_map, responses):
        incomplete = responses[responses["question"] != "F063"]
        with pytest.raises(ValueError, match="F063"):
            bootstrap_llm_positions(fitted_map, incomplete, n_boot=5, seed=1)


class TestConfidenceEllipses:
    def test_one_row_per_model_with_positive_axes(self, fitted_map, responses):
        boot = bootstrap_llm_positions(fitted_map, responses, n_boot=100, seed=4)
        ellipses = confidence_ellipses(boot)
        assert len(ellipses) == 2
        assert (ellipses["ellipse_width"] > 0).all()
        assert (ellipses["ellipse_height"] > 0).all()
        assert (ellipses["ellipse_width"] >= ellipses["ellipse_height"]).all()


def _cluster_responses():
    return pd.DataFrame(
        [
            {"llm": "cluster-model", "system_prompt_id": variant, "question": q, "value": value}
            for variant, value in [(10, 2.0), (20, 4.0), (30, 8.0)]
            for q in IV_QNS
        ]
    )


def _fixed_cluster_draws(monkeypatch, draws):
    draws = np.asarray(draws)

    class FixedDraws:
        def choice(self, population, size, replace):
            assert population == 3 and replace
            assert size == draws.shape
            return draws.copy()

    monkeypatch.setattr("app.llm_bootstrap.np.random.default_rng", lambda seed: FixedDraws())


class TestClusterPoolingRegression:
    def test_missing_groups_preserve_available_responses_and_only_empty_draw_falls_back(
        self, fitted_map, monkeypatch, capsys
    ):
        responses = _cluster_responses()
        responses = responses[
            ~((responses["system_prompt_id"] == 10) & (responses["question"] == "F120"))
        ]
        _fixed_cluster_draws(monkeypatch, [[0, 1, 1], [0, 0, 2], [0, 0, 0]])
        boot = bootstrap_llm_positions_cluster(fitted_map, responses, n_boot=3)
        # The first two draws still have valid answers; only the third has
        # zero pooled F120 responses and uses its observed overall mean, 6.
        means = pd.DataFrame([[10 / 3] * 10, [4.0] * 10, [2.0] * 10], columns=IV_QNS)
        means["F120"] = [4.0, 8.0, 6.0]
        expected = fitted_map.project(means)
        np.testing.assert_allclose(
            boot[["PC1_rescaled", "PC2_rescaled"]],
            expected[["PC1_rescaled", "PC2_rescaled"]],
        )
        assert "1/3 replicates used overall-mean fallback" in capsys.readouterr().out

    def test_unequal_counts_are_pooled_as_responses(self, fitted_map, monkeypatch, capsys):
        responses = _cluster_responses()
        extra = responses[(responses.system_prompt_id == 20) & (responses.question == "F120")]
        extra2 = responses[(responses.system_prompt_id == 30) & (responses.question == "F120")]
        responses = pd.concat([responses, extra, extra, extra2], ignore_index=True)
        _fixed_cluster_draws(monkeypatch, [[0, 1, 2], [1, 1, 0]])
        boot = bootstrap_llm_positions_cluster(fitted_map, responses, n_boot=2)
        means = pd.DataFrame([[14 / 3] * 10, [10 / 3] * 10], columns=IV_QNS)
        means["F120"] = [5.0, 26 / 7]
        expected = fitted_map.project(means)
        np.testing.assert_allclose(
            boot[["PC1_rescaled", "PC2_rescaled"]],
            expected[["PC1_rescaled", "PC2_rescaled"]],
        )
        assert "fallback" not in capsys.readouterr().out

    def test_complete_data_matches_direct_seeded_resampling(self, fitted_map):
        responses = _cluster_responses()
        draws = np.random.default_rng(23).choice(3, size=(20, 3), replace=True)
        means = pd.DataFrame(
            np.repeat(np.array([2.0, 4.0, 8.0])[draws].mean(axis=1)[:, None], 10, axis=1),
            columns=IV_QNS,
        )
        expected = fitted_map.project(means)
        actual = bootstrap_llm_positions_cluster(fitted_map, responses, n_boot=20, seed=23)
        again = bootstrap_llm_positions_cluster(fitted_map, responses, n_boot=20, seed=23)
        pd.testing.assert_frame_equal(actual, again)
        np.testing.assert_allclose(
            actual[["PC1_rescaled", "PC2_rescaled"]],
            expected[["PC1_rescaled", "PC2_rescaled"]],
        )

    def test_missing_entire_item_raises(self, fitted_map):
        responses = _cluster_responses().query("question != 'F120'")
        with pytest.raises(ValueError, match="F120"):
            bootstrap_llm_positions_cluster(fitted_map, responses, n_boot=5)

    def test_missing_variant_id_raises(self, fitted_map):
        responses = _cluster_responses()
        responses.loc[0, "system_prompt_id"] = np.nan
        with pytest.raises(ValueError, match="system_prompt_id"):
            bootstrap_llm_positions_cluster(fitted_map, responses, n_boot=5)


class TestObservedPointEstimates:
    def test_cell_means_apply_item_threshold(self, fitted_map):
        responses = _cluster_responses()
        points = project_cell_means(fitted_map, responses, min_per_item=3)
        expected = fitted_map.project(pd.DataFrame([[14 / 3] * 10], columns=IV_QNS))
        np.testing.assert_allclose(
            points[["PC1_rescaled", "PC2_rescaled"]],
            expected[["PC1_rescaled", "PC2_rescaled"]],
        )
        assert points["min_per_item"].tolist() == [3]
        assert project_cell_means(fitted_map, responses, min_per_item=4).empty
        assert project_cell_means(fitted_map, responses.query("question != 'F120'")).empty

    def test_supplied_point_controls_centre_and_distance_not_replicate_average(self):
        boot = pd.DataFrame(
            {
                "llm": ["m"] * 4,
                "PC1_rescaled": [-1.0, 1.0, -1.0, 1.0],
                "PC2_rescaled": [-1.0, -1.0, 1.0, 1.0],
            }
        )
        points = pd.DataFrame({"llm": ["m"], "PC1_rescaled": [0.2], "PC2_rescaled": [0.0]})
        ellipse = confidence_ellipses(boot, point_estimates=points).iloc[0]
        assert ellipse.PC1_rescaled == 0.2 and ellipse.PC2_rescaled == 0.0
        assert ellipse.sd_pc1 == pytest.approx(boot.PC1_rescaled.std())
        countries = pd.DataFrame(
            {
                "PC1_rescaled": [0.1, 0.5],
                "PC2_rescaled": [0.0, 0.0],
                "Cultural Region": ["Confucian", "Protestant Europe"],
            }
        )
        stats = centroid_statistics(
            boot, countries, human_mean=(0.0, 0.0), point_estimates=points
        ).iloc[0]
        assert stats.dist_human_mean == pytest.approx(0.2)
        assert stats.pct_countries_closer == 0.5
        assert stats.min_dist_nonwestern == pytest.approx(0.1)
        assert stats.dist_human_mean_lo == pytest.approx(np.sqrt(2))


class TestLoadResponses2026:
    def test_mixed_type_keys_dedup(self, fitted_map, tmp_path):
        """A resumed run serialises ids as str where the original wrote int;
        the loader must normalise before dedup or the retried row survives
        alongside its original and the item mean double-counts."""
        import json

        from app.llm_bootstrap import load_responses_2026

        original = {
            "llm": "model-a",
            "question": "A008",
            "system_prompt_id": "3",
            "repeat": "1",
            "raw_content": "2",
            "thinking": "",
            "parsed": "2",
            "error": None,
            "attempts": "1",
            "duration_ms": "10",
            "ts": "t0",
        }
        resumed = {**original, "system_prompt_id": 3, "repeat": 1, "parsed": "3", "ts": "t1"}
        path = tmp_path / "model-a.jsonl"
        path.write_text(json.dumps(original) + "\n" + json.dumps(resumed) + "\n")

        out = load_responses_2026(fitted_map, str(tmp_path))
        assert len(out) == 1
        assert out["value"].iloc[0] == 3.0


class TestClusterBootstrap:
    @pytest.fixture
    def clustered(self):
        """Ten prompt variants x ten items x 3 repeats, with variant 2 lacking
        every response on F120 (the shape of gemma4:31b [en]'s refusals)."""
        rng = np.random.default_rng(7)
        rows = []
        for variant in range(10):
            for q in IV_QNS:
                if variant == 2 and q == "F120":
                    continue
                for value in 5 + rng.standard_normal(3):
                    rows.append(
                        {
                            "llm": "model-a",
                            "question": q,
                            "value": value,
                            "system_prompt_id": variant,
                        }
                    )
        return pd.DataFrame(rows)

    def test_absent_variant_item_group_is_zero_not_nan(self, fitted_map, clustered, capsys):
        """Regression: an empty (variant, item) group used to leave NaN in the
        pivoted sums, so any replicate drawing that variant fell back to the
        overall item mean. Pooled counts stay positive here (nine of ten
        variants answer F120), so no fallback may fire."""
        from app.llm_bootstrap import bootstrap_llm_positions_cluster

        boot = bootstrap_llm_positions_cluster(fitted_map, clustered, n_boot=300, seed=1)
        assert "fallback" not in capsys.readouterr().out
        assert len(boot) == 300
        xy = boot[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        assert np.isfinite(xy).all()
        assert (xy.std(axis=0, ddof=1) > 0).all()

    def test_fallback_only_when_pooled_count_is_zero(self, fitted_map, clustered, capsys):
        """With a single item answered by exactly one variant, replicates that
        never draw that variant have a pooled count of zero: those (and only
        those) use the fallback, and it is reported."""
        from app.llm_bootstrap import bootstrap_llm_positions_cluster

        thin = clustered[
            ~((clustered["question"] == "A008") & (clustered["system_prompt_id"] != 0))
        ]
        boot = bootstrap_llm_positions_cluster(fitted_map, thin, n_boot=200, seed=1)
        out = capsys.readouterr().out
        assert "replicates used overall-mean fallback" in out
        n_fallback = int(out.split("model-a: ")[1].split("/")[0])
        # P(variant 0 never drawn in 10 draws) = 0.9**10 ~ 0.35
        assert 0 < n_fallback < 200
        assert np.isfinite(boot[["PC1_rescaled", "PC2_rescaled"]].to_numpy()).all()

    def test_item_with_no_responses_raises(self, fitted_map, clustered):
        from app.llm_bootstrap import bootstrap_llm_positions_cluster

        with pytest.raises(ValueError, match="F063"):
            bootstrap_llm_positions_cluster(
                fitted_map, clustered[clustered["question"] != "F063"], n_boot=5, seed=1
            )

    def test_missing_variant_id_raises(self, fitted_map, clustered):
        from app.llm_bootstrap import bootstrap_llm_positions_cluster

        broken = clustered.copy()
        broken.loc[broken.index[0], "system_prompt_id"] = pd.NA
        with pytest.raises(ValueError, match="system_prompt_id"):
            bootstrap_llm_positions_cluster(fitted_map, broken, n_boot=5, seed=1)
