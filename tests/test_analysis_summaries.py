"""Regression checks for denominators and displayed point estimators."""

import json

import numpy as np
import pandas as pd
import pytest

from app.culture_map import IV_QNS
from scripts.analyze_2026 import N_BOOT_CLUSTER, language_effects, parse_rates
from scripts.confirmatory_2026 import (
    _cluster_item_means,
    coherence_rate,
    mean_displacement_ci,
    origin_language_permutation,
    per_item_sign_tests,
    sign_test_delta_pc2,
)


def test_parse_rates_normalises_keys_and_counts_absent_items(tmp_path):
    rows = [
        {
            "llm": "gemma4:31b",
            "language": "en",
            "question": q,
            "system_prompt_id": 0,
            "repeat": 0,
            "parsed": 1,
            "error": None,
        }
        for q in IV_QNS[:-1]
    ]
    rows.append({**rows[0], "system_prompt_id": "0", "repeat": "0"})
    (tmp_path / "gemma.jsonl").write_text("\n".join(map(json.dumps, rows)))
    rates = parse_rates(tmp_path).iloc[0]
    assert rates["calls"] == 9
    assert rates["parsed"] == 9
    assert rates["min_per_question"] == 0


def test_language_points_are_observed_plugin_not_bootstrap_expectations():
    points = pd.DataFrame(
        {
            "llm": ["gemma4:31b", "gemma4:31b [zh]"],
            "PC1_rescaled": [0.0, 3.0],
            "PC2_rescaled": [0.0, 4.0],
        }
    )
    boot = pd.DataFrame(
        {
            "llm": ["gemma4:31b"] * 3 + ["gemma4:31b [zh]"] * 3,
            "PC1_rescaled": [0, 0, 0, 10, 11, 12],
            "PC2_rescaled": [0, 0, 0, 10, 11, 12],
        }
    )
    row = language_effects(points, boot).iloc[0]
    assert row["delta_pc1"] == 3
    assert row["delta_pc2"] == 4
    assert row["displacement"] == 5
    assert row["displacement_bootstrap_norm_mean"] > 5


def test_item_cluster_draws_include_missing_clusters_without_nan():
    frame = pd.DataFrame({"system_prompt_id": [0, 2, 2], "value": [1.0, 9.0, 9.0]})
    values = _cluster_item_means(frame, np.random.default_rng(42), 1000, np.arange(3))
    assert np.isfinite(values).all()
    assert values.min() == 1
    assert values.max() == 9
    assert np.any(values == frame["value"].mean())


def test_coherence_requires_eligibility_in_both_observed_languages():
    rates = pd.DataFrame(
        {
            "llm": ["a", "a", "b", "b"],
            "language": ["en", "zh"] * 2,
            "cohort": ["Chinese"] * 4,
            "min_per_question": [50, 9, 50, 50],
        }
    )
    now = coherence_rate(rates).query("year == '2026'").iloc[0]
    assert now["attempted"] == 2
    assert now["coherent"] == 1


def test_coherence_does_not_treat_an_absent_language_as_success():
    rates = pd.DataFrame(
        {"llm": ["a"], "language": ["en"], "cohort": ["Chinese"], "min_per_question": [50]}
    )
    now = coherence_rate(rates).query("year == '2026'").iloc[0]
    assert now["attempted"] == 1
    assert now["coherent"] == 0


def test_tied_items_remain_in_the_ten_item_bh_family():
    import pandas as pd
    from scipy.stats import binomtest

    from app.culture_map import IV_QNS
    from scripts.confirmatory_2026 import per_item_sign_tests

    effects = pd.DataFrame(
        [{"question": q, "delta": 1.0 if q == "A008" else 0.0} for q in IV_QNS for _ in range(8)]
    )
    result = per_item_sign_tests(effects).set_index("question")
    assert len(result) == 10
    assert result.loc["A165", "n_effective"] == 0
    assert result.loc["A165", "p_sign_two_sided"] == 1.0
    assert result.loc["A008", "p_bh"] == 10 * binomtest(8, 8).pvalue


def _item_effects(n_models: int = 4) -> pd.DataFrame:
    return pd.DataFrame([{"question": q, "delta": 1.0} for q in IV_QNS for _ in range(n_models)])


def test_nan_item_delta_is_rejected_not_counted_as_negative():
    effects = _item_effects()
    effects.loc[0, "delta"] = np.nan
    with pytest.raises(ValueError, match=f"{effects.loc[0, 'question']}: .*non-finite"):
        per_item_sign_tests(effects)


def test_bh_family_must_be_exactly_the_ten_items():
    effects = _item_effects()
    with pytest.raises(ValueError, match=r"missing \['Y003'\], extra \[\]"):
        per_item_sign_tests(effects[effects["question"] != "Y003"])
    extra = pd.concat([effects, pd.DataFrame([{"question": "Z999", "delta": 1.0}])])
    with pytest.raises(ValueError, match=r"missing \[\], extra \['Z999'\]"):
        per_item_sign_tests(extra)


def test_all_tied_delta_pc2_is_uninformative_not_a_scipy_error():
    row = sign_test_delta_pc2(pd.DataFrame({"delta_pc2": [0.0] * 5})).iloc[0]
    assert row["n_delta_pc2_negative"] == 0
    assert row["p_two_sided"] == 1.0
    assert row["p_directional_traditional"] == 1.0


def test_nan_delta_pc2_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        sign_test_delta_pc2(pd.DataFrame({"delta_pc2": [-1.0, np.nan, -2.0]}))


def _lang_fx(cohorts: list) -> pd.DataFrame:
    n = len(cohorts)
    pc1 = np.linspace(-1.0, 1.0, n)
    pc2 = np.linspace(0.5, -0.5, n)
    return pd.DataFrame(
        {
            "llm": ["deepseek-v4-flash", "glm-5.1", "gemma4:31b", "gpt-oss:20b"][:n],
            "cohort": cohorts,
            "delta_pc1": pc1,
            "delta_pc2": pc2,
            "displacement": np.linalg.norm(np.column_stack([pc1, pc2]), axis=1),
        }
    )


@pytest.mark.parametrize(
    "cohorts",
    [
        ["Chinese"] * 4,
        ["Chinese", "Chinese", "western", "Western"],
        ["Chinese", "Chinese", "Western", None],
    ],
)
def test_origin_permutation_needs_exactly_the_two_named_cohorts(cohorts):
    with pytest.raises(ValueError, match="both cohorts"):
        origin_language_permutation(_lang_fx(cohorts))


def test_origin_permutation_rejects_non_finite_components():
    lang_fx = _lang_fx(["Chinese", "Chinese", "Western", "Western"])
    lang_fx.loc[0, "delta_pc2"] = np.nan
    with pytest.raises(ValueError, match=r"delta_pc2: .*finite"):
        origin_language_permutation(lang_fx)


def test_language_effect_requires_the_english_arm():
    points = pd.DataFrame({"llm": ["m [zh]"], "PC1_rescaled": [1.0], "PC2_rescaled": [1.0]})
    boot = pd.DataFrame(
        {"llm": ["m [zh]"] * 3, "PC1_rescaled": [1.0] * 3, "PC2_rescaled": [1.0] * 3}
    )
    with pytest.raises(ValueError, match="m: language contrast needs both arms"):
        language_effects(points, boot)


def test_language_effect_rejects_unequal_replicate_counts():
    points = pd.DataFrame(
        {"llm": ["m", "m [zh]"], "PC1_rescaled": [0.0, 1.0], "PC2_rescaled": [0.0, 1.0]}
    )
    boot = pd.DataFrame(
        {"llm": ["m"] * 3 + ["m [zh]"] * 2, "PC1_rescaled": [0.0] * 5, "PC2_rescaled": [0.0] * 5}
    )
    with pytest.raises(ValueError, match="equal replicate counts, got 3 en and 2 zh"):
        language_effects(points, boot)


def test_paired_design_draws_as_many_replicates_as_the_primary_bootstrap():
    import scripts.language_design_sensitivity as paired

    assert paired.N_DRAWS == N_BOOT_CLUSTER
    assert paired.SEED == 20260911


def test_plugin_aliases_are_byte_copies_of_the_confirmatory_results(tmp_path, monkeypatch):
    import scripts.plugin_displacement_2026 as plugin

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    _lang_fx(["Chinese", "Chinese", "Western", "Western"]).to_csv(
        plugin.LANGUAGE_EFFECTS, index=False
    )
    effects = pd.read_csv(plugin.LANGUAGE_EFFECTS)  # parsed exactly as confirmatory_2026 does
    mean_displacement_ci(effects).to_csv(plugin.MEAN_DISPLACEMENT, index=False)
    origin_language_permutation(effects).to_csv(plugin.ORIGIN_PERMUTATION, index=False)
    assert plugin.main() == 0
    assert plugin.MEAN_DISPLACEMENT_ALIAS.read_bytes() == plugin.MEAN_DISPLACEMENT.read_bytes()
    assert plugin.ORIGIN_PERMUTATION_ALIAS.read_bytes() == plugin.ORIGIN_PERMUTATION.read_bytes()


def test_plugin_aliases_refuse_stale_confirmatory_results():
    from scripts.plugin_displacement_2026 import check_confirmatory_matches

    effects = _lang_fx(["Chinese", "Chinese", "Western", "Western"])
    ci = mean_displacement_ci(effects)
    perm = origin_language_permutation(effects)
    check_confirmatory_matches(effects, ci, perm)
    with pytest.raises(ValueError, match=r"rerun confirmatory_2026\.py"):
        check_confirmatory_matches(effects.assign(delta_pc1=effects["delta_pc1"] + 1), ci, perm)
    relabelled = effects.assign(cohort=["Chinese", "Western", "Chinese", "Western"])
    with pytest.raises(ValueError, match=r"rerun confirmatory_2026\.py"):
        check_confirmatory_matches(relabelled, ci, perm)


def test_keying_balance_rejects_unmapped_models_instead_of_unknown_cohort():
    from scripts.diagnostics_2026 import keying_balance

    parsed = pd.DataFrame(
        [{"llm": "not-a-2026-model", "language": "en", "question": q, "value": 1.0} for q in IV_QNS]
    )
    keying = pd.DataFrame(
        {
            "question": IV_QNS,
            "midpoint": 1.0,
            "half_range": 1.0,
            "d_pc1_per_unit": [1.0, -1.0] * 5,
            "d_pc2_per_unit": [1.0, -1.0] * 5,
        }
    )
    with pytest.raises(ValueError, match="not-a-2026-model"):
        keying_balance(parsed, keying)
