"""Regression checks for denominators and displayed point estimators."""

import json

import numpy as np
import pandas as pd

from app.culture_map import IV_QNS
from scripts.analyze_2026 import language_effects, parse_rates
from scripts.confirmatory_2026 import _cluster_item_means, coherence_rate


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
