import json

import numpy as np
import pandas as pd
import pytest

from app.stats import bh_adjust
from scripts.exploratory_correlations_2026 import (
    compute_correlations,
    correlation,
    raw_features,
)


def frames():
    labels = [f"{name}{suffix}" for name in ["a", "b", "c", "d"] for suffix in ["", " [zh]"]]
    points = pd.DataFrame(
        {
            "llm": labels,
            "PC1_rescaled": [0, 1, 1, 3, 2, 5, 3, 7],
            "PC2_rescaled": [0, 0, 0, 0, 0, 0, 1, 2],
            "item_sd_pc1": np.arange(1, 9),
            "item_sd_pc2": np.arange(1, 9) * 4,
        }
    )
    diagnostics = pd.DataFrame(
        {
            "llm": labels,
            "midpoint_distance": np.arange(1, 9),
            "mean_item_entropy": np.arange(8, 0, -1),
        }
    )
    features = pd.DataFrame(
        {
            "llm": labels,
            "model": [name for name in ["a", "b", "c", "d"] for _ in range(2)],
            "language": ["en", "zh"] * 4,
            "median_stored_excerpt_chars": [10, 11, 20, 21, 30, 31, np.nan, np.nan],
            "raw_answer_string_entropy_bits": np.arange(1, 9),
        }
    )
    return points, diagnostics, features


def test_exact_seven_family_and_distinct_entropy_definitions():
    result, cells = compute_correlations(*frames())
    seven = result[result.family == "historical_seven"]
    assert len(seven) == 7
    assert seven.bh_family_size.eq(7).all()
    assert len(result) == 10
    singles = result[result.family != "historical_seven"]
    assert singles.p_bh.isna().all()
    assert set(singles.x_measure) == {
        "raw_answer_string_entropy_bits",
        "numeric_index_entropy_bits",
    }
    raw = result.set_index("test_id").loc["raw_string_entropy_midpoint"]
    numeric = result.set_index("test_id").loc["numeric_entropy_midpoint_sensitivity"]
    assert raw.rho_spearman == pytest.approx(1)
    assert numeric.rho_spearman == pytest.approx(-1)
    assert np.allclose(cells.geometric_mean_item_sd, np.arange(1, 9) * 2)


def test_no_thinking_cells_excluded_not_zero_imputed_and_models_not_double_counted():
    result, _ = compute_correlations(*frames())
    result = result.set_index("test_id")
    assert result.loc["length_pc1", "n_candidate_units"] == 8
    assert result.loc["length_pc1", "n_used"] == 6
    assert result.loc["length_en_displacement", "n_candidate_units"] == 4
    assert result.loc["length_en_displacement", "n_used"] == 3
    assert json.loads(result.loc["length_pc1", "excluded_units_json"]) == ["d", "d [zh]"]
    assert result.loc["length_en_displacement", "rho_spearman"] == pytest.approx(1)
    assert "same_model_dependence_unmodelled" in result.loc["midpoint_pc1", "p_scope"]


def test_bh_undefined_test_does_not_shrink_declared_family():
    corrected = bh_adjust(np.array([0.01, 0.04, np.nan]), family_size=3)
    assert corrected[:2] == pytest.approx([0.03, 0.06])
    assert np.isnan(corrected[2])
    with pytest.raises(ValueError, match="p-values"):
        bh_adjust(np.array([-0.1]), family_size=1)


def test_constant_input_is_explicitly_unestimable():
    frame = pd.DataFrame(
        {"x": [1, 1, 1], "y": [1, 2, 3], "model": ["a", "b", "c"], "unit_id": ["a", "b", "c"]}
    )
    result = correlation(
        frame, "x", "y", test_id="constant", family="test", unit="paired_model", status="test"
    )
    assert result["estimate_status"] == "insufficient_or_constant_inputs"
    assert result["n_used"] == 3
    assert np.isnan(result["p_raw_two_sided"])


def test_raw_features_dedup_exact_strings_censoring_and_failed_excerpt_scope(tmp_path):
    base = {
        "llm": "a",
        "language": "en",
        "question": "A008",
        "system_prompt_id": 0,
        "repeat": 0,
        "error": None,
        "parsed": 1,
        "raw_content": "1",
        "thinking": "x" * 10,
    }
    rows = [
        base,
        {**base, "system_prompt_id": "0", "repeat": "0", "raw_content": "1 ", "thinking": "x" * 20},
        {**base, "repeat": 1, "thinking": ""},
        {**base, "repeat": 2, "error": "parse: invalid", "parsed": None, "thinking": "x" * 2000},
        {**base, "llm": "b", "thinking": ""},
    ]
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = raw_features(tmp_path).set_index("llm")
    a = result.loc["a"]
    assert a.n_terminal_records == 3
    assert a.n_parsed_records == 2
    assert a.raw_answer_string_entropy_bits == 1
    assert a.median_stored_excerpt_chars == 1010
    assert a.n_excerpts_at_2000_character_cap == 1
    assert pd.isna(result.loc["b", "median_stored_excerpt_chars"])


def test_missing_or_duplicate_primary_cell_inputs_raise():
    points, diagnostics, features = frames()
    with pytest.raises(ValueError, match="missing a primary"):
        compute_correlations(points, diagnostics.iloc[1:], features)
    with pytest.raises(ValueError, match="duplicate positions"):
        compute_correlations(pd.concat([points, points]), diagnostics, features)
