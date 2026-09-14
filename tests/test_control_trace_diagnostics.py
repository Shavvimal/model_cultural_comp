import json

import numpy as np
import pandas as pd
import pytest

from app.control_audit import coverage_tables
from app.culture_map import IV_QNS
from app.llm_bootstrap import bootstrap_llm_positions_cluster
from app.trace_codebook import CODES
from app.trace_diagnostics import trace_diagnostics, vote_counts
from scripts.prompt_sensitivity_2026 import child_seed, displacement, family_rows


def control_responses(n_first=50):
    rows = []
    for question in IV_QNS:
        for i in range(n_first if question == IV_QNS[0] else 50):
            rows.append(
                {
                    "llm": "model-a",
                    "question": question,
                    "system_prompt_id": i // 5,
                    "value": 1.0 + (i // 5) % 3,
                }
            )
    return pd.DataFrame(rows)


def test_control_primary_rejects_sparse_item_but_retains_labelled_sensitivity(fitted_map):
    responses = control_responses(n_first=2)
    primary, boot = family_rows(fitted_map, responses, "nosys", "min10", "nosys/min10", n_boot=20)
    relaxed, _ = family_rows(fitted_map, responses, "nosys", "min1", "nosys/min1", n_boot=20)
    assert not primary.iloc[0]["eligible"]
    assert primary.iloc[0]["min_per_question"] == 2
    assert boot.empty
    assert relaxed.iloc[0]["eligible"]
    assert relaxed.iloc[0]["required_min_per_item"] == 1


def test_equal_sized_arms_do_not_reuse_bootstrap_draws(fitted_map):
    responses = control_responses()
    _, a = family_rows(fitted_map, responses, "bare", "min1", "bare", n_boot=2000)
    _, b = family_rows(fitted_map, responses, "averaging", "min1", "averaging", n_boot=2000)
    assert child_seed("bare") != child_seed("averaging")
    assert abs(np.corrcoef(a["PC1_rescaled"], b["PC1_rescaled"])[0, 1]) < 0.1


def test_persona_cluster_uncertainty_is_not_replaced_by_item_uncertainty(fitted_map):
    responses = control_responses()
    # One varying item avoids cancellation between oppositely keyed items.
    # Five identical within-prefix repeats carry one cluster's information.
    responses.loc[responses["question"] != IV_QNS[0], "value"] = 2.0
    points, item = family_rows(fitted_map, responses, "nosys", "min10", "nosys/min10", n_boot=2000)
    cluster = bootstrap_llm_positions_cluster(fitted_map, responses, n_boot=2000, seed=42)
    _, other_item = family_rows(
        fitted_map, responses, "all_persona", "min10", "all_persona", n_boot=2000
    )
    mixed = displacement(item, cluster, points, points, "test", "min10", "item", "cluster")
    conditional = displacement(item, other_item, points, points, "test", "min10", "item", "item")
    assert mixed.iloc[0]["estimator_b"] == "cluster"
    assert (
        mixed.iloc[0]["delta_pc1_hi"] - mixed.iloc[0]["delta_pc1_lo"]
        > conditional.iloc[0]["delta_pc1_hi"] - conditional.iloc[0]["delta_pc1_lo"]
    )


def test_coverage_counts_absent_trials_duplicates_and_failures_separately(tmp_path):
    first = {
        "llm": "m",
        "question": "A008",
        "system_prompt_id": "0",
        "repeat": "0",
        "attempts": 1,
        "error": None,
        "parsed": "2",
        "ts": "2026-09-08T10:00:00Z",
    }
    replaced = {
        **first,
        "system_prompt_id": 0,
        "repeat": 0,
        "attempts": 3,
        "error": "parse: bad format",
        "parsed": None,
        "ts": "2026-09-11T10:00:00Z",
    }
    second = {**first, "repeat": 1}
    (tmp_path / "m.jsonl").write_text("\n".join(json.dumps(r) for r in [first, replaced, second]))
    cells, items = coverage_tables(str(tmp_path), "nosys", ["A008", "F063"])
    cell = cells.iloc[0]
    assert (cell["raw_record_lines"], cell["recorded_trials"]) == (3, 2)
    assert (cell["recorded_attempts_all_lines"], cell["recorded_attempts_retained_trials"]) == (
        5,
        4,
    )
    assert cell["terminal_parse_failures"] == 1
    assert cell["absent_trial_records"] == 98
    assert not cell["eligible_primary_min_10"]
    assert items.set_index("question").loc["F063", "recorded_trials"] == 0
    assert "refusals" not in cells.columns


def test_votes_never_hide_ties_or_missing_denominators():
    votes = pd.DataFrame([[1, 1, 0], [1, 0, np.nan], [np.nan] * 3])
    labels, counts = vote_counts(votes)
    assert labels.isna().sum() == 2
    assert counts["n_traces"] == 3
    assert counts["n_classified"] == 1
    assert counts["n_ties"] == counts["n_without_votes"] == 1
    assert counts["majority_share_classified"] == 1
    assert counts["positive_share_lower_all_traces"] == pytest.approx(1 / 3)
    assert counts["positive_share_upper_all_traces"] == 1


def traces_and_labels():
    sample, labels = [], []
    for i, llm in enumerate(["model-a", "other"]):
        key = {"llm": llm, "language": "en", "question": "A008", "system_prompt_id": i, "repeat": 0}
        sample.append({**key, "thinking": "x" * (2000 if i == 0 else 10), "error": None})
        for j, rater in enumerate(["model-a", "model-b", "model-c"]):
            value = int(i == 0 and j < 2)
            labels.append(
                {**key, "annotator": rater, "error": None, **{code: value for code in CODES}}
            )
    return pd.DataFrame(sample), pd.DataFrame(labels)


def test_self_rater_exclusion_reports_tie_and_preserves_sample_coverage():
    sample, labels = traces_and_labels()
    sensitivity, coverage, prefixes = trace_diagnostics(labels, sample)
    excluded = sensitivity[sensitivity["analysis"] == "exclude_self_rater"]
    assert (excluded["n_ties"] == 1).all()
    assert (excluded["n_classified"] == 1).all()
    assert (excluded["n_self_votes_excluded"] == 1).all()
    assert coverage["n_at_2000_character_cap"].sum() == 1
    assert coverage["n_complete_llm_panel"].sum() == 2
    assert "typicality_or_moderation" in set(prefixes["code_label"])


def test_trace_labels_outside_frozen_sample_are_rejected():
    sample, labels = traces_and_labels()
    labels.loc[0, "repeat"] = 99
    with pytest.raises(ValueError, match="outside"):
        trace_diagnostics(labels, sample)


@pytest.mark.parametrize("invalid", ["corrupt", "", 2, -1, np.inf])
@pytest.mark.parametrize("annotator", ["model-a", "human"])
def test_invalid_trace_labels_cannot_silently_shrink_the_panel(invalid, annotator):
    sample, labels = traces_and_labels()
    labels[CODES[0]] = labels[CODES[0]].astype(object)
    labels.loc[0, ["annotator", CODES[0]]] = [annotator, invalid]
    with pytest.raises(ValueError, match="labels must be binary or missing"):
        trace_diagnostics(labels, sample)


def test_actual_missing_labels_keep_explicit_panel_denominators():
    sample, labels = traces_and_labels()
    labels.loc[0, CODES[0]] = np.nan
    sensitivity, coverage, _ = trace_diagnostics(labels, sample)
    row = sensitivity.query("code == @CODES[0] and analysis == 'all_llm_raters'").iloc[0]
    assert row["n_traces"] == 2
    assert row["n_ties"] == 1
    assert row["min_available_raters"] == 2
    assert coverage["n_complete_llm_panel"].sum() == 1


def test_trace_agreement_cli_rejects_corruption_before_coercing_it_away(tmp_path, monkeypatch):
    from scripts import trace_agreement_2026

    _, labels = traces_and_labels()
    labels[CODES[0]] = labels[CODES[0]].astype(object)
    labels.loc[0, CODES[0]] = "corrupt"
    path = tmp_path / "labels.csv"
    labels.to_csv(path, index=False)
    monkeypatch.setattr(trace_agreement_2026, "LABELS", path)
    with pytest.raises(ValueError, match="labels must be binary or missing"):
        trace_agreement_2026.main(str(tmp_path / "output"))
    assert not list((tmp_path / "output").iterdir())


def test_prefix_family_mapping_is_unchanged():
    from app.trace_diagnostics import PREFIX_FAMILY

    assert PREFIX_FAMILY == {
        **dict.fromkeys([0, 1, 3, 4, 6, 7], "averaging"),
        **dict.fromkeys([2, 5, 8], "bare"),
        9: "world_citizen",
    }


def test_prompt_sensitivity_requires_the_no_persona_arm(tmp_path, monkeypatch):
    import scripts.prompt_sensitivity_2026 as prompt

    with pytest.raises(FileNotFoundError, match="no-persona"):
        prompt.require_nosys_collection(str(tmp_path / "absent"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no-persona"):
        prompt.require_nosys_collection(str(empty))
    (empty / "m.jsonl").write_text("{}\n")
    assert prompt.require_nosys_collection(str(empty)) == empty

    monkeypatch.setattr(prompt, "NOSYS_DIR", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="no-persona"):
        prompt.main(str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def code_traces_workspace(tmp_path, monkeypatch):
    """code_traces_2026 pointed at a scratch data/ with the two synthetic traces."""
    from scripts import code_traces_2026

    sample, labels = traces_and_labels()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(code_traces_2026, "load_traces", lambda: sample.assign(raw_content="1"))
    return code_traces_2026, labels


def write_panel(path, labels):
    path.write_text("\n".join(json.dumps(r) for r in labels.to_dict("records")) + "\n")


def test_merge_rejects_a_corrupt_code_instead_of_coercing_it_to_missing(tmp_path, monkeypatch):
    code_traces, labels = code_traces_workspace(tmp_path, monkeypatch)
    panel = labels[labels["annotator"] == "model-a"].astype(object)
    panel.iloc[0, panel.columns.get_loc(CODES[0])] = "corrupt"
    write_panel(tmp_path / "data" / "trace_labels_2026__model-a.jsonl", panel)
    with pytest.raises(ValueError, match="labels must be binary or missing"):
        code_traces.merge()
    assert not (tmp_path / "data" / "trace_labels_2026.csv").exists()


def test_merge_reports_uncoded_human_rows_it_drops(tmp_path, monkeypatch, capsys):
    code_traces, labels = code_traces_workspace(tmp_path, monkeypatch)
    write_panel(
        tmp_path / "data" / "trace_labels_2026__model-a.jsonl",
        labels[labels["annotator"] == "model-a"],
    )
    human = labels[labels["annotator"] == "model-b"].drop(columns=["annotator", "error"])
    human = human.astype({CODES[1]: float})
    human.iloc[0, human.columns.get_loc(CODES[1])] = np.nan
    human.to_csv(tmp_path / "data" / "trace_labels_human_2026.csv", index=False)
    assert code_traces.merge() == 0
    assert "dropped 1 of 2 rows with a blank code" in capsys.readouterr().out
    merged = pd.read_csv(tmp_path / "data" / "trace_labels_2026.csv")
    assert (merged["annotator"] == "human").sum() == 1


def test_resume_and_merge_both_raise_on_a_corrupt_label_line(tmp_path, monkeypatch):
    code_traces, labels = code_traces_workspace(tmp_path, monkeypatch)
    record = labels[labels["annotator"] == "model-a"].to_dict("records")[0]
    path = tmp_path / "data" / "trace_labels_2026__model-a.jsonl"
    path.write_text(json.dumps(record) + "\n\n{truncated\n")
    with pytest.raises(ValueError, match=r"trace_labels_2026__model-a\.jsonl:3: corrupt label"):
        code_traces.completed(path)
    with pytest.raises(ValueError, match=r"trace_labels_2026__model-a\.jsonl:3: corrupt label"):
        code_traces.merge()
