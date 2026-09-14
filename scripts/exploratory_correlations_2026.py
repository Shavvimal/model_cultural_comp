"""Recompute explicitly recovered exploratory correlations from release files.

The historical seven-member family comprises midpoint distance against each
axis; stored excerpt length against midpoint distance and each axis; and each
arm's length against the paired model's observed-vector displacement norm.
The 11 September repair is not a retrospective pre-registration. Historical
definitions are recorded in exploration-2026.md:267-279 and the dated plan.

The historical post-hoc entropy/midpoint result used exact answer-string
entropy, whereas the single planned item-SD diagnostic used transformed-index
entropy. Both are named; numeric-index entropy/midpoint is a separately labelled
definition sensitivity. No ambiguous six-member mechanism family is fabricated.
Cell-level Spearman p-values are nominal: same-model dependence is not modelled.
Nonsignificance does not identify a mechanism or establish equivalence.

Reads only released CSVs and persona JSONLs, with no fitted NPZ or survey rows.
Writes diag_2026_exploratory_correlations.csv and the per-cell input features.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.stats import bh_adjust
from app.study_design import TRIAL_KEY

ROOT = Path(__file__).resolve().parents[1]
KEY = list(TRIAL_KEY)
# The historical exploratory family has exactly seven declared tests.
HISTORICAL_FAMILY_SIZE = 7
CAP = 2000


def entropy_bits(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy()
    return float(-(probabilities * np.log2(probabilities)).sum())


def cell_label(model: str, language: str) -> str:
    if language not in {"en", "zh"}:
        raise ValueError(f"unexpected administration language {language!r}")
    return model + (" [zh]" if language == "zh" else "")


def raw_features(directory: Path) -> pd.DataFrame:
    """QC-compatible exact-string entropy and censored stored-excerpt lengths.

    Length medians use nonempty excerpts from all retained terminal records,
    including failures, matching the historical QC inventory. This differs
    from the selected 900-successful-excerpt annotation sample. No empty
    excerpt is replaced by a zero-length reasoning observation.
    """
    rows = []
    fields = [*KEY, "error", "parsed", "raw_content", "thinking"]
    for path in sorted(directory.glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    rows.append({field: record.get(field) for field in fields})
    raw = pd.DataFrame(rows, columns=fields)
    if raw.empty:
        raise ValueError(f"no raw records in {directory}")
    raw["language"] = raw["language"].fillna("en")
    for column in ["system_prompt_id", "repeat"]:
        raw[column] = pd.to_numeric(raw[column], errors="raise").astype(int)
    raw = raw.drop_duplicates(KEY, keep="last")
    features = []
    for (model, language), group in raw.groupby(["llm", "language"], sort=True):
        lengths = group["thinking"].fillna("").astype(str).str.len()
        lengths = lengths[lengths > 0]
        ok = group[group["error"].isna() & group["parsed"].notna()]
        entropy = ok.assign(raw_content=ok["raw_content"].fillna(""))
        per_item = entropy.groupby("question")["raw_content"].apply(entropy_bits)
        features.append(
            {
                "llm": cell_label(model, language),
                "model": model,
                "language": language,
                "n_terminal_records": len(group),
                "n_parsed_records": len(ok),
                "n_entropy_items": len(per_item),
                "raw_answer_string_entropy_bits": per_item.mean(),
                "median_stored_excerpt_chars": lengths.median() if len(lengths) else np.nan,
                "n_nonempty_excerpts": len(lengths),
                "n_excerpts_at_2000_character_cap": int(lengths.ge(CAP).sum()),
                "excerpt_scope": "all_retained_terminal_records_including_failures_nonempty_only",
            }
        )
    return pd.DataFrame(features)


def correlation(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    test_id: str,
    family: str,
    unit: str,
    status: str,
) -> dict:
    values = frame[[x, y]].apply(pd.to_numeric, errors="raise")
    valid = np.isfinite(values.to_numpy()).all(axis=1)
    selected = values.loc[valid]
    defined = len(selected) >= 3 and (selected.nunique() > 1).all()
    rho, p = spearmanr(selected[x], selected[y]) if defined else (np.nan, np.nan)
    return {
        "test_id": test_id,
        "family": family,
        "historical_status": status,
        "unit": unit,
        "x_measure": x,
        "y_measure": y,
        "n_candidate_units": len(frame),
        "n_used": int(valid.sum()),
        "n_excluded_nonfinite": int((~valid).sum()),
        "n_distinct_models": frame.loc[valid, "model"].nunique(),
        "rho_spearman": rho,
        "p_raw_two_sided": p,
        "estimate_status": "estimated" if defined else "insufficient_or_constant_inputs",
        "included_units_json": json.dumps(frame.loc[valid, "unit_id"].tolist()),
        "excluded_units_json": json.dumps(frame.loc[~valid, "unit_id"].tolist()),
        "p_scope": "nominal_cell_independence_same_model_dependence_unmodelled"
        if unit == "model_language_cell"
        else "nominal_across_selected_models_not_population_or_causal_inference",
        "point_estimator": "projected_observed_item_means_not_bootstrap_mean_or_folded_norm",
        "interpretation": "exploratory_association_not_mechanism_equivalence_or_causal_test",
    }


def compute_correlations(
    points: pd.DataFrame, diagnostics: pd.DataFrame, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, frame in [
        ("positions", points),
        ("diagnostics", diagnostics),
        ("features", features),
    ]:
        if frame["llm"].duplicated().any():
            raise ValueError(f"duplicate {name} cell labels")
    for name, frame in [("diagnostics", diagnostics), ("features", features)]:
        if not set(points["llm"]).issubset(frame["llm"]):
            raise ValueError(f"{name} is missing a primary eligible cell")
    cells = points.merge(diagnostics, on="llm", validate="one_to_one").merge(
        features, on="llm", validate="one_to_one"
    )
    cells = cells.sort_values("llm").reset_index(drop=True)
    cells["unit_id"] = cells["llm"]
    cells["numeric_index_entropy_bits"] = cells["mean_item_entropy"]
    if (cells[["item_sd_pc1", "item_sd_pc2"]] < 0).any().any():
        raise ValueError("item SDs must be nonnegative")
    cells["geometric_mean_item_sd"] = np.sqrt(cells["item_sd_pc1"] * cells["item_sd_pc2"])
    rows = []
    seven = [
        ("midpoint_pc1", "midpoint_distance", "PC1_rescaled"),
        ("midpoint_pc2", "midpoint_distance", "PC2_rescaled"),
        ("length_midpoint", "median_stored_excerpt_chars", "midpoint_distance"),
        ("length_pc1", "median_stored_excerpt_chars", "PC1_rescaled"),
        ("length_pc2", "median_stored_excerpt_chars", "PC2_rescaled"),
    ]
    for ident, x, y in seven:
        rows.append(
            correlation(
                cells,
                x,
                y,
                test_id=ident,
                family="historical_seven",
                unit="model_language_cell",
                status="historical_family_recomputed_2026_09_11",
            )
        )
    en = cells[cells["language"] == "en"].set_index("model")
    zh = cells[cells["language"] == "zh"].set_index("model")
    models = en.index.intersection(zh.index).sort_values()
    paired = pd.DataFrame({"model": models, "unit_id": models})
    delta = (
        zh.loc[models, ["PC1_rescaled", "PC2_rescaled"]]
        - en.loc[models, ["PC1_rescaled", "PC2_rescaled"]]
    )
    paired["observed_vector_displacement_norm"] = np.linalg.norm(delta.to_numpy(), axis=1)
    for language, arm in [("en", en), ("zh", zh)]:
        measure = f"median_stored_excerpt_chars_{language}"
        paired[measure] = arm.loc[models, "median_stored_excerpt_chars"].to_numpy()
        rows.append(
            correlation(
                paired,
                measure,
                "observed_vector_displacement_norm",
                test_id=f"length_{language}_displacement",
                family="historical_seven",
                unit="paired_model",
                status="historical_family_recomputed_2026_09_11",
            )
        )
    for ident, x, y, family, status in [
        (
            "raw_string_entropy_midpoint",
            "raw_answer_string_entropy_bits",
            "midpoint_distance",
            "historical_posthoc",
            "posthoc_not_in_historical_seven",
        ),
        (
            "numeric_entropy_midpoint_sensitivity",
            "numeric_index_entropy_bits",
            "midpoint_distance",
            "definition_sensitivity",
            "newly_explicit_entropy_definition_sensitivity_not_historical_test",
        ),
        (
            "numeric_entropy_geometric_item_sd",
            "numeric_index_entropy_bits",
            "geometric_mean_item_sd",
            "single_planned_diagnostic",
            "historical_single_planned_diagnostic_recomputed",
        ),
    ]:
        rows.append(
            correlation(
                cells, x, y, test_id=ident, family=family, unit="model_language_cell", status=status
            )
        )
    result = pd.DataFrame(rows)
    result["p_bh"] = np.nan
    result["bh_family_size"] = 0
    result["adjustment"] = "none_separate_single_test_status_stated"
    selected = result["family"].eq("historical_seven")
    result.loc[selected, "p_bh"] = bh_adjust(
        result.loc[selected, "p_raw_two_sided"].to_numpy(), family_size=HISTORICAL_FAMILY_SIZE
    )
    result.loc[selected, "bh_family_size"] = HISTORICAL_FAMILY_SIZE
    result.loc[selected, "adjustment"] = "BH_over_exactly_seven_historical_tests"
    return result, cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    directory = args.data_dir
    result, features = compute_correlations(
        pd.read_csv(directory / "llm_ellipses_2026.csv"),
        pd.read_csv(directory / "llm_diagnostics_2026.csv"),
        raw_features(directory / "collection_2026"),
    )
    output = args.output_dir or directory
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "diag_2026_exploratory_correlations.csv", index=False)
    features.to_csv(output / "diag_2026_exploratory_features.csv", index=False)
    print(
        result[
            ["test_id", "family", "n_used", "rho_spearman", "p_raw_two_sided", "p_bh"]
        ].to_string(index=False)
    )
    print(
        "Six-mechanism family omitted: original CJK-share denominator/aggregation not recovered unambiguously."
    )
    print(f"Wrote exploratory correlations and input features to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
