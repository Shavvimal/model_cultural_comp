"""Coverage of recorded trials, distinct from failures and unrecorded attempts."""

import json
from pathlib import Path

import pandas as pd

TRIAL_KEY = ["llm", "language", "question", "system_prompt_id", "repeat"]


def coverage_tables(directory: str, condition: str, questions: list[str]):
    """Model/item coverage after type-normalised last-record deduplication.

    Stored attempt totals omit unrecorded transport deferrals and therefore
    are lower bounds on API traffic. Parse failure does not imply refusal.
    """
    rows = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    raw["language"] = raw.get("language", pd.Series(index=raw.index, dtype=str)).fillna("en")
    for col in ["system_prompt_id", "repeat", "attempts"]:
        raw[col] = pd.to_numeric(raw[col], errors="raise").astype(int)
    retained = raw.drop_duplicates(TRIAL_KEY, keep="last")
    cells, items = [], []
    for (llm, language), group in retained.groupby(["llm", "language"], sort=True):
        original = raw[(raw["llm"] == llm) & (raw["language"] == language)]
        cell_items = []
        for question in questions:
            g = group[group["question"] == question]
            errors = g["error"].fillna("").astype(str)
            parsed = g["error"].isna() & g["parsed"].notna()
            row = {
                "condition": condition,
                "llm": llm,
                "language": language,
                "question": question,
                "scheduled_trials": 50,
                "recorded_trials": len(g),
                "absent_trial_records": 50 - len(g),
                "parsed_trials": int(parsed.sum()),
                "terminal_parse_failures": int(errors.str.startswith("parse:").sum()),
                "terminal_other_failures": int(
                    (errors.ne("") & ~errors.str.startswith("parse:")).sum()
                ),
                "recorded_attempts_retained_trials": int(g["attempts"].sum()),
                "meets_primary_min_10": bool(parsed.sum() >= 10),
                "meets_projection_min_1": bool(parsed.sum() >= 1),
            }
            cell_items.append(row)
            items.append(row)
        counts = pd.DataFrame(cell_items)
        totals = {
            key: int(counts[key].sum())
            for key in [
                "scheduled_trials",
                "recorded_trials",
                "absent_trial_records",
                "parsed_trials",
                "terminal_parse_failures",
                "terminal_other_failures",
                "recorded_attempts_retained_trials",
            ]
        }
        dates = group["ts"].dropna().astype(str).str[:10]
        cells.append(
            {
                "condition": condition,
                "llm": llm,
                "language": language,
                **totals,
                "raw_record_lines": len(original),
                "duplicate_record_lines": len(original) - len(group),
                "recorded_attempts_all_lines": int(original["attempts"].sum()),
                "attempt_counts_are_lower_bounds": True,
                "collection_date_first": dates.min(),
                "collection_date_last": dates.max(),
                "min_parsed_per_item": int(counts["parsed_trials"].min()),
                "eligible_primary_min_10": bool(counts["meets_primary_min_10"].all()),
                "estimable_relaxed_min_1": bool(counts["meets_projection_min_1"].all()),
            }
        )
    return pd.DataFrame(cells), pd.DataFrame(items)
