"""Coverage of recorded trials, distinct from failures and unrecorded attempts.

This module reads the corpus with its own loader rather than
``app.llm_bootstrap.load_trial_records``, because the two differ on inputs the
retained corpora never contain, and the published coverage tables were built
with this one:

* lines are split with ``str.splitlines`` and blank lines are skipped, where
  the shared loader iterates the file and rejects a blank line as invalid JSON;
* ``system_prompt_id``, ``repeat`` and ``attempts`` must be integers or integer
  strings, where the shared loader applies ``astype(int)``;
* a missing directory raises ``FileNotFoundError`` and a directory with no
  records raises ``ValueError``;
* both the raw lines and the deduplicated trials are kept, for the
  duplicate-line counts.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.study_design import DESIGN_CALLS, MIN_PER_QUESTION, TRIAL_KEY


def _integer_column(raw: pd.DataFrame, name: str) -> pd.Series:
    """Return ``raw[name]`` as integers, rejecting values that would be truncated.

    Accepts integers and integer strings (resumed runs serialised both); raises
    ``ValueError`` naming a float, boolean, null or non-integer string.
    """
    values = []
    for value in raw[name]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer, str))
            or (isinstance(value, str) and not re.fullmatch(r"[+-]?[0-9]+", value))
        ):
            raise ValueError(f"{name} must be an integer or integer string, got {value!r}")
        values.append(int(value))
    return pd.Series(values, index=raw.index, dtype=int)


def coverage_tables(
    directory: str, condition: str, questions: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model/item coverage after type-normalised last-record deduplication.

    Stored attempt totals omit unrecorded transport deferrals and therefore
    are lower bounds on API traffic. Parse failure does not imply refusal.
    Raises ``FileNotFoundError`` for a missing directory and ``ValueError`` when
    it holds no terminal records, rather than returning empty tables.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"coverage directory {directory!r} does not exist")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise ValueError(f"no terminal records in {directory!r}; expected *.jsonl trial records")
    raw = pd.DataFrame(rows)
    raw["language"] = raw.get("language", pd.Series(index=raw.index, dtype=str)).fillna("en")
    for col in ["system_prompt_id", "repeat", "attempts"]:
        raw[col] = _integer_column(raw, col)
    # Keep-last deduplication in file order is the published rule; changing it
    # alters prompt_control_*coverage_2026.csv. Its only effect on the retained
    # corpora: 82 duplicated nemotron-3-ultra no-persona keys, where 15 parsed
    # records are superseded by later failures and 2 failures by later parses.
    retained = raw.drop_duplicates(list(TRIAL_KEY), keep="last")
    cells: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
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
                "scheduled_trials": DESIGN_CALLS,
                "recorded_trials": len(g),
                "absent_trial_records": DESIGN_CALLS - len(g),
                "parsed_trials": int(parsed.sum()),
                "terminal_parse_failures": int(errors.str.startswith("parse:").sum()),
                "terminal_other_failures": int(
                    (errors.ne("") & ~errors.str.startswith("parse:")).sum()
                ),
                "recorded_attempts_retained_trials": int(g["attempts"].sum()),
                # The column name records the released rule, MIN_PER_QUESTION = 10.
                "meets_primary_min_10": bool(parsed.sum() >= MIN_PER_QUESTION),
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
