"""Data-integrity and QC gate for the 2026 collection — analysis-plan §0.

No downstream statistic runs unless the full expected design and stored responses
pass integrity checks. Low parse rates are reported, not fatal.
Reads data/collection_2026/*.jsonl (both arms), normalises field types
(resumed runs serialised repeat/system_prompt_id as both str and int, which
would defeat a naive drop_duplicates), and reports:

  1. design completeness per model x language cell (500 = 10 x 10 x 5)
  2. duplicate audit (duplicates are fatal and are never silently removed)
  3. parse-rate table per model x language x item (flags <95% and <10)
  4. failure taxonomy from raw_content (refusal / format / out-of-range /
     empty), with counts per model x language x item
  5. retry-intensity (attempts) distributions, en vs zh
  6. latency sanity (duration_ms; early en records measured queue-wait
     before the harness fix - treat en latencies as upper bounds)
  7. determinism census (unique values, entropy, items answered
     identically in all repeats)
  8. thinking-trace inventory (which models emit thinking, lengths,
     zh-script share under each arm, leakage heuristic)
  9. index validity for Y002 pairs and Y003 choice lists

Run:  uv run python scripts/qc_2026.py
Writes data/qc_2026_*.csv and prints the gate report.
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.cloud_survey import N_REPEATS, PARSERS, PROMPT_VARIANTS
from app.culture_map import ITEM_VALID_RANGES, IV_QNS
from app.llm_meta import CHINESE_LLMS_2026, WESTERN_LLMS_2026

RAW_DIR = Path("data/collection_2026")
# llm_meta records kimi-k3 as attempted but excluded before any responses:
# its billing-only attempts are not one of the 17 collected model cells.
EXPECTED_MODELS = (CHINESE_LLMS_2026 | WESTERN_LLMS_2026) - {"kimi-k3"}
EXPECTED_LANGUAGES = tuple(PROMPT_VARIANTS)
DESIGN_CALLS = len(IV_QNS) * len(PROMPT_VARIANTS["en"]["persona"]) * N_REPEATS
KEY_COLUMNS = ["llm", "language", "question", "system_prompt_id", "repeat"]
PARSE_TARGET = 0.95
MIN_PER_QUESTION = 10

REFUSAL_MARKERS = [
    "as an ai",
    "i cannot",
    "i can't",
    "i do not have personal",
    "i don't have personal",
    "not able to provide",
    "unable to answer",
    "i'm sorry",
    "i am sorry",
    "not going to answer",
    "can't comply",
    "cannot comply",
    "won't be answering",
    "decline to answer",
    "cannot answer",
    "can't answer",
    "作为人工智能",
    "作为一个ai",
    "我不能",
    "我无法",
    "抱歉",
    "无法回答",
]

CJK_RE = re.compile(r"[一-鿿]")


def _integer(value) -> int:
    """Accept the historical integer/string encodings, never truncate floats."""
    if isinstance(value, bool) or not isinstance(value, (int, str, np.integer)):
        raise ValueError(f"expected integer or integer string, got {value!r}")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value):
        raise ValueError(f"invalid integer string: {value!r}")
    return int(value)


def load_raw() -> pd.DataFrame:
    """Load all terminal records; normalize historical types without deduping."""
    paths = sorted(RAW_DIR.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files in {RAW_DIR}")
    records = []
    required = {
        "llm",
        "question",
        "system_prompt_id",
        "repeat",
        "attempts",
        "duration_ms",
        "raw_content",
        "thinking",
        "parsed",
        "error",
    }
    for path in paths:
        with path.open() as f:
            for lineno, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict) or not required <= rec.keys():
                        raise ValueError("record is not an object with all required fields")
                    rec["_naive_key"] = repr(tuple(rec.get(k) for k in KEY_COLUMNS))
                    rec["language"] = "en" if rec.get("language") is None else rec["language"]
                    if any(
                        not isinstance(rec[k], str) or not rec[k]
                        for k in ("llm", "question", "language")
                    ):
                        raise ValueError("model, item and language must be nonempty strings")
                    for name in ("system_prompt_id", "repeat", "attempts"):
                        rec[name] = _integer(rec[name])
                    rec["duration_ms"] = float(rec["duration_ms"])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path.name}:{lineno}: {exc}") from exc
                rec["_file"] = path.name
                rec["_line"] = lineno
                records.append(rec)
    if not records:
        raise ValueError("no terminal records found")
    df = pd.DataFrame(records)
    # Preserve JSON nulls and integer/list shapes; pandas inference must not
    # turn null errors into NaN or integer answers into floating-point values.
    for name in ("parsed", "error", "prompt_variant"):
        df[name] = pd.Series([rec.get(name) for rec in records], dtype=object)
    # A terminal parse failure may store null content/thinking.
    df["raw_content"] = df["raw_content"].fillna("")
    df["thinking"] = df["thinking"].fillna("")
    return df


def dedup_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Report normalized-key duplicates. None are removed from the corpus."""
    n_dupes = int(df.duplicated(subset=KEY_COLUMNS, keep="last").sum())
    n_naive = int(df["_naive_key"].duplicated(keep="last").sum())
    return pd.DataFrame(
        [
            {
                "duplicates_detected": n_dupes,
                "duplicates_removed": 0,
                "naive_key_would_remove": n_naive,
                "mixed_type_hazard": n_dupes - n_naive,
                "records_before": len(df),
                "records_after": len(df),
            }
        ]
    )


def cell_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Enumerate the full primary design, including entirely absent cells."""
    groups = dict(tuple(df.groupby(["llm", "language"], dropna=False)))
    cells = {(m, lang) for m in EXPECTED_MODELS for lang in EXPECTED_LANGUAGES}
    rows = []
    for llm, lang in cells | set(groups):
        g = groups.get((llm, lang), df.iloc[:0])
        is_expected = (llm, lang) in cells
        expected = (
            {
                (q, s, r)
                for q in IV_QNS
                for s in range(len(PROMPT_VARIANTS[lang]["persona"]))
                for r in range(N_REPEATS)
            }
            if is_expected
            else set()
        )
        got = set(zip(g["question"], g["system_prompt_id"], g["repeat"], strict=True))
        missing, extra = expected - got, got - expected
        duplicates = len(g) - len(got)
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "records": len(g),
                "design": len(expected),
                "expected_cell": is_expected,
                "complete": is_expected and not (missing or extra or duplicates),
                "missing_keys": len(missing),
                "extra_keys": len(extra),
                "duplicate_keys": duplicates,
            }
        )
    return pd.DataFrame(rows).sort_values(["language", "llm"])


def record_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Validate stored answer shape/range and agreement with retained raw text."""
    rows = []
    for rec in df.to_dict("records"):
        problems = []
        qn, parsed, error = rec["question"], rec["parsed"], rec["error"]
        if rec["llm"] not in EXPECTED_MODELS:
            problems.append("unknown/uncollected model")
        if rec["language"] not in EXPECTED_LANGUAGES:
            problems.append("unknown language")
        if rec["prompt_variant"] not in (None, "persona"):
            problems.append("non-persona trial in primary corpus")
        if rec["attempts"] < 1 or not np.isfinite(rec["duration_ms"]) or rec["duration_ms"] < 0:
            problems.append("invalid attempts or duration")
        if not isinstance(rec["raw_content"], str) or not isinstance(rec["thinking"], str):
            problems.append("content/thinking must be text")
        if rec.get("transport_failure", False) is True:
            problems.append("transport-only task is not a terminal answer")
        if error is None:
            try:
                if qn not in PARSERS:
                    raise ValueError("unknown item")
                if qn in ("Y002", "Y003"):
                    if (
                        not isinstance(parsed, list)
                        or not parsed
                        or any(type(v) is not int for v in parsed)
                    ):
                        raise ValueError("stored choices must be a nonempty integer list")
                elif type(parsed) is not int:
                    raise ValueError("stored scalar must be an integer")
                reparsed = PARSERS[qn].parse(rec["raw_content"])
                if isinstance(reparsed, tuple):
                    reparsed = list(reparsed)
                if reparsed != parsed:
                    raise ValueError("stored answer disagrees with reparsed raw content")
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                problems.append(f"invalid parsed response: {exc}")
        elif not isinstance(error, str) or not error or parsed is not None:
            problems.append("failed terminal record needs a nonempty error and null parsed value")
        if problems:
            rows.append(
                {
                    "file": rec["_file"],
                    "line": rec["_line"],
                    **{k: rec[k] for k in KEY_COLUMNS},
                    "problems": "; ".join(problems),
                }
            )
    return pd.DataFrame(rows, columns=["file", "line", *KEY_COLUMNS, "problems"])


def parse_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (llm, lang, qn), g in df.groupby(["llm", "language", "question"]):
        parsed = int(g["error"].isna().sum())
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "question": qn,
                "attempted": len(g),
                "parsed": parsed,
                "parse_rate": round(parsed / len(g), 4),
                "below_target": parsed / len(g) < PARSE_TARGET,
                "below_min": parsed < MIN_PER_QUESTION,
            }
        )
    return pd.DataFrame(rows)


def classify_failure(raw: str, error: str) -> str:
    low = raw.strip().lower().replace("\u2019", "'")
    if not low:
        return "empty"
    if any(m in low for m in REFUSAL_MARKERS):
        return "refusal"
    if error and "out of range" in error.lower():
        return "out_of_range"
    return "format"


def failure_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[df["error"].notna()].copy()
    if bad.empty:
        return pd.DataFrame(columns=["llm", "language", "question", "category", "n", "example"])
    bad["category"] = [
        classify_failure(r, e) for r, e in zip(bad["raw_content"], bad["error"], strict=True)
    ]
    rows = []
    for (llm, lang, qn, cat), g in bad.groupby(["llm", "language", "question", "category"]):
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "question": qn,
                "category": cat,
                "n": len(g),
                "example": g["raw_content"].iloc[0][:160],
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def retry_intensity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (llm, lang), g in df.groupby(["llm", "language"]):
        a = g["attempts"]
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "mean_attempts": round(float(a.mean()), 3),
                "share_retried": round(float((a > 1).mean()), 4),
                "p95_attempts": int(a.quantile(0.95)),
                "max_attempts": int(a.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["language", "llm"])


def latency_sanity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (llm, lang), g in df.groupby(["llm", "language"]):
        d = g["duration_ms"]
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "median_ms": int(d.median()),
                "p95_ms": int(d.quantile(0.95)),
                "max_ms": int(d.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["language", "llm"])


def _entropy(values: pd.Series) -> float:
    p = values.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def determinism_census(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["error"].isna()]
    rows = []
    for (llm, lang), g in ok.groupby(["llm", "language"]):
        per_item = g.groupby("question")["raw_content"]
        n_constant = int((per_item.nunique() == 1).sum())
        mean_h = float(per_item.apply(_entropy).mean())
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "items_constant_all_calls": n_constant,
                "mean_item_entropy_bits": round(mean_h, 3),
            }
        )
    return pd.DataFrame(
        rows, columns=["llm", "language", "items_constant_all_calls", "mean_item_entropy_bits"]
    ).sort_values(["language", "llm"])


def thinking_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (llm, lang), g in df.groupby(["llm", "language"]):
        has = g["thinking"].str.len() > 0
        lens = g.loc[has, "thinking"].str.len()
        cjk = (
            float(g.loc[has, "thinking"].apply(lambda t: bool(CJK_RE.search(t))).mean())
            if has.any()
            else math.nan
        )
        # Leakage heuristic: parsed numeric answers whose raw_content is
        # suspiciously long (>40 chars) suggest think-block bleed-through.
        ok = g[g["error"].isna() & ~g["question"].isin(["Y002", "Y003"])]
        leaky = int((ok["raw_content"].str.len() > 40).sum())
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "share_with_thinking": round(float(has.mean()), 3),
                "median_thinking_chars": int(lens.median()) if has.any() else 0,
                "p95_thinking_chars": int(lens.quantile(0.95)) if has.any() else 0,
                "share_thinking_cjk": round(cjk, 3) if has.any() else math.nan,
                "long_rawcontent_parsed_rows": leaky,
            }
        )
    return pd.DataFrame(rows).sort_values(["language", "llm"])


def index_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Verify every parsed value lands in its item's valid projected range."""
    from app.culture_map import CulturalMap

    ok = df[df["error"].isna()]
    rows = []
    for qn, g in ok.groupby("question"):
        lo, hi = ITEM_VALID_RANGES[qn]
        if qn == "Y002":
            vals = [
                CulturalMap.y002_transform(tuple(v) if isinstance(v, list) else v)
                for v in g["parsed"]
            ]
        elif qn == "Y003":
            vals = [CulturalMap.y003_transform(v) for v in g["parsed"]]
        else:
            vals = pd.to_numeric(g["parsed"], errors="raise")
        arr = np.asarray(vals, dtype=float)
        bad = int(((arr < lo) | (arr > hi)).sum())
        rows.append(
            {
                "question": qn,
                "n_parsed": len(arr),
                "out_of_range": bad,
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    return pd.DataFrame(rows, columns=["question", "n_parsed", "out_of_range", "min", "max"])


def main() -> int:
    try:
        df = load_raw()
    except (OSError, ValueError, KeyError) as exc:
        print(f"QC FAILED: {exc}", file=sys.stderr)
        return 1
    dedup_report = dedup_audit(df)
    cells = cell_completeness(df)
    integrity = record_integrity(df)
    preliminary = {
        "qc_2026_dedup": dedup_report,
        "qc_2026_cells": cells,
        "qc_2026_integrity_errors": integrity,
    }
    for name, frame in preliminary.items():
        frame.to_csv(f"data/{name}.csv", index=False)
    if not cells["complete"].all() or not integrity.empty:
        print(
            "QC FAILED: incomplete/extra/duplicate design keys or invalid responses.",
            file=sys.stderr,
        )
        print(cells.loc[~cells["complete"]].to_string(index=False))
        if not integrity.empty:
            print(integrity.head(30).to_string(index=False))
        return 1

    artefacts = {
        **preliminary,
        "qc_2026_parse_rates": parse_rate_table(df),
        "qc_2026_failures": failure_taxonomy(df),
        "qc_2026_attempts": retry_intensity(df),
        "qc_2026_latency": latency_sanity(df),
        "qc_2026_determinism": determinism_census(df),
        "qc_2026_thinking": thinking_inventory(df),
        "qc_2026_index_validity": index_validity(df),
    }

    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print("=== 1-2. Dedup audit ===")
        print(dedup_report.to_string(index=False))
        print("\n=== 1. Cell completeness ===")
        print(artefacts["qc_2026_cells"].to_string(index=False))
        pr = artefacts["qc_2026_parse_rates"]
        print("\n=== 3. Parse-rate flags (below 95% target or <10 parsed) ===")
        flagged = pr[pr["below_target"] | pr["below_min"]]
        print(flagged.to_string(index=False) if len(flagged) else "none flagged")
        print("\n=== 4. Failure taxonomy (top 30 by count) ===")
        ft = artefacts["qc_2026_failures"]
        print(ft.head(30).to_string(index=False) if len(ft) else "no failures")
        print("\n=== 5. Retry intensity ===")
        print(artefacts["qc_2026_attempts"].to_string(index=False))
        print("\n=== 6. Latency (en records pre-fix measure queue-wait; upper bounds) ===")
        print(artefacts["qc_2026_latency"].to_string(index=False))
        print("\n=== 7. Determinism census ===")
        print(artefacts["qc_2026_determinism"].to_string(index=False))
        print("\n=== 8. Thinking inventory ===")
        print(artefacts["qc_2026_thinking"].to_string(index=False))
        print("\n=== 9. Index validity ===")
        print(artefacts["qc_2026_index_validity"].to_string(index=False))

    for name, frame in artefacts.items():
        frame.to_csv(f"data/{name}.csv", index=False)
    print(f"\nWrote {len(artefacts)} data/qc_2026_*.csv artefacts.")
    print("QC PASSED: the full primary design and stored responses are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
