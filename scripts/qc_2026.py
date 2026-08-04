"""Data-integrity and QC gate for the 2026 collection — analysis-plan §0.

No downstream statistic runs until every check here has been reported.
Reads data/collection_2026/*.jsonl (both arms), normalises field types
(resumed runs serialised repeat/system_prompt_id as both str and int, which
would defeat a naive drop_duplicates), and reports:

  1. design completeness per model x language cell (500 = 10 x 10 x 5)
  2. dedup audit (duplicates removed; mixed-type duplicates a naive key
     would have missed)
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

from app.culture_map import ITEM_VALID_RANGES, IV_QNS

RAW_DIR = Path("data/collection_2026")
DESIGN_CALLS = 500  # 10 items x 10 variants x 5 repeats
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
    "作为人工智能",
    "作为一个ai",
    "我不能",
    "我无法",
    "抱歉",
    "无法回答",
]

CJK_RE = re.compile(r"[一-鿿]")


def load_raw() -> pd.DataFrame:
    """Load every record with types normalised; keep error rows."""
    paths = sorted(RAW_DIR.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files in {RAW_DIR}")
    records = []
    for path in paths:
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                rec["_file"] = path.name
                records.append(rec)
    df = pd.DataFrame(records)
    df["language"] = df.get("language", pd.Series([None] * len(df))).fillna("en")
    # Resumed runs serialised these as str; originals as int. Normalise
    # before any dedup or grouping, or duplicates survive the key.
    df["system_prompt_id"] = df["system_prompt_id"].astype(int)
    df["repeat"] = df["repeat"].astype(int)
    df["attempts"] = pd.to_numeric(df["attempts"], errors="raise").astype(int)
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="raise").astype(float)
    df["raw_content"] = df["raw_content"].fillna("")
    df["thinking"] = df["thinking"].fillna("")
    return df


def dedup_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep-last dedup on the normalised key; report what was removed."""
    key = ["llm", "language", "question", "system_prompt_id", "repeat"]
    n_dupes = int(df.duplicated(subset=key, keep="last").sum())
    # What a naive (un-normalised) key would have kept: recompute with the
    # original mixed-type values to quantify the hazard.
    naive = df.assign(
        _sp=df["system_prompt_id"].astype(str),
        _rp=df["repeat"].astype(str),
    )
    n_naive = int(
        naive.duplicated(subset=["llm", "language", "question", "_sp", "_rp"], keep="last").sum()
    )
    deduped = df.drop_duplicates(subset=key, keep="last")
    report = pd.DataFrame(
        [
            {
                "duplicates_removed": n_dupes,
                "naive_key_would_remove": n_naive,
                "mixed_type_hazard": n_dupes - n_naive,
                "records_before": len(df),
                "records_after": len(deduped),
            }
        ]
    )
    return deduped, report


def cell_completeness(df: pd.DataFrame) -> pd.DataFrame:
    expected = {(q, s, r) for q in IV_QNS for s in range(10) for r in range(5)}
    rows = []
    for (llm, lang), g in df.groupby(["llm", "language"]):
        got = set(zip(g["question"], g["system_prompt_id"], g["repeat"], strict=True))
        missing = expected - got
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "records": len(g),
                "design": DESIGN_CALLS,
                "complete": not missing,
                "missing_keys": len(missing),
                "extra_keys": len(got - expected),
            }
        )
    return pd.DataFrame(rows).sort_values(["language", "llm"])


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
    low = raw.strip().lower()
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
    return pd.DataFrame(rows).sort_values(["language", "llm"])


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
    return pd.DataFrame(rows)


def main() -> int:
    df = load_raw()
    deduped, dedup_report = dedup_audit(df)

    artefacts = {
        "qc_2026_dedup": dedup_report,
        "qc_2026_cells": cell_completeness(deduped),
        "qc_2026_parse_rates": parse_rate_table(deduped),
        "qc_2026_failures": failure_taxonomy(deduped),
        "qc_2026_attempts": retry_intensity(deduped),
        "qc_2026_latency": latency_sanity(deduped),
        "qc_2026_determinism": determinism_census(deduped),
        "qc_2026_thinking": thinking_inventory(deduped),
        "qc_2026_index_validity": index_validity(deduped),
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
