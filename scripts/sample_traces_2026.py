"""Draw the reproducible reasoning-trace sample for qualitative coding.

Analysis-plan §2 (reasoning-trace analysis): >=30 traces per model-language
cell, stratified across the ten items (3 per item), seed 42, plus every
failed (refusal/format) row's raw text for the refusal-phrasing taxonomy.

Run:  uv run python scripts/sample_traces_2026.py
Writes data/trace_samples_2026.json — one entry per sampled record with
llm, language, question, system_prompt_id, repeat, thinking, raw_content,
parsed, error. The frozen study sample is distributed separately; do not
resample it during reproduction. Five LLM panels label its 900 successful
excerpts through code_traces_2026.py; the human worksheet remains uncoded.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.study_design import TRIAL_KEY

RAW_DIR = Path("data/collection_2026")
OUT = Path("data/trace_samples_2026.json")
PER_ITEM = 3
SEED = 42
FIELDS = [
    "llm",
    "language",
    "question",
    "system_prompt_id",
    "repeat",
    "thinking",
    "raw_content",
    "parsed",
    "error",
]


def main() -> int:
    records = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with path.open() as f:
            records.extend(json.loads(line) for line in f)
    df = pd.DataFrame(records)
    df["language"] = df.get("language", pd.Series([None] * len(df))).fillna("en")
    df["system_prompt_id"] = df["system_prompt_id"].astype(int)
    df["repeat"] = df["repeat"].astype(int)
    df = df.drop_duplicates(subset=list(TRIAL_KEY), keep="last")
    df["thinking"] = df["thinking"].fillna("")

    rng = np.random.default_rng(SEED)
    sampled: list[dict] = []
    for (llm, lang), g in df.groupby(["llm", "language"]):
        with_thinking = g[(g["thinking"].str.len() > 0) & g["error"].isna()]
        for _, gq in with_thinking.groupby("question"):
            take = min(PER_ITEM, len(gq))
            idx = rng.choice(gq.index.to_numpy(), size=take, replace=False)
            sampled.extend(gq.loc[idx, FIELDS].to_dict("records"))
        failures = g[g["error"].notna()]
        sampled.extend(failures[FIELDS].to_dict("records"))
        n_traces = min(PER_ITEM, 5) * with_thinking["question"].nunique()
        print(f"{llm} [{lang}]: {n_traces} traces + {len(failures)} failure rows")

    # JSON null represents missing fields; pandas NaN is not valid JSON.
    records = pd.DataFrame(sampled).astype(object)
    records = records.where(records.notna(), None).to_dict("records")
    OUT.write_text(json.dumps(records, ensure_ascii=False, allow_nan=False, indent=1))
    print(f"\nwrote {len(sampled)} records to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
