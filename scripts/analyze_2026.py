"""Phase-3 analysis: project the 2026 cloud models onto the corrected map.

Run from the repo root after collect_cloud_2026.py has finished:

    uv run python scripts/analyze_2026.py

Reuses the fitted model from data/cultural_map_model.npz (so 2024 and 2026
positions share one coordinate space by construction) and writes:

    data/llm_ellipses_2026.csv           mean + 95% ellipse per model
    data/llm_region_stability_2026.csv   SVM region assignment + stability
    data/llm_parse_rates_2026.csv        per-model parse success rates
    figures/fig3_cultural_map_2026.{pdf,png}
"""

import json
import sys
from pathlib import Path

import pandas as pd

from app.culture_map import CulturalMap
from app.llm_bootstrap import (
    bootstrap_llm_positions,
    confidence_ellipses,
    load_transformed_responses,
)
from app.llm_meta import cohort_2026
from app.region_svm import RegionClassifier

RAW_DIR = Path("data/collection_2026")
N_BOOT = 1000
SEED = 42
MIN_PER_QUESTION = 10  # a model needs at least this many parsed responses per item


def parse_rates() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        recs = [json.loads(line) for line in path.open()]
        frame = pd.DataFrame(recs).drop_duplicates(
            subset=["question", "system_prompt_id", "repeat"], keep="last"
        )
        ok = frame["error"].isna()
        rows.append(
            {
                "llm": frame["llm"].iloc[0],
                "cohort": cohort_2026(frame["llm"].iloc[0]),
                "calls": len(frame),
                "parsed": int(ok.sum()),
                "parse_rate": round(ok.mean(), 4),
                "min_per_question": int(frame[ok].groupby("question").size().min())
                if ok.any()
                else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl", data_dir="data")
    cm.load_model("data/cultural_map_model.npz")
    country_scores = pd.read_csv("data/corrected_country_scores.csv")

    rates = parse_rates()
    print("=== Parse rates (the 2024 limitation, revisited) ===")
    print(rates.to_string(index=False))
    rates.to_csv("data/llm_parse_rates_2026.csv", index=False)

    usable = rates[rates["min_per_question"] >= MIN_PER_QUESTION]["llm"]
    excluded = sorted(set(rates["llm"]) - set(usable))
    if excluded:
        print(f"\nexcluded (fewer than {MIN_PER_QUESTION} parsed responses on "
              f"some item): {', '.join(excluded)}")

    responses = load_transformed_responses(cm, str(RAW_DIR / "pickles"))
    responses = responses[responses["llm"].isin(set(usable))]

    boot = bootstrap_llm_positions(cm, responses, n_boot=N_BOOT, seed=SEED)
    ellipses = confidence_ellipses(boot)
    ellipses["cohort"] = ellipses["llm"].map(cohort_2026)

    clf = RegionClassifier().fit(country_scores)
    stability = clf.region_stability(boot)

    summary = ellipses.merge(stability, on="llm")
    cols = ["llm", "cohort", "PC1_rescaled", "PC2_rescaled", "sd_pc1", "sd_pc2",
            "region", "stability", "runner_up", "runner_up_share"]
    with pd.option_context("display.width", 220):
        print("\n=== 2026 positions ===")
        print(summary[cols].round(3).to_string(index=False))

    ellipses.to_csv("data/llm_ellipses_2026.csv", index=False)
    stability.to_csv("data/llm_region_stability_2026.csv", index=False)
    boot.to_csv("data/llm_bootstrap_replicates_2026.csv", index=False)

    print("\n=== Cohort means ===")
    print(summary.groupby("cohort")[["PC1_rescaled", "PC2_rescaled"]].mean().round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
