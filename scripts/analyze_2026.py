"""2026-cohort analysis: cluster bootstrap (primary), both region rules,
headline statistics, language-effect contrasts.

Run from the repo root after collect_cloud_2026.py (either arm):

    uv run python scripts/analyze_2026.py

Reads data/collection_2026/*.jsonl directly (both language arms, resumed
runs deduplicated). Reuses the frozen fitted model, so 2024 and 2026
positions share one coordinate space by construction.

Writes:
    data/llm_parse_rates_2026.csv
    data/llm_bootstrap_replicates_2026.csv   cluster bootstrap (primary)
    data/llm_ellipses_2026.csv               cluster + item-bootstrap SDs
    data/llm_regions_2026.csv                both region rules
    data/llm_headline_stats_2026.csv
    data/llm_diagnostics_2026.csv
    data/llm_language_effects_2026.csv       per-model zh - en displacement
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.culture_map import CulturalMap
from app.llm_bootstrap import (
    bootstrap_llm_positions,
    bootstrap_llm_positions_cluster,
    central_tendency_diagnostics,
    centroid_statistics,
    confidence_ellipses,
    load_responses_2026,
)
from app.llm_meta import cohort_2026
from app.region_svm import RegionClassifier

RAW_DIR = Path("data/collection_2026")
N_BOOT_CLUSTER = 10_000
N_BOOT_ITEM = 1000
SEED = 42
MIN_PER_QUESTION = 10


def base_model(label: str) -> str:
    return label.split(" [")[0]


def parse_rates() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        recs = [json.loads(line) for line in path.open()]
        frame = pd.DataFrame(recs)
        frame["language"] = frame.get("language", pd.Series(["en"] * len(frame))).fillna("en")
        frame = frame.drop_duplicates(
            subset=["question", "system_prompt_id", "repeat", "language"], keep="last"
        )
        ok = frame["error"].isna()
        llm = frame["llm"].iloc[0]
        rows.append(
            {
                "llm": llm,
                "language": frame["language"].iloc[0],
                "cohort": cohort_2026(llm),
                "calls": len(frame),
                "parsed": int(ok.sum()),
                "parse_rate": round(float(ok.mean()), 4),
                "min_per_question": int(frame[ok].groupby("question").size().min())
                if ok.any()
                else 0,
            }
        )
    return pd.DataFrame(rows)


def language_effects(ellipses: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    """Per-model zh - en displacement with a bootstrap CI.

    Replicates are independent across arms, so the displacement CI pairs
    replicate i of the zh arm with replicate i of the en arm.
    """
    rows = []
    for label in ellipses["llm"]:
        if not label.endswith(" [zh]"):
            continue
        base = base_model(label)
        en = boot[boot["llm"] == base][["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        zh = boot[boot["llm"] == label][["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        if len(en) == 0 or len(zh) == 0:
            continue
        n = min(len(en), len(zh))
        delta = zh[:n] - en[:n]
        dist = np.linalg.norm(delta, axis=1)
        rows.append(
            {
                "llm": base,
                "cohort": cohort_2026(base),
                "delta_pc1": delta[:, 0].mean(),
                "delta_pc1_lo": np.quantile(delta[:, 0], 0.025),
                "delta_pc1_hi": np.quantile(delta[:, 0], 0.975),
                "delta_pc2": delta[:, 1].mean(),
                "delta_pc2_lo": np.quantile(delta[:, 1], 0.025),
                "delta_pc2_hi": np.quantile(delta[:, 1], 0.975),
                "displacement": dist.mean(),
                "displacement_lo": np.quantile(dist, 0.025),
                "displacement_hi": np.quantile(dist, 0.975),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
    cm.load_model("data/cultural_map_model.npz")
    country_scores = pd.read_csv("data/corrected_country_scores.csv")

    rates = parse_rates()
    print("=== Parse rates (the 2024 limitation, revisited) ===")
    print(rates.to_string(index=False))
    rates.to_csv("data/llm_parse_rates_2026.csv", index=False)

    responses = load_responses_2026(cm, str(RAW_DIR))
    per_item = responses.groupby("llm")["question"].value_counts().unstack(fill_value=0)
    usable = per_item[per_item.min(axis=1) >= MIN_PER_QUESTION].index
    excluded = sorted(set(responses["llm"]) - set(usable))
    if excluded:
        print(
            f"\nexcluded (<{MIN_PER_QUESTION} parsed on some item — an "
            f"outcome-dependent exclusion, reported as such): {', '.join(excluded)}"
        )
    responses = responses[responses["llm"].isin(set(usable))]

    boot = bootstrap_llm_positions_cluster(cm, responses, n_boot=N_BOOT_CLUSTER, seed=SEED)
    ellipses = confidence_ellipses(boot)

    # Item bootstrap alongside, as the explicit lower bound
    item_boot = bootstrap_llm_positions(cm, responses, n_boot=N_BOOT_ITEM, seed=SEED)
    item_sd = (
        confidence_ellipses(item_boot)[["llm", "sd_pc1", "sd_pc2"]]
        .rename(columns={"sd_pc1": "item_sd_pc1", "sd_pc2": "item_sd_pc2"})
    )
    ellipses = ellipses.merge(item_sd, on="llm")

    clf = RegionClassifier().fit(country_scores)
    print(f"\nSVM 5-fold CV accuracy: {clf.cv_accuracy:.3f}")
    regions = clf.region_assignments(boot)
    headline = centroid_statistics(boot, country_scores)
    diagnostics = central_tendency_diagnostics(cm, responses)
    lang_fx = language_effects(ellipses, boot)

    summary = ellipses.merge(regions, on="llm").merge(headline, on="llm")
    summary["cohort"] = summary["llm"].map(lambda x: cohort_2026(base_model(x)))
    cols = [
        "llm", "cohort", "PC1_rescaled", "PC2_rescaled", "sd_pc1", "sd_pc2",
        "item_sd_pc1", "svm_region", "positional_stability", "centroid_region",
        "rules_agree", "dist_human_mean", "pct_countries_closer", "min_dist_nonwestern",
    ]
    with pd.option_context("display.width", 260):
        print("\n=== 2026 positions (cluster bootstrap) ===")
        print(summary[cols].round(3).to_string(index=False))
        if len(lang_fx):
            print("\n=== Language effects (zh - en, per model) ===")
            print(lang_fx.round(3).to_string(index=False))
            print("\nCohort mean displacement:")
            print(lang_fx.groupby("cohort")[["delta_pc1", "delta_pc2", "displacement"]]
                  .mean().round(3))

    boot.to_csv("data/llm_bootstrap_replicates_2026.csv", index=False)
    ellipses.to_csv("data/llm_ellipses_2026.csv", index=False)
    regions.to_csv("data/llm_regions_2026.csv", index=False)
    headline.to_csv("data/llm_headline_stats_2026.csv", index=False)
    diagnostics.to_csv("data/llm_diagnostics_2026.csv", index=False)
    lang_fx.to_csv("data/llm_language_effects_2026.csv", index=False)
    print("\nWrote the six data/llm_*_2026.csv artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
