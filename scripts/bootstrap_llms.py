"""Phase-2: bootstrap confidence ellipses + SVM region stability.

Run from the repo root after scripts/validate_projection.py has written
data/cultural_map_model.npz:

    uv run python scripts/bootstrap_llms.py

Writes:
    data/llm_bootstrap_replicates.csv   one row per (llm, replicate)
    data/llm_ellipses.csv               mean + 95% ellipse per model
    data/llm_region_stability.csv       SVM region assignment with stability
"""

import sys

import pandas as pd

from app.culture_map import CulturalMap
from app.llm_bootstrap import (
    bootstrap_llm_positions,
    confidence_ellipses,
    load_transformed_responses,
)
from app.region_svm import RegionClassifier

N_BOOT = 1000
SEED = 42


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl", data_dir="data")
    cm.load_model("data/cultural_map_model.npz")
    country_scores = pd.read_csv("data/corrected_country_scores.csv")

    responses = load_transformed_responses(cm, "data/collection")
    print(f"{responses['llm'].nunique()} models, {len(responses)} stored responses")

    boot = bootstrap_llm_positions(cm, responses, n_boot=N_BOOT, seed=SEED)
    ellipses = confidence_ellipses(boot)

    clf = RegionClassifier().fit(country_scores)
    stability = clf.region_stability(boot)

    summary = ellipses.merge(stability, on="llm")
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string(index=False))

    boot.to_csv("data/llm_bootstrap_replicates.csv", index=False)
    ellipses.to_csv("data/llm_ellipses.csv", index=False)
    stability.to_csv("data/llm_region_stability.csv", index=False)
    print("\nWrote data/llm_bootstrap_replicates.csv, data/llm_ellipses.csv, "
          "data/llm_region_stability.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
