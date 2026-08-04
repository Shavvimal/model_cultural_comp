"""2024-cohort analysis: item bootstrap, region rules, headline statistics.

Run from the repo root after scripts/validate_projection.py:

    uv run python scripts/bootstrap_llms.py

The 2024 corpus is split by elicitation language (``c-*`` pickles are
Chinese administrations, labelled ``<llm> [zh]``) and never pooled. Only
the item bootstrap is available here — the 2024 harness did not record the
prompt-variant id — so every ellipse is a lower bound on the true
uncertainty (see docs/statistical-review.md §2.4).

Writes:
    data/llm_bootstrap_replicates.csv
    data/llm_ellipses.csv
    data/llm_regions.csv           both region rules + positional stability
    data/llm_headline_stats.csv    classifier-free centroid statistics
    data/llm_diagnostics.csv       midpoint distance + response entropy
"""

import sys

import pandas as pd

from app.culture_map import CulturalMap
from app.llm_bootstrap import (
    bootstrap_llm_positions,
    central_tendency_diagnostics,
    centroid_statistics,
    confidence_ellipses,
    load_transformed_responses,
)
from app.region_svm import RegionClassifier

N_BOOT = 1000
SEED = 42


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
    cm.load_model("data/cultural_map_model.npz")
    country_scores = pd.read_csv("data/corrected_country_scores.csv")

    responses = load_transformed_responses(cm, "data/collection")
    print(f"{responses['llm'].nunique()} model-language cells, {len(responses)} stored responses")

    boot = bootstrap_llm_positions(cm, responses, n_boot=N_BOOT, seed=SEED)
    ellipses = confidence_ellipses(boot)

    clf = RegionClassifier().fit(country_scores)
    print(f"SVM 5-fold CV accuracy: {clf.cv_accuracy:.3f} (attach to every region claim)")
    regions = clf.region_assignments(boot)
    headline = centroid_statistics(boot, country_scores)
    diagnostics = central_tendency_diagnostics(cm, responses)

    summary = ellipses.merge(regions, on="llm").merge(headline, on="llm")
    cols = [
        "llm",
        "PC1_rescaled",
        "PC2_rescaled",
        "sd_pc1",
        "sd_pc2",
        "svm_region",
        "positional_stability",
        "centroid_region",
        "rules_agree",
        "dist_human_mean",
        "pct_countries_closer",
        "min_dist_nonwestern",
    ]
    with pd.option_context("display.width", 240):
        print(summary[cols].round(3).to_string(index=False))
        print("\nDiagnostics (midpoint distance / mean item entropy):")
        print(diagnostics.round(3).to_string(index=False))

    boot.to_csv("data/llm_bootstrap_replicates.csv", index=False)
    ellipses.to_csv("data/llm_ellipses.csv", index=False)
    regions.to_csv("data/llm_regions.csv", index=False)
    headline.to_csv("data/llm_headline_stats.csv", index=False)
    diagnostics.to_csv("data/llm_diagnostics.csv", index=False)
    print(
        "\nWrote data/llm_bootstrap_replicates.csv, data/llm_ellipses.csv, "
        "data/llm_regions.csv, data/llm_headline_stats.csv, data/llm_diagnostics.csv"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
