"""Paired-variant sensitivity alongside the primary independent-arm bootstrap.

This changes the resampling target, not the observed point contrast. Both arms
use the same sampled persona IDs, retaining each ID's available item answers.
The two language versions are matched by construction but are not identical
text; neither design estimates uncertainty over every possible translation.
"""

import numpy as np
import pandas as pd

from app.culture_map import CulturalMap
from app.llm_bootstrap import load_responses_2026


def main() -> int:
    cm = CulturalMap(pd.DataFrame(), pd.DataFrame())
    cm.load_model("data/cultural_map_model.npz")
    answers = load_responses_2026(cm, "data/collection_2026")
    primary = pd.read_csv("data/llm_language_effects_2026.csv")
    boot = pd.read_csv("data/llm_bootstrap_replicates_2026.csv")
    rng = np.random.default_rng(20260911)
    rows = []
    for model in primary["llm"]:
        groups = [answers[answers["llm"] == label] for label in (model, f"{model} [zh]")]
        variants = sorted(set(groups[0]["system_prompt_id"]) | set(groups[1]["system_prompt_id"]))
        draws = rng.integers(0, len(variants), size=(10_000, len(variants)))
        positions = []
        for group in groups:
            agg = group.groupby(["system_prompt_id", "question"])["value"].agg(["sum", "count"])
            sums = (
                agg["sum"].unstack().reindex(index=variants, columns=cm.iv_qns).fillna(0).to_numpy()
            )
            counts = (
                agg["count"]
                .unstack()
                .reindex(index=variants, columns=cm.iv_qns)
                .fillna(0)
                .to_numpy()
            )
            denominator = counts[draws].sum(axis=1)
            observed = group.groupby("question")["value"].mean().reindex(cm.iv_qns).to_numpy()
            means = np.divide(
                sums[draws].sum(axis=1),
                denominator,
                out=np.broadcast_to(observed, denominator.shape).copy(),
                where=denominator > 0,
            )
            positions.append(
                cm.project(pd.DataFrame(means, columns=cm.iv_qns))[
                    ["PC1_rescaled", "PC2_rescaled"]
                ].to_numpy()
            )
        paired = positions[1] - positions[0]
        en = boot.loc[boot["llm"] == model, ["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        zh = boot.loc[boot["llm"] == f"{model} [zh]", ["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        independent = zh - en
        for j, axis in enumerate(("pc1", "pc2")):
            paired_sd = paired[:, j].std(ddof=1)
            independent_sd = independent[:, j].std(ddof=1)
            rows.append(
                {
                    "llm": model,
                    "axis": axis,
                    "paired_sd": paired_sd,
                    "independent_sd": independent_sd,
                    "paired_to_independent_sd": paired_sd / independent_sd,
                    "paired_ci_lo": np.quantile(paired[:, j], 0.025),
                    "paired_ci_hi": np.quantile(paired[:, j], 0.975),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv("data/conf_2026_paired_variant_sensitivity.csv", index=False)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
