"""Monte Carlo sensitivity of the fit to the EM initialization seed.

Run from the repo root (requires the data):

    uv run python scripts/seed_sensitivity.py

Refits the pipeline over K seeds and reports the across-seed SD and range
of every country coordinate, plus the rotation-angle spread — the
across-seed dispersion belongs in the paper's uncertainty budget.

Writes data/seed_sensitivity.csv.
"""

import sys

import numpy as np
import pandas as pd

from app.culture_map import CulturalMap

SEEDS = list(range(20))


def main() -> int:
    coords = []
    angles = []
    for seed in SEEDS:
        cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
        cm.prepare_data()
        cm.fit(seed=seed)
        cm.calculate_mean_scores()
        df = cm.country_scores_pca[["country_code", "PC1_rescaled", "PC2_rescaled"]].copy()
        df["seed"] = seed
        coords.append(df)
        angles.append(np.degrees(np.arctan2(cm.rotation[1, 0], cm.rotation[0, 0])))
        print(f"seed {seed}: angle {angles[-1]:+.2f} deg", flush=True)

    allc = pd.concat(coords)
    stats = allc.groupby("country_code")[["PC1_rescaled", "PC2_rescaled"]].agg(["std", "min", "max"])
    stats.columns = ["_".join(c) for c in stats.columns]
    stats["range_pc1"] = stats["PC1_rescaled_max"] - stats["PC1_rescaled_min"]
    stats["range_pc2"] = stats["PC2_rescaled_max"] - stats["PC2_rescaled_min"]

    print(f"\nrotation angle: mean {np.mean(angles):+.2f}, spread {np.ptp(angles):.2f} deg")
    print(
        f"country coordinate across-seed SD: "
        f"mean {stats[['PC1_rescaled_std', 'PC2_rescaled_std']].to_numpy().mean():.4f}, "
        f"max {stats[['PC1_rescaled_std', 'PC2_rescaled_std']].to_numpy().max():.4f}"
    )
    print(
        f"max coordinate range across seeds: "
        f"{stats[['range_pc1', 'range_pc2']].to_numpy().max():.4f}"
    )
    stats.to_csv("data/seed_sensitivity.csv")
    print("Wrote data/seed_sensitivity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
