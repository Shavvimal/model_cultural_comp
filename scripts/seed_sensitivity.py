"""Monte Carlo sensitivity of the likelihood fit to its initialization seed.

Run from the repo root (requires the data):

    uv run python scripts/seed_sensitivity.py

Refits the pipeline over K seeds and reports the across-seed SD and range
of every country coordinate, plus the rotation-angle spread — the
across-seed dispersion belongs in the paper's uncertainty budget.

Writes two artefacts:

  * ``data/seed_sensitivity.csv``            — per-country SD / min / max / range
  * ``data/seed_sensitivity_aggregates.csv`` — the four aggregates the paper
    quotes (rotation spread, mean and max coordinate SD, max coordinate range)

The per-seed rotation log is written together with the aggregates, at full
reported precision. Numerical optimization can still vary slightly across
BLAS/platform versions; all starts must satisfy the fitting gradient criterion.

Because the refit needs the ~5.8 GB IVS download, the aggregates can also be
recomputed from the retained per-country artefact plus the recorded
per-seed rotation angles, without refitting:

    uv run python scripts/seed_sensitivity.py --from-stored
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.culture_map import CulturalMap, check_preparation

SEEDS = list(range(20))
STATS_PATH = "data/seed_sensitivity.csv"
AGG_PATH = "data/seed_sensitivity_aggregates.csv"
RUN_LOG = "data/seed_sensitivity_run.log"

SD_COLS = ["PC1_rescaled_std", "PC2_rescaled_std"]
RANGE_COLS = ["range_pc1", "range_pc2"]


def aggregates(stats: pd.DataFrame, angles: list[float]) -> pd.DataFrame:
    """The four across-seed aggregates the paper's uncertainty budget quotes."""
    sd = stats[SD_COLS].to_numpy()
    rng = stats[RANGE_COLS].to_numpy()
    rows = [
        ("rotation_angle_mean_deg", float(np.mean(angles))),
        ("rotation_angle_spread_deg", float(np.ptp(angles))),
        ("coord_across_seed_sd_mean", float(sd.mean())),
        ("coord_across_seed_sd_max", float(sd.max())),
        ("coord_across_seed_range_max", float(rng.max())),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value"])
    out["n_seeds"] = len(angles)
    out["n_countries"] = len(stats)
    return out


def _angles_from_log(path: str = RUN_LOG) -> list[float]:
    """Recover the per-seed rotation angles from the recorded run log."""
    with open(path) as f:
        angles = [
            float(m.group(1))
            for m in re.finditer(r"^seed \d+: angle ([-+0-9.]+) deg", f.read(), re.M)
        ]
    if len(angles) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} angles in {path}, found {len(angles)}")
    return angles


def _report(stats: pd.DataFrame, agg: pd.DataFrame) -> None:
    print(agg.to_string(index=False))
    agg.to_csv(AGG_PATH, index=False)
    print(f"Wrote {AGG_PATH}")


def main(from_stored: bool = False) -> int:
    if from_stored:
        stats = pd.read_csv(STATS_PATH)
        _report(stats, aggregates(stats, _angles_from_log()))
        return 0

    coords = []
    angles = []
    log_rows = []
    # The licensed inputs are identical across seeds and need only one read.
    ivs = pd.read_pickle("data/ivs_df.pkl")
    countries = pd.read_pickle("data/country_codes.pkl")
    for seed in SEEDS:
        cm = CulturalMap(ivs, countries)
        cm.prepare_data()
        check_preparation(cm.survey_preparation_report)
        cm.fit(seed=seed)
        cm.calculate_mean_scores()
        df = cm.country_scores_pca[["country_code", "PC1_rescaled", "PC2_rescaled"]].copy()
        df["seed"] = seed
        coords.append(df)
        angles.append(np.degrees(np.arctan2(cm.rotation[1, 0], cm.rotation[0, 0])))
        line = f"seed {seed}: angle {angles[-1]:+.12f} deg"
        log_rows.append(line)
        print(line, flush=True)

    allc = pd.concat(coords)
    stats = allc.groupby("country_code")[["PC1_rescaled", "PC2_rescaled"]].agg(
        ["std", "min", "max"]
    )
    stats.columns = ["_".join(c) for c in stats.columns]
    stats["range_pc1"] = stats["PC1_rescaled_max"] - stats["PC1_rescaled_min"]
    stats["range_pc2"] = stats["PC2_rescaled_max"] - stats["PC2_rescaled_min"]

    stats.to_csv(STATS_PATH)
    Path(RUN_LOG).write_text("\n".join(log_rows) + "\n")
    print(f"Wrote {STATS_PATH}")
    _report(stats.reset_index(), aggregates(stats, angles))
    return 0


if __name__ == "__main__":
    sys.exit(main(from_stored="--from-stored" in sys.argv))
