"""Export direct plug-in displacements and model/family sensitivity summaries.

Primary points use observed item means; bootstrap norms describe uncertainty.
Norm intervals cannot be used to test a zero effect. Model-resampled intervals
assume exchangeability, not random sampling of all LLMs. Equal-family weighting
assesses sensitivity to repeated models from one developer.

The across-model interval and origin permutation on the plug-in effects are
computed once, by ``confirmatory_2026.py`` (same input and seed). The two
``_plugin`` filenames are kept because the results supplement cites them; this
stage copies those files byte for byte instead of recomputing them, after
checking that their point estimates still match the current language effects.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from app.llm_meta import FAMILY_OF
from scripts.confirmatory_2026 import mean_displacement_ci

LANGUAGE_EFFECTS = Path("data/llm_language_effects_2026.csv")
MEAN_DISPLACEMENT = Path("data/conf_2026_mean_displacement.csv")
ORIGIN_PERMUTATION = Path("data/conf_2026_origin_permutation.csv")
MEAN_DISPLACEMENT_ALIAS = Path("data/conf_2026_mean_displacement_plugin.csv")
ORIGIN_PERMUTATION_ALIAS = Path("data/conf_2026_origin_permutation_plugin.csv")
COMPONENTS = ["delta_pc1", "delta_pc2", "displacement"]


def check_confirmatory_matches(effects: pd.DataFrame, ci: pd.DataFrame, perm: pd.DataFrame) -> None:
    """Raise if the confirmatory artefacts were not built from ``effects``.

    Compares the deterministic point estimates only (cohort-mean differences
    and across-model means), with the exact arithmetic confirmatory_2026 uses,
    so a stale artefact cannot be copied under the plug-in names. effects
    must be parsed as confirmatory_2026 parses it (default pd.read_csv).
    """
    expected_means = [float(effects[col].to_numpy().mean()) for col in COMPONENTS]
    if ci["quantity"].tolist() != [f"mean_{col}" for col in COMPONENTS] or not np.array_equal(
        ci["estimate"].to_numpy(dtype=float), expected_means
    ):
        raise ValueError(
            f"{MEAN_DISPLACEMENT} does not match {LANGUAGE_EFFECTS}; rerun confirmatory_2026.py "
            f"(expected estimates {expected_means}, found {ci['estimate'].tolist()})"
        )
    is_cn = (effects["cohort"] == "Chinese").to_numpy(dtype=bool)
    expected_diffs = [
        float(effects[col].to_numpy()[is_cn].mean() - effects[col].to_numpy()[~is_cn].mean())
        for col in COMPONENTS
    ]
    if perm["component"].tolist() != COMPONENTS or not np.array_equal(
        perm["chinese_minus_western"].to_numpy(dtype=float), expected_diffs
    ):
        raise ValueError(
            f"{ORIGIN_PERMUTATION} does not match {LANGUAGE_EFFECTS}; rerun confirmatory_2026.py "
            f"(expected differences {expected_diffs}, found "
            f"{perm['chinese_minus_western'].tolist()})"
        )


def main() -> int:
    effects = pd.read_csv(LANGUAGE_EFFECTS)
    expected = np.linalg.norm(effects[["delta_pc1", "delta_pc2"]], axis=1)
    np.testing.assert_allclose(effects["displacement"], expected, rtol=1e-12, atol=1e-12)
    out = effects.rename(columns={"displacement": "displacement_plugin"}).copy()
    out["family"] = out["llm"].map(FAMILY_OF)
    if out["family"].isna().any():
        raise ValueError("every analysed model must have an explicit family mapping")
    for source in (MEAN_DISPLACEMENT, ORIGIN_PERMUTATION):
        if not source.exists():
            raise FileNotFoundError(f"{source} missing; run scripts/confirmatory_2026.py first")
    ci = pd.read_csv(MEAN_DISPLACEMENT, float_precision="round_trip")
    perm = pd.read_csv(ORIGIN_PERMUTATION, float_precision="round_trip")
    check_confirmatory_matches(effects, ci, perm)

    quantities = ["delta_pc1", "delta_pc2", "displacement"]
    family = effects.assign(family=out["family"]).groupby("family")[quantities].mean()
    family_ci = mean_displacement_ci(family.reset_index())
    family_ci["resampling_unit"] = "equal-weight developer family mean"
    family_ci["n_units"] = len(family)

    out.to_csv("data/llm_language_effects_plugin_2026.csv", index=False)
    shutil.copyfile(MEAN_DISPLACEMENT, MEAN_DISPLACEMENT_ALIAS)
    shutil.copyfile(ORIGIN_PERMUTATION, ORIGIN_PERMUTATION_ALIAS)
    family.to_csv("data/conf_2026_family_language_effects.csv")
    family_ci.to_csv("data/conf_2026_family_displacement_sensitivity.csv", index=False)
    print("Direct plug-in displacements:")
    print(out.round(4).to_string(index=False))
    print("\nAcross-model means and exploratory intervals:")
    print(ci.round(4).to_string(index=False))
    print("\nEqual-family-weight sensitivity:")
    print(family_ci.round(4).to_string(index=False))
    print("\nOrigin permutation (models exchangeable under the null):")
    print(perm.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
