"""Export direct plug-in displacements and model/family sensitivity summaries.

Primary points use observed item means; bootstrap norms describe uncertainty.
Norm intervals cannot be used to test a zero effect. Model-resampled intervals
assume exchangeability, not random sampling of all LLMs. Equal-family weighting
assesses sensitivity to repeated models from one developer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.llm_meta import FAMILY_OF
from scripts.confirmatory_2026 import mean_displacement_ci, origin_language_permutation


def main() -> int:
    effects = pd.read_csv("data/llm_language_effects_2026.csv")
    expected = np.linalg.norm(effects[["delta_pc1", "delta_pc2"]], axis=1)
    np.testing.assert_allclose(effects["displacement"], expected, rtol=1e-12, atol=1e-12)
    out = effects.rename(columns={"displacement": "displacement_plugin"}).copy()
    out["family"] = out["llm"].map(FAMILY_OF)
    if out["family"].isna().any():
        raise ValueError("every analysed model must have an explicit family mapping")
    out.to_csv("data/llm_language_effects_plugin_2026.csv", index=False)
    ci = mean_displacement_ci(effects)
    perm = origin_language_permutation(effects)
    ci.to_csv("data/conf_2026_mean_displacement_plugin.csv", index=False)
    perm.to_csv("data/conf_2026_origin_permutation_plugin.csv", index=False)

    quantities = ["delta_pc1", "delta_pc2", "displacement"]
    family = effects.assign(family=out["family"]).groupby("family")[quantities].mean()
    family.to_csv("data/conf_2026_family_language_effects.csv")
    family_ci = mean_displacement_ci(family.reset_index())
    family_ci["resampling_unit"] = "equal-weight developer family mean"
    family_ci["n_units"] = len(family)
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
