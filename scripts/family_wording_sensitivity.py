"""Offline exploratory sensitivities added after the September final review.

Family averaging changes the unit and weighting; origin permutations remain
conditional on exchangeability of the selected family labels. The Y003 exclusion
is a conservative diagnostic for documented wording misunderstandings, not a
replacement for the planned release-level analysis or corrected responses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control

from app.appendix_contrasts import origin_profile_permutation, standardise_profiles
from app.culture_map import IV_QNS
from app.llm_meta import FAMILY_OF, cohort_2026

KNOWN_Y003_WORDING_MODELS = {"glm-5.1", "glm-5.2", "kimi-k2.6", "qwen3.5:397b"}


def item_signs(deltas: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Two-sided sign tests, including all ten items in the BH family.

    Missing entries represent explicitly excluded model/item comparisons. An
    all-tied item retains p=1 so exclusion cannot silently shrink multiplicity.
    """
    if set(deltas.columns) != set(IV_QNS) or deltas.columns.duplicated().any():
        raise ValueError("deltas must contain exactly the ten instrument items")
    rows = []
    for question in IV_QNS:
        values = deltas[question].dropna().to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"{question}: no finite comparisons")
        positive = int((values > 0).sum())
        negative = int((values < 0).sum())
        effective = positive + negative
        p = binomtest(positive, effective).pvalue if effective else 1.0
        rows.append(
            {
                "question": question,
                "unit": unit,
                "n_units": len(values),
                "n_effective": effective,
                "n_ties": len(values) - effective,
                "n_positive": positive,
                "n_negative": negative,
                "mean_delta": float(values.mean()),
                "p_sign_two_sided": float(p),
            }
        )
    result = pd.DataFrame(rows)
    result["p_bh"] = false_discovery_control(result["p_sign_two_sided"].to_numpy())
    return result


def family_means(profiles: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight means within the existing explicit developer families."""
    families = profiles.index.map(FAMILY_OF)
    if families.isna().any() or profiles.index.duplicated().any():
        raise ValueError("every unique model needs an explicit family mapping")
    return profiles.groupby(families).mean()


def main() -> int:
    item_fx = pd.read_csv("data/conf_2026_item_language_effects.csv", float_precision="round_trip")
    deltas = item_fx.pivot(index="llm", columns="question", values="delta").reindex(columns=IV_QNS)
    if deltas.isna().any().any():
        raise ValueError("release item comparisons are incomplete")
    grouped = family_means(deltas)
    grouped.index.name = "family"
    grouped.to_csv("data/conf_2026_family_item_deltas.csv")
    signs = item_signs(grouped, "equal-weight developer family mean")
    signs.to_csv("data/conf_2026_family_item_sign_tests.csv", index=False)

    omitted = deltas.copy()
    if not KNOWN_Y003_WORDING_MODELS.issubset(omitted.index):
        raise ValueError("the documented Y003 wording models must be present")
    omitted.loc[sorted(KNOWN_Y003_WORDING_MODELS), "Y003"] = np.nan
    wording = item_signs(omitted, "release; four documented model/Y003 comparisons omitted")
    wording.to_csv("data/conf_2026_y003_wording_sign_tests.csv", index=False)

    paired = pd.read_csv("data/llm_language_effects_2026.csv")["llm"]
    keying = pd.read_csv("data/diag_2026_item_keying.csv", float_precision="round_trip")
    weights = keying.set_index("question").reindex(IV_QNS)[["d_pc1_per_unit", "d_pc2_per_unit"]]
    geometric_rows = []
    for mode in ["primary", "hold_Y003_constant", "exclude_four_models"]:
        values = deltas.loc[paired].copy()
        if mode == "hold_Y003_constant":
            values["Y003"] = 0.0
        elif mode == "exclude_four_models":
            values = values.drop(index=sorted(KNOWN_Y003_WORDING_MODELS))
        xy = values @ weights
        geometric_rows.append(
            {
                "sensitivity": mode,
                "n_models": len(xy),
                "n_positive_pc1": int((xy.iloc[:, 0] > 0).sum()),
                "n_negative_pc2": int((xy.iloc[:, 1] < 0).sum()),
                "mean_delta_pc1": float(xy.iloc[:, 0].mean()),
                "mean_delta_pc2": float(xy.iloc[:, 1].mean()),
                "mean_displacement": float(np.linalg.norm(xy, axis=1).mean()),
            }
        )
    geometry = pd.DataFrame(geometric_rows)
    geometry.to_csv("data/conf_2026_y003_geometry_sensitivity.csv", index=False)

    profiles = item_fx.pivot(index="llm", columns="question", values="mean_en")
    baselines = pd.read_csv(
        "data/validation_survey_item_baselines.csv", float_precision="round_trip"
    )
    origin_by_family = {}
    for model in profiles.index:
        family = FAMILY_OF[model]
        origin = cohort_2026(model)
        if family in origin_by_family and origin_by_family[family] != origin:
            raise ValueError(f"mixed origin labels within family {family}")
        origin_by_family[family] = origin
    profile_rows = []
    for scale, values in [
        ("raw", profiles),
        ("survey_standardised", standardise_profiles(profiles, baselines)),
    ]:
        means = family_means(values)
        is_chinese = np.array([origin_by_family[family] == "Chinese" for family in means.index])
        result = origin_profile_permutation(means, is_chinese)
        result["profile_scale"] = scale
        result["unit"] = "equal-weight developer family mean"
        result["n_families"] = len(means)
        profile_rows.append(result)
    origin = pd.concat(profile_rows, ignore_index=True)
    origin.to_csv("data/conf_2026_family_origin_similarity.csv", index=False)
    print("Exploratory family-unit item tests:")
    print(signs.to_string(index=False))
    print("Known Y003 wording exclusion:")
    print(wording.to_string(index=False))
    print("Y003 geometric sensitivities:")
    print(geometry.to_string(index=False))
    print("Family-unit profile permutations:")
    print(origin.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
