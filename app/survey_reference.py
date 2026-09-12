"""Aggregate survey-reference diagnostics; no fitting or file writes.

The fixed marginal-mean reference, a completed-respondent mean, a vector
of marginal modes, and scale midpoints are distinct quantities. Marginal
modes need not describe an observed joint respondent; scale midpoints need
not be admissible individual answers on categorical items.
"""

import numpy as np
import pandas as pd

from app.culture_map import ITEM_VALID_RANGES, SURVEY_REFERENCE, CulturalMap

XY = ["PC1_rescaled", "PC2_rescaled"]
# Labels from the IVS merge syntax, for the current unmapped survey codes.
UNMAPPED_LABELS = {197: "Northern Cyprus", 909: "Northern Ireland", 915: "Kosovo"}


def survey_reference_tables(
    cm: CulturalMap,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarise the already-fitted survey, with explicit populations and weights."""
    observed = cm.subset_ivs_df
    completed = cm.valid_data
    if observed is None or completed is None or len(observed) != len(completed):
        raise ValueError("aligned observed and completed fitted respondents are required")
    mapped = completed["Numeric"].notna()
    summary = {
        "fit_n_respondents": len(completed),
        "fit_n_country_codes": completed["country_code"].nunique(),
        "mapped_n_respondents": int(mapped.sum()),
        "mapped_n_country_codes": completed.loc[mapped, "country_code"].nunique(),
        "unmapped_n_respondents": int((~mapped).sum()),
        "unmapped_n_country_codes": completed.loc[~mapped, "country_code"].nunique(),
    }
    references = []

    def add(name, point, n, definition, comparison=True):
        references.append(
            {
                "reference": name,
                "PC1_rescaled": float(point[0]),
                "PC2_rescaled": float(point[1]),
                "n_respondents": int(n),
                "is_comparison_reference": comparison,
                "definition": definition,
            }
        )
        summary[f"{name}_pc1"] = float(point[0])
        summary[f"{name}_pc2"] = float(point[1])

    add(
        "survey_reference",
        SURVEY_REFERENCE,
        len(observed),
        "Projected observed-item marginal means; frozen comparison point, not a completed-row mean",
    )
    for scope, group in [("all", completed), ("mapped", completed.loc[mapped])]:
        if group.empty:
            continue
        xy = group[XY].to_numpy()
        weights = group["weight"].fillna(1.0).to_numpy()
        add(
            f"completed_{scope}_unweighted",
            xy.mean(axis=0),
            len(group),
            f"Unweighted mean of completed projected respondents ({scope} fit entities)",
        )
        add(
            f"completed_{scope}_weighted",
            np.average(xy, axis=0, weights=weights),
            len(group),
            f"S017-weighted mean of completed projected respondents ({scope}); missing weight=1",
        )

    # Export the actual frozen standardisation parameters, not recomputed
    # completed-row moments. These ten-item aggregates support replay of
    # standardised-profile analyses without the private fitted NPZ.
    fit_means = np.asarray(cm.ppca.means, dtype=float)
    fit_stds = np.asarray(cm.ppca.stds, dtype=float)
    if fit_means.shape != (len(cm.iv_qns),) or fit_stds.shape != (len(cm.iv_qns),):
        raise ValueError("one fitted standardisation mean and SD per item is required")
    if not np.isfinite(fit_means).all() or not np.isfinite(fit_stds).all() or (fit_stds <= 0).any():
        raise ValueError("fitted standardisation means must be finite and SDs positive")
    item_rows = []
    for j, question in enumerate(cm.iv_qns):
        values = observed[question].dropna()
        modes = values.mode().sort_values().to_numpy()
        weighted = (
            pd.DataFrame(
                {"value": values, "weight": observed.loc[values.index, "weight"].fillna(1.0)}
            )
            .groupby("value")["weight"]
            .sum()
        )
        weighted_modes = weighted.index[weighted == weighted.max()].sort_values().to_numpy()
        lo, hi = ITEM_VALID_RANGES[question]
        item_rows.append(
            {
                "question": question,
                "observed_n": len(values),
                "observed_marginal_mean": values.mean(),
                "fit_standardisation_mean": fit_means[j],
                "fit_standardisation_sd": fit_stds[j],
                "scale_midpoint": (lo + hi) / 2,
                "mode_unweighted": modes[0],
                "mode_unweighted_ties": "|".join(map(str, modes)),
                "mode_weighted": weighted_modes[0],
                "mode_weighted_ties": "|".join(map(str, weighted_modes)),
            }
        )
    items = pd.DataFrame(item_rows)
    for column, definition in [
        (
            "scale_midpoint",
            "Hypothetical scale midpoints; not necessarily admissible individual responses",
        ),
        (
            "mode_unweighted",
            "Observed unweighted marginal modes; smallest value breaks ties; not a joint mode",
        ),
        (
            "mode_weighted",
            "Observed S017-weighted marginal modes; smallest value breaks ties; not a joint mode",
        ),
    ]:
        vector = pd.DataFrame([items[column].to_numpy()], columns=cm.iv_qns)
        add(
            column,
            cm.project(vector)[XY].iloc[0].to_numpy(),
            len(observed),
            definition,
            comparison=False,
        )
    unmapped = (
        completed.loc[~mapped].groupby("country_code").size().rename("n_respondents").reset_index()
    )
    unmapped["survey_label"] = unmapped["country_code"].map(UNMAPPED_LABELS).fillna("unmapped code")
    return summary, pd.DataFrame(references), items, unmapped


def reference_sensitivity(
    points: pd.DataFrame,
    countries: pd.DataFrame,
    references: pd.DataFrame,
    replicates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Change only the comparison reference; keep every map coordinate fixed.

    Point summaries and optional sampled-replicate extrema are descriptive.
    Region-centroid distances are unchanged by a choice of reference.
    Country shares use strict distance comparisons, as in the main analysis.
    """
    clouds = (
        {}
        if replicates is None
        else {llm: group[XY].to_numpy() for llm, group in replicates.groupby("llm")}
    )
    cxy = countries[XY].to_numpy()
    base = np.asarray(SURVEY_REFERENCE)
    base_country_distances = np.sort(np.linalg.norm(cxy - base, axis=1))
    refs = references.loc[references["is_comparison_reference"]]
    rows = []
    for _, refrow in refs.iterrows():
        ref = refrow[XY].to_numpy(dtype=float)
        country_distances = np.sort(np.linalg.norm(cxy - ref, axis=1))
        for _, point in points.iterrows():
            xy = point[XY].to_numpy(dtype=float)
            distance = np.linalg.norm(xy - ref)
            base_distance = np.linalg.norm(xy - base)
            share = np.searchsorted(country_distances, distance, side="left") / len(cxy)
            base_share = np.searchsorted(base_country_distances, base_distance, side="left") / len(
                cxy
            )
            row = {
                "reference": refrow["reference"],
                "llm": point["llm"],
                "point_in_quadrant": bool((xy > ref).all()),
                "point_distance": distance,
                "point_country_share": share,
                "distance_change_from_fixed_reference": distance - base_distance,
                "country_share_change_from_fixed_reference": share - base_share,
                "quadrant_changed_from_fixed_reference": bool(
                    (xy > ref).all() != (xy > base).all()
                ),
            }
            if point["llm"] in clouds:
                cloud = clouds[point["llm"]]
                distances = np.linalg.norm(cloud - ref, axis=1)
                minimum = distances.min()
                row.update(
                    {
                        "n_replicates": len(cloud),
                        "replicates_outside_quadrant": int((~(cloud > ref).all(axis=1)).sum()),
                        "minimum_replicate_distance": minimum,
                        "minimum_replicate_country_share": np.searchsorted(
                            country_distances, minimum, side="left"
                        )
                        / len(cxy),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)
