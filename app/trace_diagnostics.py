"""Sensitivity of frozen typicality-or-moderation codes; no new annotation.

Self-rater exclusion can produce ties; report denominator bounds rather
than silently dropping them. Existing-prefix rates describe the selected
successful trace sample, not a causal effect of removing prompt framing.
"""

import numpy as np
import pandas as pd

from app.trace_codebook import CODES

KEY = ["llm", "language", "question", "system_prompt_id", "repeat"]
CODE_LABELS = {
    "modal_targeting": "typicality_or_moderation",
    "persona_reasoning": "first_person_persona_reasoning",
    "guideline_citation": "ai_identity_constraint_or_guideline",
}
PREFIX_FAMILY = {
    **dict.fromkeys([0, 1, 3, 4, 6, 7], "averaging"),
    **dict.fromkeys([2, 5, 8], "bare"),
    9: "world_citizen",
}


def vote_counts(votes: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Strict majority with explicit ties, missing labels and denominator bounds."""
    n = votes.notna().sum(axis=1)
    ones = votes.sum(axis=1)
    majority = pd.Series(np.nan, index=votes.index)
    majority[2 * ones > n] = 1.0
    majority[2 * ones < n] = 0.0
    tied = (n > 0) & (2 * ones == n)
    positive, classified, total = int(majority.eq(1).sum()), int(majority.notna().sum()), len(votes)
    return majority, {
        "n_traces": total,
        "n_classified": classified,
        "n_ties": int(tied.sum()),
        "n_without_votes": int(n.eq(0).sum()),
        "majority_count": positive,
        "majority_share_classified": positive / classified if classified else np.nan,
        "positive_share_lower_all_traces": positive / total if total else np.nan,
        "positive_share_upper_all_traces": (positive + total - classified) / total
        if total
        else np.nan,
        "min_available_raters": int(n.min()) if total else 0,
        "max_available_raters": int(n.max()) if total else 0,
        "n_complete_panel": int(n.eq(len(votes.columns)).sum()),
    }


def trace_diagnostics(long: pd.DataFrame, sample: pd.DataFrame):
    """Self-rater sensitivity, selected-trace coverage and prefix-code rates."""
    long, sample = long.copy(), sample.copy()
    for frame in [long, sample]:
        for col in ["system_prompt_id", "repeat"]:
            frame[col] = pd.to_numeric(frame[col]).astype(int)
    sample = sample[sample["error"].isna() & sample["thinking"].fillna("").ne("")].copy()
    if sample.duplicated(KEY).any():
        raise ValueError("sample contains duplicate trace keys")
    index = pd.MultiIndex.from_frame(sample[KEY])
    raters = sorted(set(long["annotator"]) - {"human"})
    labelled_index = pd.MultiIndex.from_frame(long[KEY].drop_duplicates())
    if not labelled_index.isin(index).all():
        raise ValueError("labels contain a key outside the successful trace sample")
    majority_codes, sensitivity, prefix_rows = {}, [], []
    for code in CODES:
        good = long[long["error"].isna() & long["annotator"].isin(raters)].copy()
        good[code] = pd.to_numeric(good[code], errors="coerce")
        if not good[code].dropna().isin([0, 1]).all():
            raise ValueError(f"{code}: labels must be binary")
        matrix = good.pivot(index=KEY, columns="annotator", values=code).reindex(
            index=index, columns=raters
        )
        for analysis in ["all_llm_raters", "exclude_self_rater"]:
            votes, excluded = matrix.copy(), 0
            if analysis == "exclude_self_rater":
                for rater in raters:
                    own = votes.index.get_level_values("llm") == rater
                    excluded += int(votes.loc[own, rater].notna().sum())
                    votes.loc[own, rater] = np.nan
            majority, counts = vote_counts(votes)
            sensitivity.append(
                {
                    "code": code,
                    "code_label": CODE_LABELS[code],
                    "analysis": analysis,
                    **counts,
                    "n_self_votes_excluded": excluded,
                }
            )
            if analysis == "all_llm_raters":
                majority_codes[code] = majority
        sample[f"complete_panel__{code}"] = matrix.notna().all(axis=1).to_numpy()
    sample["at_trace_cap"] = sample["thinking"].str.len().ge(2000)
    sample["complete_panel"] = sample[[f"complete_panel__{c}" for c in CODES]].all(axis=1)
    coverage = (
        sample.groupby(["llm", "language"], sort=True)
        .agg(
            n_sampled=("thinking", "size"),
            n_at_2000_character_cap=("at_trace_cap", "sum"),
            n_complete_llm_panel=("complete_panel", "sum"),
            n_items=("question", "nunique"),
            n_prefixes=("system_prompt_id", "nunique"),
        )
        .reset_index()
    )
    coverage["cap_share"] = coverage["n_at_2000_character_cap"] / coverage["n_sampled"]
    sample["prefix_family"] = sample["system_prompt_id"].map(PREFIX_FAMILY)
    for code, majority in majority_codes.items():
        sample["majority"] = majority.to_numpy()
        for (language, family), g in sample.groupby(["language", "prefix_family"], sort=True):
            n_valid, positive = int(g["majority"].notna().sum()), int(g["majority"].eq(1).sum())
            prefix_rows.append(
                {
                    "code": code,
                    "code_label": CODE_LABELS[code],
                    "language": language,
                    "prefix_family": family,
                    "n_sampled": len(g),
                    "n_classified": n_valid,
                    "n_unclassified": len(g) - n_valid,
                    "positive_count": positive,
                    "share_classified": positive / n_valid if n_valid else np.nan,
                    "interpretation": "selected_successful_persona_traces_descriptive_not_causal",
                }
            )
    return pd.DataFrame(sensitivity), coverage, pd.DataFrame(prefix_rows)
