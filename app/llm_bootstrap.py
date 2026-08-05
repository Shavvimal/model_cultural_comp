"""Bootstrap inference for LLM positions on the cultural map.

Two estimators, per the statistical review (paper draft repo) §2.4:

* The **item bootstrap** resamples each item's stored responses
  independently. Because the projection is affine in the ten item values, a
  model's mean position depends only on per-item means, so the point
  estimate is pairing-invariant — but independent resampling forces all
  cross-item covariances to zero, so its ellipses are a *lower bound* on
  the true uncertainty.
* The **cluster bootstrap** resamples the ten system-prompt variants with
  replacement, carrying all items and repeats within a variant, which
  propagates prompt-level correlation into the position. It is the primary
  estimator wherever the variant identifier was recorded (the 2026 corpus;
  the 2024 corpus lacks it).

Elicitation language is an experimental factor, not a nuisance: models
administered in Chinese get a distinct ``[zh]`` label and are never pooled
with their English administrations (the 2024 analysis pooled qwen2:7b
across a 2.7-map-unit language gap).
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from app.culture_map import CulturalMap

WESTERN_REGIONS = frozenset({"Protestant Europe", "English-Speaking", "Catholic Europe"})


def _label(llm: str, language: str) -> str:
    return llm if language == "en" else f"{llm} [{language}]"


def _to_value(cm: CulturalMap, question: str, response) -> float:
    if question == "Y002":
        return float(cm.y002_transform(response))
    if question == "Y003":
        return float(cm.y003_transform(response))
    return float(response)


def load_transformed_responses(cm: CulturalMap, collection_dir: str) -> pd.DataFrame:
    """Load the 2024 pickle corpus.

    Files prefixed ``c-`` hold Chinese-language administrations; their rows
    are labelled ``<llm> [zh]`` so the two languages are never pooled.
    Returns one row per (llm, language, question, numeric value).
    """
    files = glob.glob(os.path.join(collection_dir, "*.pkl"))
    if not files:
        raise FileNotFoundError(f"no collection pickles in {collection_dir}")

    frames = []
    for f in files:
        df = pd.read_pickle(f)
        df["language"] = "zh" if os.path.basename(f).startswith("c-") else "en"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # The 2024 harness recorded unparseable output as None. Those are
    # refusals/failures, not observations — drop them, but say so.
    null_mask = df["response"].isna()
    if null_mask.any():
        counts = df[null_mask].groupby("llm").size()
        print(
            f"dropping {null_mask.sum()} unparseable responses: "
            + ", ".join(f"{k}={v}" for k, v in counts.items())
        )
        df = df[~null_mask]

    df["llm"] = [_label(m, lang) for m, lang in zip(df["llm"], df["language"], strict=False)]
    df["value"] = [
        _to_value(cm, q, r) for q, r in zip(df["question"], df["response"], strict=False)
    ]
    # The 2024 corpus did not record the prompt-variant id
    df["system_prompt_id"] = pd.NA
    return df[["llm", "language", "question", "value", "system_prompt_id"]]


def load_responses_2026(cm: CulturalMap, jsonl_dir: str) -> pd.DataFrame:
    """Load the 2026 JSONL corpus directly (no pickle round-trip).

    Deduplicates resumed runs on (llm, language, question, system_prompt_id,
    repeat), keeping the last attempt, and keeps parsed rows only.
    """
    paths = sorted(Path(jsonl_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files in {jsonl_dir}")

    records = []
    for path in paths:
        with path.open() as f:
            records.extend(json.loads(line) for line in f)
    df = pd.DataFrame(records)
    df["language"] = df.get("language", pd.Series(["en"] * len(df))).fillna("en")
    # Resumed runs serialise these as str where the original run wrote int;
    # normalise before dedup or a retried row survives alongside its original.
    df["system_prompt_id"] = df["system_prompt_id"].astype(int)
    df["repeat"] = df["repeat"].astype(int)
    df = df.drop_duplicates(
        subset=["llm", "language", "question", "system_prompt_id", "repeat"], keep="last"
    )

    ok = df[df["error"].isna()].copy()
    dropped = len(df) - len(ok)
    if dropped:
        print(f"2026 corpus: {dropped} of {len(df)} calls unparsed (refusals/failures)")

    ok["response"] = [
        tuple(r) if q == "Y002" and isinstance(r, list) else r
        for q, r in zip(ok["question"], ok["parsed"], strict=False)
    ]
    ok["llm"] = [_label(m, lang) for m, lang in zip(ok["llm"], ok["language"], strict=False)]
    ok["value"] = [
        _to_value(cm, q, r) for q, r in zip(ok["question"], ok["response"], strict=False)
    ]
    return ok[["llm", "language", "question", "value", "system_prompt_id"]]


def bootstrap_llm_positions(
    cm: CulturalMap,
    responses: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Item bootstrap: resample each item's responses independently.

    A lower bound on uncertainty — see the module docstring. Returns one
    row per (llm, replicate).
    """
    rng = np.random.default_rng(seed)
    out = []
    for llm, group in responses.groupby("llm"):
        by_qn = {q: g["value"].to_numpy() for q, g in group.groupby("question")}
        missing = [q for q in cm.iv_qns if q not in by_qn or len(by_qn[q]) == 0]
        if missing:
            raise ValueError(f"{llm}: no stored responses for {missing}")

        means = np.column_stack(
            [
                rng.choice(by_qn[q], size=(n_boot, len(by_qn[q])), replace=True).mean(axis=1)
                for q in cm.iv_qns
            ]
        )
        projected = cm.project(pd.DataFrame(means, columns=cm.iv_qns))
        projected["llm"] = llm
        projected["replicate"] = np.arange(n_boot)
        out.append(projected)
    return pd.concat(out, ignore_index=True)


def bootstrap_llm_positions_cluster(
    cm: CulturalMap,
    responses: pd.DataFrame,
    n_boot: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster bootstrap over system-prompt variants — the primary estimator.

    Each replicate draws K variants with replacement (K = number observed),
    pools all their rows, and projects the per-item means. Requires
    ``system_prompt_id``; raises where it was not recorded (the 2024 corpus).
    With only ~10 clusters the interval is approximate (Cameron, Gelbach &
    Miller 2008) — report it as such.
    """
    if responses["system_prompt_id"].isna().any():
        raise ValueError(
            "system_prompt_id missing for some rows; the cluster bootstrap "
            "is only available for corpora that recorded the variant id."
        )
    rng = np.random.default_rng(seed)
    out = []
    for llm, group in responses.groupby("llm"):
        variants = np.sort(group["system_prompt_id"].unique())
        # per-variant, per-item means and counts, so replicates can pool
        # weighted by how many responses each drawn variant contributed
        sums = group.pivot_table(
            index="system_prompt_id", columns="question", values="value", aggfunc="sum"
        ).reindex(variants)
        counts = group.pivot_table(
            index="system_prompt_id", columns="question", values="value", aggfunc="count"
        ).reindex(variants)
        overall_means = group.groupby("question")["value"].mean()

        draws = rng.choice(len(variants), size=(n_boot, len(variants)), replace=True)
        sum_mat = sums[cm.iv_qns].to_numpy()
        cnt_mat = counts[cm.iv_qns].to_numpy()
        rep_sums = sum_mat[draws].sum(axis=1)  # (n_boot, 10)
        rep_counts = cnt_mat[draws].sum(axis=1)

        with np.errstate(invalid="ignore"):
            means = rep_sums / rep_counts
        # A replicate can draw only variants where an item was never parsed
        # (e.g. refusal-heavy items); fall back to the model's overall item
        # mean rather than dropping the replicate. Rare, and reported.
        n_fallback = int(np.isnan(means).any(axis=1).sum())
        if n_fallback:
            fill = overall_means[cm.iv_qns].to_numpy()
            nan_rows, nan_cols = np.where(np.isnan(means))
            means[nan_rows, nan_cols] = fill[nan_cols]
            print(f"{llm}: {n_fallback}/{n_boot} replicates used overall-mean fallback")

        projected = cm.project(pd.DataFrame(means, columns=cm.iv_qns))
        projected["llm"] = llm
        projected["replicate"] = np.arange(n_boot)
        out.append(projected)
    return pd.concat(out, ignore_index=True)


def confidence_ellipses(boot: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Mean position and confidence ellipse for the mean, per model.

    Normal-theory region at chi2(level, 2); ``mahal_q95`` gives the
    empirical Mahalanobis-squared quantile so the normality assumption is
    checkable against 5.99 per model.
    """
    k = chi2.ppf(level, df=2)
    rows = []
    for llm, g in boot.groupby("llm"):
        xy = g[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        mean = xy.mean(axis=0)
        cov = np.cov(xy.T)
        vals, vecs = np.linalg.eigh(cov)  # ascending
        width, height = 2 * np.sqrt(k * vals[::-1])
        angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
        centered = xy - mean
        mahal = np.einsum("ij,jk,ik->i", centered, np.linalg.pinv(cov), centered)
        rows.append(
            {
                "llm": llm,
                "PC1_rescaled": mean[0],
                "PC2_rescaled": mean[1],
                "ellipse_width": width,
                "ellipse_height": height,
                "angle_deg": angle,
                "sd_pc1": xy[:, 0].std(ddof=1),
                "sd_pc2": xy[:, 1].std(ddof=1),
                "mahal_q95": float(np.quantile(mahal, level)),
            }
        )
    return pd.DataFrame(rows)


def centroid_statistics(
    boot: pd.DataFrame,
    country_scores: pd.DataFrame,
    human_mean: tuple[float, float] = (0.38, -0.01),
) -> pd.DataFrame:
    """Classifier-free headline statistics, with bootstrap CIs.

    Per model: distance to the pooled human respondent mean, the share of
    countries closer to that mean than the model is, and the minimum
    distance to any non-Western region centroid. All three are computed per
    bootstrap replicate, so the reported intervals need no distributional
    assumptions and no multiplicity correction for "holds in every
    replicate" statements.
    """
    countries_xy = country_scores[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
    country_dists = np.linalg.norm(countries_xy - np.array(human_mean), axis=1)

    centroids = country_scores.groupby("Cultural Region")[["PC1_rescaled", "PC2_rescaled"]].mean()
    non_western = centroids.loc[~centroids.index.isin(WESTERN_REGIONS)].to_numpy()

    rows = []
    for llm, g in boot.groupby("llm"):
        xy = g[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        d_mean = np.linalg.norm(xy - np.array(human_mean), axis=1)
        pct_closer = (country_dists[None, :] < d_mean[:, None]).mean(axis=1)
        d_nonwest = np.linalg.norm(xy[:, None, :] - non_western[None, :, :], axis=2).min(axis=1)
        rows.append(
            {
                "llm": llm,
                "dist_human_mean": d_mean.mean(),
                "dist_human_mean_lo": np.quantile(d_mean, 0.025),
                "dist_human_mean_hi": np.quantile(d_mean, 0.975),
                "pct_countries_closer": pct_closer.mean(),
                "min_dist_nonwestern": d_nonwest.mean(),
                "min_dist_nonwestern_lo": np.quantile(d_nonwest, 0.025),
                "min_dist_nonwestern_hi": np.quantile(d_nonwest, 0.975),
            }
        )
    return pd.DataFrame(rows)


def central_tendency_diagnostics(cm: CulturalMap, responses: pd.DataFrame) -> pd.DataFrame:
    """Distance from the all-midpoint respondent, and response entropy.

    Mid-scale answering registers as strong secularity on this instrument
    (the midpoint respondent projects above Sweden), so these two
    diagnostics let readers separate expressed values from modal-response
    behaviour.
    """
    from app.culture_map import ITEM_VALID_RANGES

    midpoints = {q: (lo + hi) / 2 for q, (lo, hi) in ITEM_VALID_RANGES.items()}
    rows = []
    for llm, group in responses.groupby("llm"):
        item_means = group.groupby("question")["value"].mean()
        mid_dist = float(
            np.linalg.norm([item_means.get(q, np.nan) - midpoints[q] for q in cm.iv_qns])
        )

        def entropy(values: pd.Series) -> float:
            p = values.value_counts(normalize=True).to_numpy()
            return float(-(p * np.log2(p)).sum())

        mean_entropy = float(group.groupby("question")["value"].apply(entropy).mean())
        rows.append({"llm": llm, "midpoint_distance": mid_dist, "mean_item_entropy": mean_entropy})
    return pd.DataFrame(rows)
