"""Bootstrap inference for LLM positions on the cultural map.

Two estimators (write-up §3.4):

* The **item bootstrap** resamples each item's stored responses
  independently. Because the projection is affine in the ten item values, a
  model's mean position depends only on per-item means, so the point
  estimate is pairing-invariant — but independent resampling forces all
  cross-item covariances to zero. With mixed-sign item weights the neglected
  term can take either sign, so this is not a guaranteed lower bound.
  Empirical estimator comparisons are recomputed from the current corpus.
* The **cluster bootstrap** resamples the ten user-message prefix variants with
  replacement, carrying all items and repeats within a variant, which
  propagates prompt-level correlation into the position. It is the primary
  estimator wherever the variant identifier was recorded (the 2026 corpus;
  the 2024 corpus lacks it).

Elicitation language is an experimental factor, not a nuisance: models
administered in Chinese get a distinct ``[zh]`` label and are never pooled
with their English administrations (the 2024 analysis pooled qwen2:7b
instead of retaining their separate observed positions).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from app.culture_map import SURVEY_REFERENCE, CulturalMap

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
    """Load the portable 2024 JSONL corpus from the separate response archive.

    Files prefixed ``c-`` hold Chinese-language administrations; their rows
    are labelled ``<llm> [zh]`` so the two languages are never pooled.
    Returns one row per (llm, language, question, numeric value).
    """
    files = sorted(Path(collection_dir).glob("*_responses_df.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no 2024 response JSONLs in {collection_dir}; install the response archive "
            "described in docs/REPRODUCING.md"
        )

    frames = []
    for f in files:
        with f.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if not records or any(
            not isinstance(row, dict) or set(row) != {"llm", "question", "response"}
            for row in records
        ):
            raise ValueError(f"{f.name}: expected llm, question and response in every record")
        df = pd.DataFrame(records)
        df["language"] = "zh" if f.name.startswith("c-") else "en"
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

    Omits cross-item covariance; not a guaranteed lower bound. Returns one
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


def project_cell_means(
    cmap: CulturalMap, responses: pd.DataFrame, min_per_item: int = 1
) -> pd.DataFrame:
    """Project observed item means without Monte Carlo centring or norm bias.

    Omit cells below ``min_per_item`` on any required item. Bootstrap
    callers should apply the same eligibility rule before resampling.
    """
    if min_per_item < 1:
        raise ValueError("min_per_item must be positive")
    columns = ["llm", "PC1_rescaled", "PC2_rescaled", "min_per_item"]
    if responses.empty:
        return pd.DataFrame(columns=columns)
    counts = (
        responses.pivot_table(index="llm", columns="question", values="value", aggfunc="count")
        .reindex(columns=cmap.iv_qns)
        .fillna(0)
    )
    eligible = counts.index[(counts >= min_per_item).all(axis=1)]
    if len(eligible) == 0:
        return pd.DataFrame(columns=columns)
    means = responses.pivot_table(
        index="llm", columns="question", values="value", aggfunc="mean"
    ).reindex(index=eligible, columns=cmap.iv_qns)
    points = cmap.project(means)[["PC1_rescaled", "PC2_rescaled"]].copy()
    points.insert(0, "llm", eligible.to_numpy())
    points["min_per_item"] = counts.loc[eligible].min(axis=1).to_numpy(dtype=int)
    return points.reset_index(drop=True)


def _point_lookup(point_estimates: pd.DataFrame | None) -> pd.DataFrame | None:
    if point_estimates is None:
        return None
    if point_estimates["llm"].duplicated().any():
        raise ValueError("point_estimates must contain one row per llm")
    points = point_estimates.set_index("llm")[["PC1_rescaled", "PC2_rescaled"]]
    if not np.isfinite(points.to_numpy()).all():
        raise ValueError("point_estimates must contain finite coordinates")
    return points


def bootstrap_llm_positions_cluster(
    cm: CulturalMap,
    responses: pd.DataFrame,
    n_boot: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster bootstrap over user-message prefixes — the primary estimator.

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
        # per-variant, per-item sums and counts, so replicates can pool
        # weighted by how many responses each drawn variant contributed. A
        # (variant, item) group with no parsed row is an empty group, so it
        # contributes zero to both the sum and the count: fill with 0, never
        # leave NaN, or one absent group would poison every replicate that
        # draws that variant.
        sums = (
            group.pivot_table(
                index="system_prompt_id", columns="question", values="value", aggfunc="sum"
            )
            .reindex(index=variants, columns=cm.iv_qns)
            .fillna(0.0)
        )
        counts = (
            group.pivot_table(
                index="system_prompt_id", columns="question", values="value", aggfunc="count"
            )
            .reindex(index=variants, columns=cm.iv_qns)
            .fillna(0)
        )
        overall_means = group.groupby("question")["value"].mean().reindex(cm.iv_qns)
        unobserved = overall_means.index[overall_means.isna()].tolist()
        if unobserved:
            raise ValueError(f"{llm}: no stored responses for {unobserved}")

        draws = rng.choice(len(variants), size=(n_boot, len(variants)), replace=True)
        sum_mat = sums.to_numpy(dtype=float)
        cnt_mat = counts.to_numpy(dtype=float)
        rep_sums = sum_mat[draws].sum(axis=1)  # (n_boot, 10)
        rep_counts = cnt_mat[draws].sum(axis=1)

        # Pooled item mean wherever the drawn variants contributed at least
        # one parsed row for the item; NaN only where the pooled count is 0.
        means = np.divide(
            rep_sums, rep_counts, out=np.full_like(rep_sums, np.nan), where=rep_counts > 0
        )
        # A replicate can draw only variants where an item was never parsed
        # (e.g. refusal-heavy items); fall back to the model's overall item
        # mean rather than dropping the replicate; report every occurrence.
        empty = rep_counts == 0
        n_fallback = int(empty.any(axis=1).sum())
        if n_fallback:
            fill = overall_means.to_numpy()
            nan_rows, nan_cols = np.where(empty)
            means[nan_rows, nan_cols] = fill[nan_cols]
            print(f"{llm}: {n_fallback}/{n_boot} replicates used overall-mean fallback")

        projected = cm.project(pd.DataFrame(means, columns=cm.iv_qns))
        projected["llm"] = llm
        projected["replicate"] = np.arange(n_boot)
        out.append(projected)
    return pd.concat(out, ignore_index=True)


def confidence_ellipses(
    boot: pd.DataFrame, level: float = 0.95, point_estimates: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Mean position and nominal normal-theory ellipse, per model.

    Normal-theory region at chi2(level, 2); ``mahal_q95`` gives the
    empirical Mahalanobis-squared quantile as a replicate-cloud shape check,
    not coverage calibration. Supply ``project_cell_means`` output to centre
    on the observed plug-in estimate; otherwise the historical bootstrap
    mean is used. Covariance and quantiles always use the replicate cloud.
    """
    k = chi2.ppf(level, df=2)
    points = _point_lookup(point_estimates)
    rows = []
    for llm, g in boot.groupby("llm"):
        xy = g[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        replicate_mean = xy.mean(axis=0)
        mean = replicate_mean if points is None else points.loc[llm].to_numpy()
        cov = np.cov(xy.T)
        vals, vecs = np.linalg.eigh(cov)  # ascending
        width, height = 2 * np.sqrt(k * vals[::-1])
        angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
        centered = xy - replicate_mean
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
    human_mean: tuple[float, float] = SURVEY_REFERENCE,
    point_estimates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classifier-free headline statistics, with bootstrap CIs.

    Per model: distance to the projected observed-item marginal-mean
    reference, the share of countries closer to it than the model is, and the minimum
    distance to any non-Western region centroid. All three are computed per
    bootstrap replicate. Statements about all generated replicates are
    descriptive and do not establish simultaneous confidence coverage.
    ``human_mean`` and output column names remain for compatibility. With
    ``point_estimates``, point statistics use observed cell means; replicate
    quantiles provide uncertainty only. Without it, historical replicate-
    average point summaries are retained.
    """
    countries_xy = country_scores[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
    country_dists = np.linalg.norm(countries_xy - np.array(human_mean), axis=1)

    centroids = country_scores.groupby("Cultural Region")[["PC1_rescaled", "PC2_rescaled"]].mean()
    non_western = centroids.loc[~centroids.index.isin(WESTERN_REGIONS)].to_numpy()
    points = _point_lookup(point_estimates)

    rows = []
    for llm, g in boot.groupby("llm"):
        xy = g[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        d_mean = np.linalg.norm(xy - np.array(human_mean), axis=1)
        pct_closer = (country_dists[None, :] < d_mean[:, None]).mean(axis=1)
        d_nonwest = np.linalg.norm(xy[:, None, :] - non_western[None, :, :], axis=2).min(axis=1)
        if points is None:
            point_distance, point_share, point_nonwestern = (
                d_mean.mean(),
                pct_closer.mean(),
                d_nonwest.mean(),
            )
        else:
            point = points.loc[llm].to_numpy()
            point_distance = np.linalg.norm(point - np.array(human_mean))
            point_share = (country_dists < point_distance).mean()
            point_nonwestern = np.linalg.norm(non_western - point, axis=1).min()
        rows.append(
            {
                "llm": llm,
                "dist_human_mean": point_distance,
                "dist_human_mean_lo": np.quantile(d_mean, 0.025),
                "dist_human_mean_hi": np.quantile(d_mean, 0.975),
                "pct_countries_closer": point_share,
                "min_dist_nonwestern": point_nonwestern,
                "min_dist_nonwestern_lo": np.quantile(d_nonwest, 0.025),
                "min_dist_nonwestern_hi": np.quantile(d_nonwest, 0.975),
            }
        )
    return pd.DataFrame(rows)


def central_tendency_diagnostics(cm: CulturalMap, responses: pd.DataFrame) -> pd.DataFrame:
    """Distance from the all-midpoint respondent, and response entropy.

    Scale-midpoint answering and empirical modal answering are different
    diagnostics; this function measures the former and response entropy.
    These diagnostics alone do not establish an answering strategy.
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
