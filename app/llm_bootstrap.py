"""Bootstrap confidence regions for LLM positions on the cultural map.

The 2024 analysis assembled stored responses into ~50 pseudo-respondents per
model by arbitrary pairing. Because the map projection is affine in the ten
item values, a model's mean position depends only on its per-question mean
response — the pairing never mattered. The bootstrap below therefore
resamples each question's stored responses directly: each replicate draws
n_q responses per question with replacement, projects the resulting
mean-respondent through the corrected pipeline, and the spread of replicates
gives a 95% confidence ellipse per model.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

from app.culture_map import CulturalMap


def load_transformed_responses(cm: CulturalMap, collection_dir: str) -> pd.DataFrame:
    """Load raw stored responses and apply the WVS index transforms.

    Returns one row per (llm, question, numeric value).
    """
    files = glob.glob(os.path.join(collection_dir, "*.pkl"))
    if not files:
        raise FileNotFoundError(f"no collection pickles in {collection_dir}")
    df = pd.concat((pd.read_pickle(f) for f in files), ignore_index=True)

    # The 2024 harness recorded unparseable model output as None. Those are
    # refusals/failures, not observations — drop them, but say so, because
    # the per-question sample sizes shrink accordingly.
    null_mask = df["response"].isna()
    if null_mask.any():
        counts = df[null_mask].groupby("llm").size()
        print(f"dropping {null_mask.sum()} unparseable responses: "
              + ", ".join(f"{k}={v}" for k, v in counts.items()))
        df = df[~null_mask]

    def to_value(question, response) -> float:
        if question == "Y002":
            return float(cm.y002_transform(response))
        if question == "Y003":
            return float(cm.y003_transform(response))
        return float(response)

    df["value"] = [to_value(q, r) for q, r in zip(df["question"], df["response"])]
    return df[["llm", "question", "value"]]


def bootstrap_llm_positions(
    cm: CulturalMap,
    responses: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap replicate map positions for every model.

    For each replicate and question, draws as many values as were stored,
    with replacement, and projects the per-question means through the fitted
    pipeline. Returns a DataFrame with one row per (llm, replicate).
    """
    rng = np.random.default_rng(seed)
    out = []
    for llm, group in responses.groupby("llm"):
        by_qn = {q: g["value"].to_numpy() for q, g in group.groupby("question")}
        missing = [q for q in cm.iv_qns if q not in by_qn or len(by_qn[q]) == 0]
        if missing:
            raise ValueError(f"{llm}: no stored responses for {missing}")

        # (n_boot, 10) matrix of resampled per-question means
        means = np.column_stack([
            rng.choice(by_qn[q], size=(n_boot, len(by_qn[q])), replace=True).mean(axis=1)
            for q in cm.iv_qns
        ])
        projected = cm.project(pd.DataFrame(means, columns=cm.iv_qns))
        projected["llm"] = llm
        projected["replicate"] = np.arange(n_boot)
        out.append(projected)
    return pd.concat(out, ignore_index=True)


def confidence_ellipses(boot: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Mean position and 95% covariance ellipse per model.

    Ellipse axes are given as full widths; ``angle_deg`` is the orientation
    of the major axis, counter-clockwise from the x-axis.
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
        rows.append({
            "llm": llm,
            "PC1_rescaled": mean[0], "PC2_rescaled": mean[1],
            "ellipse_width": width, "ellipse_height": height,
            "angle_deg": angle,
            "sd_pc1": xy[:, 0].std(ddof=1), "sd_pc2": xy[:, 1].std(ddof=1),
        })
    return pd.DataFrame(rows)
