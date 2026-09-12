"""The two appendix contrasts computed from generated 2026 artefacts.

Both are pure functions of small aggregate frames, so they are tested on
synthetic fixtures and the script ``scripts/appendix_contrasts_2026.py``
only wires them to ``data/``.

``petition_contrast``
    Per model, the keyed-PC1 (map-unit) zh-en shift of one target item
    minus the mean keyed-PC1 shift of the other items in a reference set,
    with a two-sided exact binomial sign test on how many models the
    target is more survival-ward (more negative on PC1) than the reference
    mean. The keyed shift is ``delta * d_pc1_per_unit``: the raw-scale
    item shift multiplied by that item's PC1 loading per raw unit
    (``diag_2026_item_keying.csv``), i.e. the item's own contribution to
    the model's PC1 displacement.

``origin_profile_permutation``
    Exact permutation test on the pairwise Pearson correlations between
    models' ten-item profiles. Statistic: mean correlation over
    within-cohort pairs (both cohorts pooled) minus mean correlation over
    cross-cohort pairs; the null enumerates every C(n, n_chinese)
    labelling, and the p-value is the share of labellings whose statistic
    is at least the observed one (the observed labelling is one of them,
    so p >= 1 / C(n, k)). The absolute-value two-sided count is reported
    alongside; the within-Chinese, within-Western and cross means are
    returned as descriptives, together with the Chinese-minus-cross
    contrast under the same enumeration.
"""

from __future__ import annotations

import logging
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import binomtest

log = logging.getLogger(__name__)

MAX_EXACT_LABELLINGS = 5_000_000


def keyed_pc1_shift(item_fx: pd.DataFrame, keying: pd.DataFrame) -> pd.DataFrame:
    """Return ``item_fx`` with a ``keyed_pc1`` column: ``delta * d_pc1_per_unit``.

    ``item_fx`` needs ``llm``, ``question``, ``delta`` (zh minus en on the
    raw item scale); ``keying`` needs ``question`` and ``d_pc1_per_unit``.
    """
    per_unit = keying.set_index("question")["d_pc1_per_unit"]
    missing = sorted(set(item_fx["question"]) - set(per_unit.index))
    if missing:
        raise ValueError(f"no keying row for item(s) {missing}")
    out = item_fx.copy()
    out["keyed_pc1"] = out["delta"].to_numpy() * out["question"].map(per_unit).to_numpy()
    return out


def petition_contrast(
    item_fx: pd.DataFrame,
    keying: pd.DataFrame,
    target: str,
    reference: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model target-minus-reference keyed-PC1 contrast and its sign test.

    Returns ``(per_model, summary)``. ``per_model`` has one row per model
    with the target's keyed shift, the reference mean and the contrast;
    ``summary`` is one row: models where the contrast is negative (target
    more survival-ward), the number of non-tied models, the median contrast
    and the two-sided exact binomial p-value.
    """
    if target in reference:
        raise ValueError("target must not be in the reference set")
    keyed = keyed_pc1_shift(item_fx, keying)
    keyed = keyed[keyed["question"].isin([target, *reference])]
    piv = keyed.pivot(index="llm", columns="question", values="keyed_pc1")
    absent = sorted({target, *reference} - set(piv.columns))
    if absent:
        raise ValueError(f"item(s) {absent} absent from item_fx")
    piv = piv.dropna(subset=[target, *reference])

    per_model = pd.DataFrame(
        {
            "llm": piv.index,
            f"keyed_pc1_{target}": piv[target].to_numpy(),
            "keyed_pc1_reference_mean": piv[reference].mean(axis=1).to_numpy(),
        }
    )
    per_model["contrast"] = per_model[f"keyed_pc1_{target}"] - per_model["keyed_pc1_reference_mean"]
    per_model["target_more_survival_ward"] = per_model["contrast"] < 0

    c = per_model["contrast"].to_numpy()
    n_surv = int((c < 0).sum())
    n_eff = int((c != 0).sum())
    p = binomtest(n_surv, n_eff, 0.5, alternative="two-sided").pvalue if n_eff else float("nan")
    summary = pd.DataFrame(
        [
            {
                "target": target,
                "reference": "+".join(reference),
                "n_models": len(c),
                "n_effective": n_eff,
                "n_target_more_survival_ward": n_surv,
                "median_contrast": float(np.median(c)),
                "p_sign_two_sided": float(p),
            }
        ]
    )
    log.info(
        "%s vs mean(%s): %d/%d more survival-ward, median %.4f, p = %.6f",
        target,
        ",".join(reference),
        n_surv,
        n_eff,
        summary["median_contrast"].iloc[0],
        p,
    )
    return per_model, summary


def _pair_means(corr: np.ndarray, is_a: np.ndarray, iu: tuple) -> tuple[float, float, float]:
    """Mean correlation over within-A, within-B and cross pairs."""
    same_a = is_a[:, None] & is_a[None, :]
    same_b = ~is_a[:, None] & ~is_a[None, :]
    cross = is_a[:, None] ^ is_a[None, :]
    r = corr[iu]
    return (
        float(r[same_a[iu]].mean()),
        float(r[same_b[iu]].mean()),
        float(r[cross[iu]].mean()),
    )


def origin_profile_permutation(
    profiles: pd.DataFrame,
    is_chinese: np.ndarray,
) -> pd.DataFrame:
    """Exact permutation test of within- vs cross-origin profile similarity.

    ``profiles`` is a models x items frame (one row per model); ``is_chinese``
    a boolean vector aligned with its rows. Enumerates all C(n, k)
    labellings, so n and k must be small (17 choose 10 = 19,448).
    """
    x = profiles.to_numpy(dtype=float)
    is_cn = np.asarray(is_chinese, dtype=bool)
    n = len(x)
    if is_cn.shape != (n,):
        raise ValueError("is_chinese must have one entry per profile row")
    k = int(is_cn.sum())
    if k == 0 or k == n:
        raise ValueError("both cohorts must be non-empty")
    n_lab = comb(n, k)
    if n_lab > MAX_EXACT_LABELLINGS:
        raise ValueError(f"C({n}, {k}) = {n_lab:,} labellings is too many to enumerate")

    corr = np.corrcoef(x)
    iu = np.triu_indices(n, 1)
    n_within = comb(k, 2) + comb(n - k, 2)
    n_cross = k * (n - k)

    def stats(lab: np.ndarray) -> tuple[float, float]:
        wc, ww, cr = _pair_means(corr, lab, iu)
        pooled_within = (wc * comb(k, 2) + ww * comb(n - k, 2)) / n_within
        return pooled_within - cr, wc - cr

    obs_within_cross, obs_cn_cross = stats(is_cn)
    wc, ww, cr = _pair_means(corr, is_cn, iu)

    perm = np.empty((n_lab, 2))
    lab = np.zeros(n, dtype=bool)
    for i, idx in enumerate(combinations(range(n), k)):
        lab[:] = False
        lab[list(idx)] = True
        perm[i] = stats(lab)

    eps = 1e-12
    rows = []
    for j, (name, obs) in enumerate(
        [("within_minus_cross", obs_within_cross), ("chinese_minus_cross", obs_cn_cross)]
    ):
        v = perm[:, j]
        rows.append(
            {
                "statistic": name,
                "observed": obs,
                "n_labellings": n_lab,
                "n_ge_observed": int((v >= obs - eps).sum()),
                "p_exact_ge": float((v >= obs - eps).mean()),
                "p_exact_abs": float((np.abs(v) >= abs(obs) - eps).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["mean_r_within_chinese"] = wc
    out["mean_r_within_western"] = ww
    out["mean_r_cross"] = cr
    out["n_chinese"] = k
    out["n_western"] = n - k
    out["n_within_pairs"] = n_within
    out["n_cross_pairs"] = n_cross
    log.info(
        "within-CN r=%.4f within-W r=%.4f cross r=%.4f; within-cross p=%.5f (%d/%d)",
        wc,
        ww,
        cr,
        out["p_exact_ge"].iloc[0],
        out["n_ge_observed"].iloc[0],
        n_lab,
    )
    return out
