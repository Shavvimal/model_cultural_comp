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

from app.culture_map import IV_QNS

log = logging.getLogger(__name__)

MAX_EXACT_LABELLINGS = 5_000_000
# Tie tolerance for exact permutation counts. A labelling whose statistic
# equals the observed one can differ from it by floating-point rounding of
# the pooled correlation means (order 1e-15); 1e-12 counts those as ties while
# staying far below any real difference between correlation means.
PERMUTATION_TIE_TOLERANCE = 1e-12


def _require_instrument_items(name: str, items: pd.Index | pd.Series) -> None:
    """Raise unless ``items`` is exactly the ten instrument items, naming the gap."""
    present = {str(item) for item in items}
    missing = sorted(set(IV_QNS) - present)
    extra = sorted(present - set(IV_QNS))
    if missing or extra:
        raise ValueError(
            f"{name} must contain exactly the ten instrument items; missing {missing}, extra {extra}"
        )


def standardise_profiles(profiles: pd.DataFrame, item_baselines: pd.DataFrame) -> pd.DataFrame:
    """Apply the released frozen fit moments, aligned by item identifier.

    These are observed-item standardisation parameters from the fitted
    instrument, not moments recalculated from model profiles or completed
    human scores. Reject partial or invalid aggregates instead of silently
    producing a different transform.
    """
    required = {"question", "fit_standardisation_mean", "fit_standardisation_sd"}
    if not required.issubset(item_baselines.columns):
        raise ValueError(f"item baselines require columns {sorted(required)}")
    if item_baselines["question"].duplicated().any():
        raise ValueError("item baselines must contain one row per question")
    _require_instrument_items("item baselines", item_baselines["question"])
    if profiles.columns.duplicated().any():
        duplicated = sorted(map(str, profiles.columns[profiles.columns.duplicated()]))
        raise ValueError(
            f"profiles must contain each instrument item once; duplicated {duplicated}"
        )
    _require_instrument_items("profiles", profiles.columns)
    fitted = item_baselines.set_index("question").reindex(profiles.columns)
    means = fitted["fit_standardisation_mean"].astype(float)
    stds = fitted["fit_standardisation_sd"].astype(float)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("fitted standardisation means must be finite and SDs positive")
    return (profiles - means) / stds


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
    is_cn = np.asarray(is_chinese)
    n = len(x)
    if is_cn.shape != (n,) or is_cn.dtype.kind != "b":
        raise ValueError("is_chinese must have one boolean entry per profile row")
    k = int(is_cn.sum())
    if k < 2 or n - k < 2:
        raise ValueError("both cohorts need at least two models for within-cohort correlations")
    if x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("profiles must contain at least two finite item values per model")
    if not np.any(x != x[:, :1], axis=1).all():
        raise ValueError("each model profile must vary across items for Pearson correlation")
    n_lab = comb(n, k)
    if n_lab > MAX_EXACT_LABELLINGS:
        raise ValueError(f"C({n}, {k}) = {n_lab:,} labellings is too many to enumerate")

    corr = np.corrcoef(x)
    if not np.isfinite(corr).all():
        raise ValueError("profiles must produce finite Pearson correlations")
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

    eps = PERMUTATION_TIE_TOLERANCE
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
