"""Inter-annotator agreement statistics for the reasoning-trace coding.

Implemented here rather than imported so the repo gains no dependency for a
camera-ready addition: Cohen's kappa (two raters), Fleiss' kappa (a fixed
number of raters per unit) and Krippendorff's alpha (nominal metric; any
number of raters, missing labels allowed). Each takes plain arrays and is
checked against a textbook example in ``tests/test_agreement.py``.
"""

import numpy as np


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two raters' labels of the same units."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("raters must label the same units")
    cats = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa from a (units x categories) matrix of rating counts.

    Every unit must have been rated by the same number of raters.
    """
    counts = np.asarray(counts, dtype=float)
    n_raters = counts.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss' kappa needs the same number of raters per unit")
    n = n_raters[0]
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = ((counts * (counts - 1)).sum(axis=1)) / (n * (n - 1))
    p_bar = p_i.mean()
    pe = float((p_j**2).sum())
    if pe == 1.0:
        return 1.0
    return float((p_bar - pe) / (1.0 - pe))


def krippendorff_alpha_nominal(reliability: np.ndarray) -> float:
    """Krippendorff's alpha, nominal metric, from a (raters x units) matrix.

    ``np.nan`` marks a missing label. Units with fewer than two labels drop
    out, as in Krippendorff's own procedure (coincidence-matrix form).
    """
    data = np.asarray(reliability, dtype=float)
    n_units = data.shape[1]
    values = np.unique(data[~np.isnan(data)])
    index = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for u in range(n_units):
        col = data[:, u]
        col = col[~np.isnan(col)]
        m_u = len(col)
        if m_u < 2:
            continue
        for i, vi in enumerate(col):
            for j, vj in enumerate(col):
                if i != j:
                    coincidence[index[vi], index[vj]] += 1.0 / (m_u - 1)
    n = coincidence.sum()
    if n == 0:
        raise ValueError("no unit carries two or more labels")
    n_c = coincidence.sum(axis=1)
    observed = coincidence.trace()
    expected = float((n_c**2).sum() - n_c.sum()) / (n - 1)
    if expected == n:
        return 1.0
    return float((observed - expected) / (n - expected))


def percent_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Raw proportion of units on which two raters agree."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(a == b))
