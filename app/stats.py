"""Exact inference helpers shared by the 2026 confirmatory and sensitivity stages.

Each helper validates its inputs and raises ``ValueError`` rather than returning
a confident statistic from invalid data. The arithmetic of each helper matches
the inline code it replaced operation for operation, so released p-values are
reproduced to the last bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binomtest

Alternative = Literal["two-sided", "greater", "less"]
ALTERNATIVES: tuple[str, ...] = ("two-sided", "greater", "less")


def bh_adjust(p_values: ArrayLike, family_size: int) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values over a declared family of tests.

    ``family_size`` is the number of tests declared before the results were
    seen, and ``p_values`` must hold exactly one entry per declared test. A
    non-finite entry marks a declared test that could not be estimated: it
    stays NaN in the result but still counts towards m, so an undefined test
    can never make the other adjusted p-values smaller. Tied p-values are
    ranked stably and the result is capped at 1.

    Operation order is ``p * m / rank`` followed by a reversed cumulative
    minimum, the order used by every released BH column.
    """
    if isinstance(family_size, bool) or not isinstance(family_size, int | np.integer):
        raise ValueError(f"family_size must be an integer, got {family_size!r}")
    if family_size < 1:
        raise ValueError(f"family_size must be at least 1, got {family_size}")
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"p-values must be one-dimensional, got shape {p.shape}")
    if len(p) != family_size:
        raise ValueError(
            f"expected one p-value per declared test ({family_size}), got {len(p)}; "
            "a shrunken family would understate the adjustment"
        )
    valid = np.isfinite(p)
    out_of_range = p[valid][(p[valid] < 0) | (p[valid] > 1)]
    if len(out_of_range):
        raise ValueError(f"p-values must be in [0, 1], got {out_of_range.tolist()}")
    result = np.full(len(p), np.nan)
    indices = np.flatnonzero(valid)
    order = np.argsort(p[indices], kind="stable")
    sorted_indices = indices[order]
    if len(indices):
        adjusted = p[sorted_indices] * family_size / np.arange(1, len(indices) + 1)
        result[sorted_indices] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


@dataclass(frozen=True)
class SignTest:
    """Exact binomial sign test on paired differences.

    ``n_effective`` excludes zero differences (ties), which carry no sign
    information; ``n_ties`` reports how many were dropped. With no non-tied
    difference the test is uninformative and ``p_value`` is 1.
    """

    n_positive: int
    n_negative: int
    n_effective: int
    n_ties: int
    p_value: float


def sign_test(deltas: ArrayLike, alternative: Alternative = "two-sided") -> SignTest:
    """Exact sign test of H0: P(delta > 0) = 1/2 over the non-tied differences.

    ``alternative`` refers to positive differences: ``"greater"`` tests for
    more positive than negative signs. To test for negative shifts, pass the
    negated differences with ``"greater"``. Non-finite deltas raise, because a
    NaN is neither a tie nor a sign.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    d = np.asarray(deltas, dtype=float)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError(f"sign test needs a non-empty vector of deltas, got shape {d.shape}")
    finite = np.isfinite(d)
    if not finite.all():
        raise ValueError(
            f"sign test needs finite deltas; {int((~finite).sum())} of {len(d)} are non-finite: "
            f"{d[~finite].tolist()}"
        )
    n_positive = int((d > 0).sum())
    n_negative = int((d < 0).sum())
    n_effective = n_positive + n_negative
    p = (
        binomtest(n_positive, n_effective, 0.5, alternative=alternative).pvalue
        if n_effective
        else 1.0
    )
    return SignTest(
        n_positive=n_positive,
        n_negative=n_negative,
        n_effective=n_effective,
        n_ties=len(d) - n_effective,
        p_value=float(p),
    )


def permutation_mean_difference(
    values: ArrayLike,
    in_group: ArrayLike,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[float, float]:
    """Observed group-mean difference and its two-sided permutation p-value.

    The statistic is ``mean(values[in_group]) - mean(values[~in_group])``.
    Each replicate calls ``rng.permutation(in_group)`` once, in order, so the
    caller's generator stream fixes the result. The p-value carries the
    plus-one correction, ``(1 + #{|perm| >= |obs|}) / (n_permutations + 1)``.

    ``in_group`` must be a strictly boolean vector aligned with ``values``, and
    both groups must be non-empty; ``values`` must be finite.
    """
    labels = np.asarray(in_group)
    if labels.dtype.kind != "b":
        raise ValueError(f"group labels must be a boolean vector, got dtype {labels.dtype}")
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or labels.shape != v.shape:
        raise ValueError(
            f"values and group labels must be aligned vectors, got {v.shape} and {labels.shape}"
        )
    finite = np.isfinite(v)
    if not finite.all():
        raise ValueError(
            f"permutation test needs finite values; {int((~finite).sum())} of {len(v)} "
            "are non-finite"
        )
    n_in = int(labels.sum())
    if n_in == 0 or n_in == len(labels):
        raise ValueError(
            f"both groups must be non-empty, got {n_in} in group and {len(labels) - n_in} outside"
        )
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int | np.integer):
        raise ValueError(f"n_permutations must be an integer, got {n_permutations!r}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations}")
    obs = v[labels].mean() - v[~labels].mean()
    perm = np.empty(n_permutations)
    for b in range(n_permutations):
        lab = rng.permutation(labels)
        perm[b] = v[lab].mean() - v[~lab].mean()
    p = float((1 + (np.abs(perm) >= abs(obs)).sum()) / (n_permutations + 1))
    return float(obs), p
