"""Illustrative Gaussian coverage calculation at K = 10 equal-size clusters.

The published ellipses use the conventional normal-theory radius
chi2(0.95, 2) = 5.99 applied to the covariance of the cluster-bootstrap
replicate cloud (``app.llm_bootstrap.confidence_ellipses``). That radius
treats the replicate covariance as *known*. It is not: it is estimated from
only K = 10 prompt-variant clusters. This script quantifies undercoverage
in an equally weighted Gaussian benchmark, not the actual ordinal responses,
unequal retained counts, or heterogeneous prompt families in the corpus.

Three things are computed, each written to the artefact:

  1. ``hotelling_radius`` — the exact radius that would give nominal 95%
     coverage using the unbiased covariance-of-the-mean estimate from K
     Gaussian clusters: 2(K-1)/(K-2) * F(0.95; 2, K-2). For the bootstrap
     covariance, the corresponding squared radius is larger by K/(K-1).
  2. ``analytic_*`` — closed-form true coverage of the chi2 radius under two
     readings of "the covariance", both exact for Gaussian cluster means:
       - ``unbiased``: covariance estimated with divisor K-1.
       - ``bootstrap_plugin``: divisor K, which is what the *nonparametric
         cluster bootstrap* converges to as the replicate count grows. This
         matches the bootstrap covariance in the equal-cluster benchmark.
  3. ``simulated`` — a Monte Carlo of that Gaussian benchmark: draw K
     cluster means, resample K clusters with replacement B times exactly as
     ``bootstrap_llm_positions_cluster`` does, take the replicate cloud's
     mean and ddof=1 covariance exactly as ``confidence_ellipses`` does, and
     ask whether the true mean falls inside the chi2 ellipse.

Coverage of a Mahalanobis region is affine-invariant, so for Gaussian
cluster means the answer does not depend on the true covariance; the script
asserts this by simulating under both an identity and a strongly correlated
covariance and requiring the two to agree.

Needs no survey data and no corpus. Run from the repo root:

    uv run python scripts/coverage_calibration_2026.py
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import f as f_dist

K = 10  # prompt-variant clusters behind each cell's replicate cloud
LEVEL = 0.95
N_BOOT = 2_000  # bootstrap replicates per simulated cell
N_TRIALS = 20_000  # simulated cells
SEED = 42

# A deliberately non-spherical covariance, to demonstrate affine invariance.
CORRELATED = np.array([[4.0, 0.9], [0.9, 0.25]])


def hotelling_radius(k: int = K, level: float = LEVEL, divisor_k: bool = False) -> float:
    """Squared radius for unbiased or limiting bootstrap covariance of the mean.

    Exact only for equally weighted Gaussian clusters (and, for the
    bootstrap covariance, infinitely many resampling draws).
    """
    if k <= 2:
        raise ValueError("the two-dimensional Hotelling region requires K > 2")
    radius = 2 * (k - 1) / (k - 2) * f_dist.ppf(level, 2, k - 2)
    return radius * k / (k - 1) if divisor_k else radius


def analytic_coverage(
    k: int = K, level: float = LEVEL, divisor_k: bool = False, radius: float | None = None
) -> float:
    """Exact coverage of the chi2 radius for Gaussian cluster means.

    With the unbiased covariance, T^2 = 2(k-1)/(k-2) F(2, k-2), so coverage is
    the F CDF at the chi2 radius rescaled. With the plug-in (divisor k)
    covariance the ellipse shrinks by (k-1)/k in variance, which scales the
    radius the test is compared against by the same factor.
    """
    radius = chi2.ppf(level, df=2) if radius is None else radius
    if divisor_k:
        radius *= (k - 1) / k
    return float(f_dist.cdf(radius / (2 * (k - 1) / (k - 2)), 2, k - 2))


def simulate_coverage(
    cov: np.ndarray,
    k: int = K,
    n_boot: int = N_BOOT,
    n_trials: int = N_TRIALS,
    level: float = LEVEL,
    seed: int = SEED,
) -> float:
    """Gaussian equal-cluster benchmark for the nominal chi2 ellipse; true mean 0."""
    rng = np.random.default_rng(seed)
    radius = chi2.ppf(level, df=2)
    chol = np.linalg.cholesky(cov)
    covered = 0
    for _ in range(n_trials):
        clusters = rng.standard_normal((k, 2)) @ chol.T
        # The cluster bootstrap: draw k clusters with replacement, mean them.
        draws = rng.integers(0, k, size=(n_boot, k))
        cloud = clusters[draws].mean(axis=1)
        centre = cloud.mean(axis=0)
        cloud_cov = np.cov(cloud.T)
        delta = -centre  # true mean is the origin
        mahal = delta @ np.linalg.solve(cloud_cov, delta)
        covered += mahal <= radius
    return covered / n_trials


def calibration_table() -> pd.DataFrame:
    sim_identity = simulate_coverage(np.eye(2))
    sim_correlated = simulate_coverage(CORRELATED)
    if abs(sim_identity - sim_correlated) > 0.01:
        raise RuntimeError(
            "coverage is affine-invariant for Gaussian clusters but the two "
            f"simulated covariances disagree: {sim_identity} vs {sim_correlated}"
        )
    rows = [
        {
            "quantity": "chi2_radius_nominal_95",
            "value": float(chi2.ppf(LEVEL, df=2)),
            "note": "nominal normal-theory radius used in plots; not calibrated actual-corpus coverage",
        },
        {
            "quantity": "hotelling_radius",
            "value": hotelling_radius(),
            "note": f"unbiased covariance of the mean; equally weighted Gaussian clusters, K={K}",
        },
        {
            "quantity": "hotelling_radius_bootstrap_cov",
            "value": hotelling_radius(divisor_k=True),
            "note": "unbiased Hotelling radius multiplied by K/(K-1), for limiting bootstrap covariance",
        },
        {
            "quantity": "analytic_coverage_corrected_hotelling_bootstrap_cov",
            "value": analytic_coverage(divisor_k=True, radius=hotelling_radius(divisor_k=True)),
            "note": "Gaussian equal-cluster limit only; unequal observed counts are not calibrated here",
        },
        {
            "quantity": "analytic_coverage_unbiased_cov",
            "value": analytic_coverage(divisor_k=False),
            "note": "idealisation: covariance of K cluster means, divisor K-1",
        },
        {
            "quantity": "analytic_coverage_bootstrap_plugin_cov",
            "value": analytic_coverage(divisor_k=True),
            "note": "divisor K, the nonparametric cluster bootstrap's limit",
        },
        {
            "quantity": "simulated_coverage_identity_cov",
            "value": sim_identity,
            "note": f"Gaussian equal-cluster benchmark, {N_TRIALS} cells x {N_BOOT} replicates",
        },
        {
            "quantity": "simulated_coverage_correlated_cov",
            "value": sim_correlated,
            "note": "affine-invariance check; must match the identity run",
        },
    ]
    out = pd.DataFrame(rows)
    out["K_clusters"] = K
    out["scope"] = "illustrative_equal_weight_Gaussian_clusters"
    return out


def main() -> int:
    table = calibration_table()
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))
    table.to_csv("data/diag_2026_coverage_calibration.csv", index=False)
    print("\nWrote data/diag_2026_coverage_calibration.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
