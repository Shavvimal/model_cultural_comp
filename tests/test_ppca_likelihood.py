"""Independent Gaussian-density and exact-EM oracles for missing-data PPCA."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from app.ppca import PPCA


@pytest.fixture(scope="module")
def incomplete_data():
    rng = np.random.default_rng(9044)
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    loadings = 0.8 * np.c_[np.cos(angles), np.sin(angles)]
    raw = rng.normal(size=(850, 2)) @ loadings.T + 0.6 * rng.normal(size=(850, 6))
    missing = rng.random(raw.shape) < 0.13
    missing[:, 5] |= rng.random(850) < 0.24
    raw[missing] = np.nan
    return raw[np.isfinite(raw).sum(axis=1) >= 3]


@pytest.fixture(scope="module")
def fitted(incomplete_data):
    return PPCA().fit(incomplete_data, d=2, min_obs=1, seed=42)


def row_log_likelihood(data, loadings, noise):
    """SciPy densities for each row; no production likelihood helper used."""
    total = 0.0
    for row in data:
        observed = np.isfinite(row)
        if observed.any():
            w = loadings[observed]
            total += multivariate_normal.logpdf(
                row[observed], cov=w @ w.T + noise * np.eye(observed.sum())
            )
    return total


def independent_em(data, loadings, noise):
    """Exact latent-posterior EM, independent of likelihood optimization."""
    patterns, membership = np.unique(np.isfinite(data), axis=0, return_inverse=True)
    groups = [(p, data[membership == i][:, p]) for i, p in enumerate(patterns)]
    width, dimensions = loadings.shape
    for _ in range(1000):
        numerator = np.zeros_like(loadings)
        denominator = np.zeros((width, dimensions, dimensions))
        posteriors = []
        for observed, rows in groups:
            w = loadings[observed]
            posterior_cov = np.linalg.inv(np.eye(dimensions) + w.T @ w / noise)
            posterior_means = rows @ w @ posterior_cov / noise
            second_moment = len(rows) * posterior_cov + posterior_means.T @ posterior_means
            numerator[observed] += rows.T @ posterior_means
            denominator[observed] += second_moment
            posteriors.append((observed, rows, posterior_means, posterior_cov))
        new_loadings = np.array(
            [np.linalg.solve(denominator[j], numerator[j]) for j in range(width)]
        )
        residual, count = 0.0, 0
        for observed, rows, means, covariance in posteriors:
            w = new_loadings[observed]
            residual += np.sum((rows - means @ w.T) ** 2)
            residual += len(rows) * np.trace(w @ covariance @ w.T)
            count += rows.size
        new_noise = residual / count
        change = max(abs(new_loadings - loadings).max(), abs(new_noise - noise))
        loadings, noise = new_loadings, new_noise
        if change < 1e-10:
            return loadings, noise
    raise AssertionError("independent EM oracle did not converge")


def test_observed_likelihood_matches_independent_scipy_density(incomplete_data, fitted):
    standardized = (incomplete_data - fitted.means) / fitted.stds
    independent = row_log_likelihood(standardized, fitted.loadings_, fitted.noise_variance_)
    assert fitted.log_likelihood_ == pytest.approx(independent, abs=2e-9)
    # The released missing-data loop misses this fixture's optimum by about 60.
    assert independent == pytest.approx(-5183.419019391, abs=2e-7)
    assert np.min(np.diff(fitted.likelihood_history_)) >= -2e-9


def test_likelihood_optimum_agrees_with_exact_missing_data_em(incomplete_data, fitted):
    standardized = (incomplete_data - fitted.means) / fitted.stds
    initial = np.random.default_rng(73).normal(scale=0.5, size=(6, 2))
    loadings, noise = independent_em(standardized, initial, 0.7)
    covariance = loadings @ loadings.T + noise * np.eye(6)
    actual_covariance = fitted.loadings_ @ fitted.loadings_.T + fitted.noise_variance_ * np.eye(6)
    np.testing.assert_allclose(actual_covariance, covariance, atol=2e-6)
    assert fitted.noise_variance_ == pytest.approx(noise, abs=2e-7)


def test_finite_difference_of_independent_density_is_stationary(incomplete_data, fitted):
    standardized = (incomplete_data - fitted.means) / fitted.stds
    parameters = np.r_[fitted.loadings_.ravel(), np.log(fitted.noise_variance_)]
    gradients = []
    step = 1e-5
    for direction in np.eye(len(parameters)):
        plus, minus = parameters + step * direction, parameters - step * direction
        upper = row_log_likelihood(standardized, plus[:-1].reshape(6, 2), np.exp(plus[-1]))
        lower = row_log_likelihood(standardized, minus[:-1].reshape(6, 2), np.exp(minus[-1]))
        gradients.append((upper - lower) / (2 * step * len(standardized)))
    assert np.max(np.abs(gradients)) < 2e-7
    assert fitted.gradient_norm_ <= 1e-7


def test_completion_is_gaussian_conditional_mean(incomplete_data, fitted):
    standardized = (incomplete_data - fitted.means) / fitted.stds
    covariance = fitted.loadings_ @ fitted.loadings_.T + fitted.noise_variance_ * np.eye(6)
    for i, row in enumerate(standardized):
        observed = np.isfinite(row)
        expected = covariance[np.ix_(~observed, observed)] @ np.linalg.solve(
            covariance[np.ix_(observed, observed)], row[observed]
        )
        np.testing.assert_allclose(fitted.data[i, ~observed], expected, atol=1e-12)
    np.testing.assert_array_equal(
        fitted.data[np.isfinite(standardized)], standardized[np.isfinite(standardized)]
    )


def test_multiple_starts_and_seeds_agree(incomplete_data, fitted):
    assert len(fitted.start_log_likelihoods_) == 3
    assert fitted.start_converged_.all()
    assert fitted.start_gradient_norms_.max() <= 1e-7
    assert np.ptp(fitted.start_log_likelihoods_) < 1e-7
    same = PPCA().fit(incomplete_data, d=2, min_obs=1, seed=42)
    other = PPCA().fit(incomplete_data, d=2, min_obs=1, seed=12)
    np.testing.assert_array_equal(same.C, fitted.C)
    np.testing.assert_array_equal(same.loadings_, fitted.loadings_)
    np.testing.assert_allclose(other.C, fitted.C, atol=3e-6)
    assert other.log_likelihood_ == pytest.approx(fitted.log_likelihood_, abs=1e-7)


def test_complete_data_matches_analytic_covariance():
    rng = np.random.default_rng(91)
    raw = rng.normal(size=(1000, 2)) @ rng.normal(size=(2, 5)) + rng.normal(size=(1000, 5))
    model = PPCA().fit(raw, d=2, seed=3)
    standardized = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    u, singular, _ = np.linalg.svd(standardized.T, full_matrices=False)
    eigenvalues = singular**2 / len(raw)
    noise = eigenvalues[2:].mean()
    expected = (u[:, :2] * (eigenvalues[:2] - noise)) @ u[:, :2].T + noise * np.eye(5)
    actual = model.loadings_ @ model.loadings_.T + model.noise_variance_ * np.eye(5)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert model.noise_variance_ == pytest.approx(noise, abs=1e-14)
    assert model.n_iter_ == 0
    np.testing.assert_allclose(model.transform(), model.transform(raw), atol=1e-12)


def test_explained_variance_is_the_projected_centered_sum_of_squares(fitted):
    centered = fitted.data - fitted.data.mean(axis=0)
    expected = np.sum((centered @ fitted.C) ** 2) / np.sum(centered**2)
    assert fitted.var_exp[-1] == pytest.approx(expected, abs=1e-14)
    assert 0 < fitted.var_exp[-1] < 1


def test_all_missing_rows_and_filtered_columns(incomplete_data):
    raw = np.c_[incomplete_data, np.full(len(incomplete_data), np.nan)]
    raw = np.vstack([raw, np.full(raw.shape[1], np.nan)])
    model = PPCA().fit(raw, d=2, min_obs=2, seed=42)
    assert model.valid_series.tolist() == [True] * 6 + [False]
    np.testing.assert_array_equal(model.data[-1], np.zeros(6))
    complete = np.tile(model.means, (2, 1))
    np.testing.assert_array_equal(
        model.transform(complete), model.transform(np.c_[complete, [np.nan, np.nan]])
    )


def test_iteration_exhaustion_fails_without_replacing_fit(incomplete_data, fitted):
    model = PPCA().fit(incomplete_data, d=2, min_obs=1, seed=42)
    prior = model.C.copy()
    with pytest.raises(RuntimeError, match="did not converge"):
        model.fit(incomplete_data, d=2, min_obs=1, seed=42, max_iter=1)
    np.testing.assert_array_equal(model.C, prior)


def test_gaussian_and_convergence_parameters_roundtrip(fitted, tmp_path):
    path = tmp_path / "ppca.npz"
    fitted.save(path)
    loaded = PPCA().load(path)
    for name in (
        "loadings_",
        "noise_variance_",
        "log_likelihood_",
        "gradient_norm_",
        "n_iter_",
        "converged_",
        "start_log_likelihoods_",
        "start_gradient_norms_",
        "start_converged_",
        "likelihood_history_",
        "method_",
        "var_exp",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(fitted, name))
    with np.load(path, allow_pickle=False) as archive:
        assert "data" not in archive
        assert all(archive[key].dtype.kind != "O" for key in archive)
    with pytest.raises(RuntimeError, match="training data"):
        loaded.transform()


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda x: np.full_like(x, np.nan), "min_obs"),
        (lambda x: np.column_stack([np.ones(len(x)), x[:, 1:]]), "nonzero"),
        (lambda x: np.where(np.indices(x.shape)[0] == 0, np.inf, x), "infinity"),
        (lambda x: x[0], "two-dimensional"),
    ],
)
def test_bad_observations_fail_clearly(incomplete_data, change, message):
    with pytest.raises(ValueError, match=message):
        PPCA().fit(change(incomplete_data), d=2, min_obs=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d": 0},
        {"d": 6},
        {"d": 1.5},
        {"min_obs": 0},
        {"min_obs": True},
        {"tol": 0},
        {"tol": np.nan},
        {"max_iter": 0},
        {"n_init": 0},
    ],
)
def test_invalid_configuration_rejected(incomplete_data, kwargs):
    with pytest.raises(ValueError):
        PPCA().fit(incomplete_data, **kwargs)


def test_singular_complete_data_are_not_a_successful_gaussian_fit():
    x = np.arange(30.0)
    with pytest.raises(ValueError, match="nonsingular PPCA"):
        PPCA().fit(np.c_[x, x, x], d=1)


def test_one_dimension_and_legacy_projection_archive(tmp_path):
    raw = np.random.default_rng(2).normal(size=(80, 3))
    fitted = PPCA().fit(raw, d=1)
    assert fitted.C.shape == (3, 1)
    path = tmp_path / "old.npz"
    np.savez(path, C=fitted.C, means=fitted.means, stds=fitted.stds, eig_vals=fitted.eig_vals)
    loaded = PPCA().load(path)
    np.testing.assert_allclose(loaded.transform(raw), fitted.transform(raw), atol=1e-14)
    assert not loaded.converged_  # legacy projection arrays do not prove convergence
    assert loaded.noise_variance_ is None
