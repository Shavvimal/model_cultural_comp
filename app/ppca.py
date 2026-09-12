"""Gaussian probabilistic PCA with observed-data likelihood fitting.

This module retains the projection/persistence interface of the former
``pca-magic`` adaptation (Copyright Allen Tran, Apache License 2.0), but replaces
its missing-data fitting loop. Observed entries are standardized once, then the
Gaussian PPCA likelihood is optimized directly, integrating out missing entries.
Missing values are completed with conditional means only *after* fitting.

The Gaussian model is Tipping & Bishop (1999), "Probabilistic Principal Component
Analysis", JRSS B 61(3). Fitting here uses L-BFGS-B, not EM. ``C`` contains
orthonormal score axes; ``loadings_`` and ``noise_variance_`` retain the distinct
Gaussian model parameters.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, orth
from scipy.optimize import minimize

_NOISE_FLOOR = 1e-10
# Bound attempts even when floating-point objective stagnation consumes no iterations.
_MAX_OPTIMIZER_RESTARTS = 3


def _pattern_statistics(data):
    """Return observed-index/count/scatter triples, without imputing anything."""
    patterns, membership = np.unique(np.isfinite(data), axis=0, return_inverse=True)
    groups = []
    for index, observed in enumerate(patterns):
        if not observed.any():
            continue  # an entirely missing row has observed likelihood one
        rows = data[membership == index][:, observed]
        groups.append((observed, len(rows), rows.T @ rows))
    return groups


def _negative_log_likelihood(parameters, groups, width, dimensions, n_rows):
    """Mean observed-row NLL and analytic gradient for fixed standardization."""
    loadings = parameters[:-1].reshape(width, dimensions)
    noise = np.exp(parameters[-1])
    loss = 0.0
    gradient = np.zeros_like(loadings)
    noise_gradient = 0.0
    for observed, count, scatter in groups:
        w = loadings[observed]
        covariance = w @ w.T + noise * np.eye(len(w))
        cholesky = cho_factor(covariance, lower=True)
        precision = cho_solve(cholesky, np.eye(len(w)))
        logdet = 2 * np.log(np.diag(cholesky[0])).sum()
        loss += 0.5 * (count * (len(w) * np.log(2 * np.pi) + logdet) + np.sum(precision * scatter))
        covariance_gradient = 0.5 * (count * precision - precision @ scatter @ precision)
        gradient[observed] += 2 * covariance_gradient @ w
        noise_gradient += np.trace(covariance_gradient)
    return loss / n_rows, np.r_[gradient.ravel(), noise * noise_gradient] / n_rows


class PPCA:
    """Fit Gaussian PPCA, then expose the existing completed-data score axes.

    ``transform(X)`` standardizes complete raw observations and projects onto
    ``C``. These scores are orthogonal projections, not posterior latent means.
    ``transform()`` projects the conditionally completed training observations.
    """

    def __init__(self):
        self.C = None
        self.means = None
        self.stds = None
        self.eig_vals = None
        self.var_exp = None
        self.data = None
        self.valid_series = None
        self.loadings_ = None
        self.noise_variance_ = None
        self.log_likelihood_ = None
        self.gradient_norm_ = None
        self.n_iter_ = None
        self.converged_ = False
        self.start_log_likelihoods_ = None
        self.start_gradient_norms_ = None
        self.start_converged_ = None
        self.likelihood_history_ = None
        self.method_ = None
        self.tolerance_ = None
        self.n_informative_rows_ = None

    def fit(
        self,
        data,
        d=None,
        tol=1e-7,
        min_obs=10,
        seed=None,
        verbose=False,
        max_iter=1000,
        n_init=3,
    ):
        """Fit an unweighted, fixed-standardization Gaussian PPCA model.

        ``d`` must be between 1 and retained width minus 1 (default: width minus
        1). Columns with fewer than ``min_obs`` finite observations are dropped;
        retained constant columns and infinities are rejected. Entirely missing
        rows carry no likelihood information and complete to the zero mean.

        For incomplete data, ``n_init`` starts use a spectral initialization and
        seeded random perturbations. ``tol`` bounds the maximum absolute gradient
        of the negative log likelihood *per informative row*, including log noise
        variance. An objective-change message alone is not convergence. Every
        start must satisfy that gradient bound within ``max_iter`` iterations;
        objective-change stops may resume within that same iteration budget.
        Otherwise fitting raises. The converged start with greatest likelihood
        is retained. Multiple starts reduce, but do not rule out, local optima.

        Complete data use the analytic maximum-likelihood PPCA solution. Fits
        requiring noise variance at or below 1e-10 in standardized units, or a
        rank-deficient requested latent space, are rejected rather than silently
        returning a singular model. No fitted state is replaced on failure.
        """
        raw = np.array(data, dtype=float, copy=True)
        if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
            raise ValueError("data must be a two-dimensional array with at least two rows/columns")
        if np.isinf(raw).any():
            raise ValueError("data must contain finite observations or NaN, not infinity")
        for name, value in (("min_obs", min_obs), ("max_iter", max_iter), ("n_init", n_init)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        valid = np.sum(np.isfinite(raw), axis=0) >= min_obs
        retained = raw[:, valid]
        width = retained.shape[1]
        if width < 2:
            raise ValueError("at least two columns must meet min_obs")
        if d is None:
            d = width - 1
        if (
            isinstance(d, (bool, np.bool_))
            or not isinstance(d, (int, np.integer))
            or not 1 <= d < width
        ):
            raise ValueError("d must be an integer between 1 and retained columns minus 1")
        means = np.nanmean(retained, axis=0)
        stds = np.nanstd(retained, axis=0)
        if not np.isfinite(stds).all() or np.any(stds <= 0):
            raise ValueError("retained columns must have nonzero finite variance")
        standardized = (retained - means) / stds
        informative_rows = int(np.isfinite(standardized).any(axis=1).sum())
        if informative_rows < 2:
            raise ValueError("at least two rows must contain observed values")
        groups = _pattern_statistics(standardized)
        zero_filled = np.nan_to_num(standardized, nan=0.0)
        covariance = zero_filled.T @ zero_filled / informative_rows
        values, axes = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values, axes = values[order], axes[:, order]
        noise = float(values[d:].mean())
        complete = np.isfinite(standardized).all()
        if complete and noise <= _NOISE_FLOOR:
            raise ValueError(
                "no nonsingular PPCA maximum: residual noise variance is zero or too small"
            )
        initial_noise = max(noise, 0.05)
        initial_loadings = axes[:, :d] * np.sqrt(np.maximum(values[:d] - initial_noise, 0.01))
        if complete:
            if np.any(values[:d] <= noise):
                raise ValueError("requested latent space has zero-variance dimensions")
            loadings = axes[:, :d] * np.sqrt(values[:d] - noise)
            parameters = np.r_[loadings.ravel(), np.log(noise)]
            loss, gradient = _negative_log_likelihood(
                parameters, groups, width, d, informative_rows
            )
            if not np.isfinite(loss) or np.max(np.abs(gradient)) > tol:
                raise RuntimeError(
                    "analytic PPCA solution exceeds the requested numerical gradient tolerance"
                )
            histories = [[-loss * informative_rows]]
            results = [(parameters, loss, float(abs(gradient).max()), 0)]
            method = "closed-form complete-data PPCA"
        else:
            rng = np.random.default_rng(seed)
            histories, results = [], []
            initial = np.r_[initial_loadings.ravel(), np.log(initial_noise)]
            for start in range(n_init):
                x0 = initial.copy()
                if start:
                    x0[:-1] += rng.normal(scale=0.2, size=width * d)
                    x0[-1] += rng.normal(scale=0.2)
                history = []

                def objective(x):
                    return _negative_log_likelihood(x, groups, width, d, informative_rows)

                def record(x, history=history, start=start):
                    value, grad = objective(x)
                    history.append(-value * informative_rows)
                    if verbose:
                        print(
                            f"start {start + 1}: log likelihood {history[-1]:.8f}, "
                            f"gradient {abs(grad).max():.3e}"
                        )

                history.append(-objective(x0)[0] * informative_rows)
                iterations = 0
                for _ in range(_MAX_OPTIMIZER_RESTARTS + 1):
                    result = minimize(
                        objective,
                        x0,
                        jac=True,
                        method="L-BFGS-B",
                        callback=record,
                        bounds=[(None, None)] * (width * d) + [(np.log(_NOISE_FLOOR), 20.0)],
                        options={
                            "ftol": 0.0,
                            "gtol": tol,
                            "maxiter": max_iter - iterations,
                            "maxls": 50,
                        },
                    )
                    iterations += int(result.nit)
                    loss, gradient = objective(result.x)
                    norm = float(abs(gradient).max())
                    if (
                        not np.isfinite(loss)
                        or not np.isfinite(norm)
                        or norm <= tol
                        or not result.success
                        or iterations >= max_iter
                    ):
                        break
                    # L-BFGS-B may report an unchanged objective before the gradient
                    # converges. Reset its curvature history, retaining the same
                    # parameters, likelihood, bounds and strict acceptance criterion.
                    x0 = result.x
                if not np.isfinite(loss) or not np.isfinite(norm) or norm > tol:
                    raise RuntimeError(
                        f"PPCA likelihood start {start + 1}/{n_init} did not converge within "
                        f"{max_iter} iterations: gradient {norm:.3e} exceeds tol {tol:.3e} "
                        f"({result.message})."
                    )
                if np.exp(result.x[-1]) <= _NOISE_FLOOR * 1.01:
                    raise ValueError(
                        "PPCA fit reached the residual-noise floor; no accepted interior fit"
                    )
                histories.append(history)
                results.append((result.x, loss, norm, iterations))
            method = "observed-data Gaussian PPCA likelihood (L-BFGS-B)"
        best = min(range(len(results)), key=lambda index: results[index][1])
        parameters, loss, gradient_norm, iterations = results[best]
        loadings = parameters[:-1].reshape(width, d)
        noise = float(np.exp(parameters[-1]))
        completed = standardized.copy()
        patterns, membership = np.unique(np.isfinite(completed), axis=0, return_inverse=True)
        for index, observed in enumerate(patterns):
            if observed.all():
                continue
            rows = np.flatnonzero(membership == index)
            if not observed.any():
                completed[rows] = 0.0
                continue
            w = loadings[observed]
            # E[y_missing | y_observed] under the *fitted* Gaussian covariance.
            gain = np.linalg.solve(w @ w.T + noise * np.eye(len(w)), w @ loadings[~observed].T)
            completed[np.ix_(rows, ~observed)] = completed[np.ix_(rows, observed)] @ gain
        C = orth(loadings)
        if C.shape[1] != d:
            raise ValueError("fitted Gaussian loadings have a rank-deficient latent space")
        score_covariance = np.atleast_2d(np.cov((completed @ C).T))
        score_variances, score_axes = np.linalg.eigh(score_covariance)
        order = np.argsort(score_variances)[::-1]
        score_variances = score_variances[order]
        if np.any(score_variances <= 0):
            raise ValueError("fitted projection has zero-variance scores")
        C = C @ score_axes[:, order]
        C *= np.sign(C[np.argmax(np.abs(C), axis=0), np.arange(d)])
        self.C, self.means, self.stds = C, means, stds
        self.data, self.valid_series = completed, valid
        self.eig_vals = score_variances
        self.var_exp = score_variances.cumsum() / np.var(completed, axis=0, ddof=1).sum()
        self.loadings_, self.noise_variance_ = loadings, noise
        self.log_likelihood_, self.gradient_norm_ = -loss * informative_rows, gradient_norm
        self.n_iter_, self.converged_, self.method_ = iterations, True, method
        self.tolerance_, self.n_informative_rows_ = tol, informative_rows
        self.start_log_likelihoods_ = np.array([-item[1] * informative_rows for item in results])
        self.start_gradient_norms_ = np.array([item[2] for item in results])
        self.start_converged_ = np.ones(len(results), dtype=bool)
        self.likelihood_history_ = np.array(histories[best])
        return self

    def transform(self, data=None):
        """Project complete raw rows, or the completed training rows if omitted."""
        if self.C is None:
            raise RuntimeError("Fit the model first.")
        if data is None:
            if self.data is None:
                raise RuntimeError(
                    "training data are not retained by save/load; supply complete data"
                )
            return self.data @ self.C
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("transform() requires a two-dimensional array")
        if self.valid_series is not None and data.shape[1] == len(self.valid_series):
            data = data[:, self.valid_series]
        if data.shape[1] != len(self.means):
            raise ValueError(
                f"expected {len(self.means)} retained columns or the original fit width, "
                f"got {data.shape[1]}"
            )
        if not np.isfinite(data).all():
            raise ValueError("transform() requires complete finite observations")
        return ((data - self.means) / self.stds) @ self.C

    def state_dict(self):
        """Return only pickle-free fitted parameters/diagnostics, never microdata."""
        if self.C is None:
            raise RuntimeError("Fit the model first.")
        return {
            name: getattr(self, name)
            for name in (
                "C",
                "means",
                "stds",
                "eig_vals",
                "var_exp",
                "valid_series",
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
                "tolerance_",
                "n_informative_rows_",
            )
            if getattr(self, name) is not None
        }

    def save(self, fpath):
        """Save projection/Gaussian parameters and convergence evidence, not data."""
        np.savez(fpath, **self.state_dict())

    def load(self, fpath):
        """Load a new model or a legacy projection-only archive without pickle."""
        loaded = PPCA()
        with np.load(fpath, allow_pickle=False) as archive:
            for name in ("C", "means", "stds", "eig_vals"):
                setattr(loaded, name, archive[name])
            for name in vars(loaded).keys() - {"C", "means", "stds", "eig_vals", "data"}:
                if name in archive:
                    value = archive[name]
                    setattr(loaded, name, value.item() if value.ndim == 0 else value)
        self.__dict__.update(loaded.__dict__)
        return self
