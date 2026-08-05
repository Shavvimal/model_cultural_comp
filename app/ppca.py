"""Probabilistic PCA with support for missing data.

Derived from ``pca-magic`` (https://github.com/allentran/pca-magic),
Copyright Allen Tran, licensed under the Apache License, Version 2.0.
Changes from the original: standardization parameters are stored and applied
symmetrically at transform time, fitting is seedable, and model persistence
round-trips all parameters (loadings, means, stds, eigenvalues) rather than
the loadings alone.

The underlying method is the EM algorithm for PPCA of Tipping & Bishop (1999),
"Probabilistic Principal Component Analysis", J. R. Statist. Soc. B 61(3).
"""

import numpy as np
from scipy.linalg import orth


class PPCA:
    """Probabilistic PCA fitted with EM, tolerant of missing values.

    After fitting, ``transform(X)`` standardizes ``X`` with the means and
    standard deviations learned during ``fit`` before projecting onto the
    principal axes, so new data lands in the same coordinate space as the
    training scores.
    """

    def __init__(self):
        self.C = None  # (D, d) principal axes
        self.means = None  # (D,) feature means learned in fit
        self.stds = None  # (D,) feature stds learned in fit
        self.eig_vals = None  # (d,) score variances, descending
        self.var_exp = None  # (d,) cumulative explained variance ratio
        self.data = None  # (N, D) standardized training data, EM-imputed
        self.valid_series = None  # (D_in,) column mask applied during fit

    def fit(self, data, d=None, tol=1e-4, min_obs=10, seed=None, verbose=False, max_iter=1000):
        """Fit the model to ``data`` (shape N x D, NaNs allowed).

        :param d: number of latent dimensions (defaults to D)
        :param tol: relative tolerance on the EM objective for convergence
        :param min_obs: drop columns with fewer than this many observed values
        :param seed: seed for the random initialization of the loading matrix;
            set for reproducible fits
        :param verbose: print the convergence criterion each iteration
        :param max_iter: hard cap on EM iterations; raises RuntimeError rather
            than silently returning an unconverged fit
        """
        raw = np.array(data, dtype=float, copy=True)
        raw[np.isinf(raw)] = np.max(raw[np.isfinite(raw)])

        # Columns dropped here are remembered so transform() can apply the
        # same selection — otherwise fitted means/stds/C would silently
        # misalign with full-width input.
        self.valid_series = np.sum(~np.isnan(raw), axis=0) >= min_obs
        data = raw[:, self.valid_series].copy()
        N, D = data.shape

        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)
        data = (data - self.means) / self.stds

        observed = ~np.isnan(data)
        missing = np.sum(~observed)
        # NaNs are replaced with zeros so matrix operations can proceed; the
        # E-step below overwrites them with model reconstructions each pass.
        data[~observed] = 0

        if d is None:
            d = D

        rng = np.random.default_rng(seed)
        C = rng.standard_normal((D, d))

        CC = C.T @ C
        X = data @ C @ np.linalg.inv(CC)
        recon = X @ C.T
        recon[~observed] = 0
        ss = np.sum((recon - data) ** 2) / (N * D - missing)

        v0 = np.inf
        counter = 0

        while True:
            Sx = np.linalg.inv(np.eye(d) + CC / ss)

            # E-step: estimate latent variables and impute missing entries
            ss0 = ss
            if missing > 0:
                proj = X @ C.T
                data[~observed] = proj[~observed]
            X = data @ C @ Sx / ss

            # M-step: update the loading matrix
            XX = X.T @ X
            C = data.T @ X @ np.linalg.pinv(XX + N * Sx)
            CC = C.T @ C
            recon = X @ C.T
            recon[~observed] = 0

            ss = (np.sum((recon - data) ** 2) + N * np.sum(CC * Sx) + missing * ss0) / (N * D)

            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            v1 = N * (D * np.log(ss) + np.trace(Sx) - det) + np.trace(XX) - missing * np.log(ss0)
            diff = abs(v1 / v0 - 1)
            if verbose:
                print(diff)
            if (diff < tol) and (counter > 5):
                break
            if counter >= max_iter:
                raise RuntimeError(
                    f"EM did not converge within {max_iter} iterations "
                    f"(last relative change {diff:.2e}, tol {tol})."
                )

            counter += 1
            v0 = v1

        # Orthogonalize C and align it with the principal axes of the scores
        C = orth(C)
        vals, vecs = np.linalg.eig(np.cov((data @ C).T))
        order = np.flipud(np.argsort(vals))
        vecs = vecs[:, order]
        vals = vals[order]
        C = C @ vecs

        # Fix an arbitrary sign ambiguity: make the largest-magnitude loading
        # in each column positive so fits are reproducible across seeds.
        signs = np.sign(C[np.argmax(np.abs(C), axis=0), np.arange(C.shape[1])])
        C = C * signs

        self.C = C
        self.data = data
        self.eig_vals = vals
        self._calc_var()

    def transform(self, data=None):
        """Project data onto the principal axes.

        With no argument, returns the scores of the (standardized, EM-imputed)
        training data. Otherwise ``data`` must be raw (unstandardized) complete
        observations with the same columns, in the same order, as the data
        passed to ``fit``; it is standardized with the fitted means and stds
        before projection.
        """
        if self.C is None:
            raise RuntimeError("Fit the model first.")
        if data is None:
            return self.data @ self.C
        data = np.asarray(data, dtype=float)
        if self.valid_series is not None and data.shape[1] == len(self.valid_series):
            data = data[:, self.valid_series]
        if data.shape[1] != len(self.means):
            raise ValueError(
                f"expected {len(self.valid_series) if self.valid_series is not None else len(self.means)} "
                f"columns (as passed to fit), got {data.shape[1]}"
            )
        if np.isnan(data).any():
            raise ValueError(
                "transform() requires complete observations; "
                "drop or impute rows with missing values first."
            )
        return ((data - self.means) / self.stds) @ self.C

    def _calc_var(self):
        var = np.nanvar(self.data.T, axis=1)
        self.var_exp = self.eig_vals.cumsum() / var.sum()

    def save(self, fpath):
        """Save all model parameters (npz)."""
        np.savez(
            fpath,
            C=self.C,
            means=self.means,
            stds=self.stds,
            eig_vals=self.eig_vals,
            valid_series=self.valid_series,
        )

    def load(self, fpath):
        """Load model parameters saved by :meth:`save`."""
        with np.load(fpath) as npz:
            self.C = npz["C"]
            self.means = npz["means"]
            self.stds = npz["stds"]
            self.eig_vals = npz["eig_vals"]
            if "valid_series" in npz:
                self.valid_series = npz["valid_series"]
