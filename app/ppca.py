import numpy as np
from scipy.linalg import orth
import os

class PPCA():

    def __init__(self):

        self.raw = None
        self.data = None
        self.C = None
        self.means = None
        self.stds = None
        self.eig_vals = None

    def _standardize(self, X):
        # Standardize the data using precomputed means and standard deviations
        if self.means is None or self.stds is None:
            raise RuntimeError("Fit model first")

        return (X - self.means) / self.stds

    def fit(self, data, d=None, tol=1e-4, min_obs=10, verbose=False):
        """
        Fit the probabilistic PCA model to the data
        :param data: np.array, shape (N, D)
        :param d: int, number of latent dimensions
        :param tol: float, tolerance for convergence
        :param min_obs: int, minimum number of observations for a series to be included
        :param verbose: bool, print convergence information
        """

        #################################
        ####### Preprocess data ########
        #################################

        self.raw = data
        # Replace infinite values with the maximum finite value in the data
        self.raw[np.isinf(self.raw)] = np.max(self.raw[np.isfinite(self.raw)])

        # Remove series with less than min_obs observations
        valid_series = np.sum(~np.isnan(self.raw), axis=0) >= min_obs
        data = self.raw[:, valid_series].copy()

        # Take dimensions
        N = data.shape[0]
        D = data.shape[1]

        #################################
        ####### Standardize data ########
        #################################

        # Calculate means and standard deviations of each column while ignoring NaNs
        self.means = np.nanmean(data, axis=0) # Mean of each feature
        self.stds = np.nanstd(data, axis=0) # Standard deviation of each feature
        # Standardize the data
        data = self._standardize(data)

        # Determine which data points are observed
        observed = ~np.isnan(data)
        missing = np.sum(~observed)
        # Replace NaNs with zeros
        # This is done to ensure matrix operations can proceed without interruption
        # The zeros are placeholders and will be corrected during iterative updates
        data[~observed] = 0

        # Initial number of principal components
        if d is None:
            d = data.shape[1]

        # Initialize the loading matrix C
        if self.C is None:
            C = np.random.randn(D, d)
        else:
            C = self.C

        # Calculate the covariance matrix
        CC = np.dot(C.T, C)
        # Initial latent variable estimates
        X = np.dot(np.dot(data, C), np.linalg.inv(CC))
        recon = np.dot(X, C.T)
        recon[~observed] = 0
        # Initial residual variance
        ss = np.sum((recon - data) ** 2) / (N * D - missing)

        v0 = np.inf
        counter = 0

        while True:

            Sx = np.linalg.inv(np.eye(d) + CC / ss)

            # E-step: Estimate latent variables X
            ss0 = ss
            if missing > 0:
                proj = np.dot(X, C.T)
                data[~observed] = proj[~observed]  # Project missing values onto observed data
            X = np.dot(np.dot(data, C), Sx) / ss

            # M-step: Update the loading matrix C
            XX = np.dot(X.T, X)
            C = np.dot(np.dot(data.T, X), np.linalg.pinv(XX + N * Sx))
            CC = np.dot(C.T, C)
            recon = np.dot(X, C.T)
            recon[~observed] = 0  # Corrects the projection

            # Update residual variance
            ss = (np.sum((recon - data) ** 2) + N * np.sum(CC * Sx) + missing * ss0) / (N * D)
            # Check for convergence
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            v1 = N * (D * np.log(ss) + np.trace(Sx) - det) \
                 + np.trace(XX) - missing * np.log(ss0)
            diff = abs(v1 / v0 - 1)
            if verbose:
                print(diff)
            if (diff < tol) and (counter > 5):
                break

            counter += 1
            v0 = v1

        # Orthogonalize C
        C = orth(C)
        vals, vecs = np.linalg.eig(np.cov(np.dot(data, C).T))
        order = np.flipud(np.argsort(vals))
        vecs = vecs[:, order]
        vals = vals[order]

        # Rotate C using the eigenvectors of the covariance matrix
        C = np.dot(C, vecs)

        # Attach objects to the class
        self.C = C
        self.data = data
        self.eig_vals = vals
        self._calc_var()

    def transform(self, data=None):

        if self.C is None:
            raise RuntimeError('Fit the data model first.')
        if data is None:
            return np.dot(self.data, self.C)
        return np.dot(data, self.C)

    def _calc_var(self):

        if self.data is None:
            raise RuntimeError('Fit the data model first.')

        data = self.data.T

        # variance calc
        var = np.nanvar(data, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals.cumsum() / total_var

    def save(self, fpath):
        # Save the loading matrix C to a file
        np.save(fpath, self.C)

    def load(self, fpath):
        # Load the loading matrix C from a file
        assert os.path.isfile(fpath)
        self.C = np.load(fpath)