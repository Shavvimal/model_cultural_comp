"""Cultural-region assignment for map positions.

Two rules are reported side by side, per docs/statistical-review.md §2.5:
an RBF-SVM over the country coordinates (with its cross-validated accuracy
attached — around 0.55 on 109 countries in 8 classes, so a single label is
weak evidence), and the nearest region centroid. Their disagreement is a
result, not a nuisance. "Positional stability" is the share of bootstrap
replicates falling in a fixed decision region: it reflects sampling
uncertainty of the position only, never the classifier's own error rate.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

# Includes the regularised regime: the 2024 grid started at C=500, which
# never evaluates a smooth boundary at all.
PARAM_GRID = {
    "C": [0.1, 1, 10, 100, 500, 1000, 2000],
    "gamma": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    "kernel": ["rbf"],
}


class RegionClassifier:
    """Region assignment over (PC1', PC2') with honest uncertainty reporting."""

    def __init__(self):
        self.svm = None
        self.regions = None  # index -> region name
        self.centroids = None  # region -> (PC1', PC2')
        self.cv_accuracy = None  # 5-fold stratified CV accuracy of the SVM

    def fit(self, country_scores: pd.DataFrame) -> "RegionClassifier":
        data = country_scores.dropna(subset=["PC1_rescaled", "PC2_rescaled", "Cultural Region"])
        labels = pd.Categorical(data["Cultural Region"])
        self.regions = list(labels.categories)
        xy = data[["PC1_rescaled", "PC2_rescaled"]].to_numpy(dtype=float)
        codes = labels.codes.astype(int)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        search = GridSearchCV(SVC(), PARAM_GRID, refit=True, cv=cv)
        search.fit(xy, codes)
        self.svm = search.best_estimator_
        self.cv_accuracy = float(cross_val_score(search.best_estimator_, xy, codes, cv=cv).mean())

        self.centroids = data.groupby("Cultural Region")[["PC1_rescaled", "PC2_rescaled"]].mean()
        return self

    def predict_svm(self, xy: np.ndarray) -> list[str]:
        if self.svm is None:
            raise RuntimeError("Fit the classifier first.")
        return [self.regions[c] for c in self.svm.predict(np.asarray(xy, dtype=float))]

    def predict_centroid(self, xy: np.ndarray) -> list[str]:
        if self.centroids is None:
            raise RuntimeError("Fit the classifier first.")
        xy = np.asarray(xy, dtype=float)
        dists = np.linalg.norm(xy[:, None, :] - self.centroids.to_numpy()[None, :, :], axis=2)
        return [self.centroids.index[i] for i in dists.argmin(axis=1)]

    def region_assignments(self, boot: pd.DataFrame) -> pd.DataFrame:
        """Both rules per model, with the full SVM share vector.

        Returns modal SVM region + positional stability, the runner-up, the
        nearest-centroid region of the mean position, an agreement flag, and
        the classifier's CV accuracy on every row so it can never be quoted
        without it.
        """
        rows = []
        for llm, g in boot.groupby("llm"):
            xy = g[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
            shares = pd.Series(self.predict_svm(xy)).value_counts(normalize=True)
            centroid_region = self.predict_centroid(xy.mean(axis=0, keepdims=True))[0]
            rows.append(
                {
                    "llm": llm,
                    "svm_region": shares.index[0],
                    "positional_stability": float(shares.iloc[0]),
                    "svm_runner_up": shares.index[1] if len(shares) > 1 else None,
                    "svm_runner_up_share": float(shares.iloc[1]) if len(shares) > 1 else 0.0,
                    "svm_share_vector": shares.round(3).to_dict(),
                    "centroid_region": centroid_region,
                    "rules_agree": shares.index[0] == centroid_region,
                    "svm_cv_accuracy": self.cv_accuracy,
                }
            )
        return pd.DataFrame(rows)
