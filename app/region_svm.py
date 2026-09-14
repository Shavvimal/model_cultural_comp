"""Cultural-region assignment for map positions.

Two rules are reported side by side (write-up §3.4): an RBF-SVM over the
country coordinates, and the nearest region centroid. The SVM carries its
tuning score on every row: the mean stratified 5-fold accuracy of the selected
grid point, on the same folds used to select it. That score is optimistic, not
an unbiased held-out accuracy, and it sits below the training accuracy, so a
single SVM label is weak evidence. Analysis scripts print the current value. Their disagreement
is a result, not a nuisance. "Positional stability" is the share of bootstrap
replicates falling in a fixed decision region: it reflects sampling
uncertainty of the position only, never the classifier's own error rate.

Neither rule carries a headline claim. The write-up's headline statistics
route through no classifier at all (distance from the fixed survey
reference, share of countries closer, minimum distance to any non-Western region
centroid); see ``app.llm_bootstrap.centroid_statistics``.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

# Includes the regularised regime: the 2024 grid started at C=500, which
# never evaluates a smooth boundary at all. Grid search is wrapped in a
# stratified 5-fold CV. The reported score reuses the selection folds and is
# therefore a tuning score, not an unbiased held-out accuracy estimate.
PARAM_GRID = {
    "C": [0.1, 1, 10, 100, 500, 1000, 2000],
    "gamma": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    "kernel": ["rbf"],
}
# Fixed fold shuffle for the stratified 5-fold selection and its reported score.
CV_RANDOM_STATE = 0
CV_FOLDS = 5


class RegionClassifier:
    """Region assignment over (PC1', PC2') with honest uncertainty reporting."""

    def __init__(self):
        self.svm = None
        self.regions = None  # index -> region name
        self.centroids = None  # region -> (PC1', PC2')
        self.cv_accuracy = None  # selected-grid 5-fold CV score (optimistically selected)

    def fit(self, country_scores: pd.DataFrame) -> "RegionClassifier":
        data = country_scores.dropna(subset=["PC1_rescaled", "PC2_rescaled", "Cultural Region"])
        labels = pd.Categorical(data["Cultural Region"])
        self.regions = list(labels.categories)
        xy = data[["PC1_rescaled", "PC2_rescaled"]].to_numpy(dtype=float)
        codes = labels.codes.astype(int)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
        search = GridSearchCV(SVC(), PARAM_GRID, refit=True, cv=cv)
        search.fit(xy, codes)
        self.svm = search.best_estimator_
        # The selected grid point's mean fold score. Re-running cross_val_score on
        # a clone over the same folds recomputes exactly this value (the SVC is
        # deterministic), which was checked bit-identical on the released scores.
        self.cv_accuracy = float(search.best_score_)

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

    def region_assignments(
        self, boot: pd.DataFrame, point_estimates: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Both rules per model, with the full SVM share vector.

        Returns modal SVM region + positional stability, the runner-up, the
        nearest-centroid region of the mean position, an agreement flag, and
        the classifier's CV accuracy on every row so it can never be quoted
        without it.
        """
        rows = []
        if point_estimates is not None and point_estimates["llm"].duplicated().any():
            duplicated = sorted(point_estimates.loc[point_estimates["llm"].duplicated(), "llm"])
            raise ValueError(
                f"point_estimates must contain one row per llm; duplicated {duplicated}"
            )
        points = None if point_estimates is None else point_estimates.set_index("llm")
        for llm, g in boot.groupby("llm"):
            xy = g[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
            shares = pd.Series(self.predict_svm(xy)).value_counts(normalize=True)
            point = (
                xy.mean(axis=0)
                if points is None
                else points.loc[llm, ["PC1_rescaled", "PC2_rescaled"]].to_numpy(dtype=float)
            )
            centroid_region = self.predict_centroid(point[None, :])[0]
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
