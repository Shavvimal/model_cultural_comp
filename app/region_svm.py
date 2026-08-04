"""Cultural-region assignment via an SVM over country map coordinates."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

PARAM_GRID = {
    "C": [500, 1000, 1500, 2000],
    "gamma": [0.05, 0.1, 0.15, 0.2],
    "kernel": ["rbf"],
}


class RegionClassifier:
    """RBF-SVM over (PC1, PC2) country coordinates, labels = cultural regions."""

    def __init__(self):
        self.svm = None
        self.regions = None  # index -> region name

    def fit(self, country_scores: pd.DataFrame) -> "RegionClassifier":
        data = country_scores.dropna(
            subset=["PC1_rescaled", "PC2_rescaled", "Cultural Region"]
        )
        labels = pd.Categorical(data["Cultural Region"])
        self.regions = list(labels.categories)
        xy = data[["PC1_rescaled", "PC2_rescaled"]].to_numpy(dtype=float)

        search = GridSearchCV(SVC(), PARAM_GRID, refit=True, cv=5)
        search.fit(xy, labels.codes.astype(int))
        self.svm = search.best_estimator_
        return self

    def predict(self, xy: np.ndarray) -> list:
        if self.svm is None:
            raise RuntimeError("Fit the classifier first.")
        return [self.regions[c] for c in self.svm.predict(np.asarray(xy, dtype=float))]

    def region_stability(self, boot: pd.DataFrame) -> pd.DataFrame:
        """Fraction of bootstrap replicates assigned to each region, per model.

        Returns one row per model with the modal region and its share —
        a region assignment reported *with* its bootstrap stability rather
        than as a point prediction.
        """
        rows = []
        for llm, g in boot.groupby("llm"):
            preds = pd.Series(
                self.predict(g[["PC1_rescaled", "PC2_rescaled"]].to_numpy())
            )
            shares = preds.value_counts(normalize=True)
            rows.append({
                "llm": llm,
                "region": shares.index[0],
                "stability": shares.iloc[0],
                "runner_up": shares.index[1] if len(shares) > 1 else None,
                "runner_up_share": shares.iloc[1] if len(shares) > 1 else 0.0,
            })
        return pd.DataFrame(rows)
