"""The Inglehart-Welzel cultural map pipeline.

Fits a probabilistic PCA to the ten IVS items, fixes a single varimax
rotation, and projects both survey respondents and LLM survey responses
through one identical path:

    standardize (fitted means/stds) -> project onto C -> rotate by R -> rescale

The rotation is fitted exactly once, on the training score matrix, and stored.
Everything projected afterwards — country data and model data alike — reuses
the stored rotation, so all points share one coordinate space.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import Rotator

from app.ppca import PPCA

# The ten IVS items behind the Inglehart-Welzel map
IV_QNS = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]

# Valid value ranges per item. IVS microdata uses negative SPSS user-missing
# codes (-1 "don't know" ... -5 "missing"); anything outside these ranges is a
# sentinel, not a datum. Y003 is the documented offender: 31.9% of rows carry
# -3 ("not applicable"), which an earlier version of this pipeline read as
# data and which zeroed the item's loading on both axes.
ITEM_VALID_RANGES = {
    "A008": (1, 4),
    "A165": (1, 2),
    "E018": (1, 3),
    "E025": (1, 3),
    "F063": (1, 10),
    "F118": (1, 10),
    "F120": (1, 10),
    "G006": (1, 4),
    "Y002": (1, 3),
    "Y003": (-2, 2),
}

# Published WVS rescaling constants: PC' = a * PC + b
PC_RESCALE_PARAMS = {"PC1": (1.81, 0.38), "PC2": (1.61, -0.01)}

CULTURAL_REGION_COLORS = {
    "African-Islamic": "#000000",
    "Confucian": "#56b4e9",
    "Latin America": "#cc79a7",
    "Protestant Europe": "#d55e00",
    "Catholic Europe": "#e69f00",
    "English-Speaking": "#009e73",
    "Orthodox Europe": "#0072b2",
    "West & South Asia": "#f0e442",
    # Deep violet: validated CVD-distinct from all eight region colors
    # (the previous #bada55 was indistinguishable from the West & South Asia
    # yellow under protanopia).
    "AI Model": "#5e35b1",
}


class CulturalMap:
    """Fit the IW cultural map on IVS data and project new data onto it."""

    def __init__(self, ivs_df, country_codes):
        """``ivs_df`` and ``country_codes`` may be DataFrames or pickle paths."""
        self.ivs_df = ivs_df if isinstance(ivs_df, pd.DataFrame) else pd.read_pickle(ivs_df)
        self.country_codes = (
            country_codes
            if isinstance(country_codes, pd.DataFrame)
            else pd.read_pickle(country_codes)
        )

        self.subset_ivs_df = None
        self.valid_data = None
        self.country_scores_pca = None
        self.llm_scores_pca = None

        self.iv_qns = IV_QNS
        self.pc_rescale_params = PC_RESCALE_PARAMS
        self.cultural_region_colors = CULTURAL_REGION_COLORS

        self.ppca = PPCA()
        self.rotation = None  # (2, 2) varimax rotation, fitted once in fit()
        self.score_stds = None  # (2,) rotated-score SDs, fixed at fit time
        self.sentinel_counts = None  # per-item out-of-range recode counts

    ##############################################
    ################ Fitting #####################
    ##############################################

    def prepare_data(self):
        """Filter the IVS to post-2005 waves and the ten map items.

        Out-of-range values (SPSS user-missing sentinels) are recoded to NaN
        *before* the completeness filter, so a sentinel never counts as an
        answered item. Per-item recode counts are kept in
        ``self.sentinel_counts``.
        """
        subset = self.ivs_df[["S020", "S003", "S017"] + self.iv_qns]
        subset = subset.rename(columns={"S020": "year", "S003": "country_code", "S017": "weight"})
        # The waves from 2005 onwards reflect current societal norms; earlier
        # waves would blend in values measured up to four decades ago.
        subset = subset[subset["year"] >= 2005].copy()

        self.sentinel_counts = {}
        for qn, (lo, hi) in ITEM_VALID_RANGES.items():
            bad = subset[qn].notna() & ((subset[qn] < lo) | (subset[qn] > hi))
            self.sentinel_counts[qn] = int(bad.sum())
            subset.loc[bad, qn] = np.nan

        # Require at least 6 of the 10 items answered
        subset = subset.dropna(subset=self.iv_qns, thresh=6)
        self.subset_ivs_df = subset

    def fit(self, seed=42, verbose=False):
        """Fit the PPCA and fix the varimax rotation, once.

        The rotation is fitted on the training score matrix and stored in
        ``self.rotation``; :meth:`project` applies the same stored rotation to
        anything projected later. Fitting the rotator a second time on new
        data would place that data in a different, incomparable coordinate
        space — the defect this class exists to prevent.
        """
        if self.subset_ivs_df is None:
            raise RuntimeError("Call prepare_data() first.")

        self.ppca.fit(
            self.subset_ivs_df[self.iv_qns].to_numpy(),
            d=2,
            min_obs=1,
            seed=seed,
            verbose=verbose,
        )
        scores = self.ppca.transform()

        rotator = Rotator(method="varimax")
        rotator.fit_transform(scores)
        self.rotation = rotator.rotation_
        self._orient_rotation()

        rotated = scores @ self.rotation
        # The published WVS rescale constants presuppose unit-variance factor
        # scores; standardize the rotated scores before applying them, and
        # keep the SDs so projected data goes through the identical path.
        self.score_stds = rotated.std(axis=0, ddof=0)

        self.valid_data = self._rescale(rotated)
        self.valid_data["country_code"] = self.subset_ivs_df["country_code"].values
        self.valid_data["weight"] = self.subset_ivs_df["weight"].values
        n_before = len(self.valid_data)
        self.valid_data = self.valid_data.merge(
            self.country_codes, left_on="country_code", right_on="Numeric", how="left"
        )
        if len(self.valid_data) != n_before:
            raise RuntimeError(
                "country_codes merge changed the row count "
                f"({n_before} -> {len(self.valid_data)}): duplicated Numeric codes "
                "would silently corrupt every downstream coordinate."
            )

    def _orient_rotation(self):
        """Fix the rotation's sign/order ambiguity to the IW convention.

        Varimax determines the rotated axes only up to column order and sign.
        Pin both using item loadings with unambiguous placement on the map:
        F118 (justifiability of homosexuality) marks self-expression (positive
        PC1) and F063 (importance of God) marks traditional values (negative
        PC2).
        """
        loadings = self.ppca.C @ self.rotation
        f118 = self.iv_qns.index("F118")
        f063 = self.iv_qns.index("F063")

        if abs(loadings[f118, 0]) < abs(loadings[f118, 1]):
            self.rotation = self.rotation[:, ::-1]
            loadings = loadings[:, ::-1]
        signs = np.array(
            [1.0 if loadings[f118, 0] > 0 else -1.0, 1.0 if loadings[f063, 1] < 0 else -1.0]
        )
        self.rotation = self.rotation * signs

    def _rescale(self, rotated_scores) -> pd.DataFrame:
        if self.score_stds is None:
            raise RuntimeError("score_stds not set; fit() or load_model() first.")
        df = pd.DataFrame(rotated_scores / self.score_stds, columns=["PC1", "PC2"])
        for pc, (a, b) in self.pc_rescale_params.items():
            df[f"{pc}_rescaled"] = a * df[pc] + b
        return df

    ##############################################
    ############### Projection ###################
    ##############################################

    def project(self, data: pd.DataFrame) -> pd.DataFrame:
        """Project complete raw responses through the fitted pipeline.

        ``data`` must contain the ten IVS item columns with raw (1-10 scale)
        values and no missing entries. Standardization, projection, rotation
        and rescaling all reuse parameters fixed at fit time, so the output is
        directly comparable with the fitted country coordinates.
        """
        if self.rotation is None:
            raise RuntimeError("Call fit() (or load_model()) first.")
        scores = self.ppca.transform(data[self.iv_qns].to_numpy())
        return self._rescale(scores @ self.rotation)

    def calculate_mean_scores(self):
        """Country-level means of the rescaled individual scores.

        Weighted by the IVS equilibrated weight (S017), which corrects
        within-country sampling design; unweighted means would treat every
        respondent as equally representative of their country.
        """

        def weighted(group: pd.DataFrame) -> pd.Series:
            w = group["weight"].fillna(1.0)
            return pd.Series(
                {
                    "PC1_rescaled": np.average(group["PC1_rescaled"], weights=w),
                    "PC2_rescaled": np.average(group["PC2_rescaled"], weights=w),
                }
            )

        means = (
            self.valid_data.groupby("country_code")[["PC1_rescaled", "PC2_rescaled", "weight"]]
            .apply(weighted)
            .reset_index()
        )
        merged = means.merge(
            self.country_codes, left_on="country_code", right_on="Numeric", how="left"
        )
        self.country_scores_pca = merged.dropna(subset=["Numeric"])

    ##############################################
    ############ LLM survey responses ############
    ##############################################

    @staticmethod
    def y002_transform(ans) -> float:
        """Post-materialist index (Y002) from the two E-goal choices.

        Raises on invalid choices rather than returning a sentinel: the
        model-response path applies no range recode downstream, so a
        sentinel here would flow straight into a published coordinate.
        """
        first, second = ans[0], ans[1]
        if not (1 <= first <= 4 and 1 <= second <= 4):
            raise ValueError(f"Y002 choices out of range: {ans!r}")
        if (first == 1 and second == 3) or (first == 3 and second == 1):
            return 1  # materialist
        if (first == 2 and second == 4) or (first == 4 and second == 2):
            return 3  # post-materialist
        return 2  # mixed

    @staticmethod
    def y003_transform(ans: list[int]) -> float:
        """Autonomy index (Y003) from the chosen child qualities.

        Official IVS syntax: Y003 = (Q15 + Q17) - (Q8 + Q14), i.e.
        (religious faith + obedience) - (independence + determination), each
        coded 1 if the quality was mentioned and 2 if not, so higher values
        mean greater autonomy.
        """
        mentioned = {i: (1 if i in ans else 2) for i in range(1, 12)}
        independence = mentioned[2]
        determination = mentioned[8]
        faith = mentioned[9]
        obedience = mentioned[11]
        return (faith + obedience) - (independence + determination)

    ##############################################
    ############### Persistence ##################
    ##############################################

    def save_model(self, fpath):
        """Save PPCA parameters plus the fitted rotation (npz)."""
        np.savez(
            fpath,
            C=self.ppca.C,
            means=self.ppca.means,
            stds=self.ppca.stds,
            eig_vals=self.ppca.eig_vals,
            rotation=self.rotation,
            score_stds=self.score_stds,
        )

    def load_model(self, fpath):
        """Load parameters saved by :meth:`save_model`."""
        with np.load(fpath) as npz:
            self.ppca.C = npz["C"]
            self.ppca.means = npz["means"]
            self.ppca.stds = npz["stds"]
            self.ppca.eig_vals = npz["eig_vals"]
            self.rotation = npz["rotation"]
            self.score_stds = npz["score_stds"]

    ##############################################
    ############## Visualization #################
    ##############################################

    def visualize_cultural_map(
        self, title="Inglehart-Welzel Cultural Map", with_llms=False, ax=None
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 10))

        for region, color in self.cultural_region_colors.items():
            subset = self.country_scores_pca[self.country_scores_pca["Cultural Region"] == region]
            if subset.empty:
                continue
            for _, row in subset.iterrows():
                style = "italic" if row.get("Islamic", False) else "normal"
                ax.text(
                    row["PC1_rescaled"],
                    row["PC2_rescaled"],
                    row["Country"],
                    color=color,
                    fontsize=10,
                    fontstyle=style,
                )
            ax.scatter(subset["PC1_rescaled"], subset["PC2_rescaled"], label=region, color=color)

        if with_llms and self.llm_scores_pca is not None:
            color = self.cultural_region_colors["AI Model"]
            for _, row in self.llm_scores_pca.iterrows():
                style = "italic" if row["Chinese"] else "normal"
                ax.text(
                    row["PC1_rescaled"],
                    row["PC2_rescaled"],
                    row["llm"],
                    color=color,
                    fontsize=10,
                    fontstyle=style,
                )
            ax.scatter(
                self.llm_scores_pca["PC1_rescaled"],
                self.llm_scores_pca["PC2_rescaled"],
                label="AI Model",
                color=color,
            )

        ax.set_xlabel("Survival vs. Self-Expression Values")
        ax.set_ylabel("Traditional vs. Secular Values")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        return ax


if __name__ == "__main__":
    cultural_map = CulturalMap("../data/ivs_df.pkl", "../data/country_codes.pkl")
    cultural_map.prepare_data()
    cultural_map.fit(seed=42, verbose=True)
    cultural_map.calculate_mean_scores()
    cultural_map.visualize_cultural_map()
    plt.show()
