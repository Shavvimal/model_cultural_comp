"""The Inglehart-Welzel cultural map pipeline.

Fits a probabilistic PCA to the ten IVS items, fixes a single varimax
rotation, and projects both survey respondents and LLM survey responses
through one identical path:

    standardize (fitted means/stds) -> project onto C -> rotate by R -> rescale

The rotation is fitted exactly once, on the training score matrix, and stored.
Everything projected afterwards — country data and model data alike — reuses
the stored rotation, so all points share one coordinate space.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import Rotator
from matplotlib.axes import Axes

from app.ppca import PPCA
from app.survey_indices import Y003_CONSTITUENTS, recover_y003

# Declared analysis window: survey waves from this year onwards are pooled.
MIN_SURVEY_YEAR = 2005
# Completeness rule: a respondent needs at least this many of the ten items observed.
MIN_OBSERVED_ITEMS = 6
# Varimax convergence tolerance, pinned explicitly. The released instrument used
# factor_analyzer's default (tol=1e-5), which stopped at theta = -37.851 degrees
# rather than the optimum -39.397. Do not tighten it: that moves every published
# coordinate by up to 0.06 map units. Pinning guards against a library default change.
VARIMAX_TOL = 1e-5

# The ten IVS items behind the Inglehart-Welzel map
IV_QNS = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]

# Valid value ranges per item. IVS microdata uses negative SPSS user-missing
# codes (-1 "don't know" ... -5 "missing"); anything outside these ranges is a
# sentinel, not a datum. The merged EVS Y003 column contains -3 because its
# precomputed index is absent. After recoding, recover it wherever the four
# harmonized child-quality responses are observed (see survey_indices.py).
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

# WVS rescaling constants: PC' = a * PC + b, applied to unit-variance rotated
# scores. Provenance: the WVS Association's published SPSS syntax for the map
# (https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=tradrat):
#     COMPUTE SurvSAgg = 1.81 * SurvSelf + .038 .
#     COMPUTE TradAgg  = 1.61 * TradRat5 - .1 .
# Until v1.1.0 this table carried (1.81, 0.38) and (1.61, -0.01), the values
# printed in Tao et al. (2024), which misplace a decimal in both offsets; an
# external reader caught the discrepancy (GitHub issue #12, 2 Sept 2026). The
# slopes were never wrong. Because the offsets are additive and every point
# (respondent, country, model, region centroid, pooled human mean) passes
# through this one affine step, the correction is a rigid translation of the
# whole map by (-0.342, -0.090). Pairwise distances and comparisons against an
# equally translated reference are unchanged; fixed-zero crossings need not be.
# The offsets are the projection of the observed per-item marginal-mean vector,
# not necessarily the mean of completed respondent scores after PPCA imputation.
PC_RESCALE_PARAMS = {"PC1": (1.81, 0.038), "PC2": (1.61, -0.10)}
SURVEY_REFERENCE: tuple[float, float] = (PC_RESCALE_PARAMS["PC1"][1], PC_RESCALE_PARAMS["PC2"][1])
# Compatibility alias, not the mean of completed respondent coordinates.
HUMAN_MEAN = SURVEY_REFERENCE

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


def check_preparation(report: dict[str, Any]) -> None:
    """Reject a survey preparation that cannot support research artefacts.

    ``report`` is :attr:`CulturalMap.survey_preparation_report`. Raises if any
    Y003 constituent column is absent (no index could be reconstructed) or if
    any delivered Y003 disagrees with its valid constituents.
    """
    y003 = report["y003"]
    if y003["missing_constituent_columns"]:
        raise ValueError(
            "Y003 constituent columns are missing from the harmonized inputs: "
            f"{y003['missing_constituent_columns']}; supply the full harmonized inputs"
        )
    if y003["discordant_direct"]:
        raise ValueError(
            f"delivered Y003 disagrees with valid constituents in {y003['discordant_direct']} "
            "rows; resolve before fitting"
        )


def validate_weights(weights: pd.Series, groups: Any = None, *, context: str) -> pd.Series:
    """Return S017 weights unchanged after checking the published-mean contract.

    Every weight must be present, finite and non-negative (zero is allowed),
    and every group given by ``groups`` (anything ``Series.groupby`` accepts)
    must have a positive sum. Missing weights are never filled: a silent
    default would change a published weighted mean.
    """
    values = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values < 0)
    if invalid.any():
        raise ValueError(
            f"{context}: S017 weights must be present, finite and non-negative; "
            f"{int(invalid.sum())} of {len(values)} are not "
            f"(missing or non-finite {int((~np.isfinite(values)).sum())}, "
            f"negative {int((values < 0).sum())})"
        )
    if groups is not None:
        sums = weights.groupby(groups).sum()
        empty = sums.index[sums.to_numpy() <= 0].tolist()
        if empty:
            raise ValueError(f"{context}: S017 weights sum to zero for groups {empty}")
    elif values.sum() <= 0:
        raise ValueError(f"{context}: S017 weights sum to zero")
    return weights


class CulturalMap:
    """Fit the IW cultural map on IVS data and project new data onto it."""

    def __init__(
        self, ivs_df: pd.DataFrame | str | Path, country_codes: pd.DataFrame | str | Path
    ) -> None:
        """``ivs_df`` and ``country_codes`` may be DataFrames or pickle paths."""
        self.ivs_df = ivs_df if isinstance(ivs_df, pd.DataFrame) else pd.read_pickle(ivs_df)
        self.country_codes = (
            country_codes
            if isinstance(country_codes, pd.DataFrame)
            else pd.read_pickle(country_codes)
        )

        self.subset_ivs_df: pd.DataFrame | None = None
        self.valid_data: pd.DataFrame | None = None
        self.country_scores_pca: pd.DataFrame | None = None
        self.llm_scores_pca: pd.DataFrame | None = None

        self.iv_qns = IV_QNS
        self.pc_rescale_params = PC_RESCALE_PARAMS
        self.cultural_region_colors = CULTURAL_REGION_COLORS

        self.ppca = PPCA()
        # (2, 2) varimax rotation, fitted once in fit()
        self.rotation: np.ndarray | None = None
        # (2,) rotated-score SDs, fixed at fit time
        self.score_stds: np.ndarray | None = None
        # per-item out-of-range recode counts
        self.sentinel_counts: dict[str, int] | None = None
        # aggregate reconstruction/eligibility audit
        self.survey_preparation_report: dict[str, Any] | None = None

    ##############################################
    ################ Fitting #####################
    ##############################################

    def prepare_data(self) -> None:
        """Filter the IVS to post-2005 waves and the ten map items.

        Out-of-range values (SPSS user-missing sentinels) are recoded to NaN
        before recovering missing Y003 from its four valid binary survey
        constituents, then applying the completeness filter. A sentinel
        never counts as an answered item. Per-item recode counts are kept
        in ``self.sentinel_counts``; reconstruction, concordance and
        eligibility counts are in ``self.survey_preparation_report``.
        """
        columns = ["S020", "S003", "S017", *[qn for qn in self.iv_qns if qn != "Y003"]]
        columns += [column for column in ("Y003", *Y003_CONSTITUENTS) if column in self.ivs_df]
        subset = self.ivs_df[columns]
        subset = subset.rename(columns={"S020": "year", "S003": "country_code", "S017": "weight"})
        # Apply the declared post-2005 analysis window. Pooling these waves
        # does not calibrate the map to contemporary country values.
        subset = subset[subset["year"] >= MIN_SURVEY_YEAR].copy()
        if "Y003" not in subset:
            subset["Y003"] = np.nan

        self.sentinel_counts = {}
        for qn, (lo, hi) in ITEM_VALID_RANGES.items():
            bad = subset[qn].notna() & ((subset[qn] < lo) | (subset[qn] > hi))
            self.sentinel_counts[qn] = int(bad.sum())
            subset.loc[bad, qn] = np.nan

        eligible_before = subset[self.iv_qns].notna().sum(axis=1) >= MIN_OBSERVED_ITEMS
        recovery = recover_y003(subset)
        # prepare_data creates a missing column for an index absent from the
        # input; retain that distinction in the public aggregate report.
        recovery.report["input_index_column_present"] = "Y003" in self.ivs_df
        subset["Y003"] = recovery.values
        eligible_after = subset[self.iv_qns].notna().sum(axis=1) >= MIN_OBSERVED_ITEMS
        retained_provenance = recovery.provenance.loc[eligible_after]
        self.survey_preparation_report = {
            "schema_version": 1,
            "years_min": MIN_SURVEY_YEAR,
            "minimum_observed_items": MIN_OBSERVED_ITEMS,
            "post_2005_rows": len(subset),
            "eligible_before_y003_recovery": int(eligible_before.sum()),
            "eligible_after_y003_recovery": int(eligible_after.sum()),
            "added_eligible_rows": int((eligible_after & ~eligible_before).sum()),
            "y003": recovery.report,
            "retained_y003": {
                "direct": int(retained_provenance.eq("direct").sum()),
                "reconstructed": int(retained_provenance.eq("reconstructed").sum()),
                "still_missing": int(retained_provenance.eq("missing").sum()),
            },
        }
        self.subset_ivs_df = subset.loc[
            eligible_after, ["year", "country_code", "weight", *self.iv_qns]
        ].copy()

    def fit(self, seed: int | None = 42, verbose: bool = False) -> None:
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

        rotator = Rotator(method="varimax", tol=VARIMAX_TOL)
        rotator.fit_transform(scores)
        self.rotation = rotator.rotation_
        self._orient_rotation()

        rotated = scores @ self.rotation
        # The published WVS rescale constants presuppose unit-variance factor
        # scores; standardize the rotated scores before applying them, and
        # keep the SDs so projected data goes through the identical path.
        self.score_stds = rotated.std(axis=0, ddof=0)

        self.valid_data = self.rescale(rotated)
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

    def _orient_rotation(self) -> None:
        """Fix the rotation's sign/order ambiguity to the IW convention.

        Varimax determines the rotated axes only up to column order and sign.
        Pin both using projection coefficients with unambiguous placement on the map:
        F118 (justifiability of homosexuality) marks self-expression (positive
        PC1) and F063 (importance of God) marks traditional values (negative
        PC2).
        """
        if self.ppca.C is None or self.rotation is None:
            raise RuntimeError("fit the PPCA and rotation before orienting them")
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

    def rescale(self, rotated_scores: np.ndarray) -> pd.DataFrame:
        """Standardize rotated scores by the fitted SDs, then apply the WVS affine map.

        This is the single sanctioned final step of the frozen path
        (standardize, project onto C, rotate, rescale). Returns ``PC1``/``PC2``
        unit-variance scores and their ``*_rescaled`` map coordinates.
        """
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
        return self.rescale(scores @ self.rotation)

    def calculate_mean_scores(self) -> None:
        """Country-level means of the rescaled individual scores.

        Weighted by the IVS original national weight (S017), which corrects
        within-country sampling design; unweighted means would treat every
        respondent as equally representative of their country. Every retained
        weight must be present, finite and non-negative, and each country's
        weights must have a positive sum; otherwise this raises rather than
        substituting a default weight.
        """
        if self.valid_data is None:
            raise RuntimeError("Call fit() first.")
        validate_weights(
            self.valid_data["weight"],
            self.valid_data["country_code"],
            context="country mean scores",
        )

        def weighted(group: pd.DataFrame) -> pd.Series:
            w = group["weight"]
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
    def y002_transform(ans: Any) -> float:
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

    def save_model(self, fpath: str | Path) -> None:
        """Save projection, fitted Gaussian parameters and convergence evidence.

        The archive never contains completed respondent rows. Legacy archives
        remain readable, but do not acquire convergence evidence on loading.
        """
        if self.rotation is None or self.score_stds is None:
            raise RuntimeError("Call fit() before save_model().")
        state: dict[str, Any] = {
            **self.ppca.state_dict(),
            "rotation": self.rotation,
            "score_stds": self.score_stds,
        }
        np.savez(fpath, **state)

    def load_model(self, fpath: str | Path) -> None:
        """Load parameters saved by :meth:`save_model`."""
        self.ppca.load(fpath)
        with np.load(fpath, allow_pickle=False) as npz:
            self.rotation = npz["rotation"]
            self.score_stds = npz["score_stds"]

    ##############################################
    ############## Visualization #################
    ##############################################

    def visualize_cultural_map(
        self,
        title: str = "Inglehart-Welzel Cultural Map",
        with_llms: bool = False,
        ax: Axes | None = None,
    ) -> Axes:
        if self.country_scores_pca is None:
            raise RuntimeError("Call calculate_mean_scores() first.")
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
        ax.set_ylabel("Traditional vs. Secular-Rational Values")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        return ax


if __name__ == "__main__":
    # Paths are relative to the repo root, like every script under scripts/.
    cultural_map = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
    cultural_map.prepare_data()
    cultural_map.fit(seed=42, verbose=True)
    cultural_map.calculate_mean_scores()
    cultural_map.visualize_cultural_map()
    plt.show()
