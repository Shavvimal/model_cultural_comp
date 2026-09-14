"""Generate the paper/blog figures from the corrected pipeline outputs.

Run from the repo root after validate_projection.py, bootstrap_llms.py and
instrument_sensitivity.py (which writes the country-year aggregates that date
Figure 0):

    uv run python scripts/make_figures.py

Writes vector PDFs (for LaTeX) and PNGs (for the blog) to figures/:
    fig0_countries_only.{pdf,png} the IW map redrawn from our fitted IVS data
    fig1_cultural_map.{pdf,png}   corrected IW map, nominal 95% mean-position regions
    fig2_svm_regions.{pdf,png}    SVM decision regions + model positions
"""

import os
import sys
from pathlib import Path

import matplotlib as mpl

# Headless rendering. Font/library versions can affect layout across platforms.
# Must precede the pyplot import.
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from app.culture_map import CULTURAL_REGION_COLORS
from app.region_svm import RegionClassifier

# Embed TrueType rather than matplotlib's default Type 3. ACL, IEEE and several
# ACM tracks reject Type 3 outright, so the paper figures are unusable without
# this. Set at module scope so scripts/make_figures_2026.py inherits it on import.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Written by scripts/instrument_sensitivity.py earlier in `make validate-2026`;
# its year column is the retained S020 survey year of each country aggregate.
YEAR_AGGREGATES = Path("data/validation_country_year_aggregates.csv")

AI_COLOR = CULTURAL_REGION_COLORS["AI Model"]
XLIM = (-2.1, 4.0)
YLIM = (-2.6, 3.3)
XLABEL = "Survival vs. Self-Expression Values"
YLABEL = "Traditional vs. Secular-Rational Values"

# Keep every mapped point, but label a small geographically varied reference set.
# The complete names and coordinates remain available in the country-score table
# and interactive map. Dense labels hid points in the camera-ready appendix.
REFERENCE_COUNTRIES = frozenset(
    {
        "Japan",
        "China",
        "India",
        "United States",
        "Germany",
        "Sweden",
        "Egypt",
        "Qatar",
        "Ghana",
        "South Africa",
        "Brazil",
        "Mexico",
        "Russian Federation",
        "Ireland",
        "Uruguay",
    }
)

# Short display names keep model labels legible on the map
DISPLAY_NAMES = {
    "wangrongsheng/llama3-70b-chinese-chat": "llama3-70b-chinese",
    "wangshenzhi/gemma2-27b-chinese-chat": "gemma2-27b-chinese",
}

# Hand-placed label offsets (dx, dy, ha) for the crowded top-right cluster
LABEL_OFFSETS = {
    "wangshenzhi/gemma2-27b-chinese-chat": (6, 2, "left"),
    "llama2-chinese:13b": (-6, 12, "right"),
    "dolphin-llama3:8b": (9, -8, "left"),
    "gemma2:27b": (9, 7, "left"),
    "llama3:70b": (-6, -11, "right"),
    "qwen2:7b": (-6, 4, "right"),
    "qwen2:7b [zh]": (-6, -12, "right"),
    "dolphin-mistral:7b": (-9, 14, "right"),
    "mistral:7b": (8, 0, "left"),
    "wangrongsheng/llama3-70b-chinese-chat": (9, 16, "left"),
    "dolphin-mixtral:8x7b": (9, -5, "left"),
}


def _display(llm: str) -> str:
    base, _, lang = llm.partition(" [")
    name = DISPLAY_NAMES.get(base, base)
    return f"{name} [{lang}" if lang else name


def _offsets(llm: str):
    return LABEL_OFFSETS.get(llm) or LABEL_OFFSETS.get(llm.split(" [")[0]) or (5, -9, "left")


def draw_midpoint(ax, midpoint) -> None:
    """Mark a hypothetical scale-midpoint profile, not an observed respondent."""
    if midpoint is None:
        return
    ax.scatter(
        *midpoint, marker="X", s=43, facecolor="white", edgecolor="0.15", linewidth=1.0, zorder=9
    )
    ax.annotate(
        "Scale midpoints",
        midpoint,
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color="0.2",
        zorder=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.4},
    )


def draw_countries(
    ax,
    countries: pd.DataFrame,
    label_alpha: float = 1.0,
    labelled_countries: frozenset[str] | set[str] | None = REFERENCE_COUNTRIES,
    label_size: float = 7,
    monochrome: bool = False,
) -> None:
    """Scatter mapped countries by cultural region and label the reference set."""
    for region, color in CULTURAL_REGION_COLORS.items():
        if region == "AI Model":
            continue
        subset = countries[countries["Cultural Region"] == region]
        if subset.empty:
            continue
        ax.scatter(
            subset["PC1_rescaled"],
            subset["PC2_rescaled"],
            s=14,
            color="0.6" if monochrome else color,
            label=region,
            zorder=3,
        )
        for _, row in subset.iterrows():
            if labelled_countries is not None and row["Country"] not in labelled_countries:
                continue
            ax.annotate(
                row["Country"],
                (row["PC1_rescaled"], row["PC2_rescaled"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=label_size,
                color="0.2",
                alpha=label_alpha,
                zorder=8 if monochrome else 4,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.3}
                if monochrome
                else None,
            )


def _draw_models(ax, ellipses: pd.DataFrame):
    for _, row in ellipses.iterrows():
        ax.add_patch(
            Ellipse(
                (row["PC1_rescaled"], row["PC2_rescaled"]),
                width=row["ellipse_width"],
                height=row["ellipse_height"],
                angle=row["angle_deg"],
                facecolor=AI_COLOR,
                alpha=0.12,
                edgecolor=AI_COLOR,
                linewidth=0.8,
                zorder=5,
            )
        )
    ax.scatter(
        ellipses["PC1_rescaled"],
        ellipses["PC2_rescaled"],
        s=42,
        marker="D",
        color=AI_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=6,
    )
    for _, row in ellipses.iterrows():
        dx, dy, ha = _offsets(row["llm"])
        ax.annotate(
            _display(row["llm"]),
            (row["PC1_rescaled"], row["PC2_rescaled"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=7,
            fontweight="bold",
            color=AI_COLOR,
            zorder=7,
        )


def _finish(ax, title):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(XLABEL, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=8)


def survey_year_range(yearly: pd.DataFrame, countries: pd.DataFrame) -> tuple[int, int]:
    """First and last retained survey year among the countries drawn on the map.

    `yearly` has one row per country code and survey year; `countries` is the
    mapped country-score table. Raises rather than guessing a range.
    """
    if not {"country_code", "year"} <= set(yearly.columns):
        raise ValueError(f"yearly needs country_code and year columns, got {list(yearly.columns)}")
    if "country_code" not in countries.columns:
        raise ValueError(f"countries needs a country_code column, got {list(countries.columns)}")
    plotted = yearly.loc[yearly["country_code"].isin(countries["country_code"]), "year"]
    if plotted.empty:
        raise ValueError("no country-year aggregate matches a mapped country; rerun the pipeline")
    years = plotted.to_numpy(dtype=float)
    if not np.isfinite(years).all() or not np.equal(years, np.round(years)).all():
        raise ValueError(f"survey years must be finite whole numbers, got {sorted(set(years))}")
    return int(years.min()), int(years.max())


def fig0_countries_only(countries: pd.DataFrame, years: tuple[int, int]):
    """The IW map redrawn from our own fitted IVS coordinates.

    Replaces the copyrighted official WVS map figure: same layout, but every
    point is computed from the microdata by this repo's pipeline. `years` is
    the retained survey-year range from `survey_year_range`.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_countries(ax, countries)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    first, last = years
    _finish(ax, f"Inglehart–Welzel Cultural Map, reconstructed from the IVS ({first}–{last})")  # noqa: RUF001
    return fig


def fig1_cultural_map(countries, ellipses, midpoint=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_countries(ax, countries)
    _draw_models(ax, ellipses)
    draw_midpoint(ax, midpoint)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            color=AI_COLOR,
            markersize=6,
            label="LLM (nominal 95% region)",
        )
    )
    labels.append("LLM (nominal 95% region)")
    ax.legend(handles, labels, fontsize=7, loc="lower right", framealpha=0.9)
    _finish(
        ax,
        "2024 cohort on the corrected IVS map (nominal 95% mean-position regions)",
    )
    return fig


def fig2_svm_regions(countries, ellipses, midpoint=None):
    clf = RegionClassifier().fit(countries)
    xx, yy = np.meshgrid(np.linspace(*XLIM, 400), np.linspace(*YLIM, 400))
    zz = clf.svm.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([CULTURAL_REGION_COLORS[r] for r in clf.regions])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(
        xx, yy, zz, levels=np.arange(len(clf.regions) + 1) - 0.5, cmap=cmap, alpha=0.18, zorder=1
    )
    draw_countries(ax, countries, label_alpha=0.75)
    _draw_models(ax, ellipses)
    draw_midpoint(ax, midpoint)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    _finish(ax, "SVM cultural-region decision boundaries with LLM positions")
    return fig


def main() -> int:
    countries = pd.read_csv("data/corrected_country_scores.csv")
    if not YEAR_AGGREGATES.is_file():
        raise FileNotFoundError(
            f"{YEAR_AGGREGATES} is missing; run scripts/instrument_sensitivity.py "
            "(part of make validate-2026) before drawing figures"
        )
    years = survey_year_range(pd.read_csv(YEAR_AGGREGATES), countries)
    ellipses = pd.read_csv("data/llm_ellipses.csv")
    references = pd.read_csv("data/validation_survey_reference.csv").set_index("reference")
    midpoint = references.loc["scale_midpoint", ["PC1_rescaled", "PC2_rescaled"]].to_numpy(
        dtype=float
    )
    os.makedirs("figures", exist_ok=True)

    for name, fig in [
        ("fig0_countries_only", fig0_countries_only(countries, years)),
        ("fig1_cultural_map", fig1_cultural_map(countries, ellipses, midpoint)),
        ("fig2_svm_regions", fig2_svm_regions(countries, ellipses, midpoint)),
    ]:
        fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
        fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/{name}.pdf and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
