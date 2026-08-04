"""Generate the paper/blog figures from the corrected pipeline outputs.

Run from the repo root after validate_projection.py and bootstrap_llms.py:

    uv run python scripts/make_figures.py

Writes vector PDFs (for LaTeX) and PNGs (for the blog) to figures/:
    fig0_countries_only.{pdf,png} the IW map redrawn from our fitted IVS data
    fig1_cultural_map.{pdf,png}   corrected IW map, 95% CI ellipse per model
    fig2_svm_regions.{pdf,png}    SVM decision regions + model positions
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from app.culture_map import CULTURAL_REGION_COLORS
from app.region_svm import RegionClassifier

AI_COLOR = CULTURAL_REGION_COLORS["AI Model"]
XLIM = (-2.1, 4.0)
YLIM = (-2.6, 3.3)
XLABEL = "Survival vs. Self-Expression Values"
YLABEL = "Traditional vs. Secular-Rational Values"

# Short display names keep model labels legible on the map
DISPLAY_NAMES = {
    "wangrongsheng/llama3-70b-chinese-chat": "llama3-70b-chinese",
    "wangshenzhi/gemma2-27b-chinese-chat": "gemma2-27b-chinese",
}

# Hand-placed label offsets (dx, dy, ha) for the crowded top-right cluster
LABEL_OFFSETS = {
    "wangshenzhi/gemma2-27b-chinese-chat": (6, 2, "left"),
    "llama2-chinese:13b": (-6, 3, "right"),
    "dolphin-llama3:8b": (6, 5, "left"),
    "gemma2:27b": (6, -8, "left"),
    "llama3:70b": (-6, -11, "right"),
    "qwen2:7b": (-6, 4, "right"),
    "qwen2:7b [zh]": (-6, -12, "right"),
    "dolphin-mistral:7b": (7, 7, "left"),
    "mistral:7b": (6, -12, "left"),
    "wangrongsheng/llama3-70b-chinese-chat": (-8, -13, "right"),
    "dolphin-mixtral:8x7b": (6, -3, "left"),
}


def _display(llm: str) -> str:
    base, _, lang = llm.partition(" [")
    name = DISPLAY_NAMES.get(base, base)
    return f"{name} [{lang}" if lang else name


def _offsets(llm: str):
    return LABEL_OFFSETS.get(llm) or LABEL_OFFSETS.get(llm.split(" [")[0]) or (5, -9, "left")


def _draw_countries(ax, countries: pd.DataFrame, label_alpha=1.0):
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
            color=color,
            label=region,
            zorder=3,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["Country"],
                (row["PC1_rescaled"], row["PC2_rescaled"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.5,
                color=color,
                alpha=label_alpha,
                zorder=4,
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


def fig0_countries_only(countries):
    """The IW map redrawn from our own fitted IVS coordinates.

    Replaces the copyrighted official WVS map figure: same layout, but every
    point is computed from the microdata by this repo's pipeline.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_countries(ax, countries)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    _finish(ax, "Inglehart–Welzel Cultural Map, reconstructed from the IVS (2005–2022)")
    return fig


def fig1_cultural_map(countries, ellipses):
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_countries(ax, countries)
    _draw_models(ax, ellipses)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="D", linestyle="", color=AI_COLOR, markersize=6, label="LLM (95% CI)")
    )
    labels.append("LLM (95% CI)")
    ax.legend(handles, labels, fontsize=7, loc="lower right", framealpha=0.9)
    _finish(
        ax,
        "Inglehart–Welzel Cultural Map with LLM positions "
        "(corrected projection, 95% bootstrap CIs)",
    )
    return fig


def fig2_svm_regions(countries, ellipses):
    clf = RegionClassifier().fit(countries)
    xx, yy = np.meshgrid(np.linspace(*XLIM, 400), np.linspace(*YLIM, 400))
    zz = clf.svm.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([CULTURAL_REGION_COLORS[r] for r in clf.regions])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(
        xx, yy, zz, levels=np.arange(len(clf.regions) + 1) - 0.5, cmap=cmap, alpha=0.18, zorder=1
    )
    _draw_countries(ax, countries, label_alpha=0.75)
    _draw_models(ax, ellipses)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    _finish(ax, "SVM cultural-region decision boundaries with LLM positions")
    return fig


def main() -> int:
    countries = pd.read_csv("data/corrected_country_scores.csv")
    ellipses = pd.read_csv("data/llm_ellipses.csv")
    os.makedirs("figures", exist_ok=True)

    for name, fig in [
        ("fig0_countries_only", fig0_countries_only(countries)),
        ("fig1_cultural_map", fig1_cultural_map(countries, ellipses)),
        ("fig2_svm_regions", fig2_svm_regions(countries, ellipses)),
    ]:
        fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
        fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/{name}.pdf and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
