"""Generate the 2026-cohort figures from the analysis artefacts.

Run from the repo root after analyze_2026.py:

    uv run python scripts/make_figures_2026.py

Writes vector PDFs (LaTeX) and PNGs (blog) to figures/:
    fig3_map_2026.{pdf,png}      2026 map, both arms: en diamonds, zh
                                 triangles, cluster-bootstrap ellipses,
                                 en->zh displacement arrows, colour by
                                 origin cohort
    fig4_language_forest.{pdf,png}  delta_m forest plot (PC1 and PC2
                                 components with CIs), grouped by cohort
    fig5_joint_2024_2026.{pdf,png}  both cohorts, one frozen instrument -
                                 descriptive only; distinct markers per
                                 cohort AND per bootstrap estimator (the
                                 caption in the paper carries the confound
                                 list)
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from app.llm_meta import cohort_2026
from scripts.make_figures import XLABEL, XLIM, YLABEL, YLIM, _draw_countries

# Okabe-Ito, CVD-validated
COHORT_COLORS = {"Chinese": "#d55e00", "Western": "#0072b2"}

# Hand-placed label offsets (dx, dy, ha) for the crowded centre cluster;
# labels anchor on the en-arm marker.
LABEL_OFFSETS_2026 = {
    "deepseek-v4-flash": (-10, -6, "right"),
    "deepseek-v4-flash:0731": (-2, -16, "right"),
    "deepseek-v4-pro": (-8, -5, "right"),
    "gemma4:31b": (2, 10, "left"),
    "glm-5.1": (-18, -13, "right"),
    "glm-5.2": (5, -3, "left"),
    "gpt-oss:120b": (8, 4, "left"),
    "gpt-oss:20b": (4, -15, "left"),
    "kimi-k2.6": (-18, -4, "right"),
    "kimi-k2.7-code": (-6, -16.5, "right"),
    "minimax-m2.7": (-2, -17, "right"),
    "minimax-m3": (-8, -10, "right"),
    "mistral-large-3:675b": (-8, 9, "right"),
    "nemotron-3-nano:30b": (-15, -13, "left"),
    "nemotron-3-super": (-28, 22, "left"),
    "nemotron-3-ultra": (8, 0, "left"),
    "qwen3.5:397b": (-18, 3, "right"),
}
AI_2024 = "#5e35b1"


def _finish(ax, title: str) -> None:
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(XLABEL, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=8)


def _base(llm: str) -> str:
    return llm.split(" [")[0]


def _short(llm: str) -> str:
    return (
        _base(llm)
        .replace("mistral-large-3:675b", "mistral-large-3")
        .replace("nemotron-3-", "nemotron-")
        .replace(":30b", "")
        .replace("deepseek-v4-", "ds-")
    )


def _draw_cell(ax, row, marker: str, color: str) -> None:
    ax.add_patch(
        Ellipse(
            (row["PC1_rescaled"], row["PC2_rescaled"]),
            width=row["ellipse_width"],
            height=row["ellipse_height"],
            angle=row["angle_deg"],
            facecolor=color,
            alpha=0.10,
            edgecolor=color,
            linewidth=0.7,
            zorder=5,
        )
    )
    ax.scatter(
        [row["PC1_rescaled"]],
        [row["PC2_rescaled"]],
        s=46,
        marker=marker,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        zorder=6,
    )


def fig3_map_2026(countries: pd.DataFrame, ellipses: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    _draw_countries(ax, countries, label_alpha=0.3)

    ell = ellipses.copy()
    ell["language"] = ["zh" if s.endswith(" [zh]") else "en" for s in ell["llm"]]
    ell["base"] = ell["llm"].map(_base)
    ell["cohort"] = ell["base"].map(cohort_2026)

    for base, pair in ell.groupby("base"):
        color = COHORT_COLORS[cohort_2026(base)]
        en = pair[pair["language"] == "en"]
        zh = pair[pair["language"] == "zh"]
        if len(en) and len(zh):
            ax.annotate(
                "",
                xy=(zh["PC1_rescaled"].iloc[0], zh["PC2_rescaled"].iloc[0]),
                xytext=(en["PC1_rescaled"].iloc[0], en["PC2_rescaled"].iloc[0]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "alpha": 0.65,
                    "linewidth": 1.0,
                    "shrinkA": 4,
                    "shrinkB": 4,
                },
                zorder=4,
            )
        for _, row in pair.iterrows():
            _draw_cell(ax, row, "D" if row["language"] == "en" else "^", color)
        if len(en):
            dx, dy, ha = LABEL_OFFSETS_2026.get(base, (5, 4, "left"))
            ax.annotate(
                _short(base),
                (en["PC1_rescaled"].iloc[0], en["PC2_rescaled"].iloc[0]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                fontsize=6,
                fontweight="bold",
                color=color,
                zorder=7,
            )

    handles = [
        Line2D([], [], marker="D", ls="", color="0.35", label="English administration"),
        Line2D([], [], marker="^", ls="", color="0.35", label="Chinese administration"),
        Line2D([], [], marker="s", ls="", color=COHORT_COLORS["Chinese"], label="Chinese-origin"),
        Line2D([], [], marker="s", ls="", color=COHORT_COLORS["Western"], label="Western-origin"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9)
    _finish(
        ax,
        "2026 frontier cohort: both administration arms "
        "(cluster-bootstrap 95% regions; arrows en→zh)",
    )
    return fig


def fig4_language_forest(lang_fx: pd.DataFrame):
    fx = lang_fx.sort_values(["cohort", "delta_pc2"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.36 * len(fx) + 1.8), sharey=True)
    y = range(len(fx))
    for ax, comp in zip(axes, ["delta_pc1", "delta_pc2"], strict=True):
        for i, row in fx.iterrows():
            color = COHORT_COLORS[row["cohort"]]
            ax.plot([row[f"{comp}_lo"], row[f"{comp}_hi"]], [i, i], color=color, linewidth=1.6)
            ax.plot([row[comp]], [i], marker="o", color=color, markersize=4.5)
        ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--")
        ax.set_xlabel(
            {"delta_pc1": "δ PC1′ (self-expression axis)", "delta_pc2": "δ PC2′ (secular axis)"}[  # noqa: RUF001
                comp
            ],
            fontsize=9,
        )
        ax.grid(True, axis="x", linewidth=0.3, alpha=0.4)
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([_short(m) for m in fx["llm"]], fontsize=8)
    handles = [
        Line2D([], [], marker="o", ls="", color=c, label=f"{k}-origin")
        for k, c in COHORT_COLORS.items()
    ]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle(
        "Language-of-administration effect per model (zh − en, replicate-paired 95% CI)",  # noqa: RUF001
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def fig5_joint(countries: pd.DataFrame, e2024: pd.DataFrame, e2026: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    _draw_countries(ax, countries, label_alpha=0.35)
    ax.scatter(
        e2024["PC1_rescaled"],
        e2024["PC2_rescaled"],
        s=40,
        marker="o",
        facecolor="none",
        edgecolor=AI_2024,
        linewidth=1.2,
        zorder=6,
        label="2024 cohort (item bootstrap, lower-bound CIs)",
    )
    lang = ["zh" if s.endswith(" [zh]") else "en" for s in e2026["llm"]]
    for marker, arm in [("D", "en"), ("^", "zh")]:
        sub = e2026[[la == arm for la in lang]]
        ax.scatter(
            sub["PC1_rescaled"],
            sub["PC2_rescaled"],
            s=42,
            marker=marker,
            color="#009e73",
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
            label=f"2026 cohort, {arm} arm (cluster bootstrap)",
        )
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    _finish(
        ax,
        "2024 and 2026 cohorts on the one frozen instrument (descriptive; "
        "see caption for confounds)",
    )
    return fig


def main() -> int:
    countries = pd.read_csv("data/corrected_country_scores.csv")
    e2026 = pd.read_csv("data/llm_ellipses_2026.csv")
    lang_fx = pd.read_csv("data/llm_language_effects_2026.csv")
    e2024 = pd.read_csv("data/llm_ellipses.csv")
    os.makedirs("figures", exist_ok=True)

    for name, fig in [
        ("fig3_map_2026", fig3_map_2026(countries, e2026)),
        ("fig4_language_forest", fig4_language_forest(lang_fx)),
        ("fig5_joint_2024_2026", fig5_joint(countries, e2024, e2026)),
    ]:
        fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
        fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/{name}.pdf and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
