"""Prompt-cue sensitivity: does the persona prefix drive the result?

A reviewer asked whether the "You are an average human being ..." framing
itself produces the survey-statistics estimation the traces show, and asked
for a neutral baseline or alternative prompts. Two checks, both descriptive
and post-confirmatory (analysis-plan ledger, camera-ready entry):

1. Variant sub-families within the retained corpus (zero new calls). Of
   the ten persona prefixes, six carry an averaging cue ("average" /
   "typical"), three are bare ("a human being" / "a person" / "an
   individual") and one is "a world citizen". Every cell is re-projected
   from each sub-family alone and the bare-vs-averaging displacement is
   reported with an item-bootstrap CI (both sides).
2. A no-persona baseline arm (data/collection_2026_nosys/, English, fifty
   repeats per item, collected by collect_cloud_2026.py --prompt-variant=
   nosys) compared with the full persona protocol of the same model. The
   arm is required: the script raises if it is absent rather than writing
   the cited outputs without their no-persona rows.

Inclusion rules. The persona arm is filtered by the main analysis rule
(>= MIN_PER_QUESTION parsed answers on every item, as in analyze_2026). The
no-persona control is held to the same rule as its primary result
(``rule == "min10"``); the relaxed >= 1-per-item variant is written alongside
as a labelled sensitivity (``rule == "min1"``), so both are artefacts. A
persona sub-family is estimable when it has at least one parsed answer on
every item (``rule == "min1"``: a sub-family with no answer on some item is
reported with its call counts but no position — not estimable, not imputed).

Estimators. The persona sub-families and the no-persona arm are single- or
few-prefix conditions, so their intervals are item bootstraps (cross-item
covariance omitted). For the nosys-vs-persona contrast the persona side uses
the primary cluster-bootstrap replicates written by analyze_2026
(data/llm_bootstrap_replicates_2026.csv, loaded rather than recomputed), so
prompt-variant uncertainty is carried on that side; the nosys side is the
item bootstrap. The estimator is stated per side in the output. Every
family/arm draws from its own independent child of one SeedSequence.

Run from the repo root (after analyze_2026.py):

    uv run python scripts/prompt_sensitivity_2026.py

Writes data/prompt_sensitivity_variant_family_2026.csv (one row per cell x
sub-family, and per cell x rule for the nosys arm) and
data/prompt_sensitivity_2026.csv (one row per cell x contrast x rule:
displacements).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.control_audit import coverage_tables
from app.culture_map import SURVEY_REFERENCE, CulturalMap
from app.llm_bootstrap import bootstrap_llm_positions, load_responses_2026
from app.study_design import MIN_PER_QUESTION, PERSONA_PREFIX_FAMILIES, PERSONA_PREFIX_FAMILY_OF

RAW_DIR = "data/collection_2026"
NOSYS_DIR = "data/collection_2026_nosys"
MODEL_PATH = "data/cultural_map_model.npz"
CLUSTER_REPLICATES = Path("data/llm_bootstrap_replicates_2026.csv")
OUT_FAMILY = Path("data/prompt_sensitivity_variant_family_2026.csv")
OUT_DELTA = Path("data/prompt_sensitivity_2026.csv")
# Persona-prefix sub-families plus the full ten-prefix protocol, in output order.
# Named for prefixes so it cannot be confused with app.llm_meta.FAMILIES.
PREFIX_FAMILIES: dict[str, tuple[int, ...]] = {
    **PERSONA_PREFIX_FAMILIES,
    "all_persona": tuple(sorted(PERSONA_PREFIX_FAMILY_OF)),
}
N_BOOT = 2000
SEED = 42
RULES = {"min10": MIN_PER_QUESTION, "min1": 1}
# One independent RNG child per family/arm; the index is fixed so a stream
# never depends on which arms happen to be present.
STREAMS = {
    "averaging": 0,
    "bare": 1,
    "world_citizen": 2,
    "all_persona": 3,
    "nosys/min10": 4,
    "nosys/min1": 5,
}
XY = ["PC1_rescaled", "PC2_rescaled"]


def child_seed(stream: str, root: int = SEED) -> int:
    """Independent child seed for one family/arm (numpy SeedSequence spawn)."""
    children = np.random.SeedSequence(root).spawn(len(STREAMS))
    return int(children[STREAMS[stream]].generate_state(1, dtype=np.uint32)[0])


def point(cm: CulturalMap, group: pd.DataFrame) -> np.ndarray:
    means = group.groupby("question")["value"].mean().reindex(cm.iv_qns)
    return cm.project(pd.DataFrame([means.to_numpy()], columns=cm.iv_qns))[XY].to_numpy()[0]


def family_rows(
    cm: CulturalMap,
    responses: pd.DataFrame,
    family: str,
    rule: str,
    stream: str,
    *,
    n_boot: int = N_BOOT,
    universe: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimates, item-bootstrap SDs and quadrant shares for one family/arm.

    A cell is estimable under ``rule`` when every item has at least
    ``RULES[rule]`` parsed answers; otherwise it is reported with its call
    counts but no position.
    """
    min_per_item = RULES[rule]
    per_cell_q = (
        responses.groupby(["llm", "question"]).size().unstack().reindex(columns=cm.iv_qns).fillna(0)
    )
    estimable = per_cell_q.index[(per_cell_q >= min_per_item).all(axis=1)]
    boot = (
        bootstrap_llm_positions(
            cm,
            responses[responses["llm"].isin(estimable)],
            n_boot=n_boot,
            seed=child_seed(stream),
        )
        if len(estimable)
        else pd.DataFrame(columns=["llm", "replicate", *XY])
    )
    rows = []
    for llm in sorted(universe if universe is not None else responses["llm"].unique()):
        g = responses[responses["llm"] == llm]
        per_q = g.groupby("question").size().reindex(cm.iv_qns).fillna(0)
        base = {
            "llm": llm,
            "family": family,
            "rule": rule,
            "estimator": "item",
            "n_parsed_trials": len(g),
            "min_per_question": int(per_q.min()),
            "required_min_per_item": min_per_item,
            "eligible": llm in estimable,
            "n_boot": n_boot,
            "rng_stream": stream,
            "interval_scope": "conditional_on_observed_prefixes_cross_item_covariance_omitted",
        }
        if llm not in estimable:
            rows.append({**base, "in_quadrant": None, "pc1": np.nan, "pc2": np.nan})
            continue
        xy = point(cm, g)
        reps = boot[boot["llm"] == llm][XY].to_numpy()
        in_quadrant = (reps[:, 0] > SURVEY_REFERENCE[0]) & (reps[:, 1] > SURVEY_REFERENCE[1])
        rows.append(
            {
                **base,
                "pc1": xy[0],
                "pc2": xy[1],
                "pc1_sd": reps[:, 0].std(ddof=1),
                "pc2_sd": reps[:, 1].std(ddof=1),
                "dist_reference": float(np.linalg.norm(xy - np.array(SURVEY_REFERENCE))),
                "in_quadrant": bool(xy[0] > SURVEY_REFERENCE[0] and xy[1] > SURVEY_REFERENCE[1]),
                "quadrant_share": float(in_quadrant.mean()),
            }
        )
    return pd.DataFrame(rows), boot


def displacement(
    boot_a: pd.DataFrame,
    boot_b: pd.DataFrame,
    pts_a: pd.DataFrame,
    pts_b: pd.DataFrame,
    contrast: str,
    rule: str,
    estimator_a: str,
    estimator_b: str,
) -> pd.DataFrame:
    """Plug-in displacement a - b per model, with independent-replicate CIs.

    Replicates on the two sides are independent draws (possibly from
    different estimators — stated per side), so pairing is arbitrary; the
    shorter side sets the number of paired replicates.
    """
    rows = []
    estimable_a = set(pts_a.loc[pts_a["pc1"].notna(), "llm"])
    estimable_b = set(pts_b.loc[pts_b["pc1"].notna(), "llm"])
    for llm in sorted(estimable_a & estimable_b):  # estimable on both sides
        a = boot_a[boot_a["llm"] == llm][XY].to_numpy()
        b = boot_b[boot_b["llm"] == llm][XY].to_numpy()
        n = min(len(a), len(b))
        if not n:
            raise ValueError(f"{llm}: eligible cell has no replicates on one side")
        pa = pts_a[pts_a["llm"] == llm].iloc[0]
        pb = pts_b[pts_b["llm"] == llm].iloc[0]
        d = np.array([pa["pc1"] - pb["pc1"], pa["pc2"] - pb["pc2"]])
        diffs = a[:n] - b[:n]
        norms = np.linalg.norm(diffs, axis=1)
        rows.append(
            {
                "llm": llm,
                "contrast": contrast,
                "rule": rule,
                "estimator_a": estimator_a,
                "estimator_b": estimator_b,
                "n_replicates": n,
                "delta_pc1": d[0],
                "delta_pc1_lo": np.quantile(diffs[:, 0], 0.025),
                "delta_pc1_hi": np.quantile(diffs[:, 0], 0.975),
                "delta_pc2": d[1],
                "delta_pc2_lo": np.quantile(diffs[:, 1], 0.025),
                "delta_pc2_hi": np.quantile(diffs[:, 1], 0.975),
                "norm_plugin": float(np.linalg.norm(d)),
                "norm_lo": np.quantile(norms, 0.025),
                "norm_hi": np.quantile(norms, 0.975),
                "dist_reference_a": pa["dist_reference"],
                "dist_reference_b": pb["dist_reference"],
                "farther_a": bool(pa["dist_reference"] > pb["dist_reference"]),
                "same_quadrant": bool(pa["in_quadrant"] == pb["in_quadrant"]),
            }
        )
    return pd.DataFrame(rows)


def load_cluster_replicates(path: Path = CLUSTER_REPLICATES) -> pd.DataFrame:
    """The primary cluster-bootstrap replicates written by analyze_2026."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/analyze_2026.py first")
    return pd.read_csv(path)


def require_nosys_collection(directory: str = NOSYS_DIR) -> Path:
    """Return the no-persona collection directory, raising if it holds no JSONL.

    The no-persona contrast is the reason the cited CSVs exist; writing them
    without those rows would publish a partial artefact.
    """
    path = Path(directory)
    if not (path.is_dir() and any(path.glob("*.jsonl"))):
        raise FileNotFoundError(
            f"no no-persona collection in {directory}; install the response archive "
            "described in docs/REPRODUCING.md"
        )
    return path


def contrast_summary(delta: pd.DataFrame) -> pd.DataFrame:
    """Matched-cell descriptive summaries with explicit eligibility and estimators."""
    rows = []
    for (contrast, rule), g in delta.groupby(["contrast", "rule"], sort=True):
        rows.append(
            {
                "contrast": contrast,
                "rule": rule,
                "n_matched_cells": len(g),
                "n_farther_from_reference": int(g["farther_a"].sum()),
                "median_dist_reference_a": g["dist_reference_a"].median(),
                "median_dist_reference_b": g["dist_reference_b"].median(),
                "median_norm_plugin": g["norm_plugin"].median(),
                "median_delta_pc1": g["delta_pc1"].median(),
                "median_delta_pc2": g["delta_pc2"].median(),
                "n_positive_pc1": int((g["delta_pc1"] > 0).sum()),
                "n_positive_pc2": int((g["delta_pc2"] > 0).sum()),
                "n_pc1_interval_above_zero": int((g["delta_pc1_lo"] > 0).sum()),
                "n_pc2_interval_above_zero": int((g["delta_pc2_lo"] > 0).sum()),
                "n_same_quadrant": int(g["same_quadrant"].sum()),
                "estimator_a": g["estimator_a"].iloc[0],
                "estimator_b": g["estimator_b"].iloc[0],
                "interpretation": "descriptive_retained_answer_positions_not_causal_reasoning",
            }
        )
    return pd.DataFrame(rows)


def main(output_dir: str = "data") -> int:
    require_nosys_collection(NOSYS_DIR)
    cm = CulturalMap(pd.DataFrame(), pd.DataFrame())
    cm.load_model(MODEL_PATH)
    responses = load_responses_2026(cm, RAW_DIR)
    # Same inclusion rule as analyze_2026: a cell needs >= MIN_PER_QUESTION parsed answers on
    # every item under the full protocol (drops nemotron-3-ultra [zh]).
    per_q = responses.groupby(["llm", "question"]).size().unstack()
    usable = per_q.index[(per_q.fillna(0) >= MIN_PER_QUESTION).all(axis=1)]
    responses = responses[responses["llm"].isin(usable)]
    print(f"{len(usable)} usable cells under the full protocol")

    tables, boots = [], {}
    for family, ids in PREFIX_FAMILIES.items():
        sub = responses[responses["system_prompt_id"].isin(ids)]
        rule = "min10" if family == "all_persona" else "min1"
        table, boot = family_rows(cm, sub, family, rule, stream=family, universe=list(usable))
        tables.append(table)
        boots[family] = boot
    pts = pd.concat(tables, ignore_index=True)
    deltas = [
        displacement(
            boots["bare"],
            boots["averaging"],
            pts[pts["family"] == "bare"],
            pts[pts["family"] == "averaging"],
            "bare-averaging",
            "min1",
            "item",
            "item",
        )
    ]

    base = load_responses_2026(cm, NOSYS_DIR)
    base = base[base["language"] == "en"]
    # Persona side: the primary cluster-bootstrap replicates (English cells).
    cluster = load_cluster_replicates()
    cluster = cluster[~cluster["llm"].str.endswith(" [zh]")]
    persona_pts = pts[pts["family"] == "all_persona"]
    # keep the persona-arm labels so the two arms join on llm
    for rule in RULES:
        table, boot = family_rows(cm, base, "nosys", rule, stream=f"nosys/{rule}")
        tables.append(table)
        deltas.append(
            displacement(
                boot,
                cluster,
                table,
                persona_pts,
                "nosys-all_persona",
                rule,
                "item",
                "cluster",
            )
        )
        print(f"nosys arm ({rule}): {int(table['pc1'].notna().sum())} eligible cells")
    pts = pd.concat(tables, ignore_index=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pts.to_csv(destination / OUT_FAMILY.name, index=False)
    delta = pd.concat(deltas, ignore_index=True)
    delta.to_csv(destination / OUT_DELTA.name, index=False)
    contrast_summary(delta).to_csv(destination / "prompt_control_summary_2026.csv", index=False)
    coverages = [
        coverage_tables(RAW_DIR, "persona", list(cm.iv_qns)),
        coverage_tables(NOSYS_DIR, "nosys", list(cm.iv_qns)),
    ]
    pd.concat([c for c, _ in coverages], ignore_index=True).to_csv(
        destination / "prompt_control_coverage_2026.csv", index=False
    )
    pd.concat([q for _, q in coverages], ignore_index=True).to_csv(
        destination / "prompt_control_item_coverage_2026.csv", index=False
    )

    pd.set_option("display.width", 200)
    summary = pts.groupby(["family", "rule"]).agg(
        cells=("llm", "size"),
        estimable=("pc1", "count"),
        in_quadrant=("in_quadrant", "sum"),
        min_quadrant_share=("quadrant_share", "min"),
        median_dist=("dist_reference", "median"),
        min_pc1=("pc1", "min"),
        min_pc2=("pc2", "min"),
    )
    print(summary.to_string())
    for (contrast, rule), g in delta.groupby(["contrast", "rule"]):
        print(
            f"\n{contrast} [{rule}; {g['estimator_a'].iloc[0]} vs {g['estimator_b'].iloc[0]}]: "
            f"n={len(g)}, farther={int(g['farther_a'].sum())}/{len(g)}, "
            f"median dist a={g['dist_reference_a'].median():.3f} b={g['dist_reference_b'].median():.3f}, "
            f"median ||delta||={g['norm_plugin'].median():.3f}, max={g['norm_plugin'].max():.3f}, "
            f"same quadrant={int(g['same_quadrant'].sum())}/{len(g)}, "
            f"delta_pc1 median={g['delta_pc1'].median():+.3f}, "
            f"delta_pc2 median={g['delta_pc2'].median():+.3f}, "
            f"pc2 CI excl. 0={int(((g['delta_pc2_lo'] > 0) | (g['delta_pc2_hi'] < 0)).sum())}"
        )
    print(f"\nwrote prompt sensitivity, coverage and summary artefacts to {destination}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data")
    sys.exit(main(parser.parse_args().output_dir))
