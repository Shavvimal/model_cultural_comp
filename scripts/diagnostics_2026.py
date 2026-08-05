"""Robustness and measurement diagnostics for the 2026 collection.

Eight diagnostic families feeding the paper's robustness appendix, each a
typed function returning a DataFrame, written to data/diag_2026_<name>.csv:

  1. variance components (model / language / variant / repeat), in map units
  2. prompt-variant ICC(1) per model x language x item
  3. empirical item keying and an acquiescence/directional-bias table
  4. refusal rates, with an en-vs-zh comparison on the sensitive items
  5. Manski-style worst-case bounds for unparsed calls
  6. within-family descriptives for the protocol-matched model families
  7. item profiles vs the human means, leave-one-item-out displacements,
     and per-item zh-en deltas
  8. sensitivity of cell inclusion to MIN_PER_QUESTION, and headline
     robustness to neutralising the two most-refused items (F118, F120)

Run:  uv run python scripts/diagnostics_2026.py

Sources and conventions
-----------------------
* Raw corpus: data/collection_2026/*.jsonl, both arms. system_prompt_id and
  repeat are serialised as both str and int across resumed runs; they are
  normalised with astype(int) before the keep-last dedup on
  (llm, language, question, system_prompt_id, repeat).
* Values are transformed with app.llm_bootstrap._to_value, the same
  Y002/Y003 recodes as the headline pipeline. (load_responses_2026 is not
  reused directly because its output drops the repeat column.)
* Human item means are the frozen model's stored standardisation means
  (cm.ppca.means after load_model, the "means" array in
  data/cultural_map_model.npz): unweighted nanmeans over the post-2005,
  sentinel-recoded, >=6-items-answered IVS training rows, in IV_QNS order.
  The vector of these means projects to exactly (0.38, -0.01) - the pooled
  human respondent mean - so it doubles as the human-grand-mean base point.
* The variance decomposition in (1) is a balanced nested approximation to
  the full crossed model x language x variant x repeat G-study: each level
  is estimated as the mean (over higher-level units) of the ddof=1 sample
  variance of the next level's means, not by solving the crossed
  expected-mean-square equations. With near-balanced cells (10 variants x
  5 repeats) the approximation is close; it is labelled as such.
* Nothing here is stochastic - every quantity is a deterministic function
  of the corpus and the frozen model.

The zh arm may still be collecting: incomplete cells (an item with no
parsed response) are skipped where a complete 10-item mean vector is
required, and reported rather than imputed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qc_2026 import classify_failure

from app.culture_map import ITEM_VALID_RANGES, IV_QNS, CulturalMap
from app.llm_bootstrap import _to_value
from app.llm_meta import cohort_2026

RAW_DIR = Path("data/collection_2026")
MODEL_PATH = "data/cultural_map_model.npz"
HUMAN_MEAN = (0.38, -0.01)  # pooled human respondent mean, map units
SENSITIVE_QNS = ["F118", "F120", "F063", "G006", "E025"]
THRESHOLDS = (5, 10, 25)
CELL = ["llm", "language"]
PSEUDO_KEY = ["llm", "language", "system_prompt_id", "repeat"]
AXES = ("PC1_rescaled", "PC2_rescaled")

# Protocol-matched families for the within-family descriptives (6).
FAMILIES = {
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-flash:0731", "deepseek-v4-pro"],
    "gemma": ["gemma4:31b"],
    "glm": ["glm-5.1", "glm-5.2"],
    "gpt-oss": ["gpt-oss:20b", "gpt-oss:120b"],
    "kimi": ["kimi-k2.6", "kimi-k2.7-code"],
    "minimax": ["minimax-m2.7", "minimax-m3"],
    "mistral": ["mistral-large-3:675b"],
    "nemotron": ["nemotron-3-nano:30b", "nemotron-3-super", "nemotron-3-ultra"],
    "qwen": ["qwen3.5:397b"],
}


def _cohort(llm: str) -> str:
    try:
        return cohort_2026(llm)
    except ValueError:
        return "unknown"


def load_raw() -> pd.DataFrame:
    """Load every 2026 record with types normalised and resumed runs deduped."""
    paths = sorted(RAW_DIR.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files in {RAW_DIR}")
    records = []
    for path in paths:
        with path.open() as f:
            records.extend(json.loads(line) for line in f)
    df = pd.DataFrame(records)
    df["language"] = df.get("language", pd.Series([None] * len(df))).fillna("en")
    # Resumed runs serialised these as str; originals as int. Normalise
    # before the dedup, or mixed-type duplicates survive the key.
    df["system_prompt_id"] = df["system_prompt_id"].astype(int)
    df["repeat"] = df["repeat"].astype(int)
    df["raw_content"] = df["raw_content"].fillna("")
    df = df.drop_duplicates(
        subset=["llm", "language", "question", "system_prompt_id", "repeat"], keep="last"
    )
    return df.reset_index(drop=True)


def parsed_values(cm: CulturalMap, raw: pd.DataFrame) -> pd.DataFrame:
    """Parsed rows only, with the headline pipeline's value transform applied."""
    ok = raw[raw["error"].isna()].copy()
    ok["value"] = [_to_value(cm, q, r) for q, r in zip(ok["question"], ok["parsed"], strict=True)]
    return ok[["llm", "language", "question", "system_prompt_id", "repeat", "value"]]


def human_item_means(cm: CulturalMap) -> pd.Series:
    """The frozen model's standardisation means, indexed by item (see module docstring)."""
    means = np.asarray(cm.ppca.means, dtype=float)
    if means.shape[0] != len(IV_QNS):
        raise RuntimeError(f"expected {len(IV_QNS)} stored item means, got {means.shape[0]}")
    return pd.Series(means, index=IV_QNS)


def cell_item_means(parsed: pd.DataFrame) -> pd.DataFrame:
    """Per (llm, language) item means; NaN where an item never parsed in a cell."""
    means = parsed.pivot_table(index=CELL, columns="question", values="value", aggfunc="mean")
    return means.reindex(columns=IV_QNS)


def _project_cells(cm: CulturalMap, means: pd.DataFrame) -> pd.DataFrame:
    """Project complete cell mean vectors; returns a (llm, language)-indexed frame."""
    complete = means.dropna()
    coords = cm.project(complete)
    coords.index = complete.index
    return coords[list(AXES)]


# --------------------------------------------------------------------------
# 1. Variance components
# --------------------------------------------------------------------------


def build_pseudo_respondents(
    cm: CulturalMap, parsed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One pseudo-respondent per (llm, language, variant, repeat), projected.

    Missing items (refusals) are filled from the (llm, language, variant)
    item mean, else the (llm, language) cell item mean; items never parsed
    in a cell are unfillable and drop the pseudo-respondent. Returns
    (pseudo_respondents_with_coordinates, fill_counts_per_cell).
    """
    wide = parsed.pivot_table(index=PSEUDO_KEY, columns="question", values="value", aggfunc="mean")
    wide = wide.reindex(columns=IV_QNS)
    long = wide.reset_index().melt(
        id_vars=PSEUDO_KEY, value_vars=IV_QNS, var_name="question", value_name="value"
    )

    variant_means = (
        parsed.groupby([*CELL, "system_prompt_id", "question"])["value"]
        .mean()
        .rename("variant_mean")
        .reset_index()
    )
    cell_means = (
        parsed.groupby([*CELL, "question"])["value"].mean().rename("cell_mean").reset_index()
    )
    long = long.merge(variant_means, how="left", on=[*CELL, "system_prompt_id", "question"])
    long = long.merge(cell_means, how="left", on=[*CELL, "question"])
    long["fill_source"] = np.select(
        [long["value"].notna(), long["variant_mean"].notna(), long["cell_mean"].notna()],
        ["observed", "variant_mean", "cell_mean"],
        default="unfillable",
    )
    long["filled"] = long["value"].fillna(long["variant_mean"]).fillna(long["cell_mean"])

    fills = (
        long.groupby([*CELL, "fill_source"])
        .size()
        .unstack("fill_source", fill_value=0)
        .reindex(columns=["observed", "variant_mean", "cell_mean", "unfillable"], fill_value=0)
        .reset_index()
    )

    pseudo = long.pivot_table(index=PSEUDO_KEY, columns="question", values="filled", aggfunc="mean")
    pseudo = pseudo.reindex(columns=IV_QNS)
    complete = pseudo.dropna()
    n_dropped = len(pseudo) - len(complete)
    if n_dropped:
        print(
            f"pseudo-respondents: {len(complete)} built, {n_dropped} dropped "
            "(an item never parsed anywhere in their cell)"
        )
    coords = cm.project(complete)
    out = complete.reset_index()[PSEUDO_KEY].join(coords[list(AXES)])
    return out, fills


def _orthogonal_ss(pseudo: pd.DataFrame, axis: str) -> dict[str, float]:
    """Nested sums-of-squares partition of the total SS on ``axis``.

    Decomposes each pseudo-respondent's deviation from the grand mean into
    four orthogonal terms,

        y - y... = (model - grand) + (cell - model)
                   + (variant - cell) + (y - variant),

    whose squared sums add to the total SS exactly. Unlike the per-level
    mean squares this weights every level by its own degrees of freedom, so
    a two-level factor (language, 1 df per model) is not credited on the
    same footing as a 17-level one (model, 16 df) — which is why the
    mean-square convention inflates the language term roughly twofold.
    """
    d = pseudo.reset_index()
    m_variant = d.groupby([*CELL, "system_prompt_id"])[axis].transform("mean")
    m_cell = d.groupby(CELL)[axis].transform("mean")
    m_model = d.groupby("llm")[axis].transform("mean")
    grand = d[axis].mean()
    return {
        "model": float(((m_model - grand) ** 2).sum()),
        "language_within_model": float(((m_cell - m_model) ** 2).sum()),
        "variant_within_cell": float(((m_variant - m_cell) ** 2).sum()),
        "repeat_within_variant": float(((d[axis] - m_variant) ** 2).sum()),
    }


def variance_components(pseudo: pd.DataFrame) -> pd.DataFrame:
    """Nested variance decomposition per axis, in map units.

    Balanced nested approximation to the crossed G-study (module docstring):
    each component is the mean, over the units one level up, of the ddof=1
    variance of the next level's means. Language-within-model averages only
    models observed in both arms.

    Reported under both conventions. ``pct_of_total`` is the mean-square
    share (each level's per-unit variance as a share of the four summed);
    ``pct_of_total_ss`` is the orthogonal sums-of-squares share, which
    partitions the total variance exactly and does not over-credit the
    two-level language factor. The two disagree materially for language, so
    any headline comparing model identity against language must name the
    convention it uses.
    """
    rows = []
    for axis in AXES:
        ss = _orthogonal_ss(pseudo, axis)
        ss_total = float(sum(ss.values()))
        cell = pseudo.groupby(CELL)[axis].mean()
        model = cell.groupby("llm").mean()
        var_model = float(model.var(ddof=1)) if len(model) >= 2 else np.nan

        lang_vars = [float(g.var(ddof=1)) for _, g in cell.groupby("llm") if g.index.nunique() >= 2]
        var_lang = float(np.mean(lang_vars)) if lang_vars else np.nan

        variant = pseudo.groupby([*CELL, "system_prompt_id"])[axis].mean()
        variant_vars = [float(g.var(ddof=1)) for _, g in variant.groupby(CELL) if len(g) >= 2]
        var_variant = float(np.mean(variant_vars)) if variant_vars else np.nan

        repeat_vars = [
            float(g[axis].var(ddof=1))
            for _, g in pseudo.groupby([*CELL, "system_prompt_id"])
            if len(g) >= 2
        ]
        var_repeat = float(np.mean(repeat_vars)) if repeat_vars else np.nan

        components = [
            ("model", var_model, len(model)),
            ("language_within_model", var_lang, len(lang_vars)),
            ("variant_within_cell", var_variant, len(variant_vars)),
            ("repeat_within_variant", var_repeat, len(repeat_vars)),
        ]
        total = float(np.nansum([v for _, v, _ in components]))
        for name, var, n_units in components:
            rows.append(
                {
                    "axis": axis,
                    "component": name,
                    "variance": var,
                    "sd_map_units": float(np.sqrt(var)) if pd.notna(var) else np.nan,
                    "pct_of_total": 100.0 * var / total if pd.notna(var) and total > 0 else np.nan,
                    "sum_of_squares": ss[name],
                    "pct_of_total_ss": 100.0 * ss[name] / ss_total if ss_total > 0 else np.nan,
                    "n_units": n_units,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Prompt-variant ICC(1)
# --------------------------------------------------------------------------


def prompt_variant_icc(parsed: pd.DataFrame) -> pd.DataFrame:
    """One-way ICC(1) over prompt-variant groups per (llm, language, question).

    ICC(1) = (MSB - MSW) / (MSB + (k - 1) MSW), k the harmonic mean group
    size, clipped at 0. NaN where fewer than two variant groups or no
    within-group degrees of freedom.
    """
    rows = []
    for (llm, lang, qn), g in parsed.groupby([*CELL, "question"]):
        groups = [grp["value"].to_numpy() for _, grp in g.groupby("system_prompt_id")]
        k = len(groups)
        n_total = int(sum(len(x) for x in groups))
        row = {
            "llm": llm,
            "language": lang,
            "question": qn,
            "n_variants": k,
            "n_obs": n_total,
        }
        if k < 2 or n_total <= k:
            rows.append({**row, "k_harmonic": np.nan, "msb": np.nan, "msw": np.nan, "icc1": np.nan})
            continue
        grand = float(g["value"].mean())
        ssb = float(sum(len(x) * (x.mean() - grand) ** 2 for x in groups))
        ssw = float(sum(((x - x.mean()) ** 2).sum() for x in groups))
        msb = ssb / (k - 1)
        msw = ssw / (n_total - k)
        k_h = k / sum(1.0 / len(x) for x in groups)
        if msw == 0.0:
            icc = 1.0 if msb > 0.0 else 0.0
        else:
            icc = max(0.0, (msb - msw) / (msb + (k_h - 1.0) * msw))
        rows.append({**row, "k_harmonic": k_h, "msb": msb, "msw": msw, "icc1": icc})
    return pd.DataFrame(rows).sort_values(["language", "llm", "question"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Keying and acquiescence
# --------------------------------------------------------------------------


def item_keying(cm: CulturalMap) -> pd.DataFrame:
    """Empirical per-item keying from the frozen projection.

    Perturbs the human-grand-mean vector by +1 on each item and records the
    signed change in each rescaled axis. survival_pole / traditional_pole
    name which end of the item's scale pushes toward survival (low PC1) and
    traditional (low PC2) values respectively.
    """
    base = human_item_means(cm).to_numpy()
    mat = np.tile(base, (len(IV_QNS) + 1, 1))
    for j in range(len(IV_QNS)):
        mat[j + 1, j] += 1.0
    proj = cm.project(pd.DataFrame(mat, columns=IV_QNS))
    deltas = proj[list(AXES)].to_numpy()[1:] - proj[list(AXES)].to_numpy()[0]

    rows = []
    for j, qn in enumerate(IV_QNS):
        lo, hi = ITEM_VALID_RANGES[qn]
        d_pc1, d_pc2 = float(deltas[j, 0]), float(deltas[j, 1])
        rows.append(
            {
                "question": qn,
                "valid_lo": lo,
                "valid_hi": hi,
                "midpoint": (lo + hi) / 2.0,
                "half_range": (hi - lo) / 2.0,
                "d_pc1_per_unit": d_pc1,
                "d_pc2_per_unit": d_pc2,
                "survival_pole": "low" if d_pc1 > 0 else "high",
                "traditional_pole": "low" if d_pc2 > 0 else "high",
            }
        )
    return pd.DataFrame(rows)


def keying_balance(parsed: pd.DataFrame, keying: pd.DataFrame) -> pd.DataFrame:
    """Mean signed deviation from scale midpoints, in half-range units.

    Per cell: overall, and split by each axis's keying direction. Matched
    deviations across opposite-keyed item groups indicate a directional
    response bias (acquiescence) rather than a substantive position.
    """
    means = cell_item_means(parsed)
    key = keying.set_index("question")
    dev = (means - key["midpoint"]) / key["half_range"]

    pc1_pos = [q for q in IV_QNS if key.loc[q, "d_pc1_per_unit"] > 0]
    pc2_pos = [q for q in IV_QNS if key.loc[q, "d_pc2_per_unit"] > 0]
    splits = {
        "mean_dev_all": IV_QNS,
        "mean_dev_pc1_pos_keyed": pc1_pos,
        "mean_dev_pc1_neg_keyed": [q for q in IV_QNS if q not in pc1_pos],
        "mean_dev_pc2_pos_keyed": pc2_pos,
        "mean_dev_pc2_neg_keyed": [q for q in IV_QNS if q not in pc2_pos],
    }
    rows = []
    for (llm, lang), row in dev.iterrows():
        rec = {
            "llm": llm,
            "language": lang,
            "cohort": _cohort(llm),
            "n_items": int(row.notna().sum()),
        }
        for name, items in splits.items():
            rec[name] = float(row[items].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["language", "llm"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Refusals
# --------------------------------------------------------------------------


def refusal_rates(raw: pd.DataFrame) -> pd.DataFrame:
    """Attempted / parsed / refusal-classified counts per (llm, language, question)."""
    rows = []
    for (llm, lang, qn), g in raw.groupby([*CELL, "question"]):
        failed = g[g["error"].notna()]
        n_refusal = int(
            sum(
                classify_failure(r, e) == "refusal"
                for r, e in zip(failed["raw_content"], failed["error"], strict=True)
            )
        )
        attempted = len(g)
        parsed = attempted - len(failed)
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "question": qn,
                "attempted": attempted,
                "parsed": parsed,
                "n_failed": len(failed),
                "n_refusal": n_refusal,
                "n_other_failure": len(failed) - n_refusal,
                "refusal_rate": n_refusal / attempted,
            }
        )
    return pd.DataFrame(rows).sort_values(["language", "llm", "question"]).reset_index(drop=True)


def refusal_sensitive(rates: pd.DataFrame) -> pd.DataFrame:
    """Per-model en-vs-zh refusal counts on the sensitive items."""
    sub = rates[rates["question"].isin(SENSITIVE_QNS)]
    piv = sub.pivot_table(
        index=["llm", "question"],
        columns="language",
        values=["n_refusal", "attempted"],
        aggfunc="sum",
    )
    out = pd.DataFrame(index=piv.index)
    for metric in ("attempted", "n_refusal"):
        for lang in ("en", "zh"):
            col = (metric, lang)
            out[f"{metric}_{lang}"] = piv[col] if col in piv.columns else np.nan
    out["refusals_zh_minus_en"] = out["n_refusal_zh"] - out["n_refusal_en"]
    out = out.reset_index()
    out.insert(1, "cohort", out["llm"].map(_cohort))
    return out.sort_values(["llm", "question"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. Manski worst-case bounds
# --------------------------------------------------------------------------


def manski_bounds(
    cm: CulturalMap, raw: pd.DataFrame, parsed: pd.DataFrame, keying: pd.DataFrame
) -> pd.DataFrame:
    """Worst-case positions per cell if every unparsed call had answered at
    the traditional/survival extreme.

    Per axis, each item mean is replaced by (parsed_sum + n_failed * extreme)
    / attempted, with the extreme chosen from the empirical keying to push
    that axis toward survival (PC1) or traditional (PC2). Cells with zero
    failures reproduce the observed position exactly. Cells where an item
    has no parsed responses (zh arm mid-collection) carry NaN coordinates.
    """
    attempted = raw.groupby([*CELL, "question"]).size()
    n_parsed = parsed.groupby([*CELL, "question"]).size()
    val_sum = parsed.groupby([*CELL, "question"])["value"].sum()
    key = keying.set_index("question")

    rows, vectors = [], []
    for llm, lang in attempted.index.droplevel("question").unique():
        obs, worst1, worst2 = {}, {}, {}
        computable = True
        n_att = n_par = 0
        for qn in IV_QNS:
            att = int(attempted.get((llm, lang, qn), 0))
            par = int(n_parsed.get((llm, lang, qn), 0))
            n_att += att
            n_par += par
            if att == 0 or par == 0:
                computable = False
                continue
            total = float(val_sum.loc[(llm, lang, qn)])
            obs[qn] = total / par
            lo, hi = key.loc[qn, "valid_lo"], key.loc[qn, "valid_hi"]
            ext1 = lo if key.loc[qn, "d_pc1_per_unit"] > 0 else hi
            ext2 = lo if key.loc[qn, "d_pc2_per_unit"] > 0 else hi
            worst1[qn] = (total + (att - par) * ext1) / att
            worst2[qn] = (total + (att - par) * ext2) / att
        rows.append(
            {
                "llm": llm,
                "language": lang,
                "attempted": n_att,
                "parsed": n_par,
                "n_failed": n_att - n_par,
                "computable": computable,
            }
        )
        vectors.append((obs, worst1, worst2) if computable else None)

    out = pd.DataFrame(rows)
    coord_cols = [
        "obs_pc1",
        "obs_pc2",
        "worst_pc1_pc1",
        "worst_pc1_pc2",
        "worst_pc2_pc1",
        "worst_pc2_pc2",
    ]
    for col in coord_cols:
        out[col] = np.nan
    for i, vecs in enumerate(vectors):
        if vecs is None:
            continue
        mat = pd.DataFrame([[v[qn] for qn in IV_QNS] for v in vecs], columns=IV_QNS)
        proj = cm.project(mat)[list(AXES)].to_numpy()
        out.loc[i, coord_cols] = [
            proj[0, 0],
            proj[0, 1],
            proj[1, 0],
            proj[1, 1],
            proj[2, 0],
            proj[2, 1],
        ]
    # Nullable booleans: NA where the cell has an item with no parsed rows.
    out["pc1_side_vs_human_preserved"] = pd.array(
        np.sign(out["obs_pc1"] - HUMAN_MEAN[0]) == np.sign(out["worst_pc1_pc1"] - HUMAN_MEAN[0]),
        dtype="boolean",
    )
    out["pc2_side_vs_human_preserved"] = pd.array(
        np.sign(out["obs_pc2"] - HUMAN_MEAN[1]) == np.sign(out["worst_pc2_pc2"] - HUMAN_MEAN[1]),
        dtype="boolean",
    )
    side_cols = ["pc1_side_vs_human_preserved", "pc2_side_vs_human_preserved"]
    out.loc[~out["computable"], side_cols] = pd.NA
    return out.sort_values(["language", "llm"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. Within-family descriptives
# --------------------------------------------------------------------------


def within_family(cm: CulturalMap, parsed: pd.DataFrame) -> pd.DataFrame:
    """Cell-mean positions per family member per language: distance to the
    within-language family mean and the zh-en displacement. Description only."""
    positions = _project_cells(cm, cell_item_means(parsed))
    rows = []
    for family, members in FAMILIES.items():
        for lang in ("en", "zh"):
            present = [m for m in members if (m, lang) in positions.index]
            if not present:
                continue
            fam = positions.loc[[(m, lang) for m in present]].mean()
            for m in present:
                pos = positions.loc[(m, lang)]
                rec = {
                    "family": family,
                    "llm": m,
                    "language": lang,
                    "n_members_in_arm": len(present),
                    "pc1": float(pos["PC1_rescaled"]),
                    "pc2": float(pos["PC2_rescaled"]),
                    "family_mean_pc1": float(fam["PC1_rescaled"]),
                    "family_mean_pc2": float(fam["PC2_rescaled"]),
                    "dist_to_family_mean": float(
                        np.hypot(
                            pos["PC1_rescaled"] - fam["PC1_rescaled"],
                            pos["PC2_rescaled"] - fam["PC2_rescaled"],
                        )
                    ),
                    "zh_minus_en_pc1": np.nan,
                    "zh_minus_en_pc2": np.nan,
                    "zh_en_displacement": np.nan,
                }
                if lang == "zh" and (m, "en") in positions.index:
                    en = positions.loc[(m, "en")]
                    d1 = float(pos["PC1_rescaled"] - en["PC1_rescaled"])
                    d2 = float(pos["PC2_rescaled"] - en["PC2_rescaled"])
                    rec.update(
                        zh_minus_en_pc1=d1,
                        zh_minus_en_pc2=d2,
                        zh_en_displacement=float(np.hypot(d1, d2)),
                    )
                rows.append(rec)
    return pd.DataFrame(rows).sort_values(["family", "llm", "language"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 7. Item profiles, leave-one-item-out, zh-en item deltas
# --------------------------------------------------------------------------


def item_profiles(cm: CulturalMap, parsed: pd.DataFrame) -> pd.DataFrame:
    """Per-cell item means alongside the human item means (long format)."""
    human = human_item_means(cm)
    means = cell_item_means(parsed)
    long = (
        means.reset_index()
        .melt(id_vars=CELL, value_vars=IV_QNS, var_name="question", value_name="cell_mean")
        .dropna(subset=["cell_mean"])
    )
    long["human_mean"] = long["question"].map(human)
    long["deviation"] = long["cell_mean"] - long["human_mean"]
    return long.sort_values(["language", "llm", "question"]).reset_index(drop=True)


def leave_one_item_out(cm: CulturalMap, parsed: pd.DataFrame) -> pd.DataFrame:
    """Displacement when each item's cell mean is replaced by the human mean.

    Large displacements mark the items that carry a cell's placement.
    Complete cells only.
    """
    human = human_item_means(cm)
    means = cell_item_means(parsed).dropna()
    observed = _project_cells(cm, means)
    rows = []
    for (llm, lang), item_means in means.iterrows():
        mat = pd.DataFrame([item_means] * len(IV_QNS)).reset_index(drop=True)
        for j, qn in enumerate(IV_QNS):
            mat.loc[j, qn] = human[qn]
        proj = cm.project(mat)[list(AXES)].to_numpy()
        obs = observed.loc[(llm, lang)].to_numpy()
        for j, qn in enumerate(IV_QNS):
            rows.append(
                {
                    "llm": llm,
                    "language": lang,
                    "question": qn,
                    "obs_pc1": float(obs[0]),
                    "obs_pc2": float(obs[1]),
                    "loio_pc1": float(proj[j, 0]),
                    "loio_pc2": float(proj[j, 1]),
                    "displacement": float(np.hypot(*(proj[j] - obs))),
                }
            )
    return pd.DataFrame(rows).sort_values(["language", "llm", "question"]).reset_index(drop=True)


def item_language_deltas(parsed: pd.DataFrame) -> pd.DataFrame:
    """Per-item zh - en mean difference per model, where both arms have the item."""
    means = cell_item_means(parsed)
    rows = []
    for llm in means.index.get_level_values("llm").unique():
        if (llm, "en") not in means.index or (llm, "zh") not in means.index:
            continue
        en, zh = means.loc[(llm, "en")], means.loc[(llm, "zh")]
        for qn in IV_QNS:
            if pd.isna(en[qn]) or pd.isna(zh[qn]):
                continue
            rows.append(
                {
                    "llm": llm,
                    "question": qn,
                    "en_mean": float(en[qn]),
                    "zh_mean": float(zh[qn]),
                    "zh_minus_en": float(zh[qn] - en[qn]),
                }
            )
    return pd.DataFrame(rows).sort_values(["llm", "question"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 8. Sensitivity
# --------------------------------------------------------------------------


def sensitivity_thresholds(parsed: pd.DataFrame) -> pd.DataFrame:
    """Which cells' inclusion flips across MIN_PER_QUESTION in {5, 10, 25}."""
    counts = (
        parsed.groupby([*CELL, "question"])
        .size()
        .unstack("question")
        .reindex(columns=IV_QNS)
        .fillna(0)
        .astype(int)
    )
    out = counts.min(axis=1).rename("min_parsed_per_item").reset_index()
    for t in THRESHOLDS:
        out[f"include_at_{t}"] = out["min_parsed_per_item"] >= t
    flags = out[[f"include_at_{t}" for t in THRESHOLDS]]
    out["flips"] = flags.nunique(axis=1) > 1
    return out.sort_values(["language", "llm"]).reset_index(drop=True)


def sensitivity_neutralised(cm: CulturalMap, parsed: pd.DataFrame) -> pd.DataFrame:
    """Headline robustness to the two most-refused items, neutralised at
    human means: recompute each complete cell's position with F118 and F120
    set to the human item means, and report the displacement."""
    human = human_item_means(cm)
    means = cell_item_means(parsed).dropna()
    observed = _project_cells(cm, means)
    neutral = means.copy()
    neutral["F118"] = human["F118"]
    neutral["F120"] = human["F120"]
    shifted = _project_cells(cm, neutral)
    out = observed.rename(columns={"PC1_rescaled": "obs_pc1", "PC2_rescaled": "obs_pc2"}).join(
        shifted.rename(columns={"PC1_rescaled": "neut_pc1", "PC2_rescaled": "neut_pc2"})
    )
    out["displacement"] = np.hypot(
        out["neut_pc1"] - out["obs_pc1"], out["neut_pc2"] - out["obs_pc2"]
    )
    return out.reset_index().sort_values(["language", "llm"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
    cm.load_model(MODEL_PATH)

    raw = load_raw()
    parsed = parsed_values(cm, raw)
    incomplete = cell_item_means(parsed).isna().any(axis=1)
    skipped = [f"{m} [{lang}]" for (m, lang) in incomplete[incomplete].index]

    pseudo, fills = build_pseudo_respondents(cm, parsed)
    keying = item_keying(cm)
    rates = refusal_rates(raw)
    icc = prompt_variant_icc(parsed)

    artefacts = {
        "variance_components": variance_components(pseudo),
        "variance_fills": fills,
        "prompt_variant_icc": icc,
        "item_keying": keying,
        "keying_balance": keying_balance(parsed, keying),
        "refusal_rates": rates,
        "refusal_sensitive": refusal_sensitive(rates),
        "manski_bounds": manski_bounds(cm, raw, parsed, keying),
        "within_family": within_family(cm, parsed),
        "item_profiles": item_profiles(cm, parsed),
        "loio": leave_one_item_out(cm, parsed),
        "item_zh_en_delta": item_language_deltas(parsed),
        "sensitivity_thresholds": sensitivity_thresholds(parsed),
        "sensitivity_neutralised": sensitivity_neutralised(cm, parsed),
    }

    with pd.option_context("display.width", 240, "display.max_rows", 500):
        if skipped:
            print(
                "cells with an item not yet parsed (skipped where a complete "
                f"10-item vector is required): {', '.join(sorted(skipped))}"
            )
        print("\n=== 1. Variance components (balanced nested approximation) ===")
        print(artefacts["variance_components"].round(4).to_string(index=False))
        print("\n=== 1b. Pseudo-respondent fills per cell (nonzero only) ===")
        nz = fills[(fills[["variant_mean", "cell_mean", "unfillable"]] > 0).any(axis=1)]
        print(nz.to_string(index=False) if len(nz) else "no fills needed")
        print("\n=== 2. Prompt-variant ICC(1) - top 15 ===")
        top = icc.dropna(subset=["icc1"]).sort_values("icc1", ascending=False).head(15)
        print(top.round(4).to_string(index=False))
        print("\n=== 3. Item keying (empirical, from the frozen projection) ===")
        print(keying.round(4).to_string(index=False))
        print("\n=== 3b. Directional bias vs scale midpoints (half-range units) ===")
        print(artefacts["keying_balance"].round(3).to_string(index=False))
        print("\n=== 4. Refusal rates (cells x items with failures) ===")
        rf = rates[rates["n_failed"] > 0]
        print(rf.round(4).to_string(index=False) if len(rf) else "no failures")
        print("\n=== 4b. Sensitive items, en vs zh refusal counts ===")
        print(artefacts["refusal_sensitive"].to_string(index=False))
        print("\n=== 5. Manski worst-case bounds (unparsed calls at keyed extremes) ===")
        print(artefacts["manski_bounds"].round(3).to_string(index=False))
        print("\n=== 6. Within-family descriptives ===")
        print(artefacts["within_family"].round(3).to_string(index=False))
        print("\n=== 7. Item profiles: largest |deviation| from human means (top 15) ===")
        ip = artefacts["item_profiles"]
        print(
            ip.reindex(ip["deviation"].abs().sort_values(ascending=False).index)
            .head(15)
            .round(3)
            .to_string(index=False)
        )
        print("\n=== 7b. Leave-one-item-out: mean displacement by item ===")
        loio = artefacts["loio"]
        print(
            loio.groupby("question")["displacement"]
            .agg(["mean", "max"])
            .sort_values("mean", ascending=False)
            .round(3)
            .to_string()
        )
        print("\n=== 7c. Per-item zh-en deltas: largest |delta| (top 15) ===")
        dl = artefacts["item_zh_en_delta"]
        if len(dl):
            print(
                dl.reindex(dl["zh_minus_en"].abs().sort_values(ascending=False).index)
                .head(15)
                .round(3)
                .to_string(index=False)
            )
        else:
            print("no model has both arms yet")
        print("\n=== 8a. MIN_PER_QUESTION sensitivity (5/10/25) ===")
        print(artefacts["sensitivity_thresholds"].to_string(index=False))
        print("\n=== 8b. F118/F120 neutralised at human means ===")
        print(artefacts["sensitivity_neutralised"].round(3).to_string(index=False))

    for name, frame in artefacts.items():
        frame.to_csv(f"data/diag_2026_{name}.csv", index=False)
    print(f"\nWrote {len(artefacts)} data/diag_2026_*.csv artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
