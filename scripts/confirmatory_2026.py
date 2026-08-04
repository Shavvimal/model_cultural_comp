"""Confirmatory extensions for the 2026 cohort — analysis-plan §1 items not
covered by scripts/analyze_2026.py.

Consumes the artefacts analyze_2026.py writes (bootstrap replicates,
language effects) plus the raw corpus, and adds:

  1. sign test on delta_PC2 across models (directional hypothesis: Chinese
     administration shifts cells toward the traditional pole, i.e.
     delta_PC2 < 0)
  2. mean displacement vector across models with an across-model bootstrap CI
  3. origin x language interaction: two-sample permutation test (10^4
     permutations) of Chinese-origin vs Western mean language-effect vectors
  4. Confucian-centroid distances per cohort, per arm, with replicate CIs
     (no pooled cross-arm test)
  5. per-item language effects delta_mj = mean_zh - mean_en per model with
     cluster-bootstrap CIs, plus per-item cross-model sign tests
     (BH-corrected over the ten items)
  6. the one longitudinal statistic: coherence rate of attempted
     Chinese-origin models, 2024 vs 2026, with Clopper-Pearson intervals
  7. replicate-simultaneous headline bounds (the "in every replicate"
     statement, stated at whatever bound actually holds)

Run after analyze_2026.py:  uv run python scripts/confirmatory_2026.py
Self-test on synthetic data: uv run python scripts/confirmatory_2026.py --selftest
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from app.culture_map import CulturalMap
from app.llm_bootstrap import WESTERN_REGIONS, load_responses_2026
from app.llm_meta import cohort_2026

SEED = 42
N_BOOT_MODELS = 10_000
N_BOOT_ITEM_DELTA = 2_000
N_PERM = 10_000
HUMAN_MEAN = (0.38, -0.01)
# Items predicted (from the 2024 qwen2:7b contrast) to move toward the
# traditional pole under Chinese administration, with the direction that
# "traditional" takes on each item's raw scale.
DIRECTIONAL_ITEMS = {"G006": -1, "F063": +1, "E018": -1}


def sign_test_delta_pc2(lang_fx: pd.DataFrame) -> pd.DataFrame:
    d = lang_fx["delta_pc2"].to_numpy()
    n_neg = int((d < 0).sum())
    n = int((d != 0).sum())
    two_sided = binomtest(n_neg, n, 0.5, alternative="two-sided").pvalue
    directional = binomtest(n_neg, n, 0.5, alternative="greater").pvalue
    return pd.DataFrame(
        [
            {
                "n_models": len(d),
                "n_delta_pc2_negative": n_neg,
                "p_two_sided": two_sided,
                "p_directional_traditional": directional,
                "median_delta_pc2": float(np.median(d)),
            }
        ]
    )


def mean_displacement_ci(lang_fx: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Across-model bootstrap of the mean language-effect vector."""
    rng = np.random.default_rng(seed)
    vec = lang_fx[["delta_pc1", "delta_pc2", "displacement"]].to_numpy()
    idx = rng.integers(0, len(vec), size=(N_BOOT_MODELS, len(vec)))
    reps = vec[idx].mean(axis=1)  # (B, 3)
    cols = ["delta_pc1", "delta_pc2", "displacement"]
    rows = []
    for j, col in enumerate(cols):
        rows.append(
            {
                "quantity": f"mean_{col}",
                "estimate": float(vec[:, j].mean()),
                "ci_lo": float(np.quantile(reps[:, j], 0.025)),
                "ci_hi": float(np.quantile(reps[:, j], 0.975)),
            }
        )
    return pd.DataFrame(rows)


def origin_language_permutation(lang_fx: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Two-sample permutation test on the per-model language-effect vectors.

    Statistic: difference (chinese - western) of cohort means, per component
    and for the displacement magnitude. Permutes cohort labels.
    """
    rng = np.random.default_rng(seed)
    is_cn = (lang_fx["cohort"].str.lower() == "chinese").to_numpy()
    stats = {}
    for col in ["delta_pc1", "delta_pc2", "displacement"]:
        v = lang_fx[col].to_numpy()
        obs = v[is_cn].mean() - v[~is_cn].mean()
        perm = np.empty(N_PERM)
        for b in range(N_PERM):
            lab = rng.permutation(is_cn)
            perm[b] = v[lab].mean() - v[~lab].mean()
        p = float((np.abs(perm) >= abs(obs)).mean())
        stats[col] = (obs, p)
    return pd.DataFrame(
        [
            {
                "component": col,
                "chinese_minus_western": obs,
                "p_permutation_two_sided": p,
                "n_chinese": int(is_cn.sum()),
                "n_western": int((~is_cn).sum()),
            }
            for col, (obs, p) in stats.items()
        ]
    )


def confucian_distances(
    boot: pd.DataFrame, country_scores: pd.DataFrame, cohort_fn=cohort_2026
) -> pd.DataFrame:
    """Per cohort x arm: mean distance to the Confucian centroid, replicate CI.

    Cohort-mean distances are formed per replicate index (replicates are
    aligned across models by construction — same seed, same B), giving a
    bootstrap distribution of the cohort mean. Arms are never pooled.
    """
    conf = (
        country_scores[country_scores["Cultural Region"] == "Confucian"][
            ["PC1_rescaled", "PC2_rescaled"]
        ]
        .mean()
        .to_numpy()
    )
    boot = boot.copy()
    boot["language"] = np.where(boot["llm"].str.endswith(" [zh]"), "zh", "en")
    boot["base"] = boot["llm"].str.replace(" [zh]", "", regex=False)
    boot["cohort"] = boot["base"].map(cohort_fn)
    boot["dist_confucian"] = np.linalg.norm(
        boot[["PC1_rescaled", "PC2_rescaled"]].to_numpy() - conf, axis=1
    )
    rows = []
    for (cohort, lang), g in boot.groupby(["cohort", "language"]):
        per_rep = g.groupby("replicate")["dist_confucian"].mean()
        rows.append(
            {
                "cohort": cohort,
                "language": lang,
                "n_models": g["base"].nunique(),
                "mean_dist_confucian": float(per_rep.mean()),
                "ci_lo": float(per_rep.quantile(0.025)),
                "ci_hi": float(per_rep.quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


def _cluster_item_means(g: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Cluster-bootstrap replicates of one (cell, item)'s mean value."""
    by_variant = [grp["value"].to_numpy() for _, grp in g.groupby("system_prompt_id")]
    k = len(by_variant)
    sums = np.array([v.sum() for v in by_variant])
    counts = np.array([len(v) for v in by_variant])
    draws = rng.integers(0, k, size=(n_boot, k))
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def per_item_language_effects(responses: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """delta_mj = mean_zh - mean_en per model x item, cluster-bootstrap CI."""
    rng = np.random.default_rng(seed)
    responses = responses.copy()
    responses["base"] = responses["llm"].str.replace(" [zh]", "", regex=False)
    rows = []
    for base, g in responses.groupby("base"):
        langs = set(g["language"])
        if langs != {"en", "zh"}:
            continue
        for qn, gq in g.groupby("question"):
            en = gq[gq["language"] == "en"]
            zh = gq[gq["language"] == "zh"]
            if en.empty or zh.empty:
                continue
            en_reps = _cluster_item_means(en, rng, N_BOOT_ITEM_DELTA)
            zh_reps = _cluster_item_means(zh, rng, N_BOOT_ITEM_DELTA)
            delta = zh_reps - en_reps
            rows.append(
                {
                    "llm": base,
                    "cohort": cohort_2026(base),
                    "question": qn,
                    "mean_en": float(en["value"].mean()),
                    "mean_zh": float(zh["value"].mean()),
                    "delta": float(zh["value"].mean() - en["value"].mean()),
                    "delta_ci_lo": float(np.quantile(delta, 0.025)),
                    "delta_ci_hi": float(np.quantile(delta, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def per_item_sign_tests(item_fx: pd.DataFrame) -> pd.DataFrame:
    """Cross-model sign test per item, Benjamini-Hochberg over the ten items."""
    rows = []
    for qn, g in item_fx.groupby("question"):
        d = g["delta"].to_numpy()
        n_pos = int((d > 0).sum())
        n = int((d != 0).sum())
        if n == 0:
            continue
        p = binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue
        rows.append(
            {
                "question": qn,
                "n_models": len(d),
                "n_delta_positive": n_pos,
                "median_delta": float(np.median(d)),
                "p_sign_two_sided": p,
                "directional_prediction": DIRECTIONAL_ITEMS.get(qn, 0),
            }
        )
    out = pd.DataFrame(rows).sort_values("p_sign_two_sided").reset_index(drop=True)
    m = len(out)
    ranked = out["p_sign_two_sided"].to_numpy()
    bh = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    out["p_bh"] = np.minimum(bh, 1.0)
    return out


def coherence_rate(rates_2026: pd.DataFrame, min_per_question: int = 10) -> pd.DataFrame:
    """Attempted Chinese-origin models producing a usable corpus, per year.

    2024: 4 of 9 attempted Chinese-origin models produced parseable corpora
    (committed in the paper draft). 2026: computed from the parse-rate
    artefact — a model counts as coherent if every item in its English cell
    has at least ``min_per_question`` parsed responses. kimi-k3 is excluded
    from the attempted set (billing wall: zero records collected — a
    provisioning failure, not a model behaviour; reported in Appendix A).
    """
    en = rates_2026[(rates_2026["language"] == "en")]
    cn = en[en["cohort"].str.lower() == "chinese"]
    coherent = int((cn["min_per_question"] >= min_per_question).sum())
    attempted = len(cn)
    rows = []
    for year, k, n in [("2024", 4, 9), ("2026", coherent, attempted)]:
        lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
        rows.append(
            {
                "year": year,
                "coherent": k,
                "attempted": n,
                "rate": k / n,
                "ci_lo_clopper_pearson": float(lo),
                "ci_hi_clopper_pearson": float(hi),
            }
        )
    return pd.DataFrame(rows)


def simultaneous_headline(boot: pd.DataFrame, country_scores: pd.DataFrame) -> pd.DataFrame:
    """The bounds that hold in EVERY replicate of every cell, per arm.

    Reported per cell: the minimum (over replicates) share of countries
    closer to the human mean, and the minimum distance to any non-Western
    centroid — plus the across-cell floors, which are the numbers the
    abstract's simultaneous statement is entitled to.
    """
    countries_xy = country_scores[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
    country_dists = np.linalg.norm(countries_xy - np.array(HUMAN_MEAN), axis=1)
    centroids = country_scores.groupby("Cultural Region")[["PC1_rescaled", "PC2_rescaled"]].mean()
    non_western = centroids.loc[~centroids.index.isin(WESTERN_REGIONS)].to_numpy()

    rows = []
    for llm, g in boot.groupby("llm"):
        xy = g[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        d_mean = np.linalg.norm(xy - np.array(HUMAN_MEAN), axis=1)
        pct_closer = (country_dists[None, :] < d_mean[:, None]).mean(axis=1)
        d_nonwest = np.linalg.norm(xy[:, None, :] - non_western[None, :, :], axis=2).min(axis=1)
        rows.append(
            {
                "llm": llm,
                "min_pct_countries_closer": float(pct_closer.min()),
                "min_dist_nonwestern": float(d_nonwest.min()),
                "all_reps_beyond_median_country": bool((d_mean > np.median(country_dists)).all()),
            }
        )
    out = pd.DataFrame(rows)
    floor = pd.DataFrame(
        [
            {
                "llm": "== FLOOR OVER ALL CELLS ==",
                "min_pct_countries_closer": out["min_pct_countries_closer"].min(),
                "min_dist_nonwestern": out["min_dist_nonwestern"].min(),
                "all_reps_beyond_median_country": bool(out["all_reps_beyond_median_country"].all()),
            }
        ]
    )
    return pd.concat([out, floor], ignore_index=True)


def _synthetic_fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    models = [f"m{i}" for i in range(6)]
    lang_fx = pd.DataFrame(
        {
            "llm": models,
            "cohort": ["Chinese"] * 3 + ["Western"] * 3,
            "delta_pc1": rng.normal(0.3, 0.1, 6),
            "delta_pc2": rng.normal(-0.8, 0.2, 6),
            "displacement": rng.uniform(0.5, 1.5, 6),
        }
    )
    reps = []
    for m in models:
        for lang in ["", " [zh]"]:
            reps.append(
                pd.DataFrame(
                    {
                        "llm": m + lang,
                        "replicate": np.arange(200),
                        "PC1_rescaled": rng.normal(2.0, 0.1, 200),
                        "PC2_rescaled": rng.normal(1.5, 0.1, 200),
                    }
                )
            )
    boot = pd.concat(reps, ignore_index=True)
    countries = pd.DataFrame(
        {
            "Cultural Region": ["Confucian"] * 5
            + ["Protestant Europe"] * 5
            + ["Orthodox Europe"] * 5,
            "PC1_rescaled": rng.normal(0, 1, 15),
            "PC2_rescaled": rng.normal(0, 1, 15),
        }
    )
    rates = pd.DataFrame(
        {
            "llm": models,
            "language": "en",
            "cohort": ["Chinese"] * 3 + ["Western"] * 3,
            "min_per_question": [50, 50, 8, 50, 50, 50],
        }
    )
    return lang_fx, boot, countries, rates


def main(selftest: bool = False) -> int:
    if selftest:
        lang_fx, boot, country_scores, rates = _synthetic_fixtures()
        responses = None
    else:
        lang_fx = pd.read_csv("data/llm_language_effects_2026.csv")
        boot = pd.read_csv("data/llm_bootstrap_replicates_2026.csv")
        country_scores = pd.read_csv("data/corrected_country_scores.csv")
        rates = pd.read_csv("data/llm_parse_rates_2026.csv")
        cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
        cm.load_model("data/cultural_map_model.npz")
        responses = load_responses_2026(cm, "data/collection_2026")

    artefacts: dict[str, pd.DataFrame] = {
        "conf_2026_sign_test": sign_test_delta_pc2(lang_fx),
        "conf_2026_mean_displacement": mean_displacement_ci(lang_fx),
        "conf_2026_origin_permutation": origin_language_permutation(lang_fx),
        "conf_2026_confucian_distances": confucian_distances(
            boot, country_scores, cohort_fn=(lambda m: "Chinese") if selftest else cohort_2026
        ),
        "conf_2026_coherence_rate": coherence_rate(rates),
        "conf_2026_simultaneous_headline": simultaneous_headline(boot, country_scores),
    }
    if responses is not None:
        item_fx = per_item_language_effects(responses)
        artefacts["conf_2026_item_language_effects"] = item_fx
        artefacts["conf_2026_item_sign_tests"] = per_item_sign_tests(item_fx)

    with pd.option_context("display.width", 220, "display.max_rows", 250):
        for name, frame in artefacts.items():
            print(f"\n=== {name} ===")
            print(frame.round(4).to_string(index=False))

    if not selftest:
        for name, frame in artefacts.items():
            frame.to_csv(f"data/{name}.csv", index=False)
        print(f"\nWrote {len(artefacts)} data/conf_2026_*.csv artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main(selftest="--selftest" in sys.argv))
