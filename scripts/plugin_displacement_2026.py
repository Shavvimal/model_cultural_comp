"""Plug-in language-displacement estimator over the committed 2026 replicates.

Recomputes the per-model language displacement as the norm of the mean
displacement vector (the plug-in estimator), replacing the mean of
replicate-paired norms written by analyze_2026.language_effects (lines
107-109 there). The paired-norm estimator is upward-biased for small
displacements (chi-type folding of a mean near zero), and its replicate
pairing is arbitrary - the two arms' replicates are independent draws -
whereas the plug-in point estimate depends only on the two arm means.

Reads data/llm_bootstrap_replicates_2026.csv (33 cells x 10,000 cluster-
bootstrap replicates, written by analyze_2026) and the committed
data/llm_language_effects_2026.csv (for row order and cohort labels; row
order must be preserved so the seeded permutation test reproduces the
delta_pc1 / delta_pc2 rows of conf_2026_origin_permutation.csv exactly).

Per-model CI: unchanged from the committed artefact - the percentile
[2.5, 97.5] interval of the per-replicate norms. Each replicate is one
bootstrap draw of the cell-mean displacement, so the norms' quantiles are
the correct percentile CI for ||delta||; only the point estimate changes.
(Resampling the replicates and averaging would shrink the interval by
sqrt(B) - the replicates are already the estimator's distribution, not
raw data.) Intervals on a norm fold at zero and therefore cannot exclude
it; that is disclosed in the paper rather than hidden.

mean_displacement_ci and origin_language_permutation are copied verbatim
from scripts/confirmatory_2026.py (same seeds and constants) so this
script has no app.* imports and never touches data/ivs_df.pkl.

Writes data/llm_language_effects_plugin_2026.csv and prints the summary
statistics quoted in the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_BOOT_MODELS = 10_000  # confirmatory_2026.py:38
N_PERM = 10_000  # confirmatory_2026.py:40
# Protocol-matched families, diagnostics_2026.py:70-80.
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
FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}


def mean_displacement_ci(lang_fx: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Copied verbatim from confirmatory_2026.py:67-84."""
    rng = np.random.default_rng(seed)
    vec = lang_fx[["delta_pc1", "delta_pc2", "displacement"]].to_numpy()
    idx = rng.integers(0, len(vec), size=(N_BOOT_MODELS, len(vec)))
    reps = vec[idx].mean(axis=1)
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
    """Copied verbatim from confirmatory_2026.py:87-116."""
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




def main() -> int:
    boot = pd.read_csv("data/llm_bootstrap_replicates_2026.csv")
    committed = pd.read_csv("data/llm_language_effects_2026.csv")

    rows = []
    for _, row in committed.iterrows():
        base = row["llm"]
        en = boot[boot["llm"] == base][["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        zh = boot[boot["llm"] == f"{base} [zh]"][["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        n = min(len(en), len(zh))
        delta = zh[:n] - en[:n]
        # Regression guard: the committed component means must reproduce.
        assert abs(delta[:, 0].mean() - row["delta_pc1"]) < 1e-9, base
        assert abs(delta[:, 1].mean() - row["delta_pc2"]) < 1e-9, base
        point = float(np.linalg.norm(delta.mean(axis=0)))
        rows.append(
            {
                "llm": base,
                "cohort": row["cohort"],
                "family": FAMILY_OF[base],
                "delta_pc1": row["delta_pc1"],
                "delta_pc2": row["delta_pc2"],
                "displacement_plugin": point,
                "displacement_lo": row["displacement_lo"],
                "displacement_hi": row["displacement_hi"],
                "displacement_paired_norm_mean": row["displacement"],
            }
        )
    out = pd.DataFrame(rows)

    # Assertions against independently verified values.
    get = lambda m: out.loc[out["llm"] == m, "displacement_plugin"].iloc[0]  # noqa: E731
    assert abs(get("nemotron-3-nano:30b") - 0.178) < 0.002
    assert abs(get("mistral-large-3:675b") - 1.510) < 0.002
    assert abs(out["displacement_plugin"].mean() - 0.655) < 0.002

    out.to_csv("data/llm_language_effects_plugin_2026.csv", index=False)

    lang_fx_plugin = out.rename(columns={"displacement_plugin": "displacement"})[
        ["llm", "cohort", "delta_pc1", "delta_pc2", "displacement"]
    ]

    ci = mean_displacement_ci(lang_fx_plugin)
    perm = origin_language_permutation(lang_fx_plugin)

    # The component rows must be bit-identical to the committed permutation
    # artefact (same seed, same row order; only the displacement column changed
    # and the RNG stream is consumed identically across the three components).
    committed_perm = pd.read_csv("data/conf_2026_origin_permutation.csv")
    for comp in ["delta_pc1", "delta_pc2"]:
        a = perm.loc[perm["component"] == comp].iloc[0]
        b = committed_perm.loc[committed_perm["component"] == comp].iloc[0]
        assert abs(a["chinese_minus_western"] - b["chinese_minus_western"]) < 1e-12, comp
        assert a["p_permutation_two_sided"] == b["p_permutation_two_sided"], comp

    # Commit the plug-in permutation row alongside the folded one so the paper's
    # p-value for the displacement component traces to an artefact.
    perm.to_csv("data/conf_2026_origin_permutation_plugin.csv", index=False)

    pd.set_option("display.width", 200)
    print("Per-model plug-in displacements:")
    print(out.round(3).to_string(index=False))
    print("\nAcross-model mean (plug-in):")
    print(ci.round(4).to_string(index=False))
    print("\nCohort means (plug-in):")
    print(out.groupby("cohort")["displacement_plugin"].agg(["mean", "count"]).round(3))
    print("\nVendor (family) mean plug-in displacement:")
    print(out.groupby("family")["displacement_plugin"].mean().round(3).sort_values())
    print("\nOrigin x language permutation (plug-in displacement row is the new one):")
    print(perm.round(4).to_string(index=False))
    print("\nRange:", round(out["displacement_plugin"].min(), 3), "-", round(out["displacement_plugin"].max(), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
