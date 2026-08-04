"""Phase-1 validation harness for the corrected projection pipeline.

Run from the repo root with the data present:

    uv run python scripts/validate_projection.py

Three checks, in order of importance:

  A. Self-consistency (exact): raw complete-case survey rows pushed through
     the public ``project()`` API must land exactly where the fit placed
     them. This proves countries and any projected data (including LLMs)
     share one coordinate space — the property the 2024 post-hoc code broke.

  B. Agreement with the published 2024 country map: the refit is a different
     EM run on a slightly different row set (the 2024 fit included LLM
     pseudo-respondents), so agreement is reported, not asserted exact.

  C. The substantive result: corrected LLM projections vs the published 2024
     model coordinates, with per-model displacement and nearest cultural
     region, to answer whether the homogenisation finding survives the fix.

Writes corrected artefacts:
    data/cultural_map_model.npz      fitted PPCA + rotation
    data/corrected_country_scores.csv
    data/corrected_llm_scores.csv
"""

import sys

import numpy as np
import pandas as pd
from factor_analyzer import Rotator

from app.culture_map import CulturalMap

SEED = 42
TOL_EXACT = 1e-8


def check_a_self_consistency(cm: CulturalMap) -> bool:
    """Raw complete rows through project() == fitted coordinates, exactly."""
    complete = cm.subset_ivs_df.dropna(subset=cm.iv_qns)
    idx = complete.index
    projected = cm.project(complete)

    positions = cm.subset_ivs_df.index.get_indexer(idx)
    fitted = cm.valid_data.iloc[positions][["PC1_rescaled", "PC2_rescaled"]].to_numpy()
    err = np.abs(projected[["PC1_rescaled", "PC2_rescaled"]].to_numpy() - fitted).max()

    print(f"\n=== A. Self-consistency (n={len(complete):,} complete rows) ===")
    print(f"max |project() - fitted| = {err:.2e}  (tolerance {TOL_EXACT})")
    ok = err < TOL_EXACT
    print("PASS" if ok else "FAIL")
    return ok


def check_b_country_agreement(cm: CulturalMap, published: pd.DataFrame):
    """Corrected country means vs published 2024 pickle (countries only)."""
    pub = published[published["llm"] == False]  # noqa: E712 (llm col is object)
    merged = cm.country_scores_pca.merge(
        pub[["country_code", "PC1_rescaled", "PC2_rescaled"]],
        on="country_code", suffixes=("", "_pub"),
    )
    d = merged[["PC1_rescaled", "PC2_rescaled"]].to_numpy() - \
        merged[["PC1_rescaled_pub", "PC2_rescaled_pub"]].to_numpy()
    dist = np.linalg.norm(d, axis=1)
    r1 = np.corrcoef(merged["PC1_rescaled"], merged["PC1_rescaled_pub"])[0, 1]
    r2 = np.corrcoef(merged["PC2_rescaled"], merged["PC2_rescaled_pub"])[0, 1]

    print(f"\n=== B. Country agreement with published 2024 map (n={len(merged)}) ===")
    print(f"displacement: mean={dist.mean():.4f}  median={np.median(dist):.4f}  max={dist.max():.4f}")
    print(f"correlation:  PC1 r={r1:.5f}  PC2 r={r2:.5f}")
    worst = merged.assign(dist=dist).nlargest(5, "dist")[["Country", "dist"]]
    print("largest moves:")
    print(worst.to_string(index=False))
    return merged


def check_c_llm_projection(cm: CulturalMap, published: pd.DataFrame):
    """Corrected model coordinates vs published, plus nearest regions."""
    llm_data = cm.collect_llm_data()
    projected = cm.project_llm_data(llm_data)
    cm.calculate_average_llm(projected)
    corrected = cm.llm_scores_pca.copy()

    pub_llm = published[published["llm"] == True][  # noqa: E712
        ["Country", "PC1_rescaled", "PC2_rescaled"]
    ].rename(columns={"Country": "llm"})
    merged = corrected.merge(pub_llm, on="llm", suffixes=("", "_pub"), how="left")

    # nearest cultural-region centroid under the corrected map
    centroids = cm.country_scores_pca.groupby("Cultural Region")[
        ["PC1_rescaled", "PC2_rescaled"]
    ].mean()

    def nearest_region(x, y):
        d = np.linalg.norm(centroids.to_numpy() - np.array([x, y]), axis=1)
        return centroids.index[int(np.argmin(d))]

    merged["nearest_region"] = [
        nearest_region(r.PC1_rescaled, r.PC2_rescaled) for r in merged.itertuples()
    ]
    merged["displacement"] = np.linalg.norm(
        merged[["PC1_rescaled", "PC2_rescaled"]].to_numpy()
        - merged[["PC1_rescaled_pub", "PC2_rescaled_pub"]].to_numpy(),
        axis=1,
    )

    print(f"\n=== C. Corrected LLM projections (n={len(merged)}) ===")
    cols = ["llm", "PC1_rescaled", "PC2_rescaled", "PC1_rescaled_pub",
            "PC2_rescaled_pub", "displacement", "nearest_region", "Chinese"]
    with pd.option_context("display.width", 200):
        print(merged[cols].round(3).to_string(index=False))
    return merged


def report_rotation_diagnostics(cm: CulturalMap):
    """Rotated loadings (interpretability) + rotation-choice sensitivity."""
    loadings = pd.DataFrame(
        cm.ppca.C @ cm.rotation, index=cm.iv_qns, columns=["PC1", "PC2"]
    )
    print("\n=== Rotated loadings (C @ R) ===")
    print(loadings.round(3).to_string())

    angle = np.degrees(np.arctan2(cm.rotation[1, 0], cm.rotation[0, 0]))
    alt = Rotator(method="varimax")
    alt.fit_transform(cm.ppca.C * np.sqrt(cm.ppca.eig_vals))
    alt_angle = np.degrees(np.arctan2(alt.rotation_[1, 0], alt.rotation_[0, 0]))
    print(f"\nrotation fitted on score matrix: {angle:.2f} deg (used)")
    print(f"rotation fitted on scaled loadings: {alt_angle:.2f} deg (sensitivity)")
    print(f"explained variance (2 PCs): {cm.ppca.var_exp[-1]:.3f}")


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl", data_dir="data")
    print("Preparing data & fitting PPCA (seeded)...")
    cm.prepare_data()
    print(f"rows after filtering: {len(cm.subset_ivs_df):,}")
    cm.fit(seed=SEED)
    cm.calculate_mean_scores()

    published = pd.read_pickle("data/res_country_scores_pca.pkl")

    ok = check_a_self_consistency(cm)
    check_b_country_agreement(cm, published)
    llm_result = check_c_llm_projection(cm, published)
    report_rotation_diagnostics(cm)

    cm.save_model("data/cultural_map_model.npz")
    cm.country_scores_pca.to_csv("data/corrected_country_scores.csv", index=False)
    llm_result.to_csv("data/corrected_llm_scores.csv", index=False)
    print("\nArtefacts written: data/cultural_map_model.npz, "
          "data/corrected_country_scores.csv, data/corrected_llm_scores.csv")

    if not ok:
        print("\nGATE FAILED: projection path is not self-consistent.")
        return 1
    print("\nGATE PASSED: one coordinate space for countries and models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
