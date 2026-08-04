"""Validation harness for the corrected projection pipeline.

Run from the repo root with the data present:

    uv run python scripts/validate_projection.py

Checks, per the statistical review (paper draft repo):

  A. Path-identity regression test (exact): raw complete-case survey rows
     pushed through the public ``project()`` API land exactly where the fit
     placed them. This is a regression test that the model and country
     projection paths are byte-identical — the defect present in the 2024
     code — not a validation of the map itself.

  B. External agreement: per-axis affine calibration of the corrected
     country coordinates against the published 2024 coordinates. The two
     fits differ by design (Y003 sentinel recode, unit-variance rescaling),
     so agreement is a fitted affine map with R², not a displacement.

  C. Rotation diagnostics: the full sensitivity grid (scores / whitened
     scores / loadings x Kaiser on/off), the post-rotation axis
     correlation, and the rotated loadings.

Also reports the per-item sentinel recode counts and writes the corrected
artefacts:
    data/cultural_map_model.npz
    data/corrected_country_scores.csv
"""

import sys

import numpy as np
import pandas as pd
from factor_analyzer import Rotator

from app.culture_map import CulturalMap

SEED = 42
TOL_EXACT = 1e-8


def check_a_path_identity(cm: CulturalMap) -> bool:
    complete = cm.subset_ivs_df.dropna(subset=cm.iv_qns)
    projected = cm.project(complete)

    positions = cm.subset_ivs_df.index.get_indexer(complete.index)
    fitted = cm.valid_data.iloc[positions][["PC1_rescaled", "PC2_rescaled"]].to_numpy()
    err = np.abs(projected[["PC1_rescaled", "PC2_rescaled"]].to_numpy() - fitted).max()

    print(f"\n=== A. Path identity (n={len(complete):,} complete rows) ===")
    print(f"max |project() - fitted| = {err:.2e}  (tolerance {TOL_EXACT})")
    ok = err < TOL_EXACT
    print("PASS" if ok else "FAIL")
    return ok


def check_b_external_agreement(cm: CulturalMap, published: pd.DataFrame):
    pub = published[not published["llm"]]
    merged = cm.country_scores_pca.merge(
        pub[["country_code", "PC1_rescaled", "PC2_rescaled"]],
        on="country_code",
        suffixes=("", "_pub"),
    )
    print(f"\n=== B. Affine calibration vs published 2024 coordinates (n={len(merged)}) ===")
    for axis in ("PC1_rescaled", "PC2_rescaled"):
        x = merged[axis].to_numpy()
        y = merged[f"{axis}_pub"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        print(f"{axis}: published = {slope:.3f} * corrected + {intercept:+.3f}   R^2 = {r2:.5f}")


def check_c_rotation_diagnostics(cm: CulturalMap):
    loadings = pd.DataFrame(cm.ppca.C @ cm.rotation, index=cm.iv_qns, columns=["PC1", "PC2"])
    print("\n=== C. Rotated loadings (C @ R) ===")
    print(loadings.round(3).to_string())

    scores = cm.ppca.transform()
    eig = cm.ppca.eig_vals
    grid = {
        "scores (Kaiser)": (scores, True),
        "scores (no Kaiser)": (scores, False),
        "whitened scores (Kaiser)": (scores / np.sqrt(eig), True),
        "loadings C (Kaiser)": (cm.ppca.C, True),
        "loadings C*sqrt(eig) (Kaiser)": (cm.ppca.C * np.sqrt(eig), True),
        "loadings C*sqrt(eig) (no Kaiser)": (cm.ppca.C * np.sqrt(eig), False),
    }
    print("\nRotation sensitivity grid:")
    for name, (mat, kaiser) in grid.items():
        rot = Rotator(method="varimax", normalize=kaiser)
        rot.fit_transform(mat)
        angle = np.degrees(np.arctan2(rot.rotation_[1, 0], rot.rotation_[0, 0]))
        print(f"  {name:36s} {angle:+8.2f} deg")

    used_angle = np.degrees(np.arctan2(cm.rotation[1, 0], cm.rotation[0, 0]))
    cov_rot = cm.rotation.T @ np.diag(eig) @ cm.rotation
    axis_r = cov_rot[0, 1] / np.sqrt(cov_rot[0, 0] * cov_rot[1, 1])
    print(f"\nrotation used: {used_angle:+.2f} deg (scores, Kaiser)")
    print(f"post-rotation axis correlation: r = {axis_r:.4f}")
    print(f"explained variance (2 PCs): {cm.ppca.var_exp[-1]:.3f}")


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
    print("Preparing data & fitting PPCA (seeded)...")
    cm.prepare_data()
    print(f"rows after filtering: {len(cm.subset_ivs_df):,}")
    print(
        "sentinel recodes (out-of-range -> NaN): "
        + ", ".join(f"{q}={n:,}" for q, n in cm.sentinel_counts.items() if n)
    )
    cm.fit(seed=SEED)
    cm.calculate_mean_scores()

    published = pd.read_pickle("data/res_country_scores_pca.pkl")

    ok = check_a_path_identity(cm)
    check_b_external_agreement(cm, published)
    check_c_rotation_diagnostics(cm)

    cm.save_model("data/cultural_map_model.npz")
    cm.country_scores_pca.to_csv("data/corrected_country_scores.csv", index=False)
    print("\nArtefacts written: data/cultural_map_model.npz, data/corrected_country_scores.csv")

    if not ok:
        print("\nGATE FAILED: projection paths are not identical.")
        return 1
    print("\nGATE PASSED: model and country projection paths are identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
