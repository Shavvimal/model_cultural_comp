"""Validation harness for the corrected projection pipeline.

Run from the repo root with the data present:

    uv run python scripts/validate_projection.py

The three gates the write-up releases (§3.2, "Validation"):

  A. Path-identity regression test (exact): raw complete-case survey rows
     pushed through the public ``project()`` API land exactly where the fit
     placed them. This is a regression test that the model and country
     projection paths are byte-identical — the defect present in the 2024
     code — not a validation of the map itself. Both paths could share a
     wrong rotation and this check would still pass.

  B. Correction accounting: per-axis affine calibration of the corrected
     country coordinates against the *previously published 2024* coordinates.
     The two fits differ by design (Y003 sentinel recode, unit-variance
     rescaling), so agreement is a fitted affine map with R², not a
     displacement. This is an internal consistency check on the size of our
     own correction — no regression against the published WVS country
     coordinates is performed anywhere, so it is not external validation.

  C. Rotation diagnostics: six criteria using scores, whitened scores, or
     score-axis bases, with the stated Kaiser settings; the post-rotation
     axis correlation; and projection coefficients C @ R distinguished from
     completed-item/score correlations. Full matrices and score SDs support
     the later outcome sensitivity without an angle-only reconstruction.

Check B reads the 2024 published coordinates from the separately distributed
``data/published_2024_country_scores.csv`` (converted once from the original
``data/res_country_scores_pca.pkl``, which stays gitignored and is used only
as a fallback when the CSV is absent).

Also reports the per-item sentinel recode counts and writes the corrected
artefacts:
    data/cultural_map_model.npz
    data/corrected_country_scores.csv
and the generated summary of what the three checks printed:
    data/validation_summary_2024.csv   (quantity, value)
so the numbers the write-up quotes from this harness — the path-identity row
count, the check-B slopes and R², the per-item sentinel recode counts and the
rotation diagnostics — trace to a file rather than to a console transcript.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from app.culture_map import CulturalMap, check_preparation
from app.instrument_sensitivity import rotation_grid
from app.survey_reference import survey_reference_tables

SEED = 42
TOL_EXACT = 1e-8

# The country coordinates published with the 2024 blog post (119 rows: 109
# countries plus the ten 2024 model rows, flagged by ``llm``). Distributed as a
# CSV in the response archive so check B needs no old survey pickle; the original pickle is the
# gitignored file it was converted from and is only consulted as a fallback.
PUBLISHED_2024_CSV = "data/published_2024_country_scores.csv"
PUBLISHED_2024_PKL = "data/res_country_scores_pca.pkl"
# The generated (quantity, value) summary of everything the checks print.
VALIDATION_SUMMARY_CSV = "data/validation_summary_2024.csv"


def load_published_2024() -> pd.DataFrame:
    if os.path.exists(PUBLISHED_2024_CSV):
        return pd.read_csv(PUBLISHED_2024_CSV)
    if os.path.exists(PUBLISHED_2024_PKL):
        print(
            f"note: {PUBLISHED_2024_CSV} not found; falling back to the "
            f"gitignored {PUBLISHED_2024_PKL}. Regenerate the CSV with\n"
            '  uv run python -c "import pandas as pd; '
            f"pd.read_pickle('{PUBLISHED_2024_PKL}')"
            f".to_csv('{PUBLISHED_2024_CSV}', index=False)\""
        )
        return pd.read_pickle(PUBLISHED_2024_PKL)
    raise FileNotFoundError(
        f"neither {PUBLISHED_2024_CSV} (in the response archive) nor {PUBLISHED_2024_PKL} is present; "
        "check B (correction accounting against the 2024 published coordinates) "
        "cannot run"
    )


def check_a_path_identity(cm: CulturalMap) -> tuple[bool, dict[str, float]]:
    complete = cm.subset_ivs_df.dropna(subset=cm.iv_qns)
    projected = cm.project(complete)

    positions = cm.subset_ivs_df.index.get_indexer(complete.index)
    fitted = cm.valid_data.iloc[positions][["PC1_rescaled", "PC2_rescaled"]].to_numpy()
    err = np.abs(projected[["PC1_rescaled", "PC2_rescaled"]].to_numpy() - fitted).max()

    print(f"\n=== A. Path identity (n={len(complete):,} complete rows) ===")
    print(f"max |project() - fitted| = {err:.2e}  (tolerance {TOL_EXACT})")
    ok = err < TOL_EXACT
    print("PASS" if ok else "FAIL")
    summary = {
        "path_identity_n_complete_rows": len(complete),
        "path_identity_max_abs_err": float(err),
        "path_identity_tolerance": TOL_EXACT,
        "path_identity_pass": int(ok),
    }
    return ok, summary


def check_b_correction_accounting(cm: CulturalMap, published: pd.DataFrame) -> dict[str, float]:
    pub = published[~published["llm"].astype(bool)]
    merged = cm.country_scores_pca.merge(
        pub[["country_code", "PC1_rescaled", "PC2_rescaled"]],
        on="country_code",
        suffixes=("", "_pub"),
    )
    print(f"\n=== B. Affine calibration vs published 2024 coordinates (n={len(merged)}) ===")
    summary: dict[str, float] = {"correction_accounting_n_countries": len(merged)}
    for axis in ("PC1_rescaled", "PC2_rescaled"):
        x = merged[axis].to_numpy()
        y = merged[f"{axis}_pub"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        print(f"{axis}: published = {slope:.3f} * corrected + {intercept:+.3f}   R^2 = {r2:.5f}")
        key = axis.removesuffix("_rescaled").lower()
        summary[f"correction_accounting_{key}_slope"] = float(slope)
        summary[f"correction_accounting_{key}_intercept"] = float(intercept)
        summary[f"correction_accounting_{key}_r2"] = float(r2)
    return summary


def check_c_rotation_diagnostics(cm: CulturalMap) -> dict[str, float]:
    coefficients = pd.DataFrame(cm.ppca.C @ cm.rotation, index=cm.iv_qns, columns=["PC1", "PC2"])
    print("\n=== C. Rotated projection coefficients (C @ R) ===")
    print(coefficients.round(3).to_string())

    scores = cm.ppca.transform()
    eig = cm.ppca.eig_vals
    print("\nRotation sensitivity grid:")
    rotations = rotation_grid(cm)
    print(rotations[["criterion", "angle_deg"]].to_string(index=False))
    rotations.to_csv("data/validation_rotation_sensitivity.csv", index=False)
    # Keep the historical filename for consumers, but distinguish coefficients
    # from Gaussian loadings and completed-item/score correlations explicitly.
    coefficients.rename_axis("question").to_csv("data/validation_rotated_loadings.csv")
    correlations = np.corrcoef(cm.ppca.data.T, (scores @ cm.rotation).T)[: len(cm.iv_qns), -2:]
    coefficient_table = coefficients.rename(
        columns={"PC1": "coefficient_pc1", "PC2": "coefficient_pc2"}
    )
    coefficient_table["completed_item_score_correlation_pc1"] = correlations[:, 0]
    coefficient_table["completed_item_score_correlation_pc2"] = correlations[:, 1]
    coefficient_table.rename_axis("question").to_csv("data/validation_projection_coefficients.csv")

    used_angle = np.degrees(np.arctan2(cm.rotation[1, 0], cm.rotation[0, 0]))
    cov_rot = cm.rotation.T @ np.diag(eig) @ cm.rotation
    axis_r = cov_rot[0, 1] / np.sqrt(cov_rot[0, 0] * cov_rot[1, 1])
    print(f"\nrotation used: {used_angle:+.2f} deg (scores, Kaiser)")
    print(f"post-rotation axis correlation: r = {axis_r:.4f}")
    print(f"explained variance (2 PCs): {cm.ppca.var_exp[-1]:.3f}")
    return {
        "rotation_angle_deg": float(used_angle),
        "post_rotation_axis_r": float(axis_r),
        "explained_variance_2pc": float(cm.ppca.var_exp[-1]),
        "y003_god_r_observed_pairwise": float(cm.subset_ivs_df[["Y003", "F063"]].corr().iloc[0, 1]),
        "y003_god_n_observed_pairs": int(cm.subset_ivs_df[["Y003", "F063"]].dropna().shape[0]),
        "y003_god_r_completed": float(
            np.corrcoef(
                cm.ppca.data[:, cm.iv_qns.index("Y003")], cm.ppca.data[:, cm.iv_qns.index("F063")]
            )[0, 1]
        ),
        "ppca_noise_variance": float(cm.ppca.noise_variance_),
        "ppca_log_likelihood": float(cm.ppca.log_likelihood_),
        "ppca_max_abs_mean_nll_gradient": float(cm.ppca.gradient_norm_),
        "ppca_gradient_tolerance": float(cm.ppca.tolerance_),
        "ppca_converged": int(cm.ppca.converged_),
        "ppca_n_starts": len(cm.ppca.start_converged_),
        "ppca_all_starts_converged": int(np.all(cm.ppca.start_converged_)),
        "ppca_start_log_likelihood_range": float(np.ptp(cm.ppca.start_log_likelihoods_)),
    }


def write_validation_summary(rows: dict[str, float], path: str = VALIDATION_SUMMARY_CSV) -> None:
    """Write numeric values in order; keep counts integral and encode flags as 0/1.

    Textual provenance belongs in validation_survey_preparation.csv. A literal
    True/False would make pandas infer the entire value column as strings and
    break the finite claim selectors' numeric comparisons.
    """
    numeric = [
        int(value) if isinstance(value, (bool, np.bool_)) else value for value in rows.values()
    ]
    if any(not isinstance(value, (int, float, np.integer, np.floating)) for value in numeric):
        raise ValueError("validation summary values must be numeric; export text separately")
    values = pd.Series(numeric, dtype=object)
    pd.DataFrame({"quantity": list(rows), "value": values}).to_csv(path, index=False)


def main() -> int:
    cm = CulturalMap("data/ivs_df.pkl", "data/country_codes.pkl")
    print("Preparing data & fitting PPCA (seeded)...")
    cm.prepare_data()
    preparation = cm.survey_preparation_report
    check_preparation(preparation)
    print(f"rows after filtering: {len(cm.subset_ivs_df):,}")
    print(
        "sentinel recodes (out-of-range -> NaN): "
        + ", ".join(f"{q}={n:,}" for q, n in cm.sentinel_counts.items() if n)
    )
    cm.fit(seed=SEED)
    # Log only: the summary CSV and model archive schemas are frozen.
    print(f"PPCA optimizer resumes across starts: {cm.ppca.n_optimizer_restarts_}")
    cm.calculate_mean_scores()

    published = load_published_2024()

    summary: dict[str, float] = {"rows_after_filtering": len(cm.subset_ivs_df)}
    preparation_rows = {}
    for key, value in preparation.items():
        if isinstance(value, dict):
            preparation_rows.update(
                {f"{key}_{subkey}": subvalue for subkey, subvalue in value.items()}
            )
        else:
            preparation_rows[key] = value
    for key, value in preparation_rows.items():
        if isinstance(value, (int, float)):
            summary[f"preparation_{key}"] = value
    pd.DataFrame(
        [
            {
                "quantity": key,
                "value": json.dumps(value) if isinstance(value, (list, dict)) else value,
            }
            for key, value in preparation_rows.items()
        ]
    ).to_csv("data/validation_survey_preparation.csv", index=False)
    summary.update({f"sentinel_recodes_{q}": n for q, n in cm.sentinel_counts.items()})
    ok, summary_a = check_a_path_identity(cm)
    summary.update(summary_a)
    summary.update(check_b_correction_accounting(cm, published))
    summary.update(check_c_rotation_diagnostics(cm))
    reference_summary, references, items, unmapped = survey_reference_tables(cm)
    summary.update(reference_summary)

    cm.save_model("data/cultural_map_model.npz")
    cm.country_scores_pca.to_csv("data/corrected_country_scores.csv", index=False)
    write_validation_summary(summary)
    references.to_csv("data/validation_survey_reference.csv", index=False)
    # Includes the ten frozen standardisation means/SDs needed for released
    # aggregate-only profile analyses; no respondent rows are exported.
    items.to_csv("data/validation_survey_item_baselines.csv", index=False)
    unmapped.to_csv("data/validation_unmapped_entities.csv", index=False)
    print("\nSurvey comparison references (distinct estimands):")
    print(references.to_string(index=False))
    print("\nUnmapped fit entities:")
    print(unmapped.to_string(index=False))
    print(
        "\nArtefacts written: data/cultural_map_model.npz, "
        f"data/corrected_country_scores.csv, {VALIDATION_SUMMARY_CSV}, "
        "data/validation_survey_reference.csv, data/validation_survey_item_baselines.csv, "
        "data/validation_unmapped_entities.csv"
    )

    if not ok:
        print("\nGATE FAILED: projection paths are not identical.")
        return 1
    print("\nGATE PASSED: model and country projection paths are identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
