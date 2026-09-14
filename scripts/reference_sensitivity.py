"""Compare alternative respondent means without moving the frozen map.

Run after validate_projection.py and both cohort analyses. Reads only their
aggregate outputs and bootstrap replicates; writes a separate sensitivity
artifact, never replacing the primary estimand or its outputs.
"""

from pathlib import Path

import pandas as pd

from app.survey_reference import reference_sensitivity


def main() -> int:
    references = pd.read_csv("data/validation_survey_reference.csv")
    countries = pd.read_csv("data/corrected_country_scores.csv")
    tables = []
    for cohort, suffix in [(2024, ""), (2026, "_2026")]:
        points = pd.read_csv(f"data/llm_ellipses{suffix}.csv")
        replicates = pd.read_csv(f"data/llm_bootstrap_replicates{suffix}.csv")
        table = reference_sensitivity(points, countries, references, replicates)
        table.insert(0, "cohort", cohort)
        tables.append(table)
    output = Path("data/validation_reference_sensitivity.csv")
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output, index=False)
    print(
        result.groupby(["cohort", "reference"])
        .agg(
            cells=("llm", "size"),
            quadrant_changes=("quadrant_changed_from_fixed_reference", "sum"),
            max_distance_change=("distance_change_from_fixed_reference", lambda x: x.abs().max()),
            max_share_change=("country_share_change_from_fixed_reference", lambda x: x.abs().max()),
        )
        .to_string()
    )
    print(f"\nWrote {output}; primary reference and coordinates unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
