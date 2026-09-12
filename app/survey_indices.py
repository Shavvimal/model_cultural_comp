"""Recover survey indices from observed, harmonized constituent responses.

The IVS longitudinal child-quality items use 1 = mentioned and 0 = not
mentioned. This differs from the 1/2 coding in individual WVS wave files.
Only the harmonized 0/1 inputs are accepted here; missing codes are never
interpreted as non-mentions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

Y003_CONSTITUENTS = ("A029", "A039", "A040", "A042")
Y003_FORMULA = "A029 + A039 - A040 - A042"
Y003_DEFINITION_URL = "https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=autonomous"


@dataclass
class Y003Recovery:
    """In-memory values and provenance, with a respondent-free audit report."""

    values: pd.Series
    provenance: pd.Series
    report: dict


def recover_y003(frame: pd.DataFrame) -> Y003Recovery:
    """Fill absent Y003 only when all four constituents are valid 0/1 values.

    Existing in-range values are preserved, including legitimate negative
    values -2 and -1. Where constituents allow a comparison, discrepancies
    with a delivered index are counted rather than overwritten. Callers
    producing research artifacts must check ``discordant_direct``.

    A legacy or synthetic frame without all constituent columns receives no
    reconstructed values; the missing column names are explicit in the
    report. A frame with constituents but no Y003 column can be recovered.
    No input values or indices are mutated or reset. The existing numeric
    range contract for delivered indices is retained for continuous
    synthetic fixtures; reconstruction itself always yields integers.

    Formula and the complete-constituent rule follow the official WVS
    longitudinal definition at ``Y003_DEFINITION_URL``. The EVS trend file
    harmonizes these variables to the same binary coding. For example, its
    Turkey 2009 unrecorded non-mentions are coded -5, so those incomplete
    constituent sets cannot enter reconstruction.
    """
    delivered = (
        frame["Y003"].copy()
        if "Y003" in frame
        else pd.Series(np.nan, index=frame.index, name="Y003")
    )
    direct = delivered.between(-2, 2).fillna(False).astype(bool)
    values = delivered.where(direct).astype(float)
    missing_columns = [column for column in Y003_CONSTITUENTS if column not in frame]
    complete = pd.Series(False, index=frame.index)
    derived = pd.Series(np.nan, index=frame.index)
    if not missing_columns:
        parts = frame[list(Y003_CONSTITUENTS)]
        complete = parts.isin([0, 1]).all(axis=1)
        # Compute only on complete rows; do not sum with skipna=True.
        valid_parts = parts.loc[complete]
        derived.loc[complete] = (
            valid_parts["A029"] + valid_parts["A039"] - valid_parts["A040"] - valid_parts["A042"]
        )

    reconstructed = ~direct & complete
    comparable = direct & complete
    concordant = comparable & delivered.eq(derived)
    values.loc[reconstructed] = derived.loc[reconstructed]
    provenance = pd.Series(
        np.select([direct, reconstructed], ["direct", "reconstructed"], default="missing"),
        index=frame.index,
        name="y003_provenance",
    )
    return Y003Recovery(
        values=values,
        provenance=provenance,
        report={
            "formula": Y003_FORMULA,
            "definition_url": Y003_DEFINITION_URL,
            "constituents": list(Y003_CONSTITUENTS),
            "constituent_coding": "0=not mentioned; 1=mentioned; all four must be valid",
            "missing_constituent_columns": missing_columns,
            "input_index_column_present": "Y003" in frame,
            "rows": len(frame),
            "direct": int(direct.sum()),
            "complete_constituents": int(complete.sum()),
            "reconstructed": int(reconstructed.sum()),
            "still_missing": int(values.isna().sum()),
            "comparable_direct": int(comparable.sum()),
            "concordant_direct": int(concordant.sum()),
            "discordant_direct": int((comparable & ~concordant).sum()),
        },
    )
