"""Inter-annotator agreement and majority-vote counts for the frozen trace coding.

Run from the repo root after code_traces_2026.py --merge:

    uv run python scripts/trace_agreement_2026.py

Reads data/trace_labels_2026.csv (one row per annotator x trace; LLM
annotators plus, if present, the blind human subsample). Writes:

    data/trace_agreement_2026.csv       per code: pairwise Cohen's kappa and
                                        raw agreement for every annotator pair,
                                        Fleiss' kappa over the LLM annotators
                                        (complete cases), Krippendorff's alpha
                                        over all annotators (missing allowed),
                                        and each annotator's prevalence
    data/trace_coding_2026.csv          per cell: majority-vote counts of each
                                        code (LLM annotators) next to every
                                        annotator's own count, plus the
                                        mechanical reasoning-language counts
    data/trace_coding_headline_2026.csv per code: adjudicated share of the 900
                                        and the per-annotator range

Aggregation is the strict majority of the LLM annotators per trace and
code; with an even number of annotators a tie is left unadjudicated and
counted. The human subsample is never part of the adjudication - it is the
external check on the codebook, reported as kappa against each LLM
annotator and against the adjudicated label.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from app.agreement import (
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    percent_agreement,
)
from app.trace_codebook import CODES
from app.trace_diagnostics import (
    CODE_LABELS,
    trace_diagnostics,
    validate_trace_labels,
    vote_counts,
)

LABELS = Path("data/trace_labels_2026.csv")
OUT_AGREEMENT = Path("data/trace_agreement_2026.csv")
OUT_CELLS = Path("data/trace_coding_2026.csv")
OUT_HEADLINE = Path("data/trace_coding_headline_2026.csv")
KEY = ["llm", "language", "question", "system_prompt_id", "repeat"]
LANGS = ("english", "mixed", "cjk")


def wide(long: pd.DataFrame, code: str, trace_index: pd.MultiIndex | None = None) -> pd.DataFrame:
    """(trace x annotator) matrix of one code; NaN where an annotator has no label."""
    ok = long[long["error"].isna() & long[code].notna()]
    index = (
        trace_index
        if trace_index is not None
        else pd.MultiIndex.from_frame(long[KEY].drop_duplicates())
    )
    if long.duplicated(["annotator", *KEY]).any():
        raise ValueError("merged labels must contain one row per annotator and trace")
    return ok.pivot(index=KEY, columns="annotator", values=code).reindex(
        index=index, columns=sorted(long["annotator"].unique())
    )


def main(output_dir: str = "data") -> int:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    long = validate_trace_labels(pd.read_csv(LABELS))
    annotators = sorted(long["annotator"].unique())
    llm_annotators = [a for a in annotators if a != "human"]
    print(f"annotators: {annotators}  (adjudicating over {llm_annotators})")

    sample = pd.DataFrame(json.loads(Path("data/trace_samples_2026.json").read_text()))
    sensitivity, coverage, prefix_rates = trace_diagnostics(long, sample)
    successful = sample[sample["error"].isna() & sample["thinking"].fillna("").ne("")]
    trace_index = pd.MultiIndex.from_frame(successful[KEY]).sort_values()
    rows = []
    adjudicated = {}
    vote_denominators = {}
    for code in CODES:
        w = wide(long, code, trace_index)
        # pairwise
        for a, b in combinations(annotators, 2):
            both = w[[a, b]].dropna()
            if len(both) == 0:
                continue
            rows.append(
                {
                    "code": code,
                    "statistic": "cohen_kappa",
                    "raters": f"{a}|{b}",
                    "value": cohen_kappa(both[a].to_numpy(), both[b].to_numpy()),
                    "n_units": len(both),
                }
            )
            rows.append(
                {
                    "code": code,
                    "statistic": "percent_agreement",
                    "raters": f"{a}|{b}",
                    "value": percent_agreement(both[a].to_numpy(), both[b].to_numpy()),
                    "n_units": len(both),
                }
            )
        # Fleiss over LLM annotators, complete cases
        complete = w[llm_annotators].dropna()
        if len(llm_annotators) >= 2 and len(complete):
            counts = np.column_stack([(complete == 0).sum(axis=1), (complete == 1).sum(axis=1)])
            rows.append(
                {
                    "code": code,
                    "statistic": "fleiss_kappa",
                    "raters": "|".join(llm_annotators),
                    "value": fleiss_kappa(counts),
                    "n_units": len(complete),
                }
            )
        # Krippendorff over everyone, missing allowed
        rows.append(
            {
                "code": code,
                "statistic": "krippendorff_alpha",
                "raters": "|".join(annotators),
                "value": krippendorff_alpha_nominal(w[annotators].to_numpy().T),
                "n_units": int((w[annotators].notna().sum(axis=1) >= 2).sum()),
            }
        )
        for a in annotators:
            rows.append(
                {
                    "code": code,
                    "statistic": "prevalence",
                    "raters": a,
                    "value": float(w[a].mean()),
                    "n_units": int(w[a].notna().sum()),
                }
            )
        # adjudication: strict majority over the LLM annotators
        votes = w[llm_annotators]
        maj, denominators = vote_counts(votes)
        vote_denominators[code] = denominators
        adjudicated[code] = maj
        rows.append(
            {
                "code": code,
                "statistic": "n_ties_unadjudicated",
                "raters": "|".join(llm_annotators),
                "value": float(denominators["n_ties"]),
                "n_units": denominators["n_traces"] - denominators["n_without_votes"],
            }
        )
        if "human" in annotators:
            both = pd.concat([maj.rename("majority"), w["human"]], axis=1).dropna()
            rows.append(
                {
                    "code": code,
                    "statistic": "cohen_kappa",
                    "raters": "human|majority",
                    "value": cohen_kappa(both["human"].to_numpy(), both["majority"].to_numpy()),
                    "n_units": len(both),
                }
            )
    agreement = pd.DataFrame(rows)
    agreement.to_csv(destination / OUT_AGREEMENT.name, index=False)

    # per-cell table
    lang = long.drop_duplicates(subset=KEY).set_index(KEY)["reasoning_language"]
    cells = []
    index = adjudicated[CODES[0]].index
    frame = pd.DataFrame(index=index)
    for code in CODES:
        frame[f"{code}__majority"] = adjudicated[code]
        w = wide(long, code, trace_index)
        for a in llm_annotators:
            frame[f"{code}__{a}"] = w[a]
    frame["reasoning_language"] = lang.reindex(index)
    frame = frame.reset_index()
    for (llm, language), g in frame.groupby(["llm", "language"], sort=True):
        row = {"llm": llm, "language": language, "n_traces": len(g)}
        for col in frame.columns:
            if "__" in col:
                row[col] = int(g[col].sum(skipna=True))
        for lg in LANGS:
            row[f"lang_{lg}"] = int((g["reasoning_language"] == lg).sum())
        cells.append(row)
    pd.DataFrame(cells).to_csv(destination / OUT_CELLS.name, index=False)

    # headline
    head = []
    n_total = len(frame)
    for code in CODES:
        shares = {a: float(frame[f"{code}__{a}"].mean()) for a in llm_annotators}
        head.append(
            {
                "code": code,
                "code_label": CODE_LABELS[code],
                **vote_denominators[code],
                "majority_share": float(frame[f"{code}__majority"].mean()),
                "annotator_share_min": min(shares.values()),
                "annotator_share_max": max(shares.values()),
                **{f"share__{a}": v for a, v in shares.items()},
            }
        )
    for lg in LANGS:
        head.append(
            {
                "code": f"lang_{lg}",
                "n_traces": n_total,
                "majority_count": int((frame["reasoning_language"] == lg).sum()),
                "majority_share": float((frame["reasoning_language"] == lg).mean()),
            }
        )
    headline = pd.DataFrame(head)
    headline.to_csv(destination / OUT_HEADLINE.name, index=False)
    sensitivity.to_csv(destination / "trace_self_rater_sensitivity_2026.csv", index=False)
    coverage.to_csv(destination / "trace_sample_coverage_2026.csv", index=False)
    prefix_rates.to_csv(destination / "trace_prefix_code_rates_2026.csv", index=False)

    pd.set_option("display.width", 160)
    print(
        agreement[agreement["statistic"].isin(["fleiss_kappa", "krippendorff_alpha"])].to_string(
            index=False
        )
    )
    print(agreement[agreement["statistic"] == "cohen_kappa"].to_string(index=False))
    print(headline.to_string(index=False))
    print(f"\nwrote trace agreement, majority counts and sensitivity artefacts to {destination}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data")
    sys.exit(main(parser.parse_args().output_dir))
