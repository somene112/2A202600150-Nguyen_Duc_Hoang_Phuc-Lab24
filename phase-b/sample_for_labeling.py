#!/usr/bin/env python3
"""Phase B.3 Step 1 — Sample 10 pairs from pairwise_results.csv for human labeling.

Saves phase-b/to_label.csv with columns:
  question_id, question, answer_a, answer_b
  (winner columns intentionally omitted to prevent labeler bias)

Usage
-----
  python phase-b/sample_for_labeling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parents[1]

INPUT_CSV  = _root / "phase-b" / "pairwise_results.csv"
OUTPUT_CSV = _root / "phase-b" / "to_label.csv"

N_SAMPLE     = 10
RANDOM_STATE = 42


def main() -> None:
    print(f"[+] Reading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[+] {len(df)} rows available")

    sample = (
        df
        .sample(n=min(N_SAMPLE, len(df)), random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    sample.index.name = "question_id"

    # Keep only the columns a human annotator should see
    out = sample[["question", "answer_a", "answer_b"]].copy()
    out.insert(0, "question_id", range(len(out)))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[+] Saved {len(out)} rows → {OUTPUT_CSV}")
    print()
    print("Next step: fill in phase-b/human_labels.csv")
    print("  human_winner: A | B | tie")
    print("  confidence:   1=low  2=medium  3=high")


if __name__ == "__main__":
    main()
