#!/usr/bin/env python3
"""Phase B.3 Step 3 — Cohen's Kappa: human labels vs LLM judge.

Loads
-----
  phase-b/human_labels.csv   (filled by human annotator)
  phase-b/pairwise_results.csv  (LLM winner_after_swap)
  phase-b/to_label.csv       (question_id ↔ row mapping)

Outputs
-------
  phase-b/kappa_result.json  {kappa, interpretation, n_samples, agreement_pct}
  phase-b/kappa_root_cause.md   (generated only when kappa < 0.6)

Usage
-----
  python phase-b/kappa_analysis.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from textwrap import dedent

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

_root = Path(__file__).resolve().parents[1]

HUMAN_CSV    = _root / "phase-b" / "human_labels.csv"
PAIRWISE_CSV = _root / "phase-b" / "pairwise_results.csv"
TO_LABEL_CSV = _root / "phase-b" / "to_label.csv"
KAPPA_JSON   = _root / "phase-b" / "kappa_result.json"
ROOT_MD      = _root / "phase-b" / "kappa_root_cause.md"

_USE_COLOR = sys.stdout.isatty()
def _g(s): return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _r(s): return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _y(s): return f"\033[93m{s}\033[0m" if _USE_COLOR else s
def _b(s): return f"\033[1m{s}\033[0m"  if _USE_COLOR else s


# ── Label normalization ────────────────────────────────────────────────────────
_NORM_MAP = {
    # → A
    "a": "A", "answer a": "A", "answer_a": "A", "answera": "A",
    # → B
    "b": "B", "answer b": "B", "answer_b": "B", "answerb": "B",
    # → tie
    "tie": "tie", "ties": "tie", "draw": "tie", "equal": "tie",
    "neither": "tie", "both": "tie", "same": "tie", "n/a": "tie",
}

def normalize_label(raw) -> str | None:
    """Return 'A', 'B', or 'tie'; None if unrecognisable."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    return _NORM_MAP.get(s)


# ── Kappa interpretation ───────────────────────────────────────────────────────
def interpret_kappa(k: float) -> str:
    if k < 0:      return "chance (worse than random)"
    if k < 0.20:   return "slight"
    if k < 0.40:   return "fair"
    if k < 0.60:   return "moderate"
    if k < 0.80:   return "substantial"
    return "almost perfect"


# ── Root-cause template ────────────────────────────────────────────────────────
def _write_root_cause(kappa: float) -> None:
    content = dedent(f"""\
    # Cohen's Kappa Root-Cause Analysis

    **kappa = {kappa:.4f}** ({interpret_kappa(kappa)}) — below 0.60 moderate threshold

    This document templates three hypotheses for the disagreement between the
    human annotator and the LLM judge (`winner_after_swap` from B.1).  Fill in
    the Evidence and Verdict sections after inspecting the disagreement rows.

    ---

    ## Hypothesis 1 — Length Bias

    **Claim:** The LLM judge (gpt-4o-mini) systematically prefers the longer
    answer regardless of quality, inflating wins for whichever config produces
    more tokens.

    **How to test:** Compare `len(answer_a)` vs `len(answer_b)` for all rows
    where `winner_after_swap == "A"`.  If mean(len_a) > mean(len_b) by > 20%,
    length bias is likely.

    **Evidence:** <!-- fill after inspection -->

    **Verdict:** <!-- confirmed / not confirmed -->

    ---

    ## Hypothesis 2 — Residual Position Bias

    **Claim:** Despite the swap-debiasing in B.1, the judge still exhibits
    position preference (first answer benefits from primacy effect).  If Run 1
    and Run 2 frequently disagree, the swap correctly resolves to "tie"; but if
    one order consistently "wins" run1 and the opposite loses run2, the flip
    logic may expose a systematic asymmetry.

    **How to test:** Count rows where `run1_winner != run2_winner_flipped` in
    `pairwise_results.csv`.  High disagreement rate (> 40%) indicates the judge
    is position-sensitive.

    **Evidence:** <!-- fill after inspection -->

    **Verdict:** <!-- confirmed / not confirmed -->

    ---

    ## Hypothesis 3 — Style Bias (Metadata Sentence vs Factual Sentence)

    **Claim:** The Day18 pipeline returns two types of outputs:
    (a) factual extraction ("Thuế GTGT là 52.133.830 đồng") and
    (b) M5-enriched metadata preamble ("Đoạn văn nằm trong phần X của tài liệu Y").
    The LLM judge scores these consistently, but *humans* may rate (b) as
    acceptable context whereas the judge penalises it as non-responsive.
    This stylistic divergence inflates LLM-vs-human disagreement.

    **How to test:** For every disagreement row, classify answer_a and answer_b
    as "factual" or "metadata".  If disagreements cluster on rows where one
    answer is type (b), style bias is the driver.

    **Evidence:** <!-- fill after inspection -->

    **Verdict:** <!-- confirmed / not confirmed -->

    ---

    ## Recommended Fix

    Based on confirmed hypotheses, update the judge prompt in
    `phase-b/pairwise_judge.py` as follows:

    | Hypothesis | Fix |
    | ---------- | --- |
    | Length bias | Add rubric clause: "Do not prefer longer answers; conciseness is valued." |
    | Position bias | Increase swap runs to 3 (majority vote); or add explicit "position does not matter" instruction. |
    | Style bias | Add rubric clause: "An answer that only describes the document structure without extracting the actual answer should score lower than a direct factual answer." |
    """)
    ROOT_MD.write_text(content, encoding="utf-8")
    print(f"[+] Root-cause template written → {ROOT_MD}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # Validate inputs
    for p in (HUMAN_CSV, PAIRWISE_CSV, TO_LABEL_CSV):
        if not p.exists():
            print(_r(f"[ERROR] Missing: {p}"))
            if p == HUMAN_CSV:
                print(_y("  Run sample_for_labeling.py first, then fill human_labels.csv"))
            sys.exit(1)

    # ── Load files ────────────────────────────────────────────────────────────
    human_df    = pd.read_csv(HUMAN_CSV,    encoding="utf-8-sig")
    pairwise_df = pd.read_csv(PAIRWISE_CSV, encoding="utf-8-sig")
    to_label_df = pd.read_csv(TO_LABEL_CSV, encoding="utf-8-sig")

    print(f"[+] human_labels.csv   : {len(human_df)} rows")
    print(f"[+] to_label.csv       : {len(to_label_df)} rows")
    print(f"[+] pairwise_results   : {len(pairwise_df)} rows")

    # ── Merge human labels with LLM winners ───────────────────────────────────
    # to_label.csv has question_id 0..N and the same question text as pairwise_results
    # Join: human_df[question_id] → to_label_df[question_id] → pairwise question row
    merged = (
        human_df
        .merge(to_label_df[["question_id", "question"]], on="question_id", how="inner")
        .merge(pairwise_df[["question", "winner_after_swap"]], on="question", how="inner")
    )

    if len(merged) == 0:
        print(_r("[ERROR] Merge produced 0 rows — check question_id alignment"))
        sys.exit(1)

    print(f"[+] Merged rows: {len(merged)}")

    # ── Normalize labels ──────────────────────────────────────────────────────
    merged["human_norm"] = merged["human_winner"].apply(normalize_label)
    merged["llm_norm"]   = merged["winner_after_swap"].apply(normalize_label)

    bad = merged[merged["human_norm"].isna() | merged["llm_norm"].isna()]
    if len(bad) > 0:
        print(_y(f"[WARN] {len(bad)} rows with unrecognized labels — dropping:"))
        for _, row in bad.iterrows():
            print(f"  id={row.get('question_id')}  human={row.get('human_winner')!r}  "
                  f"llm={row.get('winner_after_swap')!r}")
        merged = merged.dropna(subset=["human_norm", "llm_norm"])

    n = len(merged)
    if n < 2:
        print(_r(f"[ERROR] Need at least 2 valid rows for kappa (got {n})"))
        sys.exit(1)

    y_human = merged["human_norm"].tolist()
    y_llm   = merged["llm_norm"].tolist()

    # ── Agreement table ───────────────────────────────────────────────────────
    agree = sum(h == l for h, l in zip(y_human, y_llm))
    agree_pct = agree / n * 100

    labels = sorted(set(y_human) | set(y_llm))
    print(f"\n  Agreement: {agree}/{n} ({agree_pct:.1f}%)")
    print(f"\n  Label distribution:")
    for lab in ["A", "B", "tie"]:
        h_n = y_human.count(lab)
        l_n = y_llm.count(lab)
        print(f"    {lab:>3}  human={h_n}  llm={l_n}")

    # ── Disagreement detail ───────────────────────────────────────────────────
    disagree_rows = merged[merged["human_norm"] != merged["llm_norm"]]
    if len(disagree_rows) > 0:
        print(f"\n  Disagreements ({len(disagree_rows)}):")
        for _, row in disagree_rows.iterrows():
            q_short = str(row.get("question", ""))[:60]
            print(f"    id={row.get('question_id')}  human={row.get('human_norm')}  "
                  f"llm={row.get('llm_norm')}  q={q_short!r}")

    # ── Cohen's Kappa ──────────────────────────────────────────────────────────
    # Ensure at least 2 classes; kappa is undefined for all-same
    unique_labels = set(y_human) | set(y_llm)
    if len(unique_labels) < 2:
        print(_y("[WARN] All labels identical — kappa is undefined (trivially 1.0 or 0.0)"))
        kappa = 1.0 if set(y_human) == set(y_llm) else 0.0
    else:
        kappa = float(cohen_kappa_score(y_human, y_llm, labels=sorted(unique_labels)))

    interpretation = interpret_kappa(kappa)

    sep = "=" * 56
    print(f"\n{_b(sep)}")
    print(_b("  COHEN'S KAPPA RESULT"))
    print(_b(sep))
    color = _g if kappa >= 0.6 else (_y if kappa >= 0.4 else _r)
    print(f"  κ (kappa)       = {color(f'{kappa:.4f}')}")
    print(f"  Interpretation  = {color(interpretation)}")
    print(f"  n samples       = {n}")
    print(f"  Agreement       = {agree_pct:.1f}%")
    print(_b(sep))

    # Scale reference
    print("\n  Scale reference:")
    for lo, hi, label in [
        (None, 0.0, "< 0.00  chance (worse than random)"),
        (0.00, 0.20, "< 0.20  slight"),
        (0.20, 0.40, "< 0.40  fair"),
        (0.40, 0.60, "< 0.60  moderate"),
        (0.60, 0.80, "< 0.80  substantial"),
        (0.80, None, "≥ 0.80  almost perfect"),
    ]:
        active = (lo is None or kappa >= lo) and (hi is None or kappa < hi)
        prefix = "  ▶" if active else "   "
        print(f"{prefix}  {label}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    result = {
        "kappa":           round(kappa, 6),
        "interpretation":  interpretation,
        "n_samples":       n,
        "agreement_pct":   round(agree_pct, 2),
    }
    KAPPA_JSON.parent.mkdir(parents=True, exist_ok=True)
    KAPPA_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[+] Saved → {KAPPA_JSON}")

    # ── Root-cause template (only if kappa < 0.6) ─────────────────────────────
    if kappa < 0.6:
        print(_y(f"\n[!] kappa={kappa:.4f} < 0.60 — generating root-cause analysis template..."))
        _write_root_cause(kappa)
    else:
        print(_g(f"\n[OK] kappa={kappa:.4f} ≥ 0.60 — no root-cause template needed"))

    print()


if __name__ == "__main__":
    main()
