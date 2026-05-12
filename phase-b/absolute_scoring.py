#!/usr/bin/env python3
"""Phase B.2 — Absolute scoring with 4-dimension rubric.

Scores Config A answers (RERANK_TOP_K=3) across 4 dimensions using
gpt-4o-mini as judge, on a 1–5 integer scale.

Dimensions
----------
  accuracy    — factual correctness relative to the question
  relevance   — how directly the answer addresses what was asked
  conciseness — appropriate length (not padded, not truncated)
  helpfulness — practical value to someone who asked the question

Outputs
-------
  phase-b/absolute_scores.csv  columns:
    question, answer, accuracy, relevance, conciseness, helpfulness, overall

Usage
-----
  python phase-b/absolute_scoring.py
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
load_dotenv(_root / ".env")

_DAY18 = Path(
    os.environ.get(
        "DAY18_PIPELINE_PATH",
        r"d:\Nguyen_Duc_Hoang_Phuc\aithucchien\Day18\Day18-Track3-Production-RAG",
    )
)

TESTSET_CSV = _root / "phase-a" / "testset_v1.csv"
OUTPUT_CSV  = _root / "phase-b" / "absolute_scores.csv"

N_SAMPLE     = 30
RANDOM_STATE = 42
JUDGE_MODEL  = "gpt-4o-mini"   # locked
TOP_K_A      = 3               # Config A — baseline

DIMS = ["accuracy", "relevance", "conciseness", "helpfulness"]

# ── ANSI helpers ───────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR", "")

def _g(s: str) -> str: return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _r(s: str) -> str: return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _y(s: str) -> str: return f"\033[93m{s}\033[0m" if _USE_COLOR else s
def _b(s: str) -> str: return f"\033[1m{s}\033[0m"  if _USE_COLOR else s


# ── Day18 pipeline bootstrap (same as B.1) ────────────────────────────────────
def _load_day18():
    day18_root = str(_DAY18)
    if day18_root not in sys.path:
        sys.path.insert(0, day18_root)
    try:
        from src.pipeline import _best_sentence_answer, build_pipeline  # type: ignore
        return build_pipeline, _best_sentence_answer
    except ImportError as exc:
        print(_r(f"[ERROR] Cannot import Day18 pipeline from '{day18_root}': {exc}"))
        print(_y("  Set DAY18_PIPELINE_PATH env var to the Day18 repo root."))
        sys.exit(2)


def _run_query(query: str, search, reranker, best_sentence_fn, top_k: int) -> tuple[str, list[str]]:
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=top_k)
    contexts = [r.text for r in reranked] if reranked else [d["text"] for d in docs[:top_k]]
    answer = best_sentence_fn(query, contexts)
    return answer, contexts


# ── Rubric prompt ──────────────────────────────────────────────────────────────
_RUBRIC_SYSTEM = (
    "You are a meticulous evaluator for RAG-generated answers. "
    "Score strictly and objectively. Respond with ONLY valid JSON, no prose."
)

_RUBRIC_TEMPLATE = """\
Score the following answer on four dimensions using a 1–5 integer scale.

Scoring rubric:
  1 = very poor   2 = poor   3 = acceptable   4 = good   5 = excellent

Dimensions:
  accuracy    — Are all facts in the answer correct relative to the question?
                1=contains clear errors  3=mostly correct  5=fully correct
  relevance   — Does the answer directly address what was asked?
                1=completely off-topic  3=partially on-topic  5=perfectly on-point
  conciseness — Is the answer the right length (not padded, not truncated)?
                1=extremely verbose or uselessly short  3=acceptable length  5=ideal length
  helpfulness — Would this answer actually help the person who asked?
                1=not helpful at all  3=somewhat helpful  5=very helpful

Question:
{question}

Answer:
{answer}

Respond with ONLY valid JSON (integer scores, plus optional overall as float):
{{"accuracy": <1-5>, "relevance": <1-5>, "conciseness": <1-5>, "helpfulness": <1-5>, "overall": <1.0-5.0>}}"""


# ── JSON parser ───────────────────────────────────────────────────────────────
def _parse_scores(raw: str) -> dict[str, float] | None:
    """Return dict with DIMS keys (float) + optional 'overall', or None on failure."""
    text = raw.strip()

    # Strip markdown fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    # Attempt full JSON parse
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract embedded JSON object
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    scores: dict[str, float] = {}
    for dim in DIMS:
        raw_val = obj.get(dim)
        if raw_val is None:
            return None
        try:
            v = float(raw_val)
            if not (1.0 <= v <= 5.0):
                v = max(1.0, min(5.0, v))   # clamp to valid range
            scores[dim] = v
        except (TypeError, ValueError):
            return None

    # Optional overall from LLM; compute from mean if absent or invalid
    try:
        ov = float(obj.get("overall", math.nan))
        if not (1.0 <= ov <= 5.0):
            raise ValueError
        scores["overall"] = ov
    except (TypeError, ValueError):
        scores["overall"] = float(np.mean([scores[d] for d in DIMS]))

    return scores


def _call_rubric_judge(question: str, answer: str, client, model: str = JUDGE_MODEL) -> dict[str, float]:
    """Score one (question, answer) pair.  Returns scores dict or all-3 fallback."""
    fallback = {d: 3.0 for d in DIMS}
    fallback["overall"] = 3.0

    prompt = _RUBRIC_TEMPLATE.format(question=question, answer=answer)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RUBRIC_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        scores = _parse_scores(raw)
        if scores is None:
            print(_y(f"  [WARN] Parse failed for: {raw[:120]!r} — using fallback 3.0"))
            return fallback
        return scores
    except Exception as exc:
        print(_y(f"  [WARN] Judge API error: {exc!r} — using fallback 3.0"))
        return fallback


# ── Summary stats ─────────────────────────────────────────────────────────────
def _print_summary(df: pd.DataFrame) -> None:
    cols = DIMS + ["overall"]
    sep = "=" * 62
    print(f"\n{_b(sep)}")
    print(_b(f"  ABSOLUTE SCORING SUMMARY   model={JUDGE_MODEL}   n={len(df)}"))
    print(_b(sep))
    print(f"  {'Dimension':<14}  {'Mean':>5}  {'Std':>5}  {'Min':>5}  {'Max':>5}  Bar (mean)")
    print(f"  {'-'*14}  {'-----':>5}  {'-----':>5}  {'-----':>5}  {'-----':>5}  ----------")

    means: dict[str, float] = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        mean_v = float(series.mean())
        std_v  = float(series.std())
        min_v  = float(series.min())
        max_v  = float(series.max())
        means[col] = mean_v
        bar_len = int(round((mean_v - 1) / 4 * 20))  # scale 1-5 → 0-20 chars
        bar = "█" * bar_len + "░" * (20 - bar_len)
        label = _b(col) if col == "overall" else col
        color = _g if mean_v >= 4.0 else (_r if mean_v < 3.0 else _y)
        print(f"  {label:<14}  {color(f'{mean_v:.2f}'):>5}  {std_v:.2f}  {min_v:.1f}  {max_v:.1f}  {bar}")

    # Weakest dimension (excluding overall)
    dim_means = {d: means[d] for d in DIMS}
    weakest = min(dim_means, key=lambda d: dim_means[d])
    print(_b(sep))
    print(f"\n  Weakest dimension: {_r(weakest)} ({dim_means[weakest]:.2f}/5.00)")
    print(f"  Strongest dimension: {_g(max(dim_means, key=lambda d: dim_means[d]))} "
          f"({max(dim_means.values()):.2f}/5.00)")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    from openai import OpenAI

    client = OpenAI()

    # ── Load & sample testset (identical to B.1) ───────────────────────────────
    print(f"[+] Loading testset from {TESTSET_CSV}")
    df_all = pd.read_csv(TESTSET_CSV, encoding="utf-8-sig")
    df = df_all.sample(n=min(N_SAMPLE, len(df_all)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"[+] {len(df)} questions sampled (random_state={RANDOM_STATE})")

    # ── Build pipeline ─────────────────────────────────────────────────────────
    print("\n[+] Building Day18 RAG pipeline...")
    build_pipeline, best_sentence_fn = _load_day18()
    t0 = time.perf_counter()
    search, reranker = build_pipeline()
    print(f"[+] Pipeline ready in {time.perf_counter()-t0:.1f}s\n")

    # ── Scoring loop ───────────────────────────────────────────────────────────
    sep = "─" * 64
    print(_b(sep))
    print(_b(f"  RUBRIC SCORING   model={JUDGE_MODEL}   top_k={TOP_K_A}   n={len(df)}"))
    print(_b(sep))

    records = []
    for idx, row in df.iterrows():
        question = str(row["question"])
        print(f"\n[{len(records)+1:02d}/{len(df)}] {question[:72]}")

        # Get Config A answer
        try:
            answer, _ = _run_query(question, search, reranker, best_sentence_fn, TOP_K_A)
        except Exception as exc:
            print(_y(f"  [SKIP] RAG error: {exc!r}"))
            continue

        print(f"  ans: {answer[:90]!r}")

        # Score with rubric
        scores = _call_rubric_judge(question, answer, client)
        score_str = "  ".join(f"{d[0].upper()}={scores[d]:.0f}" for d in DIMS)
        print(f"  {score_str}  overall={scores['overall']:.2f}")

        records.append({
            "question":    question,
            "answer":      answer,
            "accuracy":    scores["accuracy"],
            "relevance":   scores["relevance"],
            "conciseness": scores["conciseness"],
            "helpfulness": scores["helpfulness"],
            "overall":     scores["overall"],
        })

        time.sleep(0.2)   # gentle rate-limit guard

    # ── Save ───────────────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[+] Saved {len(results_df)} rows → {OUTPUT_CSV}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    _print_summary(results_df)


if __name__ == "__main__":
    main()
