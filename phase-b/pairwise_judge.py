#!/usr/bin/env python3
"""Phase B.1 — Pairwise LLM Judge with position-swap debiasing.

Compares two RAG configurations on 30 sampled questions:
  Config A  — RERANK_TOP_K = 3  (baseline)
  Config B  — RERANK_TOP_K = 5  (wider context window)

Each pair is judged twice (A vs B, then B vs A) so that position bias is
cancelled out.  If both runs agree, the consensus winner is recorded; if they
disagree, the result is "tie".

Outputs
-------
  phase-b/pairwise_results.csv   columns:
    question, answer_a, answer_b,
    run1_winner, run2_winner_flipped, winner_after_swap

Usage
-----
  python phase-b/pairwise_judge.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path

# UTF-8 stdout so Vietnamese text doesn't crash on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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

TESTSET_CSV  = _root / "phase-a" / "testset_v1.csv"
OUTPUT_CSV   = _root / "phase-b" / "pairwise_results.csv"

N_SAMPLE   = 30
RANDOM_STATE = 42

# Locked model versions
JUDGE_MODEL = "gpt-4o-mini"          # judge LLM (locked)

TOP_K_A = 3   # Config A — baseline
TOP_K_B = 5   # Config B — wider rerank window

# ── ANSI helpers ───────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR", "")

def _g(s: str) -> str: return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _r(s: str) -> str: return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _y(s: str) -> str: return f"\033[93m{s}\033[0m" if _USE_COLOR else s
def _b(s: str) -> str: return f"\033[1m{s}\033[0m"  if _USE_COLOR else s


# ── Day18 pipeline bootstrap ───────────────────────────────────────────────────
def _load_day18():
    """Add Day18 root to sys.path and import pipeline components."""
    day18_root = str(_DAY18)
    if day18_root not in sys.path:
        sys.path.insert(0, day18_root)
    try:
        from src.pipeline import _best_sentence_answer, build_pipeline  # type: ignore
        from src.m2_search import HybridSearch                           # type: ignore
        from src.m3_rerank import CrossEncoderReranker                   # type: ignore
        return build_pipeline, _best_sentence_answer, HybridSearch, CrossEncoderReranker
    except ImportError as exc:
        print(_r(f"[ERROR] Cannot import Day18 pipeline from '{day18_root}': {exc}"))
        print(_y("  Set DAY18_PIPELINE_PATH env var to the Day18 repo root."))
        sys.exit(2)


def _run_query(
    query: str,
    search,
    reranker,
    best_sentence_fn,
    top_k: int,
) -> tuple[str, list[str]]:
    """Search → rerank(top_k) → extract best-sentence answer."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=top_k)
    contexts = [r.text for r in reranked] if reranked else [d["text"] for d in docs[:top_k]]
    answer = best_sentence_fn(query, contexts)
    return answer, contexts


# ── Judge prompt & JSON parser ─────────────────────────────────────────────────
_JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator for RAG-generated answers. "
    "You must respond with ONLY valid JSON and nothing else."
)

_JUDGE_TEMPLATE = """\
Evaluate which answer better addresses the question. Consider: accuracy, completeness, relevance, and clarity.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Respond with ONLY valid JSON (no prose, no markdown):
{{"winner": "A"}} if Answer A is clearly better
{{"winner": "B"}} if Answer B is clearly better
{{"winner": "tie"}} if both are roughly equivalent"""


def _parse_winner(raw: str) -> str:
    """Robust parser: handles plain JSON, markdown fences, and regex fallback."""
    text = raw.strip()

    # 1. Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # 2. Try direct JSON parse
    try:
        obj = json.loads(text)
        w = str(obj.get("winner", "")).strip().lower()
        if w in ("a", "b", "tie"):
            return w.upper() if w in ("a", "b") else "tie"
    except (json.JSONDecodeError, AttributeError):
        pass

    # 3. Regex fallback: find "winner": "X" pattern
    m = re.search(r'"winner"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        w = m.group(1).strip().lower()
        if w in ("a", "b", "tie"):
            return w.upper() if w in ("a", "b") else "tie"

    # 4. Last resort: look for standalone A / B / tie in the response
    text_lower = text.lower()
    if re.search(r'\b(answer\s+)?a\b', text_lower):
        return "A"
    if re.search(r'\b(answer\s+)?b\b', text_lower):
        return "B"

    print(_y(f"  [WARN] Could not parse winner from: {raw[:120]!r} — defaulting to tie"))
    return "tie"


def _call_judge(
    question: str,
    answer_a: str,
    answer_b: str,
    client,
    model: str = JUDGE_MODEL,
) -> str:
    """Single judge call.  Returns 'A', 'B', or 'tie'."""
    prompt = _JUDGE_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=64,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return _parse_winner(raw)
    except Exception as exc:
        print(_y(f"  [WARN] Judge API error: {exc!r} — defaulting to tie"))
        return "tie"


_FLIP = {"A": "B", "B": "A", "tie": "tie"}


def pairwise_judge_with_swap(
    question: str,
    ans1: str,
    ans2: str,
    client,
    model: str = JUDGE_MODEL,
) -> tuple[str, str, str]:
    """Position-swap pairwise judge.

    Returns
    -------
    (run1_winner, run2_winner_flipped, winner_after_swap)

    Run 1  — A=ans1, B=ans2  → raw winner r1
    Run 2  — A=ans2, B=ans1  → raw winner r2_raw  → flip A↔B → r2_flipped
    Aggregate: if r1 == r2_flipped → consensus winner; else → "tie"
    """
    # Run 1: original order
    r1 = _call_judge(question, ans1, ans2, client, model)

    # Brief pause to avoid hitting rate limits
    time.sleep(0.3)

    # Run 2: swapped order
    r2_raw = _call_judge(question, ans2, ans1, client, model)
    r2_flipped = _FLIP[r2_raw]  # flip because we swapped the positions

    # Aggregate
    winner = r1 if r1 == r2_flipped else "tie"

    return r1, r2_flipped, winner


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    from openai import OpenAI

    client = OpenAI()

    # ── Load & sample testset ──────────────────────────────────────────────────
    print(f"[+] Loading testset from {TESTSET_CSV}")
    df_all = pd.read_csv(TESTSET_CSV, encoding="utf-8-sig")
    print(f"[+] {len(df_all)} total questions")

    df = df_all.sample(n=min(N_SAMPLE, len(df_all)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"[+] Sampled {len(df)} questions (random_state={RANDOM_STATE})")

    # ── Build Day18 pipeline ───────────────────────────────────────────────────
    print("\n[+] Building Day18 RAG pipeline (index once, reuse for both configs)...")
    build_pipeline, best_sentence_fn, _, _ = _load_day18()
    t0 = time.perf_counter()
    search, reranker = build_pipeline()
    print(f"[+] Pipeline ready in {time.perf_counter()-t0:.1f}s\n")

    # ── Pairwise evaluation loop ───────────────────────────────────────────────
    sep = "─" * 64
    print(_b(sep))
    print(_b(f"  PAIRWISE JUDGE   model={JUDGE_MODEL}   n={len(df)}"))
    print(_b(f"  Config A: top_k={TOP_K_A}   Config B: top_k={TOP_K_B}"))
    print(_b(sep))

    records = []
    for i, row in df.iterrows():
        question = str(row["question"])
        print(f"\n[{len(records)+1:02d}/{len(df)}] {question[:72]}")

        # Run RAG for both configs (single search call, two rerank calls)
        try:
            ans_a, _ = _run_query(question, search, reranker, best_sentence_fn, TOP_K_A)
            ans_b, _ = _run_query(question, search, reranker, best_sentence_fn, TOP_K_B)
        except Exception as exc:
            print(_y(f"  [SKIP] RAG error: {exc!r}"))
            continue

        print(f"  A (top_k={TOP_K_A}): {ans_a[:80]!r}")
        print(f"  B (top_k={TOP_K_B}): {ans_b[:80]!r}")

        # Pairwise judge with position-swap debiasing
        r1, r2f, winner = pairwise_judge_with_swap(question, ans_a, ans_b, client)

        color = _g if winner == "B" else (_r if winner == "A" else _y)
        print(f"  run1={r1}  run2_flipped={r2f}  → {color(winner)}")

        records.append({
            "question":            question,
            "answer_a":            ans_a,
            "answer_b":            ans_b,
            "run1_winner":         r1,
            "run2_winner_flipped": r2f,
            "winner_after_swap":   winner,
        })

    # ── Save results ───────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[+] Saved {len(results_df)} rows → {OUTPUT_CSV}")

    # ── Distribution check ─────────────────────────────────────────────────────
    counts = results_df["winner_after_swap"].value_counts()
    print("\n" + _b("=" * 50))
    print(_b("  WINNER DISTRIBUTION (winner_after_swap)"))
    print(_b("=" * 50))
    for label in ["A", "B", "tie"]:
        n = counts.get(label, 0)
        pct = n / len(results_df) * 100
        bar = "█" * int(pct / 4)
        marker = _g if label == "B" else (_r if label == "A" else _y)
        print(f"  {marker(label):>3}  {n:3d} ({pct:5.1f}%)  {bar}")
    print(_b("=" * 50))

    # Bias warning: if one side dominates > 75%
    for label in ["A", "B"]:
        n = counts.get(label, 0)
        if n / len(results_df) > 0.75:
            print(_y(f"\n[WARN] {label} wins {n}/{len(results_df)} ({n/len(results_df)*100:.0f}%) — "
                     f"possible {'baseline' if label=='A' else 'config-B'} dominance or position bias."))

    print()


if __name__ == "__main__":
    main()
