"""Phase A.2 — RAGAS Evaluation of Day 18 RAG pipeline.

Loads testset_v1.csv, runs every question through the Day 18 hybrid RAG
pipeline (HybridSearch + CrossEncoderReranker), then scores the results with
four RAGAS 0.4.x metrics using gpt-4o-mini as judge.

Usage:
    python phase-a/run_ragas.py

Outputs:
    phase-a/ragas_results.csv   -- per-question scores (all 4 metrics)
    phase-a/ragas_summary.json  -- 4 aggregate scores + cost metadata
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 stdout on Windows so Vietnamese characters don't crash print()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from dotenv import load_dotenv

# ─── Paths ─────────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
_day18 = Path(r"d:\Nguyen_Duc_Hoang_Phuc\aithucchien\Day18\Day18-Track3-Production-RAG")
load_dotenv(_root / ".env")

TESTSET_CSV   = _root / "phase-a" / "testset_v1.csv"
RESULTS_CSV   = _root / "phase-a" / "ragas_results.csv"
SUMMARY_JSON  = _root / "phase-a" / "ragas_summary.json"

# ─── Benchmark targets (from task spec) ────────────────────────────────────────
TARGETS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.70,
    "context_recall":    0.75,
}

# ─── gpt-4o-mini pricing (USD per 1 000 tokens) ────────────────────────────────
MODEL_ID          = "gpt-4o-mini"          # locked model version
PRICE_IN_PER_1K   = 0.00015
PRICE_OUT_PER_1K  = 0.00060
MAX_WORKERS       = 2                      # rate-limit guard


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build Day-18 RAG pipeline
# ──────────────────────────────────────────────────────────────────────────────
def _build_rag_pipeline():
    """Import and initialise the Day-18 production pipeline.

    Adds the Day 18 root to sys.path so its internal imports resolve correctly.
    """
    day18_root = str(_day18)
    if day18_root not in sys.path:
        sys.path.insert(0, day18_root)

    # Import pipeline components (these handle their own internal imports)
    from src.pipeline import build_pipeline, run_query  # type: ignore[import]

    print("[+] Building Day-18 RAG pipeline (index + reranker)...")
    t0 = time.perf_counter()
    search, reranker = build_pipeline()
    elapsed = time.perf_counter() - t0
    print(f"[+] Pipeline ready in {elapsed:.1f}s")
    return search, reranker, run_query


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Run pipeline on every testset question
# ──────────────────────────────────────────────────────────────────────────────
def _run_pipeline_on_testset(
    df: pd.DataFrame,
    search,
    reranker,
    run_query,
) -> list[dict]:
    """Return list of dicts with question / answer / retrieved_contexts columns.

    Graceful: if a question fails, it is logged and skipped (no crash).
    """
    rows: list[dict] = []
    total = len(df)
    for i, row in df.iterrows():
        q = str(row["question"])
        try:
            answer, contexts = run_query(q, search, reranker)
            rows.append({
                "question":           q,
                "answer":             answer,
                "retrieved_contexts": contexts,
                "ground_truth":       str(row["ground_truth"]),
                "reference_contexts": json.loads(row["contexts"]) if isinstance(row["contexts"], str) else [],
            })
        except Exception as exc:
            print(f"  [SKIP] Q#{i} error: {exc!r}")
            traceback.print_exc()
        print(f"  [{len(rows)}/{total}] {q[:70]}", end="\r", flush=True)
    print()
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 3.  RAGAS evaluation
# ──────────────────────────────────────────────────────────────────────────────
def _build_eval_dataset(rows: list[dict]):
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = []
    for r in rows:
        # retrieved_contexts: what the RAG actually returned  (for faithfulness, AR, CP)
        # reference_contexts: gold contexts from testset       (for context_recall)
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["retrieved_contexts"] if r["retrieved_contexts"] else [""],
                reference_contexts=r["reference_contexts"] if r["reference_contexts"] else [""],
                reference=r["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


def _run_ragas(eval_ds, llm, embeddings):
    """Evaluate with RAGAS 0.4.x.

    RAGAS 0.4.x has two metric class hierarchies:
    - ragas.metrics.*  (old singletons) → satisfy isinstance(m, Metric) check
    - ragas.metrics.collections.* (new) → do NOT satisfy the check

    The evaluate() validator requires isinstance(m, Metric), so we use the old
    singletons and pre-assign the judge LLM + embeddings manually.
    """
    import warnings
    from ragas import evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.run_config import RunConfig

    # Suppress the deprecation warnings for old-path imports
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (  # type: ignore[attr-defined]
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

    # Pre-assign the judge LLM + embeddings (RAGAS cast-assigns at runtime)
    for m in [faithfulness, context_precision, context_recall]:
        m.llm = llm  # type: ignore[assignment]
    answer_relevancy.llm = llm  # type: ignore[assignment]
    answer_relevancy.embeddings = embeddings  # type: ignore[assignment]

    run_config = RunConfig(max_workers=MAX_WORKERS, max_retries=5, timeout=180)

    print(f"\n[+] Running RAGAS evaluate on {len(eval_ds)} samples "
          f"(judge: {MODEL_ID})...")
    t0 = time.perf_counter()
    # Pass llm=None so evaluate() does not override our pre-assigned llm
    result = evaluate(
        dataset=eval_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=None,
        embeddings=None,
        run_config=run_config,
        token_usage_parser=get_token_usage_for_openai,
        raise_exceptions=False,
        show_progress=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"[+] Evaluation done in {elapsed:.1f}s")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Cost extraction helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extract_cost(result) -> dict:
    """Extract token counts and estimate USD cost from EvaluationResult."""
    cb = result.cost_cb
    if cb is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}

    try:
        usage = cb.total_tokens()
        # total_tokens() returns a single TokenUsage or a list
        if isinstance(usage, list):
            inp  = sum(u.input_tokens  for u in usage)
            out  = sum(u.output_tokens for u in usage)
        else:
            inp  = usage.input_tokens
            out  = usage.output_tokens
    except Exception:
        inp, out = 0, 0

    cost = (inp * PRICE_IN_PER_1K + out * PRICE_OUT_PER_1K) / 1000
    return {"input_tokens": inp, "output_tokens": out, "total_cost_usd": round(cost, 6)}


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    from openai import OpenAI
    from ragas.embeddings import OpenAIEmbeddings as RagasEmbeddings
    from ragas.llms import llm_factory

    # ── LLM setup (locked: gpt-4o-mini) ───────────────────────────────────────
    openai_client = OpenAI()
    llm = llm_factory(MODEL_ID, provider="openai", client=openai_client)

    # Old-style answer_relevancy needs .embed_query() (LangChain interface)
    from langchain_openai import OpenAIEmbeddings as LCEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    embeddings = LangchainEmbeddingsWrapper(
        LCEmbeddings(model="text-embedding-3-small")
    )
    # Also keep native embeddings for building eval dataset (not currently used)
    _native_embeddings = RagasEmbeddings(
        client=openai_client, model="text-embedding-3-small"
    )

    # ── Load testset ───────────────────────────────────────────────────────────
    print(f"[+] Loading testset from {TESTSET_CSV}")
    df = pd.read_csv(TESTSET_CSV, encoding="utf-8-sig")
    print(f"[+] {len(df)} questions loaded")

    # ── Build RAG pipeline ─────────────────────────────────────────────────────
    search, reranker, run_query = _build_rag_pipeline()

    # ── Run pipeline on all questions ──────────────────────────────────────────
    print("\n[+] Running RAG pipeline on testset...")
    rows = _run_pipeline_on_testset(df, search, reranker, run_query)
    print(f"[+] {len(rows)}/{len(df)} questions answered "
          f"({len(df)-len(rows)} skipped)")
    if not rows:
        sys.exit("[ERROR] No rows to evaluate — check RAG pipeline.")

    # ── Build RAGAS EvaluationDataset ──────────────────────────────────────────
    eval_ds = _build_eval_dataset(rows)

    # ── RAGAS evaluate ─────────────────────────────────────────────────────────
    result = _run_ragas(eval_ds, llm, embeddings)

    # ── Per-question scores CSV ────────────────────────────────────────────────
    scores_df = result.to_pandas()
    # Attach original question column for readability
    scores_df.insert(0, "question_text",
                     [r["question"] for r in rows[:len(scores_df)]])
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"[+] Per-question scores saved -> {RESULTS_CSV}")

    # ── Aggregate scores ───────────────────────────────────────────────────────
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    aggregates: dict[str, float] = {}
    for m in metric_names:
        col_values = [v for v in scores_df.get(m, pd.Series(dtype=float)) if v is not None and str(v) != "nan"]
        aggregates[m] = round(float(sum(col_values) / len(col_values)), 4) if col_values else 0.0

    # ── Cost ───────────────────────────────────────────────────────────────────
    cost_info = _extract_cost(result)

    # ── Summary JSON ───────────────────────────────────────────────────────────
    summary = {
        "model":         MODEL_ID,
        "num_questions": len(rows),
        "scores":        aggregates,
        "cost":          cost_info,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[+] Summary saved -> {SUMMARY_JSON}")

    # ── Benchmark comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RAGAS BENCHMARK  (judge: {MODEL_ID}, n={len(rows)})")
    print("=" * 60)
    print(f"{'Metric':<22} {'Score':>7}  {'Target':>7}  {'Status'}")
    print("-" * 60)
    for m, target in TARGETS.items():
        score  = aggregates.get(m, 0.0)
        status = "PASS" if score >= target else ("WARN" if score >= target * 0.7 else "FAIL")
        marker = "[OK] " if status == "PASS" else ("[!]  " if status == "WARN" else "[X]  ")
        print(f"{marker}{m:<20} {score:>7.4f}  {target:>7.2f}")

    print("=" * 60)
    print(f"\nAPI Cost ({MODEL_ID}):")
    print(f"  Input tokens  : {cost_info['input_tokens']:,}")
    print(f"  Output tokens : {cost_info['output_tokens']:,}")
    print(f"  Total cost    : ${cost_info['total_cost_usd']:.4f} USD")
    print()

    # ── Observation for low metrics ────────────────────────────────────────────
    low = {m: v for m, v in aggregates.items() if v < 0.5}
    if low:
        print("Metrics below 0.5 — observations:")
        for m, v in low.items():
            print(f"  [{m}={v:.4f}] "
                  "Likely cause: extractive pipeline returns sentence fragments "
                  "rather than full generative answers, reducing faithfulness/relevancy.")


if __name__ == "__main__":
    main()
