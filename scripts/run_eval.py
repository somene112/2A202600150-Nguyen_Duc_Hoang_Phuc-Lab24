#!/usr/bin/env python3
"""CI/CD RAGAS Eval Gate — check metrics against thresholds, exit 1 on failure.

Modes
-----
Fast (CI default) — load pre-computed results, no API calls needed:
    python scripts/run_eval.py \\
        --threshold faithfulness=0.85 --threshold answer_relevancy=0.80 \\
        --results-csv phase-a/ragas_results.csv

Full pipeline run — requires Day18 codebase + OPENAI_API_KEY:
    python scripts/run_eval.py \\
        --threshold faithfulness=0.85 --threshold answer_relevancy=0.80

Environment variables
---------------------
  OPENAI_API_KEY       OpenAI key (required for full-pipeline and RAGAS judge)
  DAY18_PIPELINE_PATH  Override path to Day18 repo root
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── UTF-8 stdout (prevents Vietnamese character crashes on Windows) ────────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
load_dotenv(_root / ".env")

_DEFAULT_DAY18 = Path(
    os.environ.get("DAY18_PIPELINE_PATH",
                   r"d:\Nguyen_Duc_Hoang_Phuc\aithucchien\Day18\Day18-Track3-Production-RAG")
)

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR", "")

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m" if _USE_COLOR else s

def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m" if _USE_COLOR else s

def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m" if _USE_COLOR else s

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOR else s


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RAGAS eval gate: compare metrics against thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--threshold",
        action="append",
        metavar="KEY=VALUE",
        help="Threshold in 'metric=value' form; repeat for multiple metrics "
             "(e.g. --threshold faithfulness=0.85 --threshold answer_relevancy=0.80)",
    )
    p.add_argument(
        "--results-csv",
        metavar="PATH",
        default=None,
        help="Path to pre-computed ragas_results.csv; skips full pipeline run",
    )
    p.add_argument(
        "--testset-csv",
        metavar="PATH",
        default=str(_root / "phase-a" / "testset_v1.csv"),
        help="Path to testset CSV (used in full-pipeline mode)",
    )
    p.add_argument(
        "--output-csv",
        metavar="PATH",
        default=str(_root / "phase-a" / "ragas_results.csv"),
        help="Where to save per-question scores in full-pipeline mode",
    )
    p.add_argument(
        "--pipeline-path",
        metavar="PATH",
        default=str(_DEFAULT_DAY18),
        help="Path to Day18 RAG repo root (full-pipeline mode only)",
    )
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI judge model (default: gpt-4o-mini)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Max concurrent RAGAS workers (default: 2)",
    )
    return p


def _parse_thresholds(raw: list[str] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in (raw or []):
        try:
            k, v = item.split("=", 1)
            thresholds[k.strip()] = float(v.strip())
        except ValueError:
            sys.exit(f"[ERROR] Bad --threshold format '{item}'. Expected metric=value")
    return thresholds


# ── Full pipeline helpers (mirrors phase-a/run_ragas.py) ─────────────────────
def _build_rag_pipeline(pipeline_path: str):
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    try:
        from src.pipeline import build_pipeline, run_query  # type: ignore[import]
    except ImportError as exc:
        print(_red(f"[ERROR] Cannot import Day18 pipeline from '{pipeline_path}': {exc}"))
        print(_yellow("  Tip: set DAY18_PIPELINE_PATH env var or use --pipeline-path PATH"))
        print(_yellow("  Tip: use --results-csv phase-a/ragas_results.csv for fast mode"))
        sys.exit(2)
    t0 = time.perf_counter()
    search, reranker = build_pipeline()
    print(f"[+] Pipeline ready in {time.perf_counter()-t0:.1f}s")
    return search, reranker, run_query


def _run_pipeline_on_testset(df: pd.DataFrame, search, reranker, run_query) -> list[dict]:
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
                "reference_contexts": json.loads(row["contexts"])
                    if isinstance(row["contexts"], str) else [],
            })
        except Exception as exc:
            print(f"  [SKIP] Q#{i}: {exc!r}")
            traceback.print_exc()
        print(f"  [{len(rows)}/{total}] {q[:70]}", end="\r", flush=True)
    print()
    return rows


def _build_eval_dataset(rows: list[dict]):
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["retrieved_contexts"] or [""],
            reference_contexts=r["reference_contexts"] or [""],
            reference=r["ground_truth"],
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)


def _run_ragas(eval_ds, llm, embeddings, max_workers: int, model_id: str) -> pd.DataFrame:
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.cost import get_token_usage_for_openai

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (  # type: ignore[attr-defined]
            answer_relevancy, context_precision, context_recall, faithfulness,
        )

    for m in [faithfulness, context_precision, context_recall]:
        m.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings

    run_config = RunConfig(max_workers=max_workers, max_retries=5, timeout=180)
    print(f"\n[+] RAGAS evaluate on {len(eval_ds)} samples (judge: {model_id})...")
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
    return result.to_pandas()


def _full_pipeline_run(args: argparse.Namespace) -> pd.DataFrame:
    """Run Day18 RAG + RAGAS evaluation end-to-end (Phase A.2 flow)."""
    from openai import OpenAI
    from ragas.llms import llm_factory
    from langchain_openai import OpenAIEmbeddings as LCEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    client = OpenAI()
    llm = llm_factory(args.model, provider="openai", client=client)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        embeddings = LangchainEmbeddingsWrapper(
            LCEmbeddings(model="text-embedding-3-small")
        )

    testset_path = Path(args.testset_csv)
    if not testset_path.exists():
        sys.exit(f"[ERROR] Testset not found: {testset_path}")

    print(f"[+] Loading testset from {testset_path}")
    df = pd.read_csv(testset_path, encoding="utf-8-sig")
    print(f"[+] {len(df)} questions")

    print("[+] Building Day18 RAG pipeline...")
    search, reranker, run_query = _build_rag_pipeline(args.pipeline_path)

    print("[+] Running RAG pipeline on testset...")
    rows = _run_pipeline_on_testset(df, search, reranker, run_query)
    if not rows:
        sys.exit("[ERROR] No rows answered — check RAG pipeline.")

    eval_ds = _build_eval_dataset(rows)
    scores_df = _run_ragas(eval_ds, llm, embeddings, args.max_workers, args.model)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.insert(0, "question_text",
                     [r["question"] for r in rows[:len(scores_df)]])
    scores_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[+] Scores saved -> {output_path}")
    return scores_df


# ── Threshold comparison and reporting ───────────────────────────────────────
METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _aggregate(scores_df: pd.DataFrame, metric: str) -> float:
    col = pd.to_numeric(scores_df.get(metric, pd.Series(dtype=float)), errors="coerce")
    valid = col.dropna()
    return float(valid.mean()) if len(valid) > 0 else float("nan")


def _print_results(scores: dict[str, float], thresholds: dict[str, float],
                   n: int, model: str) -> bool:
    sep = "=" * 62
    print(f"\n{_bold(sep)}")
    print(_bold(f"  RAGAS EVAL GATE   judge={model}   n={n}"))
    print(_bold(sep))
    print(f"  {'Metric':<24} {'Score':>7}  {'Threshold':>9}  Status")
    print(f"  {'-'*24} {'-------':>7}  {'---------':>9}  ------")

    all_pass = True
    for metric in METRIC_COLS:
        score = scores.get(metric, float("nan"))
        if metric not in thresholds:
            score_str = "N/A  " if np.isnan(score) else f"{score:.4f}"
            print(f"  {'':2}{metric:<24} {score_str:>7}  {'(no gate)':>9}")
            continue
        target = thresholds[metric]
        passed = not np.isnan(score) and score >= target
        if not passed:
            all_pass = False
        if np.isnan(score):
            score_str = "N/A  "
            status_str = _yellow("  SKIP")
        elif passed:
            score_str = f"{score:.4f}"
            status_str = _green("  ✓ PASS")
        else:
            score_str = f"{score:.4f}"
            status_str = _red("  ✗ FAIL")
        print(f"  {metric:<24} {score_str:>7}  {target:>9.2f}  {status_str}")

    print(_bold(sep))
    return all_pass


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    args = _build_parser().parse_args()
    thresholds = _parse_thresholds(args.threshold)

    if not thresholds:
        print(_yellow("[WARN] No --threshold args provided — will report scores but always pass."))

    # Load or compute scores
    if args.results_csv:
        csv_path = Path(args.results_csv)
        if not csv_path.exists():
            sys.exit(f"[ERROR] --results-csv not found: {csv_path}")
        print(f"[+] Loading pre-computed results from {csv_path}")
        scores_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"[+] {len(scores_df)} rows loaded")
        model_label = args.model + " (pre-computed)"
    else:
        print("[+] Full pipeline mode (Day18 RAG + RAGAS evaluate)...")
        scores_df = _full_pipeline_run(args)
        model_label = args.model

    # Compute aggregate scores
    scores = {m: _aggregate(scores_df, m) for m in METRIC_COLS}
    n = len(scores_df)

    all_pass = _print_results(scores, thresholds, n, model_label)

    if all_pass:
        print(_green("\n[OK] All gated metrics passed. Merge allowed.\n"))
        sys.exit(0)
    else:
        print(_red("\n[FAIL] One or more metrics below threshold. Blocking merge.\n"))
        sys.exit(1)


if __name__ == "__main__":
    main()
