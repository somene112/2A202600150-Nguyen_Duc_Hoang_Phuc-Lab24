"""Phase A.1 — Synthetic Test Set Generation with RAGAS 0.4.x and gpt-4o-mini.

Usage:
    python phase-a/generate_testset.py

Outputs:
    phase-a/testset_v1.csv   (≥50 rows, columns: question, ground_truth, contexts, evolution_type)

Handles:
    - OOM: docs larger than MAX_CHARS_PER_DOC are split into sub-chunks
    - RateLimit: RunConfig(max_workers=2) caps concurrent OpenAI calls
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ─── Paths ────────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
load_dotenv(_root / ".env")

CORPUS_DIR = Path(r"d:\Nguyen_Duc_Hoang_Phuc\aithucchien\Day18\Day18-Track3-Production-RAG\data")
OUTPUT_CSV = _root / "phase-a" / "testset_v1.csv"
TESTSET_SIZE = 50
MAX_WORKERS = 2          # rate-limit guard (max_concurrent=2 as required)
MAX_CHARS_PER_DOC = 120_000  # OOM guard: split docs larger than this

# ─── Map RAGAS 0.4.x synthesizer names → canonical evolution_type labels ──────
SYNTH_LABEL: dict[str, str] = {
    "single_hop_specific_query_synthesizer": "simple",
    "multi_hop_abstract_query_synthesizer": "reasoning",
    "multi_hop_specific_query_synthesizer": "multi_context",
}

# ─── GPT-4o-mini pricing (USD per 1 000 tokens, as of mid-2024) ───────────────
PRICE_INPUT_PER_1K = 0.00015
PRICE_OUTPUT_PER_1K = 0.00060


# ──────────────────────────────────────────────────────────────────────────────
def _split_by_headers(text: str, source: str) -> list[dict]:
    """Split markdown into sections by H2 headers (## ...).

    RAGAS 0.4.x needs ≥5 nodes in the knowledge graph to form clusters for
    multi-hop synthesis. Splitting each doc into its sections gives enough
    independent nodes even from a small corpus.
    """
    import re

    sections: list[dict] = []
    current_header = source
    current_lines: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+", line):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append({"text": body, "header": current_header})
            current_header = re.sub(r"^#{1,3}\s+", "", line).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append({"text": body, "header": current_header})
    return sections


def load_documents(corpus_dir: Path) -> list:
    """Load markdown/text docs; split by sections for richer knowledge graph.

    RAGAS 0.4.x builds a knowledge graph from the supplied documents. With only
    2 large docs, multi-hop synthesizers cannot form entity clusters. Splitting
    each document into its H2 sections (typically 5-10 per file) gives enough
    nodes while preserving coherent context within each node.

    OOM guard: any section larger than MAX_CHARS_PER_DOC is further split.
    """
    from langchain_core.documents import Document as LCDocument

    docs: list[LCDocument] = []
    for pattern in ("*.md", "*.txt"):
        for fp in sorted(corpus_dir.glob(pattern)):
            raw = fp.read_text(encoding="utf-8").strip()
            if not raw:
                continue

            sections = _split_by_headers(raw, fp.name)
            # Fallback: if no headers found, treat whole file as one doc
            if not sections:
                sections = [{"text": raw, "header": fp.name}]

            for sec in sections:
                text = sec["text"]
                # OOM guard: split very large sections
                if len(text) > MAX_CHARS_PER_DOC:
                    parts = [
                        text[i : i + MAX_CHARS_PER_DOC]
                        for i in range(0, len(text), MAX_CHARS_PER_DOC)
                    ]
                    for idx, part in enumerate(parts):
                        docs.append(
                            LCDocument(
                                page_content=part,
                                metadata={
                                    "source": fp.name,
                                    "section": f"{sec['header']}_part{idx}",
                                },
                            )
                        )
                else:
                    docs.append(
                        LCDocument(
                            page_content=text,
                            metadata={"source": fp.name, "section": sec["header"]},
                        )
                    )

    print(f"  -> Expanded corpus to {len(docs)} document sections")
    return docs


def _manual_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_INPUT_PER_1K + completion_tokens * PRICE_OUTPUT_PER_1K) / 1000


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    from openai import OpenAI
    from ragas.cost import CostCallbackHandler, get_token_usage_for_openai
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    # ── LLMs (generator = critic = gpt-4o-mini, native RAGAS 0.4.x API) ──────
    openai_client = OpenAI()
    llm = llm_factory("gpt-4o-mini", provider="openai", client=openai_client)
    embeddings = RagasOpenAIEmbeddings(
        client=openai_client, model="text-embedding-3-small"
    )

    # ── Cost callback (RAGAS built-in) ─────────────────────────────────────────
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)

    # ── Generator (direct constructor, not from_langchain) ────────────────────
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    # ── Distribution: simple=0.5, reasoning=0.25, multi_context=0.25 ──────────
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
    ]

    # ── RunConfig: cap workers to handle RateLimit ─────────────────────────────
    run_config = RunConfig(
        max_workers=MAX_WORKERS,
        max_retries=5,
        max_wait=60,
        timeout=180,
    )

    # ── Load corpus ────────────────────────────────────────────────────────────
    print(f"[+] Loading corpus from: {CORPUS_DIR}")
    docs = load_documents(CORPUS_DIR)
    if not docs:
        sys.exit(f"[ERROR] No documents found in {CORPUS_DIR}")
    print(f"[+] Loaded {len(docs)} document(s)")

    # ── Generate testset ───────────────────────────────────────────────────────
    print(f"\n[+] Generating {TESTSET_SIZE} questions "
          f"(max_workers={MAX_WORKERS}, gpt-4o-mini)...")
    t0 = time.perf_counter()

    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=TESTSET_SIZE,
        query_distribution=query_distribution,
        run_config=run_config,
        callbacks=[cost_cb],
        raise_exceptions=False,
    )

    elapsed = time.perf_counter() - t0
    print(f"[+] Done in {elapsed:.1f}s — {len(testset.samples)} samples returned")

    # ── Log API cost ───────────────────────────────────────────────────────────
    print("\n=== API Cost ===")
    if testset.cost_cb is not None:
        cb = testset.cost_cb
        total_cost = cb.total_cost()
        total_tok = cb.total_tokens()
        print(f"  Total tokens : {total_tok:,}")
        print(f"  Total cost   : ${total_cost:.4f} USD")
    else:
        # Fallback: compute from cost_cb we passed
        prompt_tok = sum(
            getattr(u, "input_tokens", 0) for u in getattr(cost_cb, "_usages", [])
        )
        comp_tok = sum(
            getattr(u, "output_tokens", 0) for u in getattr(cost_cb, "_usages", [])
        )
        est_cost = _manual_cost(prompt_tok, comp_tok)
        print(f"  Prompt tokens     : {prompt_tok:,}")
        print(f"  Completion tokens : {comp_tok:,}")
        print(f"  Estimated cost    : ${est_cost:.4f} USD  "
              f"(gpt-4o-mini @ $0.15/$0.60 per 1M tokens)")

    # ── Build DataFrame with required columns ──────────────────────────────────
    df_raw = testset.to_pandas()
    # Rename to required schema: question, ground_truth, contexts, evolution_type
    df = pd.DataFrame(
        {
            "question": df_raw["user_input"],
            "ground_truth": df_raw["reference"],
            "contexts": df_raw["reference_contexts"].apply(
                lambda x: json.dumps(x if isinstance(x, list) else [], ensure_ascii=False)
            ),
            "evolution_type": df_raw["synthesizer_name"].map(SYNTH_LABEL).fillna(
                df_raw["synthesizer_name"]
            ),
        }
    )

    # Drop rows where question is null (failed generations)
    before = len(df)
    df = df.dropna(subset=["question"]).reset_index(drop=True)
    if len(df) < before:
        print(f"[!] Dropped {before - len(df)} failed samples (null question)")

    print(f"\n[+] Final row count: {len(df)}")
    if len(df) < TESTSET_SIZE:
        print(f"[!] Warning: {len(df)} rows < target {TESTSET_SIZE}. "
              "Consider expanding the corpus or retrying.")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[+] Saved -> {OUTPUT_CSV}")

    # ── Verify distribution ────────────────────────────────────────────────────
    print("\n=== evolution_type distribution ===")
    vc = df["evolution_type"].value_counts()
    print(vc.to_string())
    print(f"\n  Expected: simple~25, reasoning~12-13, multi_context~12-13")
    print(f"  Actual  : {dict(vc)}")

    target_pct = {"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25}
    ok = True
    for label, pct in target_pct.items():
        actual_pct = vc.get(label, 0) / max(len(df), 1)
        deviation = abs(actual_pct - pct)
        status = "OK" if deviation <= 0.10 else "WARN"
        print(f"  [{status}] {label}: {actual_pct:.0%} (target {pct:.0%})")
        if deviation > 0.10:
            ok = False
    if not ok:
        print("[!] Distribution deviates >10% from target. Check corpus size.")


if __name__ == "__main__":
    main()
