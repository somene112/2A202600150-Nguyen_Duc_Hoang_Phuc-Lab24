"""Phase A.3 — Failure Cluster Analysis of RAGAS bottom-10 questions.

Usage:
    python phase-a/analyze_failures.py

Outputs:
    phase-a/failure_analysis.md
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_root = Path(__file__).resolve().parents[1]
load_dotenv(_root / ".env")

RESULTS_CSV = _root / "phase-a" / "ragas_results.csv"
TESTSET_CSV  = _root / "phase-a" / "testset_v1.csv"
OUTPUT_MD    = _root / "phase-a" / "failure_analysis.md"

METRICS       = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {"faithfulness": "F", "answer_relevancy": "AR",
                 "context_precision": "CP", "context_recall": "CR"}
N_BOTTOM   = 10
N_CLUSTERS = 2

# ── Thresholds: below these a metric is considered "bad" ──────────────────────
BAD_THRESHOLDS = {
    "faithfulness":      0.70,
    "answer_relevancy":  0.60,
    "context_precision": 0.55,
    "context_recall":    0.60,
}

# ── Pre-written technical descriptions keyed by dominant failure metric ───────
FAILURE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "faithfulness": {
        "pattern": "Extractive Answer Misselection — Wrong Sentence Chosen by `_best_sentence_answer()`",
        "root_cause": (
            "Day18 pipeline calls `_best_sentence_answer()` which scores every sentence "
            "in the top-3 chunks by cosine similarity to the query embedding and returns "
            "the single highest-scoring one. For multi-part questions (e.g. "
            "'value AND VAT amount'), the function selects the sentence that best "
            "matches one half of the question, ignoring the other half. The RAGAS "
            "faithfulness judge then finds claims in the reference answer that are "
            "absent from the one-sentence response, driving faithfulness < 0.5."
        ),
        "proposed_fix": (
            "Switch from extractive to generative answers: replace `_best_sentence_answer()` "
            "in `src/pipeline.py` with a `ChatOpenAI(model='gpt-4o-mini', temperature=0)` "
            "call: `answer = llm.predict(f'Based ONLY on these contexts, answer concisely: "
            "{question}\\n\\nContexts:\\n{joined_contexts}')`. "
            "To keep cost low, also raise `RERANK_TOP_K` from 3→5 in `config.py` "
            "so more evidence reaches the answer step. Add sentence-window expansion "
            "(±1 sentences around each retrieved chunk boundary) via a custom "
            "`SentenceWindowRetriever` wrapping `HybridSearch` in `m2_search.py`."
        ),
    },
    "context_recall": {
        "pattern": "Retrieval Miss — BM25 Fails to Recover All Gold Contexts for Multi-Hop Queries",
        "root_cause": (
            "BM25 in `m2_search.py:HybridSearch` scores chunks by exact token overlap "
            "with the query string. Multi-hop and multi-context questions require chunks "
            "from two or more distinct document sections (e.g., 'rights' + 'obligations') "
            "that rarely share tokens with a single query. The dense-fallback embedding "
            "encodes a single query vector that cannot simultaneously bias toward two "
            "separated topic clusters, so at least one gold context is ranked below "
            "`HYBRID_TOP_K=20` and never reaches the reranker."
        ),
        "proposed_fix": (
            "Implement HyDE (Hypothetical Document Embeddings) in `m2_search.py`: "
            "before dense retrieval, call `ChatOpenAI(model='gpt-4o-mini').predict("
            "f'Write a short answer to: {question}')` and embed the hypothetical "
            "answer text with `sentence_transformers.SentenceTransformer` "
            "instead of the raw question. Increase `HYBRID_TOP_K` from 20→35 in "
            "`config.py` to widen the candidate pool before reranking. "
            "Alternatively, use multi-query retrieval: generate 3 sub-questions with "
            "`langchain.retrievers.MultiQueryRetriever` and union the result sets."
        ),
    },
    "context_precision": {
        "pattern": "Context Precision Failure — Irrelevant Chunks Ranked Ahead of Gold by Reranker",
        "root_cause": (
            "With `HYBRID_TOP_K=20`, BM25 floods the reranking pool with term-matching "
            "but semantically distant chunks. `CrossEncoderReranker` in `m3_rerank.py` "
            "uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (English-only) to score "
            "Vietnamese passages, producing unreliable relevance scores. The gold chunk "
            "is retrieved but ranked below the `RERANK_TOP_K=3` cutoff, so context "
            "precision collapses even when recall is adequate."
        ),
        "proposed_fix": (
            "Replace `CrossEncoderReranker` in `m3_rerank.py` with "
            "`CohereRerank(model='rerank-multilingual-v3.0', top_n=5)` from "
            "`langchain_cohere` (pip install langchain-cohere). Set `RERANK_TOP_K=5` "
            "in `config.py`. Add a hard score threshold: in `rerank()`, filter out "
            "chunks where `rerank_score < 0.25` before returning. "
            "Alternatively, use `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` "
            "(multilingual cross-encoder, supports Vietnamese) via `sentence_transformers`."
        ),
    },
    "answer_relevancy": {
        "pattern": "Answer Relevancy Failure — Response Doesn't Address the Actual Question",
        "root_cause": (
            "The extractive `_best_sentence_answer()` picks the sentence with highest "
            "embedding cosine similarity to the question, but for ambiguous or cross-lingual "
            "questions (mixed Vietnamese/English), the embedding space is noisy and the "
            "top-scoring sentence is topically adjacent rather than actually answering "
            "the question. RAGAS answer_relevancy reverse-engineers a question from the "
            "response and measures similarity to the original; if the response answers "
            "a related but different question, the score drops."
        ),
        "proposed_fix": (
            "Add language normalization: use `langdetect.detect(question)` to identify "
            "language, then translate Vietnamese→English with `deep_translator.GoogleTranslator` "
            "before embedding lookup in `_best_sentence_answer()`. "
            "Switch the answer step to `ChatOpenAI(model='gpt-4o-mini', temperature=0)` "
            "with prompt: 'Answer in the SAME language as the question. Use ONLY the "
            "provided contexts. Be concise (max 2 sentences).' "
            "Also add `answer_length_penalty` post-processing: if `len(answer.split()) < 5`, "
            "retry with top-2 chunks concatenated."
        ),
    },
    "mixed": {
        "pattern": "Mixed Multi-Metric Failures — Ambiguous or Under-Specified Question Scope",
        "root_cause": (
            "Questions with vague scope (e.g., 'Hàng hóa là gì và ...' or broad legal "
            "questions asking about 'trách nhiệm' without specifying which party) cause "
            "multiple RAGAS metrics to fail simultaneously: the retriever returns plausible "
            "but semantically scattered chunks (CR and CP drop), the reranker picks a "
            "verbose chunk over the precise one (CP drops further), and the extractive "
            "selector picks an adjacent non-answer sentence (F drops). No single fix "
            "targets all three failure modes."
        ),
        "proposed_fix": (
            "Add a query rewriting step at the top of `run_query()` in `src/pipeline.py`: "
            "use `ChatOpenAI(model='gpt-4o-mini').predict(f'Rewrite to be specific and "
            "unambiguous, keep the same language: {question}')` — cache with "
            "`functools.lru_cache` keyed on the question hash. "
            "Also implement MMR (Maximal Marginal Relevance) diversity in the retrieval "
            "step: after BM25+dense hybrid, apply "
            "`langchain_community.vectorstores.utils.maximal_marginal_relevance` "
            "with `lambda_mult=0.5` to ensure the top-5 chunks cover diverse aspects. "
            "Set `RERANK_TOP_K=5`, `HYBRID_TOP_K=30` in `config.py`."
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
def _get_embeddings(texts: list[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([e.embedding for e in resp.data])


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(f):
        return "N/A"
    return f"{f:.3f}"


def _dominant_failure(cluster_df: pd.DataFrame) -> str:
    """Return key of FAILURE_DESCRIPTIONS for this cluster's most widespread failure.

    Scores each metric by (a) number of rows below its bad-threshold,
    then breaks ties by the cluster's average for that metric.
    Returns the metric that is bad in the most rows (lowest avg if tied).
    Falls back to 'mixed' only when no metric clears its threshold.
    """
    bad_counts: dict[str, int] = {}
    for m in METRICS:
        col = pd.to_numeric(cluster_df[m], errors="coerce").dropna()
        if len(col) == 0:
            bad_counts[m] = 0
            continue
        threshold = BAD_THRESHOLDS.get(m, 0.65)
        bad_counts[m] = int((col < threshold).sum())

    max_count = max(bad_counts.values(), default=0)
    if max_count == 0:
        return "mixed"

    # Among tied candidates, pick the one with the lowest average (worst cluster mean)
    candidates = [m for m, c in bad_counts.items() if c == max_count]
    candidates.sort(
        key=lambda m: float(
            pd.to_numeric(cluster_df[m], errors="coerce").dropna().mean()
            if len(pd.to_numeric(cluster_df[m], errors="coerce").dropna()) > 0
            else 1.0
        )
    )
    return candidates[0]


def _alternate_failure(cluster_df: pd.DataFrame, exclude: str) -> str:
    """Fallback label when exclude is already taken by another cluster.

    Picks the metric with the single worst (minimum) individual score,
    ignoring the excluded metric.
    """
    worst_m = "mixed"
    worst_min = 1.0
    for m in METRICS:
        if m == exclude:
            continue
        col = pd.to_numeric(cluster_df[m], errors="coerce").dropna()
        if len(col) == 0:
            continue
        m_min = float(col.min())
        threshold = BAD_THRESHOLDS.get(m, 0.65)
        if m_min < threshold and m_min < worst_min:
            worst_min = m_min
            worst_m = m
    return worst_m


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[+] Loading {RESULTS_CSV.name}...")
    scores_df = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")
    testset_df = pd.read_csv(TESTSET_CSV, encoding="utf-8-sig")
    print(f"    {len(scores_df)} scored rows, {len(testset_df)} testset rows")

    # Merge evolution_type from testset (deduplicate first to avoid fan-out)
    testset_dedup = testset_df.drop_duplicates(subset=["question"])
    merged = scores_df.merge(
        testset_dedup[["question", "evolution_type"]],
        left_on="question_text",
        right_on="question",
        how="left",
    )

    # Compute avg_score = nanmean(F, AR, CP, CR) per row
    def row_avg(row: pd.Series) -> float:
        vals = [float(row[m]) for m in METRICS
                if m in row and pd.notna(row[m])]
        return float(np.mean(vals)) if vals else 0.0

    merged["avg_score"] = merged.apply(row_avg, axis=1)

    # Bottom N
    bottom = merged.nsmallest(N_BOTTOM, "avg_score").reset_index(drop=True)
    print(f"[+] Bottom-{N_BOTTOM} avg score range: "
          f"{bottom['avg_score'].min():.3f} – {bottom['avg_score'].max():.3f}")

    # Embed and cluster
    print(f"[+] Embedding {N_BOTTOM} questions with text-embedding-3-small...")
    embs = _get_embeddings(bottom["question_text"].tolist())

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    bottom["cluster"] = km.fit_predict(embs)
    sizes = {i: int((bottom["cluster"] == i).sum()) for i in range(N_CLUSTERS)}
    print(f"[+] KMeans(k={N_CLUSTERS}) => " +
          ", ".join(f"C{k}: {v} questions" for k, v in sizes.items()))

    # ── Assign failure labels, ensuring each cluster gets a distinct label ──
    cluster_ids = sorted(bottom["cluster"].unique())
    cluster_labels: dict[int, str] = {}
    used_labels: set[str] = set()
    for cl_id in cluster_ids:
        cl_df = bottom[bottom["cluster"] == cl_id]
        label = _dominant_failure(cl_df)
        if label in used_labels:
            label = _alternate_failure(cl_df, exclude=label)
        cluster_labels[cl_id] = label
        used_labels.add(label)
    print(f"[+] Cluster labels: " +
          ", ".join(f"C{k}={v}" for k, v in cluster_labels.items()))

    # ── Build markdown ──────────────────────────────────────────────────────
    lines: list[str] = []
    lines += [
        "# Failure Cluster Analysis — Phase A.3",
        "",
        f"**Source:** `phase-a/ragas_results.csv`  (n={len(merged)}, bottom-{N_BOTTOM} analyzed)",
        f"**Clustering:** KMeans(k={N_CLUSTERS}) on OpenAI `text-embedding-3-small` of question text",
        f"**Avg score:** nanmean(F, AR, CP, CR) per row — NaN metrics skipped",
        "",
    ]

    # Bottom-10 table
    lines += [
        "## Bottom 10 Questions by Average RAGAS Score",
        "",
        "| # | Question | Type | F | AR | CP | CR | Avg | Cluster |",
        "|---|----------|------|---|----|----|----|----|---------|",
    ]
    for seq, (_, row) in enumerate(bottom.iterrows(), 1):
        q = str(row["question_text"])[:60].replace("|", "/")
        etype = str(row.get("evolution_type", "?"))
        lines.append(
            f"| {seq} | {q} | {etype} | "
            f"{_fmt(row.get('faithfulness'))} | "
            f"{_fmt(row.get('answer_relevancy'))} | "
            f"{_fmt(row.get('context_precision'))} | "
            f"{_fmt(row.get('context_recall'))} | "
            f"{row['avg_score']:.3f} | {int(row['cluster'])} |"
        )

    lines += ["", "---", ""]

    # Cluster sections
    lines += ["## Clusters Identified", ""]

    for cl_id in cluster_ids:
        cl_df = bottom[bottom["cluster"] == cl_id]
        dominant = cluster_labels[cl_id]
        desc = FAILURE_DESCRIPTIONS[dominant]
        etypes = ", ".join(sorted(cl_df["evolution_type"].dropna().unique()))

        lines += [
            f"### Cluster {cl_id} — {desc['pattern']}",
            "",
            f"**Size:** {len(cl_df)} questions &nbsp;|&nbsp; "
            f"**Evolution types:** {etypes or 'N/A'}",
            "",
        ]

        # Metric averages
        lines.append("**Avg metric scores in cluster:**")
        for m, lbl in METRIC_LABELS.items():
            col = pd.to_numeric(cl_df[m], errors="coerce").dropna()
            val = f"{col.mean():.3f}" if len(col) > 0 else "N/A"
            lines.append(f"- **{lbl}** ({m}): {val}")
        lines.append("")

        # Examples (at least 2)
        lines.append("**Examples:**")
        for _, ex in cl_df.head(max(2, len(cl_df))).iterrows():
            q = str(ex["question_text"])[:80].replace("|", "/")
            lines.append(
                f"- `{q}`  "
                f"(F={_fmt(ex.get('faithfulness'))}, "
                f"AR={_fmt(ex.get('answer_relevancy'))}, "
                f"CP={_fmt(ex.get('context_precision'))}, "
                f"CR={_fmt(ex.get('context_recall'))}, "
                f"avg={ex['avg_score']:.3f})"
            )
        lines.append("")

        lines += [
            f"**Root cause:** {desc['root_cause']}",
            "",
            f"**Proposed fix:** {desc['proposed_fix']}",
            "",
            "---",
            "",
        ]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[+] Written -> {OUTPUT_MD}")


if __name__ == "__main__":
    main()
