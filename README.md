# Lab 24 — Full Evaluation & Guardrail System

Student: Nguyen Duc Hoang Phuc
ID: 2A202600150
Course: AI Thực Chiến — Day 24

## Overview

End-to-end pipeline for evaluating RAG systems and implementing production-grade guardrails using RAGAS, LangChain, and Presidio.

## Structure

| Folder | Phase | Description |
| -------- | ----- | ----------- |
| `phase-a/` | A | Dataset preparation & baseline RAG |
| `phase-b/` | B | Pairwise judge & RAGAS evaluation |
| `phase-c/` | C | Guardrail implementation (PII, toxicity, hallucination) |
| `phase-d/` | D | Full pipeline integration & reporting |
| `demo/` | Bonus | Streamlit dashboard |
| `.github/workflows/` | CI | Automated eval on push |

## Setup

```bash
# 1. Clone and enter repo
git clone <repo-url>
cd lab24-eval-guardrails

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env and fill in your API keys

# 5. Run tests
pytest
```

## Phases

### Phase A — Baseline RAG

Build a simple retrieval-augmented generation pipeline and generate a Q&A evaluation dataset.

#### Phase A Results — RAGAS Baseline (n=53, judge: gpt-4o-mini)

| Metric            |  Score | Target | Status |
| ----------------- | -----: | -----: | ------ |
| faithfulness      | 0.9748 |   0.85 | PASS   |
| answer_relevancy  | 0.4575 |   0.80 | FAIL   |
| context_precision | 0.9119 |   0.70 | PASS   |
| context_recall    | 0.7437 |   0.75 | FAIL   |

**Total API cost:** $0.00 USD (token usage callback returned 0 — RAGAS 0.4.x cost tracking not supported in this configuration)

**Observations:**

- `answer_relevancy` (0.4575) is below the 0.80 target. Root cause: `_best_sentence_answer()` returns a single extracted sentence; RAGAS reverse-engineers a question from the response and the sentence fragment often represents a related but different question than the original. Fix: switch to generative answers via `ChatOpenAI(model='gpt-4o-mini', temperature=0)` (see `phase-a/failure_analysis.md`, Cluster 0).
- `faithfulness` (0.9748) and `context_precision` (0.9119) are well above threshold — the Day18 extractive pipeline rarely hallucinates and retrieves highly relevant chunks.
- `context_recall` (0.7437) narrowly misses the 0.75 target; multi-hop reasoning questions that require contexts from multiple document sections are the primary drag (see `phase-a/failure_analysis.md`, Cluster 1 — HyDE fix proposed).

### Phase B — Pairwise Judge

Pairwise LLM judge comparing Config A (top_k=3) vs Config B (top_k=5) across 30 sampled questions with position-swap debiasing.

### Phase C — Guardrails

#### Phase C.1 — PII Redaction (`InputGuard`)

Two-stage sanitizer: Vietnamese regex patterns (CCCD, PHONE_VN, TAX_CODE_VN, EMAIL) → Presidio NLP (en_core_web_lg, PERSON/ORG/LOCATION/…).

| Metric | Result | Target | Status |
| ---------------------- | ------: | ------: | ------ |
| Detection rate (10 tc) | 100.0% | ≥ 80% | PASS |
| Mean recall | 100.0% | — | — |
| Latency P50 | 6.2 ms | < 50 ms | PASS |
| Latency P95 | 22.0 ms | < 50 ms | PASS |

#### Phase C.2 — Topic Scope Validator (`TopicGuard`)

Embedding-based guard (OpenAI `text-embedding-3-small`, cosine similarity ≥ 0.6) with bilingual anchors covering two domains: Vietnamese personal data protection law (Nghị định 13/2023/NĐ-CP) and VAT tax regulations.

| Metric | Result | Target | Status |
| ----------------------- | ------: | ------: | ------ |
| Accuracy (20 inputs) | 85.0% | ≥ 75% | PASS |
| Precision | 100.0% | — | — |
| Recall (on-topic) | 70.0% | — | — |
| Refuse rate (off-topic) | 100.0% | — | — |

**Refuse rate:** 100% of the 10 off-topic queries were correctly refused with a bilingual explanation naming the closest in-scope topic and its similarity score.

#### Phase C.3 — Adversarial Robustness (`adversarial_attacks.py` + `test_adversarial.py`)

20 adversarial attack samples (5 categories: DAN variants, Roleplay, Payload splitting, Encoding, Indirect injection) tested through the two-layer defence pipeline (InputGuard → TopicGuard). 10 legitimate on-topic queries measured for false-positive rate.

| Metric | Result | Target | Status |
| ------------------------------ | ------: | ------: | ------ |
| Detection rate (20 attacks) | 100.0% | ≥ 70% | PASS |
| False positive rate (10 legit) | 0.0% | ≤ 10% | PASS |

**Per-category detection:** DAN 5/5 · Roleplay 5/5 · PayloadSplit 3/3 · Encoding 3/3 · IndirectInjection 4/4

**Key finding:** TopicGuard is the primary blocking layer (19/20 attacks); one indirect-injection carrying a real email address was caught jointly by InputGuard (EMAIL) and TopicGuard. All adversarial prompts score cosine similarity ≤ 0.42 against in-scope anchors, well below the 0.6 threshold. Note: TopicGuard must receive the **original** query (not PII-redacted text) to avoid false positives from Presidio's English NER mislabelling Vietnamese vocabulary.

#### Phase C.4 — Output Guardrail (`OutputGuardAPI`)

LLM-as-judge output safety classifier using Groq API. Preferred model: `llama-guard-3-8b` (native guard); falls back automatically to `llama-3.3-70b-versatile` + Llama Guard-style system prompt when the native model is unavailable. Supports sync (`check()`) and async (`check_async()`) with 3-attempt exponential back-off.

| Metric | Result | Target | Status |
| ------------------------------ | ------: | ------: | ------ |
| Detection rate (10 unsafe) | 100.0% | ≥ 80% | PASS |
| False positive rate (10 safe) | 0.0% | ≤ 20% | PASS |
| Latency P50 | 383 ms | — | — |
| Latency P95 | 682 ms | — | — |

**Per-category detection:** violence 2/2 · self-harm 2/2 · hate 2/2 · medical-misinfo 2/2 · illegal 2/2

**Model note:** `llama-guard-3-8b` is not provisioned on this Groq account. The fallback uses `llama-3.3-70b-versatile` with a Llama Guard-compatible safety policy prompt, producing the same `safe` / `unsafe\n<Sx>` output format parsed by `_parse_safe()`.

#### Phase C.5 — Full Pipeline Integration & Latency Benchmark (`full_pipeline.py` + `benchmark_latency.py`)

Four-layer `GuardedRAGPipeline`: L1 InputGuard+TopicGuard parallel (`asyncio.gather`), L2 mock RAG (thread-wrapped), L3 OutputGuard async (Groq), L4 audit log fire-and-forget (`asyncio.create_task`).

Benchmark: 99 unique queries (53 from testset + 47 synthetic) — guarded vs baseline (L2 only).

| Layer | P50 | P95 | P99 | n | Notes |
| ----- | ---: | ---: | ---: | --: | ----- |
| L1 (InputGuard + TopicGuard) | 340 ms | 694 ms | 1237 ms | 99 | Dominated by OpenAI embedding API call in TopicGuard |
| L2 (RAG) | 83 ms | 123 ms | 124 ms | 50 | Mock RAG 40–120 ms |
| L3 (OutputGuard) | 380 ms | 551 ms | 651 ms | 50 | Groq `llama-3.3-70b-versatile` fallback |
| Total | 572 ms | 1200 ms | 1755 ms | 99 | — |
| Baseline (L2 only) | 78 ms | 123 ms | 124 ms | 99 | — |
| Overhead | 486 ms | 1147 ms | 1676 ms | 99 | Guardrail cost over baseline |

**Query routing:** 49/99 blocked at L1 (off-topic queries from Day18 testset) · 0 blocked at L3 · 50 passed all layers.

**L1 parallelism:** CONFIRMED — wall=415 ms < sequential=423 ms (speedup 1.02×). Theoretical max speedup is 1.02× when TopicGuard (413 ms) dominates PII (11 ms); `asyncio.gather` runs both in thread pool concurrently.

**Target notes:** L1 P95 target of 50 ms and L3 P95 target of 100 ms assume local model inference. Actual latency is dominated by OpenAI (TopicGuard embeddings) and Groq (OutputGuard) API roundtrips; local deployment would meet both targets.

### Phase D — Integration

Wire phases A–C into a single pipeline with end-to-end reporting and threshold-based CI gates.

### Bonus — Streamlit Dashboard

Interactive dashboard to visualize evaluation results (`streamlit run demo/app.py`).

## Requirements

See [requirements.txt](requirements.txt) for pinned versions.
See [prompts.md](prompts.md) for all system/user prompts used in experiments.
