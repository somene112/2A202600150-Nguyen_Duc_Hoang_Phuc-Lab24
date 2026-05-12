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

- **Input guardrails**: PII detection (Presidio), prompt injection detection
- **Output guardrails**: hallucination check, toxicity filter, PII scrubbing

### Phase D — Integration

Wire phases A–C into a single pipeline with end-to-end reporting and threshold-based CI gates.

### Bonus — Streamlit Dashboard

Interactive dashboard to visualize evaluation results (`streamlit run demo/app.py`).

## Requirements

See [requirements.txt](requirements.txt) for pinned versions.
See [prompts.md](prompts.md) for all system/user prompts used in experiments.
