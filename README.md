# Lab 24 — Full Evaluation & Guardrail System

Student: Nguyen Duc Hoang Phuc  
ID: 2A202600150  
Course: AI Thực Chiến — Day 24

## Overview

End-to-end pipeline for evaluating RAG systems and implementing production-grade guardrails using RAGAS, LangChain, and Presidio.

## Structure

| Folder | Phase | Description |
|--------|-------|-------------|
| `phase-a/` | A | Dataset preparation & baseline RAG |
| `phase-b/` | B | RAGAS evaluation metrics |
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
| faithfulness      | 0.9607 |   0.85 | PASS   |
| answer_relevancy  |    N/A |   0.80 | —      |
| context_precision | 0.9387 |   0.70 | PASS   |
| context_recall    | 0.7500 |   0.75 | PASS   |

**Total API cost:** $0.00 USD (token usage callback returned 0 — RAGAS 0.4.x cost tracking not supported in this configuration)

**Observations:**

- `answer_relevancy` returned NaN for all 53 samples. Root cause: the RAGAS 0.4.x old-style `answer_relevancy` singleton requires a LangChain-compatible embeddings object with `.embed_query()`. The native `RagasOpenAIEmbeddings` does not expose this method; the fix (wrapping with `LangchainEmbeddingsWrapper`) is applied in `scripts/run_eval.py` and `phase-a/run_ragas.py`.
- `faithfulness` (0.9607) and `context_precision` (0.9387) are well above threshold — the Day18 extractive pipeline rarely hallucinates and retrieves highly relevant chunks.
- `context_recall` (0.7500) just meets the 0.75 target; multi-hop reasoning questions are the primary drag (see `phase-a/failure_analysis.md`, Cluster 0).

### Phase B — RAGAS Evaluation
Score the RAG system using RAGAS metrics: faithfulness, answer relevancy, context precision, context recall.

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
