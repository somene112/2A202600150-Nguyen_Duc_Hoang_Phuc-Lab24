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
