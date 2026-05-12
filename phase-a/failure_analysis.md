# Failure Cluster Analysis — Phase A.3

**Source:** `phase-a/ragas_results.csv`  (n=53, bottom-10 analyzed)
**Clustering:** KMeans(k=2) on OpenAI `text-embedding-3-small` of question text
**Avg score:** nanmean(F, AR, CP, CR) per row — NaN metrics skipped

## Bottom 10 Questions by Average RAGAS Score

| # | Question | Type | F | AR | CP | CR | Avg | Cluster |
|---|----------|------|---|----|----|----|----|---------|
| 1 | What is the relationship between the value of goods purchase | reasoning | 1.000 | N/A | 0.000 | 0.000 | 0.333 | 0 |
| 2 | Giá trị hàng hóa trong kỳ là bao nhiêu và thuế GTGT của hàng | simple | 0.000 | N/A | 1.000 | 0.500 | 0.500 | 1 |
| 3 | How does the principle of fairness and transparency apply to | reasoning | 1.000 | N/A | 0.583 | 0.000 | 0.528 | 0 |
| 4 | Dữ liệu cá nhân cơ bản là gì? | multi_context | 0.750 | N/A | 1.000 | 0.000 | 0.583 | 1 |
| 5 | What are the responsibilities of data controllers in relatio | reasoning | 1.000 | N/A | 1.000 | 0.250 | 0.750 | 0 |
| 6 | What responsibilities do data controllers and processors hav | reasoning | 1.000 | N/A | 1.000 | 0.333 | 0.778 | 0 |
| 7 | Thuế giá trị gia tăng là bao nhiêu? | simple | 1.000 | N/A | 0.333 | 1.000 | 0.778 | 1 |
| 8 | Tại sao sự đồng ý của chủ thể dữ liệu là cần thiết trong việ | multi_context | 1.000 | N/A | 1.000 | 0.333 | 0.778 | 1 |
| 9 | Dữ liệu cá nhân cơ bản bao gồm những gì, đặc biệt là ngày th | simple | 1.000 | N/A | 1.000 | 0.500 | 0.833 | 1 |
| 10 | Trong dữ liệu cá nhân cơ bản, số chứng minh nhân dân có vai  | simple | 1.000 | N/A | 1.000 | 0.500 | 0.833 | 1 |

---

## Clusters Identified

### Cluster 0 — Retrieval Miss — BM25 Fails to Recover All Gold Contexts for Multi-Hop Queries

**Size:** 4 questions &nbsp;|&nbsp; **Evolution types:** reasoning

**Avg metric scores in cluster:**
- **F** (faithfulness): 1.000
- **AR** (answer_relevancy): N/A
- **CP** (context_precision): 0.646
- **CR** (context_recall): 0.146

**Examples:**
- `What is the relationship between the value of goods purchased and the VAT obliga`  (F=1.000, AR=N/A, CP=0.000, CR=0.000, avg=0.333)
- `How does the principle of fairness and transparency apply to the processing of p`  (F=1.000, AR=N/A, CP=0.583, CR=0.000, avg=0.528)
- `What are the responsibilities of data controllers in relation to personal data p`  (F=1.000, AR=N/A, CP=1.000, CR=0.250, avg=0.750)
- `What responsibilities do data controllers and processors have in relation to per`  (F=1.000, AR=N/A, CP=1.000, CR=0.333, avg=0.778)

**Root cause:** BM25 in `m2_search.py:HybridSearch` scores chunks by exact token overlap with the query string. Multi-hop and multi-context questions require chunks from two or more distinct document sections (e.g., 'rights' + 'obligations') that rarely share tokens with a single query. The dense-fallback embedding encodes a single query vector that cannot simultaneously bias toward two separated topic clusters, so at least one gold context is ranked below `HYBRID_TOP_K=20` and never reaches the reranker.

**Proposed fix:** Implement HyDE (Hypothetical Document Embeddings) in `m2_search.py`: before dense retrieval, call `ChatOpenAI(model='gpt-4o-mini').predict(f'Write a short answer to: {question}')` and embed the hypothetical answer text with `sentence_transformers.SentenceTransformer` instead of the raw question. Increase `HYBRID_TOP_K` from 20→35 in `config.py` to widen the candidate pool before reranking. Alternatively, use multi-query retrieval: generate 3 sub-questions with `langchain.retrievers.MultiQueryRetriever` and union the result sets.

---

### Cluster 1 — Extractive Answer Misselection — Wrong Sentence Chosen by `_best_sentence_answer()`

**Size:** 6 questions &nbsp;|&nbsp; **Evolution types:** multi_context, simple

**Avg metric scores in cluster:**
- **F** (faithfulness): 0.792
- **AR** (answer_relevancy): N/A
- **CP** (context_precision): 0.889
- **CR** (context_recall): 0.472

**Examples:**
- `Giá trị hàng hóa trong kỳ là bao nhiêu và thuế GTGT của hàng hóa đó là bao nhiêu`  (F=0.000, AR=N/A, CP=1.000, CR=0.500, avg=0.500)
- `Dữ liệu cá nhân cơ bản là gì?`  (F=0.750, AR=N/A, CP=1.000, CR=0.000, avg=0.583)
- `Thuế giá trị gia tăng là bao nhiêu?`  (F=1.000, AR=N/A, CP=0.333, CR=1.000, avg=0.778)
- `Tại sao sự đồng ý của chủ thể dữ liệu là cần thiết trong việc xử lý dữ liệu cá n`  (F=1.000, AR=N/A, CP=1.000, CR=0.333, avg=0.778)
- `Dữ liệu cá nhân cơ bản bao gồm những gì, đặc biệt là ngày tháng năm sinh?`  (F=1.000, AR=N/A, CP=1.000, CR=0.500, avg=0.833)
- `Trong dữ liệu cá nhân cơ bản, số chứng minh nhân dân có vai trò gì và tại sao nó`  (F=1.000, AR=N/A, CP=1.000, CR=0.500, avg=0.833)

**Root cause:** Day18 pipeline calls `_best_sentence_answer()` which scores every sentence in the top-3 chunks by cosine similarity to the query embedding and returns the single highest-scoring one. For multi-part questions (e.g. 'value AND VAT amount'), the function selects the sentence that best matches one half of the question, ignoring the other half. The RAGAS faithfulness judge then finds claims in the reference answer that are absent from the one-sentence response, driving faithfulness < 0.5.

**Proposed fix:** Switch from extractive to generative answers: replace `_best_sentence_answer()` in `src/pipeline.py` with a `ChatOpenAI(model='gpt-4o-mini', temperature=0)` call: `answer = llm.predict(f'Based ONLY on these contexts, answer concisely: {question}\n\nContexts:\n{joined_contexts}')`. To keep cost low, also raise `RERANK_TOP_K` from 3→5 in `config.py` so more evidence reaches the answer step. Add sentence-window expansion (±1 sentences around each retrieved chunk boundary) via a custom `SentenceWindowRetriever` wrapping `HybridSearch` in `m2_search.py`.

---

