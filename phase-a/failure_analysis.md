# Failure Cluster Analysis — Phase A.3

**Source:** `phase-a/ragas_results.csv`  (n=53, bottom-10 analyzed)
**Clustering:** KMeans(k=2) on OpenAI `text-embedding-3-small` of question text
**Avg score:** nanmean(F, AR, CP, CR) per row — NaN metrics skipped

## Bottom 10 Questions by Average RAGAS Score

| # | Question | Type | F | AR | CP | CR | Avg | Cluster |
|---|----------|------|---|----|----|----|----|---------|
| 1 | What is the relationship between the value of goods purchase | reasoning | 1.000 | 0.186 | 0.000 | 0.000 | 0.296 | 0 |
| 2 | How does the principle of fairness and transparency apply to | reasoning | 1.000 | 0.383 | 0.500 | 0.000 | 0.471 | 1 |
| 3 | What responsibilities do data controllers and processors hav | reasoning | 1.000 | 0.442 | 0.500 | 0.333 | 0.569 | 1 |
| 4 | What are the responsibilities of data controllers in relatio | reasoning | 1.000 | 0.585 | 0.500 | 0.250 | 0.584 | 1 |
| 5 | Giá trị hàng hóa trong kỳ là bao nhiêu và thuế GTGT của hàng | simple | 0.500 | 0.345 | 1.000 | 0.500 | 0.586 | 0 |
| 6 | Bên kiểm soát và xử lý dữ liệu cá nhân phải phối hợp với cơ  | reasoning | 1.000 | 0.000 | 1.000 | 0.500 | 0.625 | 1 |
| 7 | Thuế giá trị gia tăng là bao nhiêu? | simple | 1.000 | 0.282 | 0.333 | 1.000 | 0.654 | 0 |
| 8 | Dữ liệu cá nhân cơ bản là gì? | multi_context | 1.000 | 0.664 | 1.000 | 0.000 | 0.666 | 1 |
| 9 | Cái gì là trách nhiệm của bên kiểm soát dữ liệu khi chuyển d | reasoning | 1.000 | 0.403 | 1.000 | 0.333 | 0.684 | 1 |
| 10 | What are the requirements for obtaining consent from data su | multi_context | 1.000 | 0.329 | 1.000 | 0.500 | 0.707 | 1 |

---

## Clusters Identified

### Cluster 0 — Answer Relevancy Failure — Response Doesn't Address the Actual Question

**Size:** 3 questions &nbsp;|&nbsp; **Evolution types:** reasoning, simple

**Avg metric scores in cluster:**
- **F** (faithfulness): 0.833
- **AR** (answer_relevancy): 0.271
- **CP** (context_precision): 0.444
- **CR** (context_recall): 0.500

**Examples:**
- `What is the relationship between the value of goods purchased and the VAT obliga`  (F=1.000, AR=0.186, CP=0.000, CR=0.000, avg=0.296)
- `Giá trị hàng hóa trong kỳ là bao nhiêu và thuế GTGT của hàng hóa đó là bao nhiêu`  (F=0.500, AR=0.345, CP=1.000, CR=0.500, avg=0.586)
- `Thuế giá trị gia tăng là bao nhiêu?`  (F=1.000, AR=0.282, CP=0.333, CR=1.000, avg=0.654)

**Root cause:** The extractive `_best_sentence_answer()` picks the sentence with highest embedding cosine similarity to the question, but for ambiguous or cross-lingual questions (mixed Vietnamese/English), the embedding space is noisy and the top-scoring sentence is topically adjacent rather than actually answering the question. RAGAS answer_relevancy reverse-engineers a question from the response and measures similarity to the original; if the response answers a related but different question, the score drops.

**Proposed fix:** Add language normalization: use `langdetect.detect(question)` to identify language, then translate Vietnamese→English with `deep_translator.GoogleTranslator` before embedding lookup in `_best_sentence_answer()`. Switch the answer step to `ChatOpenAI(model='gpt-4o-mini', temperature=0)` with prompt: 'Answer in the SAME language as the question. Use ONLY the provided contexts. Be concise (max 2 sentences).' Also add `answer_length_penalty` post-processing: if `len(answer.split()) < 5`, retry with top-2 chunks concatenated.

---

### Cluster 1 — Retrieval Miss — BM25 Fails to Recover All Gold Contexts for Multi-Hop Queries

**Size:** 7 questions &nbsp;|&nbsp; **Evolution types:** multi_context, reasoning

**Avg metric scores in cluster:**
- **F** (faithfulness): 1.000
- **AR** (answer_relevancy): 0.401
- **CP** (context_precision): 0.786
- **CR** (context_recall): 0.274

**Examples:**
- `How does the principle of fairness and transparency apply to the processing of p`  (F=1.000, AR=0.383, CP=0.500, CR=0.000, avg=0.471)
- `What responsibilities do data controllers and processors have in relation to per`  (F=1.000, AR=0.442, CP=0.500, CR=0.333, avg=0.569)
- `What are the responsibilities of data controllers in relation to personal data p`  (F=1.000, AR=0.585, CP=0.500, CR=0.250, avg=0.584)
- `Bên kiểm soát và xử lý dữ liệu cá nhân phải phối hợp với cơ quan có thẩm quyền n`  (F=1.000, AR=0.000, CP=1.000, CR=0.500, avg=0.625)
- `Dữ liệu cá nhân cơ bản là gì?`  (F=1.000, AR=0.664, CP=1.000, CR=0.000, avg=0.666)
- `Cái gì là trách nhiệm của bên kiểm soát dữ liệu khi chuyển dữ liệu cá nhân ra nư`  (F=1.000, AR=0.403, CP=1.000, CR=0.333, avg=0.684)
- `What are the requirements for obtaining consent from data subjects for processin`  (F=1.000, AR=0.329, CP=1.000, CR=0.500, avg=0.707)

**Root cause:** BM25 in `m2_search.py:HybridSearch` scores chunks by exact token overlap with the query string. Multi-hop and multi-context questions require chunks from two or more distinct document sections (e.g., 'rights' + 'obligations') that rarely share tokens with a single query. The dense-fallback embedding encodes a single query vector that cannot simultaneously bias toward two separated topic clusters, so at least one gold context is ranked below `HYBRID_TOP_K=20` and never reaches the reranker.

**Proposed fix:** Implement HyDE (Hypothetical Document Embeddings) in `m2_search.py`: before dense retrieval, call `ChatOpenAI(model='gpt-4o-mini').predict(f'Write a short answer to: {question}')` and embed the hypothetical answer text with `sentence_transformers.SentenceTransformer` instead of the raw question. Increase `HYBRID_TOP_K` from 20→35 in `config.py` to widen the candidate pool before reranking. Alternatively, use multi-query retrieval: generate 3 sub-questions with `langchain.retrievers.MultiQueryRetriever` and union the result sets.

---

