# LinkedIn post draft (fill in the X/Y numbers from YOUR eval run)

I built a RAG assistant over Indian government scheme documents. It answers in English and Hindi, cites the exact page, and says "I don't know" when the documents don't cover the question.

The part I found most useful wasn't the chatbot, it was the evaluation:

- Hybrid search (BM25 + BGE-M3 embeddings) lifted hit@4 from X% to Y% over embeddings alone
- Section-aware chunking mattered more than adding a reranker: [your finding]
- Hindi questions only work with multilingual embeddings: TF-IDF got Z% on Hindi
- The hardest failure: questions that sound in-scope but aren't answered in the documents (e.g. "current interest rate"). Here's what I did about it: [your finding]

Stack: PyMuPDF, rank_bm25, BGE-M3, Claude API, Gradio.

Demo: [Hugging Face Space link]
Code and eval harness: [GitHub link]

#RAG #NLP #MachineLearning #LLM #India

---
Tips: attach the ablation table as an image, and one screenshot of a funny failure case.
