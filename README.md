# Cited: a multilingual RAG assistant with citations, refusal, and an evaluation harness

Ask about Indian government schemes in **English or Hindi**. Every answer cites the exact document, page and passage. If the documents don't contain the answer, the bot **refuses instead of guessing**.

Most RAG demos stop at "it works". This one measures *how well* it works: retrieval hit rate, MRR, refusal accuracy, and ablations over chunking, retrieval mode and reranking.

## Architecture

```
PDFs / MD / TXT -> parse (PyMuPDF, optional OCR) -> chunk (fixed | section-aware)
                                                        |
        question --> BM25 ----+                         v
                              +--> RRF fusion --> [reranker] --> top-k passages --> LLM (cited answer)
                   dense  ----+                                          ^
                   (BGE-M3)                                              |
                      +--> best cosine < threshold --> refuse -----------+
```

- **Hybrid search:** BM25 (exact terms like "Section 80C") + multilingual dense embeddings (meaning, and Hindi -> English), fused with reciprocal rank fusion.
- **Refusal gate:** if the best dense similarity is below a tunable threshold, answer "not found". The LLM is also told to output `NOT_FOUND` when the passages don't support an answer.
- **Citations:** the model must tag claims with `[n]`; the UI shows only the cited passages.
- **Graceful degradation:** no API key -> extractive answers; no embedding model -> word-level TF-IDF (offline only, no cross-lingual retrieval).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # optional; without it you get extractive answers
python app.py                       # http://127.0.0.1:7860
python -m cited.cli "Who is eligible for Atal Pension Yojana?"
```

First run downloads BGE-M3 (~2 GB) and caches the embeddings in `data/cache/`.

## Use your own documents

1. Delete the demo files in `data/raw/` (they are simplified summaries, **not official documents**).
2. Drop in official PDFs (PIB, ministry portals, India Code). Scanned PDFs need `pytesseract` + the Tesseract binary with `eng` and `hin` language data.
3. Rewrite `eval/questions.jsonl` for your corpus (aim for 50-100 questions, written from a real user's point of view, plus questions the corpus can't answer).
4. Re-run the eval and tune the refusal threshold.

## Evaluation

```bash
python eval/run_eval.py            # chunking x {bm25, dense, hybrid}
python eval/run_eval.py --rerank   # add cross-encoder reranking (BGE reranker)
python eval/run_eval.py --sweep    # refusal threshold sweep
python eval/run_eval.py --judge    # end-to-end correctness + LLM faithfulness (needs API key)
```

### Results

`eval/results.md` currently holds an **offline baseline** (TF-IDF fallback, no neural embeddings), generated with `CITED_BACKEND=tfidf`. It shows the harness works, and why the neural model matters: Hindi hit@4 is poor without multilingual embeddings.

**To fill in before you publish:** run the commands above with BGE-M3 on your machine and paste your table here.

| config | hit@4 | MRR | hit@4 HI | correct refusal | false refusal |
|---|---|---|---|---|---|
| fixed + bm25 | | | | | |
| fixed + hybrid | | | | | |
| section + hybrid | | | | | |
| section + hybrid + rerank | | | | | |

### Known limitations

- The demo corpus is tiny, so scores are easy to saturate. Use a real corpus before drawing conclusions.
- The confidence gate uses embedding similarity, which can pass an *in-domain but unanswered* question (e.g. "current interest rate" when the docs don't state it). The LLM `NOT_FOUND` instruction is the second line of defence; measure it with `--judge`.
- Tables in PDFs may parse poorly. Check chunks for your own documents.
- Not legal, financial or eligibility advice: always confirm on the official portal.

## Project layout

```
app.py                 Gradio UI
cited/ingest.py        parsing + fixed / section-aware chunking
cited/retrieval.py     BM25, dense, hybrid (RRF), reranker
cited/generate.py      cited answers (Claude or extractive fallback)
cited/pipeline.py      CitedBot: retrieval -> refusal gate -> answer
eval/                  questions.jsonl + run_eval.py (+ results.md)
tests/                 smoke tests (run offline)
```

## Deploy on Hugging Face Spaces

Create a Gradio Space, push this repo, add `ANTHROPIC_API_KEY` as a Space secret. Spaces needs YAML front matter at the top of the README (`sdk: gradio`, `app_file: app.py`); add it to the Space's copy only.

## License

MIT
