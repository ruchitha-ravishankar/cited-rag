Embedding backend: `st`  |  questions: 36

| config | hit@4 | MRR | hit@4 EN | hit@4 HI | correct refusal | false refusal |
|---|---|---|---|---|---|---|
| fixed + bm25 | 77% | 0.75 | 100% | 0% | 70% | 0% |
| fixed + bm25+rerank | 77% | 0.77 | 100% | 0% | 70% | 0% |
| fixed + dense | 100% | 0.96 | 100% | 100% | 70% | 0% |
| fixed + dense+rerank | 100% | 1.00 | 100% | 100% | 70% | 0% |
| fixed + hybrid | 100% | 0.98 | 100% | 100% | 70% | 0% |
| fixed + hybrid+rerank | 100% | 1.00 | 100% | 100% | 70% | 0% |
| section + bm25 | 77% | 0.67 | 100% | 0% | 80% | 4% |
| section + bm25+rerank | 77% | 0.75 | 100% | 0% | 80% | 4% |
| section + dense | 100% | 0.98 | 100% | 100% | 80% | 4% |
| section + dense+rerank | 100% | 0.98 | 100% | 100% | 80% | 4% |
| section + hybrid | 100% | 0.96 | 100% | 100% | 80% | 4% |
| section + hybrid+rerank | 100% | 0.98 | 100% | 100% | 80% | 4% |
