"""Evaluation harness + ablations.

    python eval/run_eval.py                 # chunking x retrieval-mode grid (retrieval metrics + refusal)
    python eval/run_eval.py --rerank        # also include cross-encoder reranking
    python eval/run_eval.py --sweep         # refusal-threshold sweep for the best config
    python eval/run_eval.py --judge         # end-to-end answers scored by keyword match + LLM faithfulness (needs ANTHROPIC_API_KEY)

Metrics
  hit@k      : a retrieved passage from the gold doc contains a gold keyword
  MRR        : reciprocal rank of the first such passage
  refusal    : share of unanswerable questions correctly refused
  false-ref  : share of answerable questions wrongly refused
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cited import config  # noqa: E402
from cited.ingest import build_chunks  # noqa: E402
from cited.pipeline import CitedBot  # noqa: E402
from cited.retrieval import Indexes, Retriever  # noqa: E402

QFILE = Path(__file__).parent / "questions.jsonl"


def load_questions():
    return [json.loads(l) for l in QFILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_gold(chunk, item) -> bool:
    return chunk.doc == item["gold_doc"] and any(k.lower() in chunk.text.lower() for k in item["keywords"])


def evaluate(retriever: Retriever, questions, min_conf: float) -> dict:
    ans = [q for q in questions if q["answerable"]]
    unans = [q for q in questions if not q["answerable"]]
    stats = {"hit": [], "rr": [], "hit_en": [], "hit_hi": [], "false_ref": [], "ref": []}
    for q in ans:
        hits, conf = retriever.search(q["q"])
        rank = next((i for i, h in enumerate(hits, 1) if is_gold(h.chunk, q)), None)
        hit = rank is not None
        stats["hit"].append(hit)
        stats["rr"].append(1 / rank if rank else 0.0)
        stats["hit_" + q["lang"]].append(hit)
        stats["false_ref"].append(conf < min_conf)
    for q in unans:
        _, conf = retriever.search(q["q"])
        stats["ref"].append(conf < min_conf)
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {"hit": avg(stats["hit"]), "mrr": avg(stats["rr"]), "hit_en": avg(stats["hit_en"]),
            "hit_hi": avg(stats["hit_hi"]), "refusal": avg(stats["ref"]), "false_ref": avg(stats["false_ref"])}


def table(rows: list[tuple[str, dict]]) -> str:
    head = f"| config | hit@{config.TOP_K} | MRR | hit@{config.TOP_K} EN | hit@{config.TOP_K} HI | correct refusal | false refusal |\n|---|---|---|---|---|---|---|\n"
    pct = lambda x: f"{x * 100:.0f}%"  # noqa: E731
    return head + "\n".join(
        f"| {n} | {pct(r['hit'])} | {r['mrr']:.2f} | {pct(r['hit_en'])} | {pct(r['hit_hi'])} | {pct(r['refusal'])} | {pct(r['false_ref'])} |"
        for n, r in rows)


def run_grid(questions, rerank: bool):
    rows = []
    for chunking in ("fixed", "section"):
        ix = Indexes(build_chunks(chunking))
        min_conf = config.MIN_CONF[ix.backend]
        for mode in ("bm25", "dense", "hybrid"):
            for rr in ([False, True] if rerank else [False]):
                r = Retriever(ix, mode, rr)
                rows.append((f"{chunking} + {r.name}", evaluate(r, questions, min_conf)))
    return rows, ix.backend


def sweep(questions):
    ix = Indexes(build_chunks("section"))
    r = Retriever(ix, "hybrid")
    lo, hi = (0.55, 0.65) if ix.backend == "st" else (0.0, 0.3)
    out = []
    for i in range(11):
        t = lo + (hi - lo) * i / 10
        m = evaluate(r, questions, t)
        out.append(f"| {t:.2f} | {m['refusal'] * 100:.0f}% | {m['false_ref'] * 100:.0f}% |")
    return "| threshold | correct refusal | false refusal |\n|---|---|---|\n" + "\n".join(out)


def judge(questions):
    from cited.generate import format_sources, llm_available

    if not llm_available():
        sys.exit("Set ANTHROPIC_API_KEY to use --judge.")
    import anthropic

    client = anthropic.Anthropic()
    bot = CitedBot()
    correct = faithful = refused_ok = n_a = n_u = 0
    for q in questions:
        a = bot.ask(q["q"])
        if not q["answerable"]:
            n_u += 1
            refused_ok += a.refused
            continue
        n_a += 1
        if a.refused:
            continue
        correct += any(k.lower() in a.text.lower() for k in q["keywords"])
        prompt = (f"Sources:\n{format_sources(a.sources)}\n\nAnswer: {a.text}\n\n"
                  "Is every claim in the answer supported by the sources? Reply only YES or NO.")
        v = client.messages.create(model=config.LLM_MODEL, max_tokens=5,
                                   messages=[{"role": "user", "content": prompt}])
        faithful += v.content[0].text.strip().upper().startswith("YES")
    print(f"answer correctness (keyword): {correct}/{n_a}\nfaithfulness (LLM judge): {faithful}/{n_a}\n"
          f"correct refusals: {refused_ok}/{n_u}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    qs = load_questions()
    if args.judge:
        return judge(qs)
    if args.sweep:
        print(sweep(qs))
        return
    rows, backend = run_grid(qs, args.rerank)
    md = f"Embedding backend: `{backend}`  |  questions: {len(qs)}\n\n" + table(rows)
    print(md)
    (Path(__file__).parent / "results.md").write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
