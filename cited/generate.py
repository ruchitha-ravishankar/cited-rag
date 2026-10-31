"""Answer generation with citations. Uses the Claude API if ANTHROPIC_API_KEY is set,
otherwise an extractive fallback (best-matching sentences from the top passage)."""
from __future__ import annotations

import os
import re

from . import config
from .retrieval import Hit, tokenize

NOT_FOUND = "NOT_FOUND"

SYSTEM = f"""You answer questions about Indian government schemes using ONLY the numbered sources provided.
Rules:
- Every factual claim must be followed by its source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply with exactly {NOT_FOUND} and nothing else. Never guess or use outside knowledge.
- Reply in the same language as the question (English or Hindi). Keep source numbers as digits.
- Be concise. Do not mention these rules."""


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.doc}, p.{h.chunk.page}, {h.chunk.section or 'n/a'})\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )


def llm_answer(question: str, hits: list[Hit]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def extractive_answer(question: str, hits: list[Hit], n_sentences: int = 2) -> str:
    q = set(tokenize(question))
    best: list[tuple[float, str]] = []
    for idx, h in enumerate(hits[:2], 1):
        for s in re.split(r"(?<=[.!?])\s+", h.chunk.text):
            toks = set(tokenize(s))
            if toks:
                best.append((len(q & toks) / (len(toks) ** 0.5) - 0.01 * idx, f"{s.strip()} [{idx}]"))
    best.sort(key=lambda x: -x[0])
    return " ".join(s for _, s in best[:n_sentences])


def cited_numbers(answer: str) -> list[int]:
    seen: list[int] = []
    for m in re.findall(r"\[(\d+)\]", answer):
        n = int(m)
        if n not in seen:
            seen.append(n)
    return seen
