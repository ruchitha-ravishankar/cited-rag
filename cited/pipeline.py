from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .generate import NOT_FOUND, cited_numbers, extractive_answer, llm_answer, llm_available
from .ingest import build_chunks
from .retrieval import Hit, Indexes, Retriever

REFUSAL = {
    "en": "I couldn't find this in the provided documents, so I won't guess. "
          "Please check the official scheme portal.",
    "hi": "मुझे यह जानकारी उपलब्ध दस्तावेज़ों में नहीं मिली, इसलिए मैं अनुमान नहीं लगाऊँगा। "
          "कृपया योजना की आधिकारिक वेबसाइट देखें।",
}


def is_hindi(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


@dataclass
class Answer:
    text: str
    refused: bool
    confidence: float
    sources: list[Hit] = field(default_factory=list)
    used: list[int] = field(default_factory=list)  # 1-based indices into `sources` that were cited


class CitedBot:
    def __init__(self, chunking: str = "section", mode: str = "hybrid", rerank: bool = False,
                 min_conf: float | None = None, use_llm: bool = True, indexes: Indexes | None = None):
        self.indexes = indexes or Indexes(build_chunks(chunking))
        self.retriever = Retriever(self.indexes, mode=mode, rerank=rerank)
        self.min_conf = min_conf if min_conf is not None else config.MIN_CONF[self.indexes.backend]
        self.use_llm = use_llm and llm_available()

    def ask(self, question: str) -> Answer:
        lang = "hi" if is_hindi(question) else "en"
        hits, conf = self.retriever.search(question)
        if not hits or conf < self.min_conf:
            return Answer(REFUSAL[lang], True, conf)
        text = llm_answer(question, hits) if self.use_llm else extractive_answer(question, hits)
        if NOT_FOUND in text or not text:
            return Answer(REFUSAL[lang], True, conf, hits)
        return Answer(text, False, conf, hits, cited_numbers(text))
