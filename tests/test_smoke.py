import os

os.environ["CITED_BACKEND"] = "tfidf"  # no model downloads in tests

from cited.ingest import build_chunks  # noqa: E402
from cited.pipeline import CitedBot  # noqa: E402


def test_chunks_have_sections_and_no_cross_heading_mixing():
    chunks = build_chunks("section")
    assert len(chunks) >= 15
    assert all(c.section for c in chunks)
    pm = [c for c in chunks if c.doc == "PM_KISAN" and c.section == "Benefit"]
    assert pm and "6,000" in pm[0].text and "Exclusions" not in pm[0].text


def test_answers_with_citation():
    bot = CitedBot(use_llm=False)
    a = bot.ask("What is the maximum yearly deposit in Sukanya Samriddhi?")
    assert not a.refused and a.used
    assert any("1.5 lakh" in h.chunk.text for h in a.sources)


def test_refuses_out_of_corpus():
    bot = CitedBot(use_llm=False)
    assert bot.ask("Who won the 2011 cricket world cup?").refused
