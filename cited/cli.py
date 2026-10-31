"""Quick terminal usage:  python -m cited.cli "How much money does PM-KISAN give?" """
import sys

from .pipeline import CitedBot


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: python -m cited.cli "your question"')
    bot = CitedBot()
    print(f"[backend={bot.indexes.backend} llm={bot.use_llm} chunks={len(bot.indexes.chunks)}]")
    a = bot.ask(" ".join(sys.argv[1:]))
    print(a.text, f"\n(confidence={a.confidence:.2f}, refused={a.refused})")
    for i in a.used:
        h = a.sources[i - 1].chunk
        print(f"  [{i}] {h.doc} p.{h.page} - {h.section}")


if __name__ == "__main__":
    main()
