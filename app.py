"""Gradio UI. Run: python app.py   (set ANTHROPIC_API_KEY for LLM answers)"""
import gradio as gr

from cited.generate import llm_available
from cited.pipeline import CitedBot

bot = CitedBot()

EXAMPLES = [
    "How much money does PM-KISAN give each year?",
    "Who is eligible for Atal Pension Yojana?",
    "अटल पेंशन योजना में कौन आवेदन कर सकता है?",
    "What is the current interest rate on Sukanya Samriddhi?",  # not in corpus -> should refuse
]


def respond(question: str):
    if not question.strip():
        return "", ""
    a = bot.ask(question)
    status = f"*confidence {a.confidence:.2f}*" + (" - **refused**" if a.refused else "")
    if not a.sources:
        return f"{a.text}\n\n{status}", ""
    cited = a.used or list(range(1, len(a.sources) + 1))
    src = "\n\n".join(
        f"**[{i}] {a.sources[i - 1].chunk.doc.replace('_', ' ')}** - p.{a.sources[i - 1].chunk.page}"
        f"{' - ' + a.sources[i - 1].chunk.section if a.sources[i - 1].chunk.section else ''}\n\n> "
        f"{a.sources[i - 1].chunk.text}"
        for i in cited if i <= len(a.sources))
    return f"{a.text}\n\n{status}", src


with gr.Blocks(title="Cited") as demo:
    gr.Markdown(
        "# Cited: ask about Indian government schemes\n"
        "Answers come only from the indexed documents, with citations. English and Hindi questions work. "
        f"Mode: `{bot.indexes.backend}` embeddings, {'Claude' if llm_available() else 'extractive (no API key)'} answers.\n\n"
        "*Demo corpus is simplified and may be out of date. Not legal or financial advice.*")
    q = gr.Textbox(label="Your question", placeholder="Ask in English or Hindi")
    btn = gr.Button("Ask", variant="primary")
    ans = gr.Markdown()
    with gr.Accordion("Sources", open=True):
        src = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=q)
    btn.click(respond, q, [ans, src])
    q.submit(respond, q, [ans, src])

if __name__ == "__main__":
    demo.launch()
