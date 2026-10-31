"""Parsing + chunking. Two strategies so they can be ablated: fixed windows vs section-aware."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config

HEADING = re.compile(
    r"^(#{1,6}\s+.+|\d+(\.\d+)*[.)]?\s+[A-Z][^.]{3,80}|[A-Z][A-Z0-9 &/\-]{4,80})$"
)
BULLET = re.compile(r"^(\s*[-*•]\s+|\s*\d+[.)]\s+)")


@dataclass
class Page:
    doc: str
    page: int
    text: str


@dataclass
class Chunk:
    id: str
    doc: str
    page: int
    section: str
    text: str

    def index_text(self) -> str:
        """Text used for indexing: body plus doc/section context (helps short chunks)."""
        head = f"{self.doc.replace('_', ' ')} | {self.section}".strip(" |")
        return f"{head}\n{self.text}"


def _ocr(page) -> str:
    try:
        import io

        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="eng+hin")
    except Exception:
        return ""


def load_pages(data_dir: Path | str = config.DATA_DIR) -> list[Page]:
    pages: list[Page] = []
    for p in sorted(Path(data_dir).glob("**/*")):
        suf = p.suffix.lower()
        if suf == ".pdf":
            import fitz  # PyMuPDF

            with fitz.open(p) as d:
                for i, pg in enumerate(d, 1):
                    t = pg.get_text("text")
                    if len(t.strip()) < 20:  # probably a scanned page
                        t = _ocr(pg)
                    pages.append(Page(p.stem, i, t))
        elif suf in {".md", ".txt"}:
            pages.append(Page(p.stem, 1, p.read_text(encoding="utf-8")))
    return pages


def _units(text: str) -> list[tuple[str | None, str]]:
    """Split page text into (heading, paragraph) units. heading is set for heading lines."""
    text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate PDF line breaks
    out: list[tuple[str | None, str]] = []
    buf: list[str] = []

    def flush():
        if buf:
            out.append((None, " ".join(buf).strip()))
            buf.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif HEADING.match(line):
            flush()
            out.append((line.lstrip("# ").strip(), ""))
        elif BULLET.match(raw):
            flush()
            buf.append(line)
        else:
            buf.append(line)
    flush()
    return out


def chunk_section_aware(pages: list[Page], max_chars=config.CHUNK_CHARS,
                        overlap=config.CHUNK_OVERLAP) -> list[Chunk]:
    """Never merges text across headings; splits long sections at paragraph boundaries."""
    chunks: list[Chunk] = []
    section_by_doc: dict[str, str] = {}
    for pg in pages:
        section = section_by_doc.get(pg.doc, "")
        cur: list[str] = []
        fresh = 0  # paragraphs in `cur` that have not been emitted yet

        def emit():
            nonlocal cur, fresh
            if fresh:
                body = " ".join(cur).strip()
                chunks.append(Chunk(f"{pg.doc}-p{pg.page}-{len(chunks)}", pg.doc, pg.page, section, body))
            # carry the last paragraph over as overlap if it is small enough
            cur = [cur[-1]] if cur and len(cur[-1]) <= overlap else []
            fresh = 0

        for heading, para in _units(pg.text):
            if heading is not None:
                emit()
                cur = []
                section = heading
                section_by_doc[pg.doc] = section
                continue
            if cur and len(" ".join(cur)) + len(para) > max_chars:
                emit()
            cur.append(para)
            fresh += 1
        emit()
    return chunks


def chunk_fixed(pages: list[Page], size=config.CHUNK_CHARS, overlap=config.CHUNK_OVERLAP) -> list[Chunk]:
    """Naive sliding window baseline (ignores structure)."""
    chunks: list[Chunk] = []
    for pg in pages:
        t = re.sub(r"\s+", " ", pg.text).strip()
        step = max(size - overlap, 1)
        for start in range(0, len(t), step):
            body = t[start:start + size].strip()
            if body:
                chunks.append(Chunk(f"{pg.doc}-p{pg.page}-{len(chunks)}", pg.doc, pg.page, "", body))
            if start + size >= len(t):
                break
    return chunks


def build_chunks(strategy: str = "section", data_dir=config.DATA_DIR) -> list[Chunk]:
    pages = load_pages(data_dir)
    if not pages:
        raise FileNotFoundError(f"No .pdf/.md/.txt documents found in {data_dir}")
    return chunk_section_aware(pages) if strategy == "section" else chunk_fixed(pages)
