from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as LangChainRecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - optional dependency.
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter as LangChainRecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - optional dependency.
        LangChainRecursiveCharacterTextSplitter = None

from src.tv1_data.parse_clean import ParsedArticle, ParsedDocument, clean_text

CLAUSE_PATTERN = r"^\d+\.\s+"
POINT_PATTERN = r"^[a-zđ]\)\s+"


@dataclass(slots=True, frozen=True)
class ChunkConfig:
    """Chunking settings aligned with the current report baseline."""

    chunk_size: int = 800
    chunk_overlap: int = 150


class RecursiveCharacterTextSplitterCompat:
    """Lightweight fallback when LangChain is unavailable locally."""

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str] | None = None) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))
        self.separators = separators or ["\n\n", "\n", "; ", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        normalized = clean_text(text)
        if len(normalized) <= self.chunk_size:
            return [normalized] if normalized else []

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk = normalized[start:end]

            if end < len(normalized):
                for separator in self.separators:
                    if not separator:
                        continue
                    split_at = chunk.rfind(separator)
                    if split_at >= max(self.chunk_size // 2, 1):
                        end = start + split_at + len(separator)
                        chunk = normalized[start:end]
                        break

            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(normalized):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks


def _build_text_splitter(chunk_size: int, chunk_overlap: int) -> Any:
    separators = ["\n\n", "\n", "; ", ". ", " ", ""]
    if LangChainRecursiveCharacterTextSplitter is not None:
        return LangChainRecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    return RecursiveCharacterTextSplitterCompat(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def _split_preamble_and_sections(lines: list[str], section_pattern: str) -> tuple[list[str], list[list[str]]]:
    import re

    pattern = re.compile(section_pattern, re.IGNORECASE)
    preamble: list[str] = []
    sections: list[list[str]] = []
    current: list[str] = []
    started = False

    for line in lines:
        if pattern.match(line):
            started = True
            if current:
                sections.append(current)
            current = [line]
            continue

        if started:
            current.append(line)
        else:
            preamble.append(line)

    if current:
        sections.append(current)

    return preamble, sections


def _compose_section_body(article: ParsedArticle, body: str) -> str:
    normalized_body = clean_text(body)
    if not normalized_body:
        return ""

    heading = article.heading
    if not heading:
        return normalized_body
    return f"{heading}\n{normalized_body}".strip()


def _split_with_heading(article: ParsedArticle, body: str, config: ChunkConfig) -> list[str]:
    normalized_body = clean_text(body)
    if not normalized_body:
        return [""]

    heading = article.heading
    if not heading:
        splitter = _build_text_splitter(config.chunk_size, config.chunk_overlap)
        return splitter.split_text(normalized_body)

    available_size = max(200, config.chunk_size - len(heading) - 1)
    available_overlap = min(config.chunk_overlap, max(available_size - 1, 0))
    splitter = _build_text_splitter(available_size, available_overlap)
    body_chunks = splitter.split_text(normalized_body)
    if not body_chunks:
        return [""]

    return [f"{heading}\n{chunk}".strip() for chunk in body_chunks]


def _build_article_sections(article: ParsedArticle) -> list[str]:
    lines = [line for line in clean_text(article.raw_content).splitlines() if line.strip()]
    if not lines:
        return [""]

    article_preamble, clauses = _split_preamble_and_sections(lines, CLAUSE_PATTERN)
    if not clauses:
        return ["\n".join(lines)]

    article_preamble_text = "\n".join(article_preamble).strip()
    sections: list[str] = []
    for clause in clauses:
        clause_header = clause[0]
        clause_tail = clause[1:]
        clause_preamble, points = _split_preamble_and_sections(clause_tail, POINT_PATTERN)

        if not points:
            body_parts = [article_preamble_text, "\n".join(clause)]
            sections.append("\n".join(part for part in body_parts if part).strip())
            continue

        clause_preamble_text = "\n".join(clause_preamble).strip()
        for point in points:
            body_parts = [
                article_preamble_text,
                clause_header,
                clause_preamble_text,
                "\n".join(point),
            ]
            sections.append("\n".join(part for part in body_parts if part).strip())

    return sections or ["\n".join(lines)]


def _build_metadata(article: ParsedArticle) -> dict[str, Any]:
    return {
        "de_muc": article.de_muc,
        "file_id": article.file_id,
        "article_code": article.article_code,
        "article_name": article.article_name,
        "mapc": article.mapc,
        "law_id": article.law_id,
        "topic_id": article.topic_id,
        "title": article.title,
        "article": article.article,
        "issuer": article.issuer,
        "effective_date": article.effective_date,
        "source_note": article.source_note,
        "related_articles": list(article.related_articles or []),
    }


def chunk_article(article: ParsedArticle, config: ChunkConfig | None = None) -> list[dict[str, Any]]:
    """Chunk a parsed article following Điều / Khoản / Điểm boundaries first."""

    chunk_config = config or ChunkConfig()
    metadata = _build_metadata(article)
    chunk_records: list[dict[str, Any]] = []

    for section in _build_article_sections(article):
        normalized_section = clean_text(section)
        split_sections = _split_with_heading(article, normalized_section, chunk_config)
        for content in split_sections:
            chunk_records.append(
                {
                    "content": content,
                    "metadata": metadata.copy(),
                }
            )

    return chunk_records or [{"content": "", "metadata": metadata}]


def chunk_document(document: ParsedDocument, config: ChunkConfig | None = None) -> list[dict[str, Any]]:
    """Chunk every parsed article in a document into TV1-ready Qdrant records."""

    chunk_config = config or ChunkConfig()
    all_chunks: list[dict[str, Any]] = []
    for article in document.articles:
        all_chunks.extend(chunk_article(article, chunk_config))
    return all_chunks


__all__ = ["ChunkConfig", "chunk_article", "chunk_document"]
