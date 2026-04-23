from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

ARTICLE_PATTERN = re.compile(r"(Điều\s+[0-9A-Za-z.\-]+)", re.IGNORECASE)
KHOAN_PATTERN = re.compile(r"(Khoản\s+\d+)", re.IGNORECASE)
DIEM_PATTERN = re.compile(r"(Điểm\s+[a-zđ])", re.IGNORECASE)
LAW_ID_PATTERN = re.compile(
    r"(Luật số\s+[^\s,.;)]+|Bộ luật số\s+[^\s,.;)]+|Nghị định số\s+[^\s,.;)]+|Thông tư số\s+[^\s,.;)]+)",
    re.IGNORECASE,
)
GENERIC_CITATION_PATTERN = re.compile(
    r"(theo quy định pháp luật|theo pháp luật hiện hành|theo quy định hiện hành)",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_citation_text(text: str) -> str:
    return _clean_text(text).replace(" ,", ",")


def extract_citations_from_answer(answer: str) -> list[str]:
    """Extract legal citations that appear in the answer text."""

    normalized = _clean_text(answer)
    citations: list[str] = []

    for match in ARTICLE_PATTERN.finditer(normalized):
        citation_parts = [match.group(1)]
        trailing_text = normalized[match.end() : match.end() + 80]
        khoan_match = KHOAN_PATTERN.search(trailing_text)
        diem_match = DIEM_PATTERN.search(trailing_text)
        law_match = LAW_ID_PATTERN.search(trailing_text)
        if khoan_match:
            citation_parts.append(khoan_match.group(1))
        if diem_match:
            citation_parts.append(diem_match.group(1))
        if law_match:
            citation_parts.append(law_match.group(1))
        citation = _normalize_citation_text(", ".join(citation_parts))
        if citation and citation not in citations:
            citations.append(citation)

    for match in LAW_ID_PATTERN.finditer(normalized):
        citation = _normalize_citation_text(match.group(1))
        if citation and citation not in citations:
            citations.append(citation)
    return citations


def _expected_citations_from_docs(reranked_docs: Sequence[Mapping[str, Any]]) -> list[str]:
    citations: list[str] = []
    for doc in reranked_docs or []:
        metadata = dict(doc.get("metadata") or {})
        article_code = _clean_text(str(metadata.get("article_code") or metadata.get("article") or ""))
        article_name = _clean_text(str(metadata.get("article_name") or ""))
        law_id = _clean_text(str(metadata.get("law_id") or metadata.get("title") or ""))
        citation_parts = [part for part in (article_code, article_name, law_id) if part]
        if citation_parts:
            citation = " - ".join(citation_parts)
            if citation not in citations:
                citations.append(citation)
    return citations


def _normalize_sources(sources: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    for source in sources or []:
        cleaned = _clean_text(str(source))
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _extract_article_or_law_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for pattern in (ARTICLE_PATTERN, LAW_ID_PATTERN):
        for match in pattern.finditer(text or ""):
            tokens.add(_normalize_citation_text(match.group(1)).lower())
    return tokens


def _citation_matches_expected(citation: str, expected_pool: Sequence[str]) -> bool:
    normalized_citation = _normalize_citation_text(citation).lower()
    citation_tokens = _extract_article_or_law_tokens(normalized_citation)
    for expected in expected_pool:
        normalized_expected = _normalize_citation_text(expected).lower()
        if normalized_citation in normalized_expected or normalized_expected in normalized_citation:
            return True
        expected_tokens = _extract_article_or_law_tokens(normalized_expected)
        if citation_tokens and expected_tokens and citation_tokens & expected_tokens:
            return True
    return False


def inspect_citations(
    answer: str,
    sources: Sequence[str] | None,
    reranked_docs: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Inspect and normalize legal citations found in an answer."""

    answer_text = _clean_text(answer)
    answer_citations = extract_citations_from_answer(answer_text)
    source_citations = _normalize_sources(sources)
    doc_citations = _expected_citations_from_docs(reranked_docs or [])
    expected_pool = source_citations + [item for item in doc_citations if item not in source_citations]

    missing_citations: list[str] = []
    weak_citations: list[str] = []
    normalized_citations: list[str] = []

    has_generic_citation = bool(GENERIC_CITATION_PATTERN.search(answer_text))
    if not answer_citations and expected_pool:
        missing_citations.append("Câu trả lời chưa nêu Điều/Khoản/Điểm hoặc nguồn pháp lý cụ thể.")

    for citation in answer_citations:
        normalized = _normalize_citation_text(citation)
        normalized_citations.append(normalized)
        if expected_pool and not _citation_matches_expected(normalized, expected_pool):
            weak_citations.append(f"Trích dẫn '{normalized}' chưa khớp rõ với nguồn truy xuất.")

    if has_generic_citation and not answer_citations:
        weak_citations.append("Câu trả lời chỉ viện dẫn chung chung, chưa có căn cứ cụ thể.")

    citation_ok = (not missing_citations) and (not weak_citations)
    return {
        "citation_ok": citation_ok,
        "weak_citations": weak_citations,
        "missing_citations": missing_citations,
        "normalized_citations": normalized_citations,
        "expected_citations": expected_pool,
    }


__all__ = ["extract_citations_from_answer", "inspect_citations"]
