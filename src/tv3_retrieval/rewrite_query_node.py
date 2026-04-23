from __future__ import annotations

import logging
import re
from typing import Any, Callable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

LAW_ID_PATTERN = re.compile(
    r"\b("
    r"Bộ luật số\s+[^\s,.;)]+|"
    r"Luật số\s+[^\s,.;)]+|"
    r"Pháp lệnh số\s+[^\s,.;)]+|"
    r"Nghị quyết số\s+[^\s,.;)]+|"
    r"Nghị định số\s+[^\s,.;)]+|"
    r"Thông tư số\s+[^\s,.;)]+|"
    r"Quyết định số\s+[^\s,.;)]+"
    r")",
    re.IGNORECASE,
)
ARTICLE_CODE_PATTERN = re.compile(r"\b(?:Điều\s+)?(\d+(?:\.\d+)+\.[A-Z]{1,6}\.\d+\.?)\b", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"\b(Điều\s+\d+[A-Za-z]?)(?![.\dA-Za-z])\b", re.IGNORECASE)
TITLE_PATTERN = re.compile(
    r"\b("
    r"Luật\s+[A-ZÀ-Ỵ][A-Za-zÀ-ỹ0-9\s\-]+?|"
    r"Bộ luật\s+[A-ZÀ-Ỵ][A-Za-zÀ-ỹ0-9\s\-]+?|"
    r"Nghị định\s+[A-ZÀ-Ỵ][A-Za-zÀ-ỹ0-9\s\-]+?|"
    r"Thông tư\s+[A-ZÀ-Ỵ][A-Za-zÀ-ỹ0-9\s\-]+?"
    r")(?=\s+(?:quy định|là gì|được quy định|nói gì|thế nào|ra sao)\b|[,.?;:]|$)",
    re.IGNORECASE,
)
DE_MUC_PATTERN = re.compile(r"(Đề mục\s+[0-9A-Za-z.\-]+(?:\s*-\s*[^,.;]+)?)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")

KNOWN_ISSUERS = (
    "Quốc hội",
    "Chính phủ",
    "Ủy ban Thường vụ Quốc hội",
    "Thủ tướng Chính phủ",
    "Bộ Công an",
    "Bộ Tư pháp",
    "Bộ Tài chính",
    "Bộ Giáo dục và Đào tạo",
    "Bộ Lao động - Thương binh và Xã hội",
    "Bộ Y tế",
    "Tòa án nhân dân tối cao",
    "Viện kiểm sát nhân dân tối cao",
)

INTENT_EXPANSIONS: dict[str, Sequence[str]] = {
    "hoi_dinh_nghia": ("khái niệm", "định nghĩa", "giải thích từ ngữ", "quy định"),
    "hoi_thu_tuc_hanh_chinh": ("trình tự", "thủ tục", "hồ sơ", "cách thực hiện", "điều kiện"),
    "hoi_muc_phat": ("xử phạt", "mức phạt", "chế tài", "vi phạm"),
    "hoi_so_sanh_luat": ("so sánh", "khác nhau", "phân biệt", "áp dụng"),
    "hoi_tinh_huong_thuc_te": ("trách nhiệm", "quyền và nghĩa vụ", "đối tượng áp dụng"),
}

TERM_EXPANSIONS: dict[str, Sequence[str]] = {
    "quyền": ("quyền", "nghĩa vụ", "trách nhiệm"),
    "nghĩa vụ": ("nghĩa vụ", "trách nhiệm", "quy định"),
    "trách nhiệm": ("trách nhiệm", "nghĩa vụ", "quy định"),
    "xử phạt": ("xử phạt", "mức phạt", "chế tài"),
    "phạt": ("mức phạt", "xử phạt", "vi phạm"),
    "đối tượng": ("đối tượng áp dụng", "phạm vi điều chỉnh", "chủ thể"),
    "hiệu lực": ("hiệu lực", "hiệu lực thi hành", "ngày có hiệu lực"),
    "thủ tục": ("trình tự thủ tục", "hồ sơ", "cách thức thực hiện"),
    "điều kiện": ("điều kiện", "đối tượng áp dụng", "trình tự thủ tục"),
    "khái niệm": ("khái niệm", "định nghĩa", "giải thích từ ngữ"),
}


def normalize_legal_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_match_text(text: str) -> str:
    return normalize_legal_text(text).lower()


def _canonical_article(article_text: str) -> str:
    match = ARTICLE_PATTERN.search(normalize_legal_text(article_text))
    if not match:
        return ""
    number = match.group(1).split(maxsplit=1)[1].strip()
    return f"Điều {number}"


def _canonical_article_code(text: str) -> str:
    match = ARTICLE_CODE_PATTERN.search(normalize_legal_text(text))
    if not match:
        return ""
    code = match.group(1).strip().rstrip(".")
    return f"Điều {code}."


def _extract_title(text: str, *, law_id: str = "") -> str:
    normalized = normalize_legal_text(text)
    match = TITLE_PATTERN.search(normalized)
    if match:
        return match.group(1).strip(" ,.;:")
    if law_id:
        return ""
    if normalized.lower().startswith("luật "):
        return normalized.strip(" ,.;:")
    return ""


def extract_metadata_filters(question: str) -> dict[str, str]:
    text = normalize_legal_text(question)
    filters: dict[str, str] = {}

    law_match = LAW_ID_PATTERN.search(text)
    if law_match:
        filters["law_id"] = law_match.group(1).strip()

    article_code = _canonical_article_code(text)
    if article_code:
        filters["article_code"] = article_code

    article = _canonical_article(text)
    if article:
        filters["article"] = article

    title = _extract_title(text, law_id=filters.get("law_id", ""))
    if title:
        filters["title"] = title

    de_muc_match = DE_MUC_PATTERN.search(text)
    if de_muc_match:
        filters["de_muc"] = de_muc_match.group(1).strip()

    if re.search(r"hiệu lực", text, re.IGNORECASE):
        date_match = DATE_PATTERN.search(text)
        if date_match:
            filters["effective_date"] = date_match.group(1)

    for issuer in KNOWN_ISSUERS:
        if _normalize_match_text(issuer) in _normalize_match_text(text):
            filters["issuer"] = issuer
            break

    return filters


def extract_legal_query_features(question: str, metadata_filters: Mapping[str, str]) -> dict[str, Any]:
    normalized = _normalize_match_text(question)
    return {
        "has_article_ref": bool(metadata_filters.get("article")),
        "has_article_code": bool(metadata_filters.get("article_code")),
        "has_law_title": bool(metadata_filters.get("title")),
        "has_law_id": bool(metadata_filters.get("law_id")),
        "has_de_muc": bool(metadata_filters.get("de_muc")),
        "is_structured_legal_query": bool(
            metadata_filters.get("article")
            or metadata_filters.get("article_code")
            or metadata_filters.get("title")
            or metadata_filters.get("law_id")
        ),
        "mentions_quy_dinh": "quy định" in normalized,
        "mentions_la_gi": "là gì" in normalized,
    }


def _extract_focus_terms(question: str, intent: str) -> list[str]:
    lowered = question.lower()
    expansions: list[str] = []
    for token, related_terms in TERM_EXPANSIONS.items():
        if token in lowered:
            expansions.extend(related_terms)
    for token in INTENT_EXPANSIONS.get(intent.strip().lower(), ()):
        expansions.append(token)

    unique_terms: list[str] = []
    for term in expansions:
        cleaned = normalize_legal_text(term)
        if cleaned and cleaned not in unique_terms:
            unique_terms.append(cleaned)
    return unique_terms[:4]


def _dedupe_queries(queries: Sequence[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    for query in queries:
        cleaned = normalize_legal_text(query)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
        if len(deduped) >= limit:
            break
    return deduped


def _build_structured_queries(
    base_question: str,
    metadata_filters: Mapping[str, str],
    legal_query_features: Mapping[str, Any],
) -> list[str]:
    queries: list[str] = [base_question]

    article_code = str(metadata_filters.get("article_code") or "")
    article = str(metadata_filters.get("article") or "")
    law_id = str(metadata_filters.get("law_id") or "")
    title = str(metadata_filters.get("title") or "")

    if article_code:
        queries.append(" ".join(part for part in (article_code, law_id or title) if part).strip())

    if article and (title or law_id):
        queries.append(" ".join(part for part in (article, title or law_id) if part).strip())

    if legal_query_features.get("mentions_la_gi") and (title or law_id):
        queries.append(" ".join(part for part in (title or law_id, "định nghĩa", base_question) if part).strip())

    if legal_query_features.get("mentions_quy_dinh") and article:
        queries.append(" ".join(part for part in (article, title or law_id, "quy định") if part).strip())

    return queries


def _build_rule_based_queries(
    base_question: str,
    intent: str,
    metadata_filters: Mapping[str, str],
    legal_query_features: Mapping[str, Any],
) -> list[str]:
    queries = _build_structured_queries(base_question, metadata_filters, legal_query_features)
    focus_terms = _extract_focus_terms(base_question, intent)

    if focus_terms:
        queries.append(f"{base_question} {' '.join(focus_terms[:3])}".strip())

    if not metadata_filters and not focus_terms:
        queries.append(f"{base_question} quy định pháp luật".strip())

    return _dedupe_queries(queries, limit=4)


def _build_fast_queries(
    base_question: str,
    intent: str,
    metadata_filters: Mapping[str, str],
    legal_query_features: Mapping[str, Any],
) -> list[str]:
    queries = _build_structured_queries(base_question, metadata_filters, legal_query_features)

    if len(queries) == 1:
        focus_terms = _extract_focus_terms(base_question, intent)
        if focus_terms:
            queries.append(f"{base_question} {' '.join(focus_terms[:2])}".strip())

    return _dedupe_queries(queries, limit=2)


def rewrite_query(
    question: str,
    *,
    normalized_question: str = "",
    intent: str = "",
    execution_profile: str = "full",
    llm_rewriter: Callable[[str, str, Mapping[str, str]], Sequence[str]] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Rewrite a legal question into retrieval-friendly variants."""

    resolved_logger = logger or LOGGER
    base_question = normalize_legal_text(normalized_question or question)
    metadata_filters = extract_metadata_filters(base_question)
    legal_query_features = extract_legal_query_features(base_question, metadata_filters)
    normalized_profile = "fast" if execution_profile == "fast" else "full"

    llm_queries: list[str] = []
    if llm_rewriter is not None and normalized_profile != "fast":
        try:
            llm_queries = [
                normalize_legal_text(item)
                for item in llm_rewriter(base_question, intent, metadata_filters)
                if normalize_legal_text(item)
            ]
        except Exception as exc:  # pragma: no cover
            resolved_logger.warning("LLM rewrite failed, falling back to rule-based rewrite: %s", exc)

    if normalized_profile == "fast":
        queries = _build_fast_queries(base_question, intent, metadata_filters, legal_query_features)
        rewrite_strategy = "fast_rule_based"
    else:
        queries = _dedupe_queries(
            [
                base_question,
                *llm_queries,
                *_build_rule_based_queries(base_question, intent, metadata_filters, legal_query_features),
            ],
            limit=4,
        )
        rewrite_strategy = "llm+rule_based" if llm_queries else "rule_based"

    return {
        "rewritten_queries": queries,
        "metadata_filters": dict(metadata_filters),
        "legal_query_features": dict(legal_query_features),
        "retrieval_debug": {
            "metadata_filters": dict(metadata_filters),
            "legal_query_features": dict(legal_query_features),
            "rewrite_strategy": rewrite_strategy,
            "execution_profile": normalized_profile,
        },
    }


def rewrite_query_node(
    state: Mapping[str, Any],
    *,
    llm_rewriter: Callable[[str, str, Mapping[str, str]], Sequence[str]] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """LangGraph-friendly TV3 node for query rewriting."""

    question = str(state.get("question") or "").strip()
    normalized_question = str(state.get("normalized_question") or "").strip()
    intent = str(state.get("intent") or "").strip()
    execution_profile = str(state.get("execution_profile") or "full").strip().lower()
    result = rewrite_query(
        question,
        normalized_question=normalized_question,
        intent=intent,
        execution_profile=execution_profile,
        llm_rewriter=llm_rewriter,
        logger=logger,
    )

    existing_debug = state.get("retrieval_debug") or {}
    merged_debug = dict(existing_debug) if isinstance(existing_debug, Mapping) else {}
    merged_debug.update(result["retrieval_debug"])
    return {
        "rewritten_queries": result["rewritten_queries"],
        "metadata_filters": result["metadata_filters"],
        "legal_query_features": result["legal_query_features"],
        "retrieval_debug": merged_debug,
    }


__all__ = [
    "extract_legal_query_features",
    "extract_metadata_filters",
    "normalize_legal_text",
    "rewrite_query",
    "rewrite_query_node",
]
