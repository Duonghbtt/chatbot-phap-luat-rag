from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.tv2_index.search_with_filters import result_matches_filters
from src.tv3_retrieval.fallback_policy import RetrievalConfig
from src.tv3_retrieval.rerank_node import rerank_node
from src.tv3_retrieval.retrieval_check_node import retrieval_check_node
from src.tv3_retrieval.retrieve_node import retrieve_node, tokenize_legal_text
from src.tv3_retrieval.rewrite_query_node import rewrite_query_node

LOW_SIGNAL_QUERY_TOKENS = {
    "quy",
    "định",
    "khái",
    "niệm",
    "giải",
    "thích",
    "từ",
    "ngữ",
    "quyền",
    "nghĩa",
    "vụ",
    "trách",
    "nhiệm",
}


def _sample_tv1_records() -> list[dict[str, Any]]:
    return [
        {
            "content": "Điều 36.3.LQ.1. Thanh niên\nThanh niên là công dân Việt Nam từ đủ 16 tuổi đến 30 tuổi.",
            "metadata": {
                "de_muc": "Đề mục 36.3 - Thanh niên",
                "file_id": "dm-thanh-nien",
                "article_code": "Điều 36.3.LQ.1.",
                "article_name": "Thanh niên",
                "mapc": "mapc-1",
                "law_id": "Luật số 57/2020/QH14",
                "topic_id": "topic-thanh-nien",
                "title": "Luật Thanh niên",
                "issuer": "Quốc hội",
                "article": "Điều 1",
                "effective_date": "01/01/2021",
                "source_note": "",
                "related_articles": [],
            },
        },
        {
            "content": "Điều 36.3.LQ.10. Đối thoại với thanh niên\n1. Thủ tướng Chính phủ, Chủ tịch Ủy ban nhân dân các cấp có trách nhiệm đối thoại với thanh niên ít nhất mỗi năm một lần.",
            "metadata": {
                "de_muc": "Đề mục 36.3 - Thanh niên",
                "file_id": "dm-thanh-nien",
                "article_code": "Điều 36.3.LQ.10.",
                "article_name": "Đối thoại với thanh niên",
                "mapc": "mapc-10-1",
                "law_id": "Luật số 57/2020/QH14",
                "topic_id": "topic-thanh-nien",
                "title": "Luật Thanh niên",
                "issuer": "Quốc hội",
                "article": "Điều 10",
                "effective_date": "01/01/2021",
                "source_note": "",
                "related_articles": [],
            },
        },
        {
            "content": "Điều 36.3.LQ.10. Đối thoại với thanh niên\n2. Nội dung đối thoại với thanh niên bao gồm việc thực hiện chính sách, pháp luật đối với thanh niên.",
            "metadata": {
                "de_muc": "Đề mục 36.3 - Thanh niên",
                "file_id": "dm-thanh-nien",
                "article_code": "Điều 36.3.LQ.10.",
                "article_name": "Đối thoại với thanh niên",
                "mapc": "mapc-10-2",
                "law_id": "Luật số 57/2020/QH14",
                "topic_id": "topic-thanh-nien",
                "title": "Luật Thanh niên",
                "issuer": "Quốc hội",
                "article": "Điều 10",
                "effective_date": "01/01/2021",
                "source_note": "",
                "related_articles": [],
            },
        },
        {
            "content": "Điều 36.3.LQ.11. Áp dụng điều ước quốc tế\nViệc áp dụng điều ước quốc tế về quyền trẻ em và thanh niên thực hiện theo điều ước quốc tế mà Việt Nam là thành viên.",
            "metadata": {
                "de_muc": "Đề mục 36.3 - Thanh niên",
                "file_id": "dm-thanh-nien",
                "article_code": "Điều 36.3.LQ.11.",
                "article_name": "Áp dụng điều ước quốc tế",
                "mapc": "mapc-11",
                "law_id": "Luật số 57/2020/QH14",
                "topic_id": "topic-thanh-nien",
                "title": "Luật Thanh niên",
                "issuer": "Quốc hội",
                "article": "Điều 11",
                "effective_date": "01/01/2021",
                "source_note": "",
                "related_articles": [],
            },
        },
        {
            "content": "Điều 4. Công dân phục vụ tại ngũ\nCông dân nam đủ 18 tuổi được gọi nhập ngũ theo Luật Nghĩa vụ quân sự.",
            "metadata": {
                "de_muc": "Đề mục 12.1 - Nghĩa vụ quân sự",
                "file_id": "dm-nghia-vu-qs",
                "article_code": "Điều 4.",
                "article_name": "Công dân phục vụ tại ngũ",
                "mapc": "mapc-nvqs-4",
                "law_id": "Luật số 78/2015/QH13",
                "topic_id": "topic-nvqs",
                "title": "Luật Nghĩa vụ quân sự",
                "issuer": "Quốc hội",
                "article": "Điều 4",
                "effective_date": "01/01/2016",
                "source_note": "",
                "related_articles": [],
            },
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _build_article_docs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        metadata = row["metadata"]
        grouped[(metadata["law_id"], metadata["article_code"])].append(row)

    docs: list[dict[str, Any]] = []
    for grouped_rows in grouped.values():
        metadata = dict(grouped_rows[0]["metadata"])
        content = "\n\n".join(row["content"] for row in grouped_rows)
        docs.append({"content": content, "metadata": metadata})
    return docs


def _overlap_score(query_text: str, content: str) -> float:
    query_tokens = {
        token for token in tokenize_legal_text(query_text) if token not in LOW_SIGNAL_QUERY_TOKENS
    }
    content_tokens = tokenize_legal_text(content)
    if not query_tokens or not content_tokens:
        return 0.0
    matched_tokens = {token for token in content_tokens if token in query_tokens}
    if len(matched_tokens) < 2:
        return 0.0
    if max((len(token) for token in matched_tokens), default=0) < 4:
        return 0.0
    overlap = sum(1 for token in content_tokens if token in matched_tokens)
    return float(overlap) / max(len(query_tokens), 1)


class FakeBM25Retriever:
    def __init__(self, article_docs: list[dict[str, Any]], chunk_docs: list[dict[str, Any]]) -> None:
        self.article_docs = article_docs
        self.chunk_docs = chunk_docs

    def search(
        self,
        query_text: str,
        *,
        level: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if level == "article" and "mỗi năm một lần" in query_text.lower():
            return []
        docs = self.article_docs if level == "article" else self.chunk_docs
        scored = []
        for doc in docs:
            if filters and not result_matches_filters(doc, filters):
                continue
            score = _overlap_score(query_text, doc["content"])
            if score <= 0:
                continue
            scored.append(
                {
                    "content": doc["content"],
                    "metadata": dict(doc["metadata"]),
                    "bm25_score": score,
                    "vector_score": 0.0,
                    "combined_score": 0.0,
                    "level": level,
                }
            )
        scored.sort(key=lambda item: item["bm25_score"], reverse=True)
        return scored[:top_k]


class FakeVectorSearchService:
    def __init__(
        self,
        article_docs: list[dict[str, Any]],
        chunk_docs: list[dict[str, Any]],
        *,
        emit_duplicates: bool = False,
    ) -> None:
        self.article_docs = article_docs
        self.chunk_docs = chunk_docs
        self.emit_duplicates = emit_duplicates

    def _search(self, docs: list[dict[str, Any]], query_text: str, filters: Mapping[str, Any] | None, top_k: int, *, level: str) -> list[dict[str, Any]]:
        if level == "article" and "mỗi năm một lần" in query_text.lower():
            return []
        scored = []
        for doc in docs:
            if filters and not result_matches_filters(doc, filters):
                continue
            score = _overlap_score(query_text, doc["content"])
            if score <= 0:
                continue
            scored.append(
                {
                    "content": doc["content"],
                    "metadata": dict(doc["metadata"]),
                    "score": score + (0.1 if level == "article" else 0.2),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        results = scored[:top_k]
        if self.emit_duplicates and results:
            results = results + [dict(results[0])]
        return results

    def search_article_level(self, query_text: str, *, filters: Mapping[str, Any] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        return self._search(self.article_docs, query_text, filters, top_k, level="article")

    def search_chunk_level(self, query_text: str, *, filters: Mapping[str, Any] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        return self._search(self.chunk_docs, query_text, filters, top_k, level="chunk")


def _fake_reranker(pairs: list[tuple[str, str]]) -> list[float]:
    return [_overlap_score(query_text, content) for query_text, content in pairs]


def _make_config(tmp_path: Path, records: list[dict[str, Any]], **overrides: Any) -> RetrievalConfig:
    corpus_path = tmp_path / "all_chunks.jsonl"
    _write_jsonl(corpus_path, records)
    config_kwargs = {
        "local_corpus_path": str(corpus_path),
        "top_k": 4,
        "bm25_top_k": 5,
        "vector_top_k": 5,
        "rerank_top_n": 3,
        "max_retry_loops": 3,
        "min_valid_sources": 1,
        "min_unique_sources": 1,
        "min_top_rerank_score": 0.2,
    }
    config_kwargs.update(overrides)
    return RetrievalConfig(**config_kwargs)


def _run_flow(
    question: str,
    *,
    config: RetrievalConfig,
    bm25_retriever: FakeBM25Retriever,
    vector_service: FakeVectorSearchService,
    intent: str = "definition",
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question": question,
        "normalized_question": question,
        "intent": intent,
        "intent_score": 0.92,
        "rewritten_queries": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "context": "",
        "sources": [],
        "retrieval_ok": False,
        "loop_count": 0,
        "need_clarify": False,
        "human_review_required": False,
        "history": [],
    }
    state.update(rewrite_query_node(state))

    for _ in range(config.max_retry_loops + 2):
        state.update(
            retrieve_node(
                state,
                retrieval_config=config,
                bm25_retriever=bm25_retriever,
                vector_search_service=vector_service,
            )
        )
        state.update(rerank_node(state, retrieval_config=config, reranker_backend=_fake_reranker))
        state.update(retrieval_check_node(state, retrieval_config=config))
        if state.get("retrieval_ok") or state.get("next_action") != "retry":
            break
    return state


def test_easy_query_end_to_end(tmp_path: Path) -> None:
    records = _sample_tv1_records()
    config = _make_config(tmp_path, records)
    article_docs = _build_article_docs(records)
    flow_state = _run_flow(
        "quyền của thanh niên là gì",
        config=config,
        bm25_retriever=FakeBM25Retriever(article_docs, records),
        vector_service=FakeVectorSearchService(article_docs, records),
        intent="definition",
    )

    assert flow_state["retrieval_ok"] is True
    assert flow_state["reranked_docs"]
    assert "Luật số 57/2020/QH14" in " ".join(flow_state["sources"])


def test_ambiguous_query_generates_legal_rewrites(tmp_path: Path) -> None:
    records = _sample_tv1_records()
    config = _make_config(tmp_path, records)
    article_docs = _build_article_docs(records)
    state = _run_flow(
        "thanh niên là ai",
        config=config,
        bm25_retriever=FakeBM25Retriever(article_docs, records),
        vector_service=FakeVectorSearchService(article_docs, records),
        intent="definition",
    )

    assert any(term in " ".join(state["rewritten_queries"]).lower() for term in ("khái niệm", "định nghĩa"))
    assert state["retrieval_ok"] is True


def test_query_with_law_id_filter(tmp_path: Path) -> None:
    records = _sample_tv1_records()
    config = _make_config(tmp_path, records)
    article_docs = _build_article_docs(records)
    state = _run_flow(
        "Theo Luật số 57/2020/QH14, đối thoại với thanh niên được quy định thế nào?",
        config=config,
        bm25_retriever=FakeBM25Retriever(article_docs, records),
        vector_service=FakeVectorSearchService(article_docs, records),
        intent="scenario",
    )

    assert state["retrieval_debug"]["metadata_filters"]["law_id"] == "Luật số 57/2020/QH14"
    assert all(doc["metadata"]["law_id"] == "Luật số 57/2020/QH14" for doc in state["retrieved_docs"])


def test_article_level_can_fallback_to_chunk_level(tmp_path: Path) -> None:
    records = _sample_tv1_records()
    config = _make_config(tmp_path, records, article_first=True, allow_chunk_fallback=True)
    article_docs = _build_article_docs(records)
    state = _run_flow(
        "đối thoại với thanh niên ít nhất mỗi năm một lần như thế nào",
        config=config,
        bm25_retriever=FakeBM25Retriever(article_docs, records),
        vector_service=FakeVectorSearchService(article_docs, records),
        intent="scenario",
    )

    assert state["retrieval_ok"] is True
    assert state["loop_count"] >= 1
    assert state["retrieval_debug"]["current_plan"]["level"] == "chunk"


def test_duplicate_docs_are_merged(tmp_path: Path) -> None:
    records = _sample_tv1_records()
    config = _make_config(tmp_path, records)
    article_docs = _build_article_docs(records)
    initial_state = {
        "question": "đối thoại với thanh niên",
        "normalized_question": "đối thoại với thanh niên",
        "intent": "scenario",
        "rewritten_queries": ["đối thoại với thanh niên"],
        "retrieval_debug": {},
    }

    updates = retrieve_node(
        initial_state,
        retrieval_config=config,
        bm25_retriever=FakeBM25Retriever(article_docs, records),
        vector_search_service=FakeVectorSearchService(article_docs, records, emit_duplicates=True),
    )

    dedup_keys = {
        (doc["metadata"].get("law_id"), doc["metadata"].get("article_code"), doc["metadata"].get("mapc"))
        for doc in updates["retrieved_docs"]
    }
    assert len(updates["retrieved_docs"]) == len(dedup_keys)


def test_no_result_case_stops_with_fallback(tmp_path: Path) -> None:
    records = _sample_tv1_records()
    config = _make_config(tmp_path, records, max_retry_loops=2)
    article_docs = _build_article_docs(records)
    state = _run_flow(
        "quy định về tàu vũ trụ dân sự",
        config=config,
        bm25_retriever=FakeBM25Retriever(article_docs, records),
        vector_service=FakeVectorSearchService(article_docs, records),
        intent="definition",
    )

    assert state["retrieval_ok"] is False
    assert state["next_action"] == "fallback"
    assert state["retrieval_failure_reason"] in {"no_results", "weak_evidence", "insufficient_results"}
