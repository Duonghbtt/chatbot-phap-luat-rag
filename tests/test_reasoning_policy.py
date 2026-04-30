from __future__ import annotations

from typing import Any, Mapping

from src.graph.builder import citation_format_node
from src.tv5_reasoning.generate_draft_node import LLMConfig, generate_draft
from src.tv5_reasoning.grounding_check_node import grounding_check_node


class _FakePromptLibrary:
    class _Config:
        shared: dict[str, str] = {}
        draft: dict[str, str] = {}

    config = _Config()

    def get_draft_prompt(
        self,
        intent: str,
        risk_level: str,
        *,
        question: str,
        context: str,
        sources: list[str] | None,
    ) -> str:
        return "draft prompt"

    def get_grounding_prompt(
        self,
        intent: str,
        risk_level: str,
        *,
        question: str,
        context: str,
        sources: list[str] | None,
        draft_answer: str,
    ) -> str:
        return "grounding prompt"

    def get_system_prompt(self) -> str:
        return "system prompt"


class _FakeClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate_with_retry(self, *, prompt: str, system_prompt: str = "") -> str:
        return self.response_text


def _sample_reranked_doc() -> dict[str, Any]:
    return {
        "content": "Điều 36.3.LQ.1. Thanh niên Thanh niên là công dân Việt Nam từ đủ 16 tuổi đến 30 tuổi.",
        "metadata": {
            "article": "Điều 1",
            "article_code": "Điều 36.3.LQ.1.",
            "article_name": "Thanh niên",
            "title": "Luật Thanh niên",
            "law_id": "Luật số 57/2020/QH14",
            "issuer": "Quốc hội",
            "effective_date": "01/01/2021",
        },
        "exact_hit_fields": ["article", "title"],
        "matched_filters": {"article": "Điều 1", "title": "Luật Thanh niên"},
        "combined_score": 1.42,
        "rerank_score": 2.85,
    }


def _sample_sources() -> list[str]:
    return ["Điều 36.3.LQ.1. - Thanh niên - Luật số 57/2020/QH14 - Quốc hội, 01/01/2021"]


def test_generate_draft_rewrites_direct_legal_lookup_answer() -> None:
    reranked_docs = [_sample_reranked_doc()]
    sources = _sample_sources()
    client = _FakeClient(
        """
        {
          "draft_answer": "Điều 1 của Luật Thanh niên không được đề cập trong ngữ cảnh cung cấp. Tuy nhiên, theo Điều 36.3.LQ.1 của Luật số 57/2020/QH14 - Quốc hội, thanh niên được định nghĩa là công dân Việt Nam từ đủ 16 tuổi đến 30 tuổi.",
          "draft_citations": [],
          "draft_confidence": 0.88
        }
        """
    )

    result = generate_draft(
        question="Điều 1 Luật Thanh niên quy định gì?",
        context=reranked_docs[0]["content"],
        sources=sources,
        reranked_docs=reranked_docs,
        intent="hoi_dinh_nghia",
        risk_level="low",
        execution_profile="full",
        llm_config=LLMConfig(),
        prompt_library=_FakePromptLibrary(),
        client=client,
    )

    assert "không được đề cập" not in result["draft_answer"].lower()
    assert "Theo Điều 1 của Luật Thanh niên" in result["draft_answer"]
    assert "Luật số 57/2020/QH14" in result["draft_answer"]


def test_grounding_check_skips_human_review_for_low_risk_exact_hit() -> None:
    reranked_docs = [_sample_reranked_doc()]
    state: Mapping[str, Any] = {
        "question": "Điều 1 Luật Thanh niên quy định gì?",
        "normalized_question": "Điều 1 Luật Thanh niên quy định gì?",
        "intent": "hoi_dinh_nghia",
        "risk_level": "low",
        "draft_answer": "Theo Điều 1 của Luật Thanh niên (Luật số 57/2020/QH14), Thanh niên là công dân Việt Nam từ đủ 16 tuổi đến 30 tuổi.",
        "context": reranked_docs[0]["content"],
        "sources": _sample_sources(),
        "reranked_docs": reranked_docs,
        "execution_profile": "full",
    }
    client = _FakeClient(
        """
        {
          "grounding_score": 0.2,
          "unsupported_claims": [],
          "missing_evidence": ["LLM critic would like human review."],
          "next_action": "human_review",
          "notes": "forced human review for test"
        }
        """
    )

    result = grounding_check_node(
        state,
        llm_config=LLMConfig(),
        prompt_library=_FakePromptLibrary(),
        client=client,
    )

    assert result["next_action"] == "proceed"
    assert result["grounding_ok"] is True
    assert result["human_review_required"] is False


def test_grounding_check_skips_human_review_for_medium_risk_definition_overview() -> None:
    reranked_docs = [
        {
            **_sample_reranked_doc(),
            "exact_hit_fields": [],
            "matched_filters": {},
        },
        {
            **_sample_reranked_doc(),
            "metadata": {
                **_sample_reranked_doc()["metadata"],
                "article": "Điều 5",
                "article_code": "Điều 36.3.LQ.5.",
                "article_name": "Nguyên tắc bảo đảm thực hiện quyền, nghĩa vụ của thanh niên và chính sách của Nhà nước đối với thanh niên",
            },
            "exact_hit_fields": [],
            "matched_filters": {},
        },
    ]
    state: Mapping[str, Any] = {
        "question": "Nghĩa vụ của thanh niên được quy định như thế nào?",
        "normalized_question": "Nghĩa vụ của thanh niên được quy định như thế nào?",
        "intent": "hoi_dinh_nghia",
        "risk_level": "medium",
        "draft_answer": (
            "Nghĩa vụ của thanh niên được quy định trong Luật số 57/2020/QH14. "
            "Theo Điều 36.3.LQ.4, thanh niên có quyền và nghĩa vụ của công dân theo quy định của Hiến pháp và pháp luật."
        ),
        "context": "\n\n".join(str(doc["content"]) for doc in reranked_docs),
        "sources": [
            "Điều 36.3.LQ.5. - Nguyên tắc bảo đảm thực hiện quyền, nghĩa vụ của thanh niên và chính sách của Nhà nước đối với thanh niên - Luật số 57/2020/QH14 - Quốc hội, 01/01/2021",
            "Điều 36.3.LQ.4. - Vai trò, quyền và nghĩa vụ của thanh niên - Luật số 57/2020/QH14 - Quốc hội, 01/01/2021",
        ],
        "reranked_docs": reranked_docs,
        "execution_profile": "full",
    }
    client = _FakeClient(
        """
        {
          "grounding_score": 0.75,
          "unsupported_claims": [],
          "missing_evidence": [],
          "next_action": "human_review",
          "notes": "forced human review for medium-risk definition overview"
        }
        """
    )

    result = grounding_check_node(
        state,
        llm_config=LLMConfig(),
        prompt_library=_FakePromptLibrary(),
        client=client,
    )

    assert result["next_action"] == "proceed"
    assert result["grounding_ok"] is True
    assert result["human_review_required"] is False


def test_citation_format_node_prunes_redundant_sources_for_direct_lookup() -> None:
    state: Mapping[str, Any] = {
        "status": "ok",
        "response_status": "ok",
        "execution_profile": "full",
        "sources": [
            "Điều 36.3.LQ.1. - Thanh niên - Luật số 57/2020/QH14 - Quốc hội, 01/01/2021",
            "Luật số 57/2020/QH14",
        ],
        "citation_findings": {
            "normalized_citations": [
                "Điều 1",
                "Luật số 57/2020/QH14",
            ]
        },
        "reranked_docs": [_sample_reranked_doc()],
    }

    result = citation_format_node(state)

    assert "Điều 1" not in result["sources"]
    assert "Luật số 57/2020/QH14" not in result["sources"]
    assert any("Điều 36.3.LQ.1." in source for source in result["sources"])
    assert len(result["sources"]) <= 2
