from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

import pytest

from src.app.api.routes.chat import build_chat_response
from src.graph.builder import AppConfig, GraphDependencies, build_graph
from src.graph.checkpointing import BaseCheckpointStore, CheckpointNotFoundError
from src.graph.human_review_node import human_review_node
from src.graph.state import AgentState, clone_state, create_initial_state


class InMemoryCheckpointStore(BaseCheckpointStore):
    """Dict-backed checkpoint store for TV6 resume tests."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], AgentState] = {}
        self.save_calls = 0
        self.load_calls = 0

    def save_state(self, state: Mapping[str, Any]) -> str:
        self.save_calls += 1
        snapshot = clone_state(state)
        checkpoint_id = str(snapshot.get("app_checkpoint_id") or f"ckpt-{self.save_calls}")
        snapshot["app_checkpoint_id"] = checkpoint_id
        key = (str(snapshot.get("session_id") or ""), str(snapshot.get("thread_id") or ""))
        self.records[key] = deepcopy(snapshot)
        return checkpoint_id

    def load_state(self, *, thread_id: str, session_id: str | None = None) -> AgentState:
        self.load_calls += 1
        key = (str(session_id or ""), str(thread_id))
        if key not in self.records:
            raise CheckpointNotFoundError(
                f"Checkpoint not found for thread_id={thread_id} session_id={session_id or ''}"
            )
        return deepcopy(self.records[key])

    def delete_state(self, *, thread_id: str, session_id: str | None = None) -> None:
        self.records.pop((str(session_id or ""), str(thread_id)), None)

    def exists(self, *, thread_id: str, session_id: str | None = None) -> bool:
        return (str(session_id or ""), str(thread_id)) in self.records

    def get_record(self, *, thread_id: str, session_id: str) -> AgentState:
        return deepcopy(self.records[(session_id, thread_id)])


def _sample_retrieved_doc() -> dict[str, Any]:
    return {
        "content": (
            "Điều 36.3.LQ.10. Đối thoại với thanh niên\n"
            "Thủ tướng Chính phủ, Chủ tịch Ủy ban nhân dân các cấp có trách nhiệm "
            "đối thoại với thanh niên ít nhất mỗi năm một lần."
        ),
        "metadata": {
            "article_code": "Điều 36.3.LQ.10.",
            "article_name": "Đối thoại với thanh niên",
            "law_id": "Luật số 57/2020/QH14",
            "title": "Luật Thanh niên",
            "issuer": "Quốc hội",
            "effective_date": "01/01/2021",
        },
        "combined_score": 0.94,
        "rerank_score": 0.93,
    }


def _sample_sources() -> list[str]:
    return ["Điều 36.3.LQ.10. - Đối thoại với thanh niên - Luật số 57/2020/QH14"]


def _make_trace() -> dict[str, int]:
    return {
        "route": 0,
        "rewrite": 0,
        "retrieve": 0,
        "rerank": 0,
        "retrieval_check": 0,
        "generate": 0,
        "grounding": 0,
        "revise": 0,
    }


def _build_post_reasoning_dependencies(trace: dict[str, int]) -> GraphDependencies:
    """Build fake TV3/TV5 nodes that require review after reasoning."""

    doc = _sample_retrieved_doc()
    sources = _sample_sources()

    def fake_route_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["route"] += 1
        return {
            "intent": "hoi_tinh_huong_thuc_te",
            "intent_score": 0.91,
            "risk_level": "high",
            "need_clarify": False,
            "unsupported_query": False,
            "human_review_required": False,
            "next_route": "legal-agent-path",
            "route_reason": "Đưa vào legal-agent-path rồi human review sau reasoning.",
        }

    def fake_rewrite_query_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["rewrite"] += 1
        return {"rewritten_queries": [str(state.get("question") or ""), "đối thoại với thanh niên luật thanh niên"]}

    def fake_retrieve_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["retrieve"] += 1
        return {
            "retrieved_docs": [deepcopy(doc)],
            "retrieval_debug": {"mode": "fake_post_reasoning"},
        }

    def fake_rerank_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["rerank"] += 1
        return {
            "reranked_docs": [deepcopy(doc)],
            "context": doc["content"],
            "sources": list(sources),
        }

    def fake_retrieval_check_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["retrieval_check"] += 1
        return {
            "retrieval_ok": True,
            "next_action": "proceed",
        }

    def fake_generate_draft_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["generate"] += 1
        return {
            "draft_answer": (
                "Theo Điều 36.3.LQ.10. của Luật số 57/2020/QH14, "
                "Thủ tướng Chính phủ và Chủ tịch Ủy ban nhân dân các cấp "
                "có trách nhiệm đối thoại với thanh niên ít nhất mỗi năm một lần."
            ),
            "draft_citations": list(sources),
            "draft_confidence": 0.89,
        }

    def fake_grounding_check_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["grounding"] += 1
        return {
            "grounding_ok": False,
            "grounding_score": 0.72,
            "unsupported_claims": [],
            "missing_evidence": ["Cần human review để xác nhận trước khi phát hành."],
            "next_action": "human_review",
            "human_review_required": True,
            "review_note": "Cần chuyên viên pháp lý xác nhận trước khi phát hành câu trả lời.",
        }

    return GraphDependencies(
        route_node=fake_route_node,
        rewrite_query_node=fake_rewrite_query_node,
        retrieve_node=fake_retrieve_node,
        rerank_node=fake_rerank_node,
        retrieval_check_node=fake_retrieval_check_node,
        generate_draft_node=fake_generate_draft_node,
        grounding_check_node=fake_grounding_check_node,
        revise_answer_node=lambda state: {"final_answer": str(state.get("draft_answer") or "")},
        human_review_node=human_review_node,
    )


def _build_pre_review_dependencies(trace: dict[str, int]) -> GraphDependencies:
    """Build fake nodes where review happens before retrieval/reasoning.

    This helper uses a tiny pre-review human-review node so the test can focus on
    TV6 resume orchestration after an early review gate, without depending on
    downstream reasoning state to infer the review stage.
    """

    doc = _sample_retrieved_doc()
    sources = _sample_sources()

    def fake_route_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["route"] += 1
        return {
            "intent": "hoi_tinh_huong_thuc_te",
            "intent_score": 0.88,
            "risk_level": "high",
            "need_clarify": False,
            "unsupported_query": False,
            "human_review_required": True,
            "next_route": "human-review-path",
            "route_reason": "Câu hỏi rủi ro cao, cần duyệt trước khi chạy legal-agent-path.",
            "review_note": "Cần chuyên viên duyệt trước khi hệ thống tiếp tục xử lý.",
        }

    def fake_rewrite_query_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["rewrite"] += 1
        return {"rewritten_queries": [str(state.get("question") or ""), "quy định khởi kiện tranh chấp đất đai"]}

    def fake_retrieve_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["retrieve"] += 1
        return {
            "retrieved_docs": [deepcopy(doc)],
            "retrieval_debug": {"mode": "fake_pre_review"},
        }

    def fake_rerank_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["rerank"] += 1
        return {
            "reranked_docs": [deepcopy(doc)],
            "context": doc["content"],
            "sources": list(sources),
        }

    def fake_retrieval_check_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["retrieval_check"] += 1
        return {"retrieval_ok": True, "next_action": "proceed"}

    def fake_generate_draft_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["generate"] += 1
        review_response = str(state.get("review_response") or "")
        return {
            "final_answer": "",
            "draft_answer": (
                "Đã được chuyên viên xác nhận tiếp tục xử lý. "
                f"Phản hồi review: {review_response}. "
                "Theo Điều 36.3.LQ.10. của Luật số 57/2020/QH14, "
                "có trách nhiệm đối thoại với thanh niên ít nhất mỗi năm một lần."
            ),
            "draft_citations": list(sources),
            "draft_confidence": 0.93,
        }

    def fake_grounding_check_node(state: Mapping[str, Any]) -> dict[str, Any]:
        trace["grounding"] += 1
        return {
            "grounding_ok": True,
            "grounding_score": 0.93,
            "unsupported_claims": [],
            "missing_evidence": [],
            "next_action": "proceed",
        }

    def fake_pre_review_human_review_node(state: Mapping[str, Any]) -> dict[str, Any]:
        if not bool(state.get("human_review_required")):
            return {}
        review_response = str(state.get("review_response") or "").strip()
        if review_response:
            return {
                "human_review_required": False,
                "interrupt_payload": None,
                "next_action": "resume_legal_agent",
                "response_status": "",
                "status": "",
                "resume_kind": "",
                "resume_question": "",
                "review_note": f"Đã nhận phản hồi human review: {review_response}",
            }
        return {
            "interrupt_payload": {
                "kind": "human_review",
                "stage": "pre_retrieval",
                "thread_id": str(state.get("thread_id") or ""),
                "session_id": str(state.get("session_id") or ""),
                "question": str(state.get("normalized_question") or state.get("question") or ""),
                "review_note": str(state.get("review_note") or ""),
                "answer_preview": "",
                "sources": [],
            },
            "next_action": "human_review",
            "response_status": "review_required",
            "status": "review_required",
        }

    return GraphDependencies(
        route_node=fake_route_node,
        rewrite_query_node=fake_rewrite_query_node,
        retrieve_node=fake_retrieve_node,
        rerank_node=fake_rerank_node,
        retrieval_check_node=fake_retrieval_check_node,
        generate_draft_node=fake_generate_draft_node,
        grounding_check_node=fake_grounding_check_node,
        revise_answer_node=lambda state: {"final_answer": str(state.get("draft_answer") or "")},
        human_review_node=fake_pre_review_human_review_node,
    )


def _make_runtime(
    *,
    dependencies: GraphDependencies,
    checkpoint_store: InMemoryCheckpointStore | None = None,
) -> tuple[Any, InMemoryCheckpointStore]:
    store = checkpoint_store or InMemoryCheckpointStore()
    runtime = build_graph(
        app_config=replace(AppConfig(), max_reasoning_loops=1, max_retrieval_rounds=2),
        checkpoint_store=store,
        dependencies=dependencies,
    )
    return runtime, store


def _invoke_review_interrupt(runtime: Any, *, question: str, thread_id: str, session_id: str) -> AgentState:
    initial_state = create_initial_state(question=question, thread_id=thread_id, session_id=session_id)
    return runtime.invoke(initial_state)


def test_unsupported_route_is_not_overridden_by_clarify_waiting_state() -> None:
    def fake_unsupported_route_node(state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "intent": "hoi_tinh_huong_thuc_te",
            "intent_score": 0.82,
            "risk_level": "medium",
            "need_clarify": True,
            "clarify_reason": "muc_tieu_tra_loi",
            "clarify_question": "Bạn muốn hệ thống tập trung vào nội dung pháp luật nào?",
            "missing_slots": ["muc_tieu_tra_loi"],
            "unsupported_query": True,
            "next_route": "unsupported-path",
            "route_reason": "Câu hỏi có dấu hiệu nằm ngoài phạm vi hỗ trợ pháp luật của hệ thống.",
            "status": "",
            "response_status": "",
            "resume_kind": "",
            "resume_question": "",
        }

    runtime, _ = _make_runtime(
        dependencies=GraphDependencies(
            route_node=fake_unsupported_route_node,
            human_review_node=human_review_node,
        )
    )

    state = runtime.invoke(
        create_initial_state(
            question="Thời tiết Hà Nội hôm nay thế nào?",
            thread_id="thread-unsupported-001",
            session_id="session-unsupported-001",
        )
    )

    assert state["next_route"] == "unsupported-path"
    assert state["unsupported_query"] is True
    assert state["response_status"] == "unsupported"
    assert state["status"] == "unsupported"
    assert state["resume_kind"] == ""
    assert state["resume_question"] == ""
    assert state["interrupt_payload"] is None
    assert state["sources"] == []
    assert state["citation_findings"] == {}
    assert "ngoài phạm vi" in str(state["final_answer"]).lower()


def test_interrupt_when_human_review_required() -> None:
    trace = _make_trace()
    runtime, store = _make_runtime(dependencies=_build_post_reasoning_dependencies(trace))

    state = _invoke_review_interrupt(
        runtime,
        question="Theo Luật Thanh niên, đối thoại với thanh niên được quy định thế nào?",
        thread_id="thread-review-001",
        session_id="session-review-001",
    )

    assert state["response_status"] == "waiting_user_input"
    assert state["interrupt_payload"] is not None
    assert state["interrupt_payload"]["kind"] == "human_review"
    assert state["interrupt_payload"]["stage"] == "post_reasoning"
    assert state["thread_id"] == "thread-review-001"
    assert state["session_id"] == "session-review-001"
    assert state["review_note"] == "Cần chuyên viên pháp lý xác nhận trước khi phát hành câu trả lời."
    assert state["draft_answer"]
    assert state["human_review_required"] is True
    assert state["status"] == "waiting_user_input"
    assert state["resume_kind"] == "human_review"
    assert state["resume_question"]
    assert store.exists(thread_id="thread-review-001", session_id="session-review-001") is True


def test_checkpoint_saved_before_resume() -> None:
    trace = _make_trace()
    runtime, store = _make_runtime(dependencies=_build_post_reasoning_dependencies(trace))

    _invoke_review_interrupt(
        runtime,
        question="Theo Luật Thanh niên, đối thoại với thanh niên được quy định thế nào?",
        thread_id="thread-review-002",
        session_id="session-review-002",
    )
    saved = store.get_record(thread_id="thread-review-002", session_id="session-review-002")

    assert saved["thread_id"] == "thread-review-002"
    assert saved["session_id"] == "session-review-002"
    assert saved["question"] == "Theo Luật Thanh niên, đối thoại với thanh niên được quy định thế nào?"
    assert saved["next_route"] == "legal-agent-path"
    assert saved["interrupt_payload"]["kind"] == "human_review"
    assert saved["interrupt_payload"]["stage"] == "post_reasoning"
    assert saved["response_status"] == "waiting_user_input"
    assert saved["status"] == "waiting_user_input"
    assert saved["resume_kind"] == "human_review"
    assert saved["resume_question"]
    assert saved["loop_count"] == 0
    assert isinstance(saved["history"], list)


def test_resume_from_checkpoint_success() -> None:
    trace = _make_trace()
    runtime, store = _make_runtime(dependencies=_build_post_reasoning_dependencies(trace))

    _invoke_review_interrupt(
        runtime,
        question="Theo Luật Thanh niên, đối thoại với thanh niên được quy định thế nào?",
        thread_id="thread-review-003",
        session_id="session-review-003",
    )
    resumed = runtime.resume(
        thread_id="thread-review-003",
        session_id="session-review-003",
        review_response="Đồng ý phát hành câu trả lời này.",
    )

    assert resumed["interrupt_payload"] is None
    assert resumed["human_review_required"] is False
    assert resumed["response_status"] == "ok"
    assert resumed["status"] == "ok"
    assert resumed["resume_kind"] == ""
    assert resumed["resume_question"] == ""
    assert resumed["final_answer"]
    assert "Điều 36.3.LQ.10." in resumed["final_answer"]
    assert "Đồng ý phát hành câu trả lời này." in resumed["review_note"]
    assert store.load_calls >= 1


def test_resume_missing_checkpoint_returns_error() -> None:
    trace = _make_trace()
    runtime, _ = _make_runtime(dependencies=_build_post_reasoning_dependencies(trace))

    with pytest.raises(CheckpointNotFoundError):
        runtime.resume(
            thread_id="thread-does-not-exist",
            session_id="session-does-not-exist",
            review_response="Đồng ý",
        )


def test_resume_does_not_restart_from_beginning() -> None:
    trace = _make_trace()
    runtime, _ = _make_runtime(dependencies=_build_post_reasoning_dependencies(trace))

    _invoke_review_interrupt(
        runtime,
        question="Theo Luật Thanh niên, đối thoại với thanh niên được quy định thế nào?",
        thread_id="thread-review-004",
        session_id="session-review-004",
    )
    counts_before_resume = dict(trace)

    resumed = runtime.resume(
        thread_id="thread-review-004",
        session_id="session-review-004",
        review_response="Đồng ý phát hành.",
    )

    assert resumed["response_status"] == "ok"
    assert trace["route"] == counts_before_resume["route"]
    assert trace["rewrite"] == counts_before_resume["rewrite"]
    assert trace["retrieve"] == counts_before_resume["retrieve"]
    assert trace["rerank"] == counts_before_resume["rerank"]
    assert trace["retrieval_check"] == counts_before_resume["retrieval_check"]
    assert trace["generate"] == counts_before_resume["generate"]
    assert trace["grounding"] == counts_before_resume["grounding"]


def test_resume_preserves_state_fields() -> None:
    trace = _make_trace()
    runtime, store = _make_runtime(dependencies=_build_post_reasoning_dependencies(trace))

    interrupted = _invoke_review_interrupt(
        runtime,
        question="Theo Luật Thanh niên, đối thoại với thanh niên được quy định thế nào?",
        thread_id="thread-review-005",
        session_id="session-review-005",
    )
    resumed = runtime.resume(
        thread_id="thread-review-005",
        session_id="session-review-005",
        review_response="Đồng ý phát hành.",
    )

    assert resumed["question"] == interrupted["question"]
    assert resumed["normalized_question"] == interrupted["normalized_question"]
    assert resumed["intent"] == interrupted["intent"]
    assert resumed["risk_level"] == interrupted["risk_level"]
    assert resumed["retrieved_docs"] == interrupted["retrieved_docs"]
    assert resumed["reranked_docs"] == interrupted["reranked_docs"]
    assert resumed["context"] == interrupted["context"]
    assert resumed["draft_answer"] == interrupted["draft_answer"]
    assert resumed["thread_id"] == interrupted["thread_id"]
    assert resumed["session_id"] == interrupted["session_id"]
    assert resumed["review_note"].startswith("Đã nhận phản hồi human review:")
    assert store.exists(thread_id="thread-review-005", session_id="session-review-005") is True


def test_resume_updates_final_answer_after_review() -> None:
    trace = _make_trace()
    runtime, store = _make_runtime(dependencies=_build_pre_review_dependencies(trace))

    interrupted = _invoke_review_interrupt(
        runtime,
        question="Tôi có nên khởi kiện tranh chấp đất đai không?",
        thread_id="thread-review-006",
        session_id="session-review-006",
    )
    resumed = runtime.resume(
        thread_id="thread-review-006",
        session_id="session-review-006",
        review_response="Đồng ý cho hệ thống tiếp tục xử lý và phát hành câu trả lời.",
    )

    assert interrupted["response_status"] == "waiting_user_input"
    assert interrupted["interrupt_payload"]["stage"] == "pre_retrieval"
    assert resumed["response_status"] == "ok"
    assert "Đồng ý cho hệ thống tiếp tục xử lý" in resumed["final_answer"]
    assert resumed["final_answer"] != interrupted["final_answer"]
    assert resumed["interrupt_payload"] is None
    assert store.exists(thread_id="thread-review-006", session_id="session-review-006") is True


def test_resume_session_thread_mismatch_handled() -> None:
    trace = _make_trace()
    runtime, _ = _make_runtime(dependencies=_build_post_reasoning_dependencies(trace))

    _invoke_review_interrupt(
        runtime,
        question="Theo Luật Thanh niên, đối thoại với thanh niên được quy định thế nào?",
        thread_id="thread-review-007",
        session_id="session-review-007",
    )

    with pytest.raises(CheckpointNotFoundError):
        runtime.resume(
            thread_id="thread-review-007",
            session_id="session-other",
            review_response="Đồng ý",
        )


def test_build_chat_response_exposes_intent_and_clarify_fields() -> None:
    state = create_initial_state(
        question="Quyền của thanh niên là gì?",
        thread_id="thread-chat-response-001",
        session_id="session-chat-response-001",
    )
    state.update(
        {
            "response_status": "waiting_user_input",
            "final_answer": "",
            "intent": "hoi_dinh_nghia",
            "intent_score": 0.91,
            "next_route": "clarify-path",
            "risk_level": "low",
            "need_clarify": True,
            "missing_slots": ["van_ban_phap_luat"],
            "clarify_question": "Bạn đang muốn hỏi theo luật hoặc văn bản pháp luật nào cụ thể?",
        }
    )

    response = build_chat_response(state)

    assert response.intent == "hoi_dinh_nghia"
    assert response.intent_score == pytest.approx(0.91)
    assert response.route == "clarify-path"
    assert response.risk_level == "low"
    assert response.need_clarify is True
    assert response.missing_slots == ["van_ban_phap_luat"]
    assert response.clarify_question == "Bạn đang muốn hỏi theo luật hoặc văn bản pháp luật nào cụ thể?"
    assert response.resume_question == "Bạn đang muốn hỏi theo luật hoặc văn bản pháp luật nào cụ thể?"
