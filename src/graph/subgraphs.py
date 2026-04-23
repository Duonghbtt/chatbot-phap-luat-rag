from __future__ import annotations

import logging
from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph

from src.graph.state import AgentState
from src.tv3_retrieval.rerank_node import rerank_node
from src.tv3_retrieval.retrieve_node import retrieve_node
from src.tv3_retrieval.retrieval_check_node import retrieval_check_node
from src.tv3_retrieval.rewrite_query_node import rewrite_query_node
from src.tv5_reasoning.generate_draft_node import generate_draft_node
from src.tv5_reasoning.grounding_check_node import grounding_check_node
from src.tv5_reasoning.revise_answer_node import revise_answer_node

LOGGER = logging.getLogger(__name__)


def _prepare_revision_retry_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Promote the revised answer into `draft_answer` before another grounding pass."""

    revised_answer = str(state.get("final_answer") or "").strip()
    if not revised_answer:
        return {}
    return {
        "draft_answer": revised_answer,
        "loop_count": int(state.get("loop_count") or 0) + 1,
    }


def _route_retrieval_decision(state: Mapping[str, Any]) -> str:
    return str(state.get("next_action") or "fallback")


def _make_grounding_router(max_reasoning_loops: int):
    def route_grounding_decision(state: Mapping[str, Any]) -> str:
        action = str(state.get("next_action") or "proceed")
        if action == "revise" and int(state.get("loop_count") or 0) >= max_reasoning_loops:
            return "human_review"
        return action

    return route_grounding_decision


def build_legal_agent_subgraph(*, logger: logging.Logger | None = None, max_reasoning_loops: int = 2) -> Any:
    """Build the reusable legal-agent subgraph by composing TV3 + TV5 nodes."""

    _ = logger or LOGGER
    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query_node", rewrite_query_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("rerank_node", rerank_node)
    graph.add_node("retrieval_check_node", retrieval_check_node)
    graph.add_node("generate_draft_node", generate_draft_node)
    graph.add_node("grounding_check_node", grounding_check_node)
    graph.add_node("revise_answer_node", revise_answer_node)
    graph.add_node("prepare_revision_retry_node", _prepare_revision_retry_node)

    graph.add_edge(START, "rewrite_query_node")
    graph.add_edge("rewrite_query_node", "retrieve_node")
    graph.add_edge("retrieve_node", "rerank_node")
    graph.add_edge("rerank_node", "retrieval_check_node")
    graph.add_conditional_edges(
        "retrieval_check_node",
        _route_retrieval_decision,
        {
            "retry": "retrieve_node",
            "proceed": "generate_draft_node",
            "fallback": END,
        },
    )
    graph.add_edge("generate_draft_node", "grounding_check_node")
    graph.add_conditional_edges(
        "grounding_check_node",
        _make_grounding_router(max_reasoning_loops),
        {
            "proceed": END,
            "revise": "revise_answer_node",
            "retrieve_again": "retrieve_node",
            "human_review": END,
        },
    )
    graph.add_edge("revise_answer_node", "prepare_revision_retry_node")
    graph.add_edge("prepare_revision_retry_node", "grounding_check_node")
    return graph.compile()


def build_review_subgraph(*, logger: logging.Logger | None = None) -> Any:
    """Build the reusable human-review subgraph."""

    _ = logger or LOGGER
    from src.graph.human_review_node import human_review_node

    graph = StateGraph(AgentState)
    graph.add_node("human_review_node", human_review_node)
    graph.add_edge(START, "human_review_node")
    graph.add_edge("human_review_node", END)
    return graph.compile()


__all__ = ["build_legal_agent_subgraph", "build_review_subgraph"]
