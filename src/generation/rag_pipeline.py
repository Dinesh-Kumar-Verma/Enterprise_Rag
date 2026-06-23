"""
LangGraph-based generation pipeline:
  query_router → generate → hallucination_check → respond
"""

from __future__ import annotations

from typing import AsyncIterator, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from loguru import logger

from config.settings import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are EnterpriseRAG, an intelligent knowledge assistant.
Answer questions accurately using ONLY the provided context.
Always cite sources using [Source N] notation.
If the context doesn't contain enough information, say so clearly — do not hallucinate.
Be concise but thorough. Use markdown formatting for clarity."""

GENERATION_PROMPT = """Context:
{context}

Question: {query}

Provide a comprehensive answer with source citations [Source N].
If multiple sources support a point, cite all relevant ones."""

HALLUCINATION_CHECK_PROMPT = """You are a factual accuracy checker.
Review this answer and determine if it contains claims NOT supported by the provided context.

Context:
{context}

Answer:
{answer}

Respond with ONLY: "GROUNDED" if all claims are supported, or "HALLUCINATED" if any claim lacks support."""

ROUTER_PROMPT = """Classify this query into one of: 
- RETRIEVAL: needs knowledge base search
- CONVERSATIONAL: general chat, greetings, thanks
- OUT_OF_SCOPE: completely unrelated to enterprise knowledge

Query: {query}
Answer with ONLY the category name."""


class RAGState(TypedDict):
    query: str
    context: str
    sources: list[dict]
    answer: str
    query_type: str
    is_grounded: bool
    attempt: int


class RAGPipeline:
    def __init__(self):
        self._llm: ChatGoogleGenerativeAI | None = None
        self._graph = None

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.gemini_api_key,
                temperature=settings.temperature,
            )
        return self._llm

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _route_query(self, state: RAGState) -> RAGState:
        try:
            response = self.llm.invoke(ROUTER_PROMPT.format(query=state["query"]))
            query_type = response.content.strip().upper()
            if query_type not in {"RETRIEVAL", "CONVERSATIONAL", "OUT_OF_SCOPE"}:
                query_type = "RETRIEVAL"
        except Exception:
            query_type = "RETRIEVAL"
        logger.debug(f"Query type: {query_type}")
        return {**state, "query_type": query_type}

    def _generate(self, state: RAGState) -> RAGState:
        if state["query_type"] == "CONVERSATIONAL":
            response = self.llm.invoke(
                [HumanMessage(content=state["query"])]
            )
            return {**state, "answer": response.content, "is_grounded": True}

        if state["query_type"] == "OUT_OF_SCOPE":
            return {
                **state,
                "answer": "I can only answer questions about the enterprise knowledge base.",
                "is_grounded": True,
            }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=GENERATION_PROMPT.format(
                    context=state["context"], query=state["query"]
                )
            ),
        ]
        response = self.llm.invoke(messages)
        return {**state, "answer": response.content}

    def _check_hallucination(self, state: RAGState) -> RAGState:
        if state["query_type"] != "RETRIEVAL" or state.get("is_grounded"):
            return {**state, "is_grounded": True}

        try:
            prompt = (
                HALLUCINATION_CHECK_PROMPT
                .replace("{context}", state["context"][:4000])
                .replace("{answer}", state["answer"])
            )
            response = self.llm.invoke(prompt)
    
            is_grounded = "GROUNDED" in response.content.upper()
        except Exception:
            is_grounded = True

        logger.debug(f"Hallucination check: {'GROUNDED' if is_grounded else 'HALLUCINATED'}")
        return {**state, "is_grounded": is_grounded}

    def _should_retry(self, state: RAGState) -> str:
        if state["is_grounded"] or state.get("attempt", 0) >= 1:
            return "end"
        return "regenerate"

    def _regenerate(self, state: RAGState) -> RAGState:
        logger.warning("Regenerating due to hallucination detection")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT + "\nIMPORTANT: Only use information explicitly stated in the context."),
            HumanMessage(
                content=GENERATION_PROMPT.format(
                    context=state["context"], query=state["query"]
                )
            ),
        ]
        response = self.llm.invoke(messages)
        return {**state, "answer": response.content, "attempt": 1}

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("generate", self._generate)
        workflow.add_node("check_hallucination", self._check_hallucination)
        workflow.add_node("regenerate", self._regenerate)

        workflow.set_entry_point("route_query")
        workflow.add_edge("route_query", "generate")
        workflow.add_edge("generate", "check_hallucination")
        workflow.add_conditional_edges(
            "check_hallucination",
            self._should_retry,
            {"end": END, "regenerate": "regenerate"},
        )
        workflow.add_edge("regenerate", "check_hallucination")


        return workflow.compile()

    def run(self, query: str, context: str, sources: list[dict]) -> dict:
        initial_state: RAGState = {
            "query": query,
            "context": context,
            "sources": sources,
            "answer": "",
            "query_type": "RETRIEVAL",
            "is_grounded": False,
            "attempt": 0,
        }
        result = self.graph.invoke(initial_state)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "query_type": result["query_type"],
            "is_grounded": result["is_grounded"],
        }

    async def stream(
        self, query: str, context: str, sources: list[dict]
    ) -> AsyncIterator[str]:
        """Stream tokens from the generation step."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=GENERATION_PROMPT.format(context=context, query=query)
            ),
        ]
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content
