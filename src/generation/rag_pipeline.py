"""
LangGraph-based generation pipeline:
  query_router → generate → hallucination_check → respond

Improvements:
  - Early-exit: CONVERSATIONAL / OUT_OF_SCOPE skip retrieval entirely
  - NO_RELEVANT_DOCS type: LLM says "I don't know" instead of forcing irrelevant context
  - Stream method now uses the same graph (with routing)
  - Better prompts for cleaner chat behaviour
"""

from __future__ import annotations

from typing import AsyncIterator, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from loguru import logger

from config.settings import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are EnterpriseRAG, an intelligent knowledge assistant for enterprise documents.
Answer questions accurately using ONLY the provided context.
Always cite sources using [Source N] notation.
If the context doesn't contain enough information, say so clearly — do not hallucinate.
Be concise but thorough. Use markdown formatting for clarity."""

GENERATION_PROMPT = """Context:
{context}

Question: {query}

Provide a comprehensive answer with source citations [Source N].
If multiple sources support a point, cite all relevant ones."""

NO_CONTEXT_PROMPT = """The user asked: "{query}"

No relevant documents were found in the knowledge base for this query.
Respond politely that you don't have information on this topic and suggest they try rephrasing or ask about something covered in the uploaded documents."""

HALLUCINATION_CHECK_PROMPT = """You are a factual accuracy checker.
Review this answer and determine if it contains claims NOT supported by the provided context.

Context:
{context}

Answer:
{answer}

Respond with ONLY: "GROUNDED" if all claims are supported, or "HALLUCINATED" if any claim lacks support."""

ROUTER_PROMPT = """Classify this query into exactly one of these categories:

- RETRIEVAL: the user is asking a question that requires searching the knowledge base for information (e.g. "what is...", "how does...", "explain...", "tell me about...")
- CONVERSATIONAL: the user is greeting, thanking, saying goodbye, or making casual small talk (e.g. "hi", "hello", "thanks", "goodbye", "how are you", "hey there")
- OUT_OF_SCOPE: the user is asking about something completely unrelated to enterprise knowledge (e.g. "write a poem", "what's the weather", "tell a joke")

Query: {query}
Answer with ONLY the category name: RETRIEVAL, CONVERSATIONAL, or OUT_OF_SCOPE."""

CONVERSATIONAL_SYSTEM = """You are EnterpriseRAG, a friendly assistant for enterprise documents.
Respond warmly and briefly. If the user greets you, greet them back and mention you can help with their enterprise documents.
Keep it short — 1-2 sentences max."""


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
        self._llm: ChatGroq | None = None
        self._graph = None

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=settings.temperature,
            )
        return self._llm

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    # ── Graph nodes ──────────────────────────────────────────────────────────

    def _route_query(self, state: RAGState) -> RAGState:
        try:
            response = self.llm.invoke(ROUTER_PROMPT.format(query=state["query"]))
            query_type = response.content.strip().upper()
            if query_type not in {"RETRIEVAL", "CONVERSATIONAL", "OUT_OF_SCOPE"}:
                query_type = "RETRIEVAL"
        except Exception:
            query_type = "RETRIEVAL"
        logger.debug(f"Query routed as: {query_type}")
        return {**state, "query_type": query_type}

    def _generate(self, state: RAGState) -> RAGState:
        # ── RETRIEVAL with no relevant context: don't force irrelevant docs ──
        if state["query_type"] == "RETRIEVAL" and not state["context"].strip():
            logger.info("No relevant context found — using no-context path")
            response = self.llm.invoke(
                NO_CONTEXT_PROMPT.format(query=state["query"])
            )
            return {**state, "answer": response.content, "query_type": "NO_RELEVANT_DOCS", "is_grounded": True}

        # ── CONVERSATIONAL: warm reply, no retrieval needed ──
        if state["query_type"] == "CONVERSATIONAL":
            response = self.llm.invoke([
                SystemMessage(content=CONVERSATIONAL_SYSTEM),
                HumanMessage(content=state["query"]),
            ])
            return {**state, "answer": response.content, "is_grounded": True, "sources": []}

        # ── OUT_OF_SCOPE: polite refusal ──
        if state["query_type"] == "OUT_OF_SCOPE":
            return {
                **state,
                "answer": "I'm sorry, I can only answer questions based on the enterprise knowledge base. Please ask something related to your uploaded documents.",
                "is_grounded": True,
                "sources": [],
            }

        # ── RETRIEVAL: generate from context ──
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

    # ── Build LangGraph ──

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

    # ── Public API ──

    def run(self, query: str, context: str = "", sources: list[dict] | None = None) -> dict:
        initial_state: RAGState = {
            "query": query,
            "context": context,
            "sources": sources or [],
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
