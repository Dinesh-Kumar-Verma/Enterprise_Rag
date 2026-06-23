"""
EnterpriseRAG — Streamlit Chat Frontend
Features: streaming answers, source citation cards, feedback, stats sidebar
"""

import json
import time
from typing import Any

import os
import requests
import streamlit as st
import websocket

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="EnterpriseRAG",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.source-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
}
.score-badge {
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
}
.grounded-badge {
    background: #e8f5e9;
    color: #1b5e20;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 12px;
}
.hallucinated-badge {
    background: #ffebee;
    color: #b71c1c;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "avg_latency" not in st.session_state:
    st.session_state.avg_latency = 0.0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏢 EnterpriseRAG")
    st.caption("Production-grade Knowledge Assistant")

    st.divider()
    st.subheader("⚙️ Settings")
    use_hyde = st.toggle("HyDE Query Expansion", value=True,
                         help="Generates a hypothetical document to improve retrieval")
    streaming = st.toggle("Streaming Response", value=True)

    st.divider()
    st.subheader("📥 Ingest Documents")

    ingest_tab = st.radio("Source type", ["File", "URL", "Text"], horizontal=True)

    if ingest_tab == "File":
        uploaded = st.file_uploader(
            "Upload document", type=["pdf", "docx", "txt", "md", "csv"]
        )
        if st.button("Ingest File", disabled=not uploaded, use_container_width=True):
            with st.spinner("Ingesting..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest/file",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
                        timeout=120,
                    )
                    data = resp.json()
                    st.success(f"✓ {data['chunks']} chunks added from {uploaded.name}")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    elif ingest_tab == "URL":
        url_input = st.text_input("URL", placeholder="https://docs.example.com/api")
        if st.button("Ingest URL", disabled=not url_input, use_container_width=True):
            with st.spinner("Fetching and ingesting..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest/url",
                        json={"url": url_input},
                        timeout=60,
                    )
                    data = resp.json()
                    st.success(f"✓ {data['chunks']} chunks added from URL")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    elif ingest_tab == "Text":
        text_input = st.text_area("Paste text", height=120)
        source_name = st.text_input("Source name", value="manual")
        if st.button("Ingest Text", disabled=not text_input, use_container_width=True):
            with st.spinner("Ingesting..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest/text",
                        json={"text": text_input, "source_name": source_name},
                        timeout=30,
                    )
                    data = resp.json()
                    st.success(f"✓ {data['chunks']} chunks added")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()
    st.subheader("📊 Stats")
    if st.button("Refresh Stats", use_container_width=True):
        try:
            resp = requests.get(f"{API_BASE}/stats", timeout=5)
            stats = resp.json()
            vs = stats.get("vector_store", {})
            st.metric("Total Chunks", vs.get("total_chunks", "N/A"))
            st.metric("Session Queries", st.session_state.total_queries)
        except Exception:
            st.warning("API not reachable")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat UI ──────────────────────────────────────────────────────────────
st.title("Ask your knowledge base")

def _render_sources(sources: list[dict], meta: dict) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} sources", expanded=False):
        cols = st.columns(min(len(sources), 3))
        for i, src in enumerate(sources):
            with cols[i % 3]:
                score = src.get("relevance_score", 0)
                st.markdown(
                    f"""<div class="source-card">
                    <strong>[{src['index']}] {src['source_name']}</strong><br>
                    <span class="score-badge">score: {score:.2f}</span>
                    &nbsp;<em>{src['source_type']}</em><br>
                    <small>{src['preview']}</small>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if src.get("url"):
                    st.caption(f"🔗 [{src['url'][:40]}...]({src['url']})")

        latencies = meta.get("latencies", {})
        if latencies:
            cols2 = st.columns(3)
            for col, (stage, t) in zip(cols2, latencies.items()):
                col.metric(stage.title(), f"{t:.2f}s")

        grounded = meta.get("is_grounded", True)
        badge = "grounded-badge" if grounded else "hallucinated-badge"
        label = "✓ Grounded" if grounded else "⚠ Hallucination risk"
        st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            _render_sources(msg["sources"], msg.get("meta", {}))




def _query_streaming(query: str) -> tuple[str, list[dict], dict]:
    ws_url = API_BASE.replace("http", "ws") + "/ws/query"
    full_answer = ""
    sources: list[dict] = []
    meta: dict = {}

    placeholder = st.empty()

    try:
        ws = websocket.create_connection(ws_url, timeout=60)
        ws.send(json.dumps({"query": query, "use_hyde": use_hyde}))

        while True:
            raw = ws.recv()
            chunk = json.loads(raw)

            if chunk["type"] == "sources":
                sources = chunk["data"]
            elif chunk["type"] == "token":
                full_answer += chunk["data"]
                placeholder.markdown(full_answer + "▌")
            elif chunk["type"] == "done":
                break
            elif chunk["type"] == "error":
                st.error(chunk["data"])
                break

        ws.close()
        placeholder.markdown(full_answer)
    except Exception as e:
        st.error(f"Streaming error: {e}. Falling back to REST...")
        full_answer, sources, meta = _query_rest(query)

    return full_answer, sources, meta


def _query_rest(query: str) -> tuple[str, list[dict], dict]:
    resp = requests.post(
        f"{API_BASE}/query",
        json={"query": query, "use_hyde": use_hyde},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data["sources"], {
        "latencies": data["latencies"],
        "is_grounded": data["is_grounded"],
        "query_type": data["query_type"],
    }


if prompt := st.chat_input("Ask anything about your knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if streaming:
                answer, sources, meta = _query_streaming(prompt)
            else:
                with st.spinner("Thinking..."):
                    answer, sources, meta = _query_rest(prompt)
                st.markdown(answer)

            _render_sources(sources, meta)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "meta": meta}
            )
            st.session_state.total_queries += 1

        except Exception as e:
            st.error(f"Error: {e}")
