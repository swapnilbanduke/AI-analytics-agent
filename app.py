"""
Streamlit frontend for the AI Data Analyst Assistant.
"""

import os

import streamlit as st

from config import (
    DEFAULT_PROVIDER,
    get_default_model,
    get_model_values,
    get_provider_config,
    get_provider_ids,
    resolve_api_key,
    resolve_model,
)
from graph import build_agent_graph, process_question
from styles import inject_styles


# File extensions that go to SQL (structured data)
DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}
# File extensions that go to RAG (documents)
DOC_EXTENSIONS = {".pdf", ".txt", ".md", ".doc", ".docx"}


# ========================
# Session state
# ========================

def init_session_state():
    defaults = {
        "chat_history": [],
        "provider": DEFAULT_PROVIDER,
        "model_name": "",
        "api_key": "",
        "tavily_api_key": "",
        "processing": False,
        "tables": [],
        "has_documents": False,
        "uploaded_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ========================
# Smart file upload — auto-detect type
# ========================

def handle_file_upload(uploaded_file):
    """Auto-detect file type and route to SQL ingestion or RAG indexing."""
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower()

    if ext in DATA_EXTENSIONS:
        _ingest_data(uploaded_file, file_name, ext)
    elif ext in DOC_EXTENSIONS:
        _index_document(uploaded_file, file_name, ext)
    else:
        st.error(f"Unsupported file type: {ext}")


def _ingest_data(uploaded_file, file_name, ext):
    """Ingest CSV/Excel into SQLite database."""
    from database import list_tables
    from ingestion import ingest_csv, ingest_excel

    file_bytes = uploaded_file.getvalue()
    table_name = file_name.rsplit(".", 1)[0]

    if ext == ".csv":
        result = ingest_csv(file_bytes, table_name=table_name)
    else:
        result = ingest_excel(file_bytes)

    st.session_state["tables"] = list_tables()
    st.session_state["uploaded_files"].append(file_name)

    if isinstance(result, dict) and result.get("success"):
        rows = result.get("rows", "?")
        tbl = result.get("table_name", table_name)
        st.success(f"**{file_name}** → SQL table `{tbl}` ({rows} rows)")
    elif isinstance(result, dict) and result.get("tables"):
        for t in result["tables"]:
            st.success(f"Sheet **{t['original_sheet_name']}** → `{t['sql_table_name']}` ({t['rows']} rows)")


def _index_document(uploaded_file, file_name, ext):
    """Index PDF/text/Word into RAG vector store."""
    import tempfile

    try:
        from rag import add_documents

        file_bytes = uploaded_file.getvalue()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        chunks = add_documents(tmp_path, collection_name="default")
        os.unlink(tmp_path)

        st.session_state["has_documents"] = True
        st.session_state["uploaded_files"].append(file_name)
        st.success(f"**{file_name}** → Indexed for document search ({chunks} chunks)")

    except Exception as exc:
        st.error(f"Error indexing document: {exc}")


# ========================
# Sidebar
# ========================

def render_sidebar():
    with st.sidebar:
        st.markdown("### Configuration")

        # Provider selection
        providers = get_provider_ids()
        provider_labels = [get_provider_config(p)["label"] for p in providers]
        selected_idx = st.selectbox(
            "Provider",
            range(len(providers)),
            format_func=lambda i: provider_labels[i],
            index=providers.index(st.session_state["provider"]),
        )
        provider = providers[selected_idx]
        st.session_state["provider"] = provider

        # API key
        config = get_provider_config(provider)
        api_key_input = st.text_input(
            config["api_key_label"],
            type="password",
            placeholder=config["api_key_help"],
            value=st.session_state.get("api_key", ""),
        )
        st.session_state["api_key"] = api_key_input

        resolved_key = resolve_api_key(provider, api_key_input)
        if resolved_key:
            st.session_state["_resolved_api_key"] = resolved_key
            # Also set env var so tools can find it regardless of session state timing
            config = get_provider_config(provider)
            os.environ[config["env_var"]] = resolved_key
            st.success("API key set")
        else:
            st.warning("Enter API key or set in .env")

        # Model selection
        models = get_model_values(provider)
        default_model = get_default_model(provider)
        default_idx = models.index(default_model) if default_model in models else 0
        selected_model = st.selectbox("Model", models, index=default_idx)
        st.session_state["model_name"] = selected_model

        # Tavily API key for web search
        tavily_key = st.text_input(
            "Tavily API Key (web search)",
            type="password",
            placeholder="tvly-...",
            value=st.session_state.get("tavily_api_key", os.environ.get("TAVILY_API_KEY", "")),
        )
        st.session_state["tavily_api_key"] = tavily_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        st.divider()

        # --- Single unified file upload ---
        st.markdown("### Upload Files")
        st.caption("CSV/Excel → SQL database | PDF/Text → Document search")
        uploaded = st.file_uploader(
            "Drop any file",
            type=["csv", "xlsx", "xls", "pdf", "txt", "md"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded:
            for f in uploaded:
                if f.name not in st.session_state.get("uploaded_files", []):
                    handle_file_upload(f)

        # Show what's loaded
        if st.session_state.get("tables"):
            st.caption(f"SQL tables: {', '.join(st.session_state['tables'])}")
        if st.session_state.get("has_documents"):
            doc_names = [n for n in st.session_state.get("uploaded_files", [])
                         if os.path.splitext(n)[1].lower() in DOC_EXTENSIONS]
            if doc_names:
                st.caption(f"Documents: {', '.join(doc_names)}")

        st.divider()

        # Clear chat
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

        st.divider()
        st.caption("AI Data Analyst Assistant")


# ========================
# Welcome screen
# ========================

def render_welcome():
    st.markdown("""
    <div class="welcome-container">
        <h1>AI Data Analyst</h1>
        <p>Ask me anything — I'll calculate, search the web, query your data,
        or search your documents to find the answer.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    features = [
        ("🔢", "Calculator", "Math & percentages"),
        ("🌐", "Web Search", "Current events & data"),
        ("🗄️", "SQL Queries", "Ask about your data"),
        ("📄", "Doc Search", "Search your documents"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 2rem">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ========================
# Graph visualization
# ========================

def render_graph_tab():
    """Render the LangGraph agent architecture as a visual diagram."""
    st.markdown("### Agent Graph Architecture")
    st.markdown("This is the live LangGraph that processes every question.")

    has_db = bool(st.session_state.get("tables"))
    has_docs = st.session_state.get("has_documents", False)

    # Render the real graph from LangGraph as a visual Mermaid diagram
    try:
        graph = build_agent_graph(has_database=has_db, has_documents=has_docs)

        # Build clean Mermaid from the actual graph edges
        # (LangGraph's draw_mermaid() has syntax bugs that break browser rendering)
        g = graph.get_graph()
        lines = ["graph TD"]
        # Define styled nodes
        lines.append('    START(["Start"]) :::startNode')
        node_emojis = {
            "classify": "Classify",
            "agent": "Agent (LLM)",
            "tools": "Tools",
            "handle_error": "Error Handler",
        }
        for node_id in g.nodes:
            if node_id in ("__start__", "__end__"):
                continue
            label = node_emojis.get(node_id, node_id)
            lines.append(f'    {node_id}["{label}"]')
        lines.append('    FINISH(["End"]) :::endNode')

        # Add edges from the real graph
        for edge in g.edges:
            src = edge.source if hasattr(edge, "source") else edge[0]
            tgt = edge.target if hasattr(edge, "target") else edge[1]
            cond = getattr(edge, "data", None) if hasattr(edge, "data") else (edge[2] if len(edge) > 2 else None)

            src_id = "START" if src == "__start__" else src
            tgt_id = "FINISH" if tgt == "__end__" else tgt

            # Improve edge labels for readability
            label = cond
            if cond == "end":
                label = "done"
            if src_id == "agent" and tgt_id == "tools":
                label = "tool calls"

            if label:
                lines.append(f'    {src_id} -->|"{label}"| {tgt_id}')
            else:
                lines.append(f"    {src_id} --> {tgt_id}")

        # Style
        lines.append("    classDef startNode fill:#2d6a4f,stroke:#52b788,color:#fff")
        lines.append("    classDef endNode fill:#9d4edd,stroke:#c77dff,color:#fff")

        mermaid_code = "\n".join(lines)

        import streamlit.components.v1 as components
        mermaid_html = f"""
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            <style>
                body {{ margin: 0; background: transparent; }}
                .mermaid {{ display: flex; justify-content: center; }}
            </style>
        </head>
        <body>
            <div class="mermaid">
{mermaid_code}
            </div>
            <script>
                mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
            </script>
        </body>
        </html>
        """
        components.html(mermaid_html, height=450, scrolling=True)
        st.caption("Live graph — auto-generated from LangGraph StateGraph")
    except Exception as exc:
        st.warning(f"Could not render graph: {exc}")

    st.divider()
    st.markdown("### Detailed Flow")
    st.markdown("""
```
User Question
     │
     ▼
┌──────────────┐
│   Classify   │  LLM picks the best route
└──────┬───────┘
       │
       ├── calculation ──→ 🔢 Calculator Tool
       ├── web_search ───→ 🌐 Tavily Web Search
       ├── sql ──────────→ 🗄️ SQL Query Tool (NL → SQL → Execute)
       ├── document ─────→ 📄 Document Search (ChromaDB RAG)
       └── direct ───────→ 💬 LLM Direct Answer
              │
              ▼
       ┌─────────────┐
       │ Agent Loop   │  LLM ↔ Tools (repeat until done)
       └──────┬──────┘
              │
              ▼
         Final Answer
```
""")

    # Show active tools
    st.markdown("### Active Tools")
    tools_data = [
        ("calculator", "Safe math evaluation via AST", True),
        ("web_search", "Tavily search API", bool(os.environ.get("TAVILY_API_KEY"))),
        ("sql_query", f"NL2SQL on tables: {', '.join(st.session_state.get('tables', []))}" if has_db else "No data uploaded", has_db),
        ("document_search", "ChromaDB vector similarity", has_docs),
    ]
    for name, desc, active in tools_data:
        status = "Active" if active else "Inactive"
        icon = "🟢" if active else "⚪"
        st.markdown(f"{icon} **{name}** — {desc} *({status})*")


# ========================
# Chat display
# ========================

ROUTE_LABELS = {
    "calculation": "Calculator",
    "web_search": "Web Search",
    "sql": "SQL Query",
    "document": "Doc Search",
    "direct": "Direct Answer",
}


def render_chat():
    for entry in st.session_state["chat_history"]:
        role = entry["role"]
        content = entry["content"]

        with st.chat_message(role):
            if role == "assistant" and entry.get("route"):
                route = entry["route"]
                label = ROUTE_LABELS.get(route, route)
                st.markdown(
                    f'<span class="route-badge route-{route}">{label}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(content)


# ========================
# Main
# ========================

def main():
    st.set_page_config(
        page_title="AI Data Analyst",
        page_icon="📊",
        layout="wide",
    )

    inject_styles()
    init_session_state()
    render_sidebar()

    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1>📊 AI Data Analyst Assistant</h1>
        <p>Ask questions — I'll pick the right tool and deliver a polished answer.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs: Chat | Agent Graph
    chat_tab, graph_tab = st.tabs(["Chat", "Agent Graph"])

    with graph_tab:
        render_graph_tab()

    with chat_tab:
        if not st.session_state["chat_history"]:
            render_welcome()

        render_chat()

        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            resolved_key = resolve_api_key(
                st.session_state["provider"],
                st.session_state.get("api_key", ""),
            )
            if not resolved_key:
                st.error("Please provide an API key in the sidebar.")
                return

            st.session_state["_resolved_api_key"] = resolved_key

            st.session_state["chat_history"].append({
                "role": "user",
                "content": prompt,
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = process_question(
                            question=prompt,
                            provider=st.session_state["provider"],
                            model_name=resolve_model(
                                st.session_state["provider"],
                                st.session_state["model_name"],
                            ),
                            api_key=resolved_key,
                            tables=st.session_state.get("tables", []),
                            has_documents=st.session_state.get("has_documents", False),
                        )
                    except Exception as exc:
                        err = str(exc).lower()
                        if "429" in err or "rate limit" in err or "rate_limit" in err:
                            result = {
                                "answer": "Rate limit reached — the AI provider needs a moment. Please wait a few seconds and try again.",
                                "route": None,
                                "error": "rate_limit",
                            }
                        elif "api key" in err or "authentication" in err or "401" in err:
                            result = {
                                "answer": "API key issue — please check the provider and key in the sidebar.",
                                "route": None,
                                "error": "auth",
                            }
                        else:
                            result = {
                                "answer": f"Something went wrong: {exc}",
                                "route": None,
                                "error": str(exc),
                            }

                route = result.get("route")
                if route:
                    label = ROUTE_LABELS.get(route, route)
                    st.markdown(
                        f'<span class="route-badge route-{route}">{label}</span>',
                        unsafe_allow_html=True,
                    )

                answer = result.get("answer", "I couldn't generate an answer.")
                st.markdown(answer)

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": answer,
                "route": route,
            })


if __name__ == "__main__":
    main()
