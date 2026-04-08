"""
Streamlit frontend for the AI Data Analyst Assistant.
"""

import os

import streamlit as st

from config import (
    get_default_model,
    get_model_values,
    get_provider_config,
    resolve_api_key,
    resolve_model,
)
from graph import process_question
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
        "provider": "openai",
        "model_name": "",
        "api_key": "",
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
        st.markdown('<div class="sidebar-brand"><span>MultiAgent</span> AI</div>', unsafe_allow_html=True)

        st.markdown("#### Model")
        models = get_model_values("openai")
        default_model = get_default_model("openai")
        default_idx = models.index(default_model) if default_model in models else 0
        selected_model = st.selectbox("Model", models, index=default_idx, label_visibility="collapsed")
        st.session_state["model_name"] = selected_model

        st.markdown("#### OpenAI API Key")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            value=st.session_state.get("api_key", ""),
            label_visibility="collapsed",
        )
        st.session_state["api_key"] = api_key_input

        resolved_key = resolve_api_key("openai", api_key_input)
        if resolved_key:
            st.session_state["_resolved_api_key"] = resolved_key
            os.environ["OPENAI_API_KEY"] = resolved_key
        else:
            st.info("Enter your OpenAI API key to get started")

        st.divider()

        # --- File upload ---
        st.markdown("#### Upload Files")
        st.caption("CSV/Excel → SQL | PDF/Text → Doc search")
        uploaded = st.file_uploader(
            "Drop any file",
            type=["csv", "xlsx", "xls", "pdf", "txt", "md"],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed",
        )
        if uploaded:
            for f in uploaded:
                if f.name not in st.session_state.get("uploaded_files", []):
                    handle_file_upload(f)

        # Show what's loaded
        if st.session_state.get("tables"):
            st.caption(f"Tables: {', '.join(st.session_state['tables'])}")
        if st.session_state.get("has_documents"):
            doc_names = [n for n in st.session_state.get("uploaded_files", [])
                         if os.path.splitext(n)[1].lower() in DOC_EXTENSIONS]
            if doc_names:
                st.caption(f"Docs: {', '.join(doc_names)}")

        st.divider()

        if st.button("Clear Chat", use_container_width=True, type="secondary"):
            st.session_state["chat_history"] = []
            st.rerun()


# ========================
# Welcome screen
# ========================

def render_welcome():
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-logo">&#x26A1;</div>
        <h1>MultiAgent AI</h1>
        <p class="subtitle">Ask anything. Upload data, search the web, or query your documents &mdash; powered by intelligent agents that verify every answer.</p>
        <p class="subtitle-accent">Web Search &bull; SQL Analytics &bull; Document Q&A</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        ("icon-web", "&#x1F310;", "Web Search", "Verified answers from the web"),
        ("icon-sql", "&#x1F5C4;", "SQL Queries", "Natural language data analytics"),
        ("icon-doc", "&#x1F4C4;", "Doc Search", "Instant answers from your files"),
    ]
    for col, (icon_cls, icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon {icon_cls}">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ========================
# Chat display
# ========================

ROUTE_LABELS = {
    "web_search": "Web Search",
    "sql": "SQL Query",
    "document": "Doc Search",
    "direct": "Direct Answer",
}


def render_chat():
    for entry in st.session_state["chat_history"]:
        role = entry["role"]
        content = entry["content"]

        with st.chat_message(role, avatar="user" if role == "user" else "assistant"):
            if role == "assistant" and entry.get("route"):
                route = entry["route"]
                label = ROUTE_LABELS.get(route, route)
                badge_html = f'<span class="route-badge route-{route}">{label}</span>'
                confidence = entry.get("confidence")
                if confidence and route == "web_search":
                    badge_html += (
                        f' <span class="confidence-badge confidence-{confidence}">'
                        f'Confidence: {confidence.title()}</span>'
                    )
                st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(content)


# ========================
# Main
# ========================

def main():
    st.set_page_config(
        page_title="MultiAgent AI",
        page_icon="&#x1F4CA;",
        layout="wide",
        initial_sidebar_state="auto",
    )

    inject_styles()
    init_session_state()
    render_sidebar()

    # Show welcome if no chat yet, otherwise show chat
    if not st.session_state["chat_history"]:
        render_welcome()

    render_chat()

    # Chat input — always at the bottom
    if prompt := st.chat_input("Ask me anything..."):
        resolved_key = resolve_api_key(
            "openai",
            st.session_state.get("api_key", ""),
        )
        if not resolved_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
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
                        provider="openai",
                        model_name=resolve_model(
                            "openai",
                            st.session_state["model_name"],
                        ),
                        api_key=resolved_key,
                        tables=st.session_state.get("tables", []),
                        has_documents=st.session_state.get("has_documents", False),
                        chat_history=st.session_state.get("chat_history", []),
                    )
                except Exception as exc:
                    err = str(exc).lower()
                    exc_detail = f"{type(exc).__name__}: {exc}"
                    if "429" in err or "rate limit" in err or "rate_limit" in err:
                        result = {
                            "answer": f"Rate limit reached — please wait a moment and try again.\n\n`{exc_detail}`",
                            "route": None,
                            "error": "rate_limit",
                        }
                    elif "api key" in err or "authentication" in err or "401" in err:
                        result = {
                            "answer": f"API key issue — please check your key in the sidebar.\n\n`{exc_detail}`",
                            "route": None,
                            "error": "auth",
                        }
                    else:
                        result = {
                            "answer": f"Something went wrong:\n\n`{exc_detail}`",
                            "route": None,
                            "error": str(exc),
                        }

            route = result.get("route")
            confidence = result.get("confidence_label")
            if route:
                label = ROUTE_LABELS.get(route, route)
                badge_html = f'<span class="route-badge route-{route}">{label}</span>'
                if confidence and route == "web_search":
                    badge_html += (
                        f' <span class="confidence-badge confidence-{confidence}">'
                        f'Confidence: {confidence.title()}</span>'
                    )
                st.markdown(badge_html, unsafe_allow_html=True)

            answer = result.get("answer", "I couldn't generate an answer.")
            st.markdown(answer)

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "route": route,
            "confidence": confidence,
        })


if __name__ == "__main__":
    main()
