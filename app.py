"""
Streamlit frontend for the AI Data Analyst Assistant.
"""

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
from graph import process_question
from styles import inject_styles


# ========================
# Session state
# ========================

def init_session_state():
    defaults = {
        "chat_history": [],
        "provider": DEFAULT_PROVIDER,
        "model_name": "",
        "api_key": "",
        "processing": False,
        "tables": [],
        "has_documents": False,
        "uploaded_files": [],
        "uploaded_docs": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ========================
# Data upload helpers
# ========================

def handle_data_upload(uploaded_file):
    """Ingest an uploaded CSV/Excel file into the SQLite database."""
    from database import list_tables
    from ingestion import ingest_csv, ingest_excel

    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    table_name = file_name.rsplit(".", 1)[0]

    if file_name.endswith(".csv"):
        result = ingest_csv(file_bytes, table_name=table_name)
    elif file_name.endswith((".xlsx", ".xls")):
        result = ingest_excel(file_bytes)
    else:
        st.error(f"Unsupported file type: {file_name}")
        return

    st.session_state["tables"] = list_tables()
    st.session_state["uploaded_files"].append(file_name)

    if isinstance(result, dict) and result.get("success"):
        rows = result.get("rows", "?")
        tbl = result.get("table_name", table_name)
        st.success(f"Loaded **{file_name}** → table `{tbl}` ({rows} rows)")
    elif isinstance(result, dict) and result.get("tables"):
        for t in result["tables"]:
            st.success(f"Sheet **{t['original_sheet_name']}** → `{t['sql_table_name']}` ({t['rows']} rows)")


def handle_doc_upload(uploaded_file):
    """Ingest an uploaded document into the RAG vector store."""
    import os
    import tempfile

    try:
        from rag import add_documents

        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        add_documents(tmp_path, collection_name="default")
        os.unlink(tmp_path)

        st.session_state["has_documents"] = True
        st.session_state["uploaded_docs"].append(file_name)
        st.success(f"Indexed **{file_name}** for document search")

    except ImportError:
        st.warning("Document search not available yet (Layer 4)")
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
            st.success("API key set")
        else:
            st.warning("Enter API key or set in .env")

        # Model selection
        models = get_model_values(provider)
        default_model = get_default_model(provider)
        default_idx = models.index(default_model) if default_model in models else 0
        selected_model = st.selectbox("Model", models, index=default_idx)
        st.session_state["model_name"] = selected_model

        st.divider()

        # --- Data Upload ---
        st.markdown("### Data")
        data_file = st.file_uploader(
            "Upload CSV / Excel",
            type=["csv", "xlsx", "xls"],
            key="data_uploader",
        )
        if data_file and data_file.name not in st.session_state.get("uploaded_files", []):
            handle_data_upload(data_file)

        if st.session_state.get("tables"):
            st.caption(f"Tables: {', '.join(st.session_state['tables'])}")

        st.divider()

        # --- Document Upload ---
        st.markdown("### Documents")
        doc_file = st.file_uploader(
            "Upload PDF / Text",
            type=["pdf", "txt", "md"],
            key="doc_uploader",
        )
        if doc_file and doc_file.name not in st.session_state.get("uploaded_docs", []):
            handle_doc_upload(doc_file)

        if st.session_state.get("uploaded_docs"):
            st.caption(f"Docs: {', '.join(st.session_state['uploaded_docs'])}")

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

    # Show welcome screen or chat history
    if not st.session_state["chat_history"]:
        render_welcome()

    render_chat()

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Check API key
        resolved_key = resolve_api_key(
            st.session_state["provider"],
            st.session_state.get("api_key", ""),
        )
        if not resolved_key:
            st.error("Please provide an API key in the sidebar.")
            return

        # Store resolved key for tools to access
        st.session_state["_resolved_api_key"] = resolved_key

        # Add user message
        st.session_state["chat_history"].append({
            "role": "user",
            "content": prompt,
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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

            route = result.get("route")
            if route:
                label = ROUTE_LABELS.get(route, route)
                st.markdown(
                    f'<span class="route-badge route-{route}">{label}</span>',
                    unsafe_allow_html=True,
                )

            answer = result.get("answer", "I couldn't generate an answer.")
            st.markdown(answer)

        # Save to history
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "route": route,
        })


if __name__ == "__main__":
    main()
