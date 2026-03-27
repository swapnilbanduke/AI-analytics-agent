"""
RAG pipeline — document loading, chunking, embedding, and retrieval.

Uses ChromaDB for persistent vector storage and OpenAI embeddings.
"""

import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "data", "vectorstore")

# Chunking config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"


def _get_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY", "")
    # Also check Streamlit session state (user may have entered key in sidebar)
    if not api_key:
        try:
            import streamlit as st
            # If provider is OpenAI, the resolved key is the OpenAI key
            if st.session_state.get("provider", "openai") == "openai":
                api_key = st.session_state.get("_resolved_api_key", "")
            # Also try the raw api_key input (works if user typed OpenAI key)
            if not api_key:
                api_key = st.session_state.get("api_key", "")
        except Exception:
            pass
    if not api_key:
        raise ValueError(
            "OpenAI API key is required for document embeddings. "
            "Set OPENAI_API_KEY in .env or enter it in the sidebar."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)


def _get_vectorstore(collection_name: str = "default"):
    from langchain_community.vectorstores import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=VECTORSTORE_PATH,
    )


def load_document(file_path: str):
    """Load a document and split it into chunks."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported document type: {ext}")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def add_documents(file_path: str, collection_name: str = "default") -> int:
    """Ingest a document into the vector store. Returns number of chunks added."""
    chunks = load_document(file_path)
    if not chunks:
        return 0

    vectorstore = _get_vectorstore(collection_name)
    vectorstore.add_documents(chunks)
    return len(chunks)


def search_documents(query: str, k: int = 4, collection_name: str = "default") -> list[str]:
    """Retrieve the top-k relevant document chunks for a query."""
    vectorstore = _get_vectorstore(collection_name)
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def list_collections() -> list[str]:
    """List available collection names."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
        return [c.name for c in client.list_collections()]
    except Exception:
        return []


def clear_vectorstore(collection_name: str = "default"):
    """Delete all documents from a collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
        client.delete_collection(collection_name)
    except Exception:
        pass


def has_documents(collection_name: str = "default") -> bool:
    """Check if the vector store has any documents."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
        collection = client.get_collection(collection_name)
        return collection.count() > 0
    except Exception:
        return False
