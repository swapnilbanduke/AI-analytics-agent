"""
FastAPI REST server for the AI Data Analyst.

Run: uvicorn api:app --reload
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, UploadFile
from pydantic import BaseModel

from auth import verify_token

load_dotenv()

app = FastAPI(
    title="AI Data Analyst API",
    version="1.0.0",
    description="Query databases, search documents, and analyze data with AI.",
)


# ========================
# Request / Response models
# ========================

class QueryRequest(BaseModel):
    question: str
    provider: str = "openai"
    model_name: str = ""
    api_key: str = ""


class QueryResponse(BaseModel):
    answer: str
    route: str | None = None
    sql: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    success: bool
    message: str
    table_name: str | None = None
    rows: int | None = None


class HealthResponse(BaseModel):
    status: str
    tables: list[str]
    has_documents: bool


# ========================
# Endpoints
# ========================

@app.get("/health", response_model=HealthResponse)
async def health():
    from database import list_tables
    from rag import has_documents

    return HealthResponse(
        status="healthy",
        tables=list_tables(),
        has_documents=has_documents(),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, _=Depends(verify_token)):
    from graph import process_question

    result = process_question(
        question=request.question,
        provider=request.provider,
        model_name=request.model_name,
        api_key=request.api_key or None,
        tables=_get_tables(),
        has_documents=_has_docs(),
    )

    return QueryResponse(
        answer=result.get("answer", "No answer generated."),
        route=result.get("route"),
        sql=None,
        error=result.get("error"),
    )


@app.post("/upload/data", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...), _=Depends(verify_token)):
    from ingestion import ingest_csv, ingest_excel

    content = await file.read()
    file_name = file.filename or "upload"
    table_name = file_name.rsplit(".", 1)[0]

    if file_name.endswith(".csv"):
        result = ingest_csv(content, table_name=table_name)
        return UploadResponse(
            success=result["success"],
            message=f"Loaded {result['rows']} rows into table '{result['table_name']}'",
            table_name=result["table_name"],
            rows=result["rows"],
        )
    elif file_name.endswith((".xlsx", ".xls")):
        result = ingest_excel(content)
        tables = result.get("tables", [])
        names = [t["sql_table_name"] for t in tables]
        total_rows = sum(t["rows"] for t in tables)
        return UploadResponse(
            success=result["success"],
            message=f"Loaded {len(tables)} sheet(s) ({total_rows} total rows): {', '.join(names)}",
            table_name=names[0] if names else None,
            rows=total_rows,
        )
    else:
        return UploadResponse(success=False, message=f"Unsupported file type: {file_name}")


@app.post("/upload/document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), _=Depends(verify_token)):
    from rag import add_documents

    content = await file.read()
    file_name = file.filename or "upload.pdf"
    ext = os.path.splitext(file_name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        num_chunks = add_documents(tmp_path)
        return UploadResponse(
            success=True,
            message=f"Indexed '{file_name}' ({num_chunks} chunks)",
            rows=num_chunks,
        )
    finally:
        os.unlink(tmp_path)


@app.get("/tables")
async def get_tables(_=Depends(verify_token)):
    from database import list_tables
    return {"tables": list_tables()}


@app.get("/schema")
async def get_schema(_=Depends(verify_token)):
    from database import get_sql_database, list_tables

    tables = list_tables()
    if not tables:
        return {"schema": "No tables loaded."}
    db = get_sql_database()
    return {"schema": db.get_table_info()}


# ========================
# Helpers
# ========================

def _get_tables() -> list[str]:
    from database import list_tables
    return list_tables()


def _has_docs() -> bool:
    try:
        from rag import has_documents
        return has_documents()
    except Exception:
        return False


@app.get("/graph")
async def get_graph():
    """Return the agent graph as a Mermaid diagram."""
    from graph import build_agent_graph

    tables = _get_tables()
    has_docs = _has_docs()
    graph = build_agent_graph(has_database=bool(tables), has_documents=has_docs)

    try:
        mermaid = graph.get_graph().draw_mermaid()
    except Exception:
        mermaid = "graph TD\n    A[classify] --> B[agent]\n    B --> C[tools]\n    C --> B\n    B --> D[END]"

    return {
        "mermaid": mermaid,
        "active_tools": {
            "calculator": True,
            "web_search": bool(os.environ.get("TAVILY_API_KEY")),
            "sql_query": bool(tables),
            "document_search": has_docs,
        },
        "tables": tables,
    }
