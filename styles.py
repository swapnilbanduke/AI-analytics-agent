"""
Streamlit custom CSS for a polished analytics UI.

Adapted from Analytics platform/styles.py with a distinct color scheme.
"""

import streamlit as st


def inject_styles():
    """Inject custom CSS for the AI Data Analyst UI."""
    st.markdown("""
    <style>
    /* ===== Global ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ===== Hero Header ===== */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.8rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .hero-header p {
        margin: 0.4rem 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
        color: #c8d6e5 !important;
    }

    /* ===== Metric Cards ===== */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b !important;
    }

    /* ===== Chat Messages ===== */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }

    /* ===== Code / SQL Output Block ===== */
    .code-output-block {
        background: #f1f5f9;
        border-left: 3px solid #0f3460;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        color: #334155;
        line-height: 1.5;
    }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* ===== Expander ===== */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.9rem;
        color: #334155;
    }

    /* ===== Chat Input ===== */
    [data-testid="stChatInput"] textarea {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #0f3460;
        box-shadow: 0 0 0 2px rgba(15, 52, 96, 0.15);
    }

    /* ===== Welcome Screen ===== */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem 1.5rem;
    }
    .welcome-container h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .welcome-container p {
        font-size: 1.05rem;
        color: #64748b;
        max-width: 600px;
        margin: 0.5rem auto;
        line-height: 1.6;
    }

    /* ===== Feature Cards ===== */
    .feature-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-color: #0f3460;
    }
    .feature-card h3 {
        font-size: 1rem;
        margin: 0.5rem 0 0.25rem;
        color: #1e293b;
    }
    .feature-card p {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0;
    }

    /* ===== Route Badge ===== */
    .route-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .route-calculation { background: #fef3c7; color: #92400e; }
    .route-web_search { background: #dbeafe; color: #1e40af; }
    .route-sql { background: #d1fae5; color: #065f46; }
    .route-document { background: #ede9fe; color: #5b21b6; }
    .route-direct { background: #f1f5f9; color: #475569; }
    </style>
    """, unsafe_allow_html=True)
