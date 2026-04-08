"""
Streamlit custom CSS — clean, modern light interface.
"""

import streamlit as st


def inject_styles():
    """Inject custom CSS for a polished, modern UI."""
    st.markdown("""
    <style>
    /* ===== Global ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h4 {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6366f1;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }

    .sidebar-brand {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1e1e2e;
        padding: 0.5rem 0 1.25rem;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.25rem;
        letter-spacing: -0.02em;
    }
    .sidebar-brand span {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        border-radius: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        border-radius: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
    }
    [data-testid="stSidebar"] hr {
        border-color: #e5e7eb;
        margin: 1rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border-radius: 12px;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* ===== Welcome Screen ===== */
    .welcome-container {
        text-align: center;
        padding: 4.5rem 2rem 1.5rem;
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .welcome-logo {
        width: 64px;
        height: 64px;
        margin: 0 auto 1.25rem;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        box-shadow: 0 6px 24px rgba(99,102,241,0.2);
    }
    .welcome-container h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e1e2e;
        margin-bottom: 0.6rem;
        letter-spacing: -0.03em;
    }
    .welcome-container .subtitle {
        font-size: 1rem;
        color: #6b7280;
        max-width: 460px;
        margin: 0 auto 0.4rem;
        line-height: 1.65;
    }
    .welcome-container .subtitle-accent {
        font-size: 0.82rem;
        color: #6366f1;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* ===== Feature Cards ===== */
    .feature-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.5rem 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.25s ease;
        cursor: default;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(99,102,241,0.1);
        border-color: #c7d2fe;
    }
    .feature-icon {
        width: 44px;
        height: 44px;
        margin: 0 auto 0.7rem;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        color: white;
    }
    .icon-web { background: linear-gradient(135deg, #3b82f6, #2563eb); }
    .icon-sql { background: linear-gradient(135deg, #10b981, #059669); }
    .icon-doc { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
    .feature-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e1e2e;
        margin-bottom: 0.2rem;
    }
    .feature-desc {
        font-size: 0.76rem;
        color: #9ca3af;
    }

    /* ===== Chat Messages ===== */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.9rem 1.15rem;
        margin-bottom: 0.4rem;
        max-width: 100%;
    }

    /* ===== Chat Input ===== */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1.5px solid #e5e7eb !important;
        font-size: 0.95rem;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
        border-radius: 12px !important;
        color: white !important;
        border: none !important;
    }

    /* ===== Route Badges ===== */
    .route-badge {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .route-web_search { background: #eff6ff; color: #2563eb; }
    .route-sql { background: #ecfdf5; color: #059669; }
    .route-document { background: #f5f3ff; color: #7c3aed; }
    .route-direct { background: #f3f4f6; color: #6b7280; }

    /* ===== Confidence Badges ===== */
    .confidence-badge {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 6px;
        margin-bottom: 0.4rem;
    }
    .confidence-high { background: #ecfdf5; color: #059669; }
    .confidence-medium { background: #fffbeb; color: #d97706; }
    .confidence-low { background: #fef2f2; color: #dc2626; }

    /* ===== Alerts ===== */
    [data-testid="stAlert"] { border-radius: 10px; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
    </style>
    """, unsafe_allow_html=True)
