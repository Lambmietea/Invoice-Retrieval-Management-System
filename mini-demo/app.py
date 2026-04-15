"""
COMP6708 Assignment 3 — LLM-based Invoice Intelligence Assistant (Streamlit UI).

Mini-demo 副本：在 `mini-demo` 目录下运行：
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config
from ui.dashboard_page import render_dashboard_page
from ui.search_page import render_search_page
from ui.tasks_page import render_tasks_page
from ui.upload_page import render_upload_page

st.set_page_config(
    page_title="Invoice Intelligence Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — API configuration status
# ---------------------------------------------------------------------------


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "❌ 未配置"
    return f"{key[:4]}****{key[-4:]}"


with st.sidebar:
    st.header("⚙️ API 配置")

    st.markdown("**Gemini** (Module C · 智能分析)")
    st.caption(f"Key: `{_mask_key(config.GEMINI_API_KEY)}`")
    st.caption(f"Model: `{config.GEMINI_MODEL}`")

    st.markdown("**Module B LLM** (查询解析)")
    st.caption(f"Key: `{_mask_key(config.MODULE_B_API_KEY)}`")
    st.caption(f"Base URL: `{config.MODULE_B_API_BASE}`")
    st.caption(f"Model: `{config.MODULE_B_MODEL}`")

    st.markdown("**DashScope** (Module A · PDF 解析)")
    st.caption(f"Key: `{_mask_key(config.DASHSCOPE_API_KEY)}`")
    st.caption(f"Model: `{config.DASHSCOPE_MODEL}`")

    st.divider()
    st.caption(
        "通过环境变量配置：\n"
        "- `GEMINI_API_KEY`\n"
        "- `MODULE_B_API_KEY`\n"
        "- `MODULE_B_API_BASE`\n"
        "- `MODULE_B_MODEL`\n"
        "- `DASHSCOPE_API_KEY`\n"
        "- `GEMINI_MODEL`\n"
        "- `DASHSCOPE_MODEL`"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Invoice Intelligence Assistant")
st.markdown(
    "**COMP6708 Assignment 3** — 发票智能助手。"
    "集成 Module A（PDF 解析）、Module B（混合 RAG 检索）、Module C（LLM 智能分析）。"
)

tab_upload, tab_search, tab_tasks, tab_dash = st.tabs(
    ["Upload & Parse", "Search", "Tasks", "Dashboard"]
)

with tab_upload:
    render_upload_page()

with tab_search:
    render_search_page()

with tab_tasks:
    render_tasks_page()

with tab_dash:
    render_dashboard_page()

st.divider()
st.caption(
    "数据默认来自 `data/*.json`（30 张发票）；上传 PDF 发票后会自动解析并索引到 Module B，"
    "可在 Search、Tasks、Dashboard 中使用。"
)
