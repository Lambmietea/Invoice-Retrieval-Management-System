"""Search tab — natural language invoice queries powered by Module B hybrid RAG."""

from __future__ import annotations

import streamlit as st

from services.api_adapter import query_invoices

EXAMPLE_QUERIES = [
    "2024年7月的发票",
    "金额超过500的发票",
    "和差旅相关的票据",
    "京东的发票",
    "2024年10月超过100元的发票",
    "食品饮料类的消费",
]


def render_search_page() -> None:
    st.header("Search")
    st.caption(
        "自然语言智能检索 — 基于 Module B 混合 RAG 引擎（SQL 结构化过滤 + 向量语义搜索）。"
    )

    with st.expander("示例查询（点击复制）", expanded=False):
        for q in EXAMPLE_QUERIES:
            st.code(q, language=None)

    user_query = st.text_input(
        "输入查询",
        placeholder="例如：2024年7月的发票、金额超过500、和差旅相关的票据 ...",
        help="支持日期、金额、供应商、类别等结构化条件，也支持纯语义模糊搜索。",
    )

    col1, _ = st.columns([1, 5])
    with col1:
        search_clicked = st.button("搜索", type="primary", use_container_width=True)

    if not search_clicked:
        if user_query.strip():
            st.caption("点击 **搜索** 执行查询。")
        else:
            st.info("输入查询内容并点击 **搜索**，或参考上方示例。")
        return

    q = user_query.strip() or EXAMPLE_QUERIES[0]
    with st.spinner("正在检索发票..."):
        try:
            result = query_invoices(q)
        except Exception as e:
            st.error(f"检索失败: {e}")
            return

    st.caption(
        f"检索模式: **{result.get('query_type', '—')}** · "
        f"状态: **{result.get('status', '—')}**"
    )

    if result["status"] == "error":
        st.error(result["summary_context"])
        return

    st.subheader("检索摘要")
    st.write(result["summary_context"])

    matched = result.get("matched_invoices") or []
    st.subheader(f"匹配结果（{len(matched)} 张）")
    if not matched:
        st.dataframe([], use_container_width=True)
    else:
        rows = []
        for inv in matched:
            rows.append(
                {
                    "发票号码": inv.get("invoice_id"),
                    "供应商": inv.get("vendor"),
                    "日期": inv.get("date"),
                    "金额": inv.get("amount"),
                    "税额": inv.get("tax"),
                    "币种": inv.get("currency"),
                    "类别": inv.get("category"),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
