"""Search tab — natural language invoice queries powered by Module B hybrid RAG."""

from __future__ import annotations

import streamlit as st

from services.api_adapter import get_invoice_detail, query_invoices

EXAMPLE_QUERIES = [
    "2024年7月的发票",
    "金额超过500的发票",
    "京东的发票",
    "2024年10月超过100元的发票",
    "食品类的消费",
    "和会议相关的票据",
]


def _render_invoice_detail(inv: dict) -> None:
    """Render detailed view for a single invoice inside an expander."""
    invoice_id = inv.get("invoice_id") or ""
    cn_dict = get_invoice_detail(invoice_id)

    if cn_dict is None:
        st.info("未找到该发票的详细数据。")
        return

    # Key metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("发票号码", cn_dict.get("发票号码") or "—")
    with col2:
        st.metric("价税合计", cn_dict.get("价税合计小写") or "—")
    with col3:
        st.metric("开票日期", cn_dict.get("开票日期") or "—")

    # Full fields table
    display_fields = {
        "发票类型": cn_dict.get("发票类型"),
        "发票号码": cn_dict.get("发票号码"),
        "开票日期": cn_dict.get("开票日期"),
        "购买方名称": cn_dict.get("购买方名称"),
        "销售方名称": cn_dict.get("销售方名称"),
        "价税合计小写": cn_dict.get("价税合计小写"),
        "价税合计大写": cn_dict.get("价税合计大写"),
        "备注": cn_dict.get("备注"),
        "开票人": cn_dict.get("开票人"),
    }
    st.dataframe(
        [{"字段": k, "值": v or ""} for k, v in display_fields.items()],
        use_container_width=True,
        hide_index=True,
    )

    # Line items (项目明细)
    items = cn_dict.get("项目明细") or []
    if items and isinstance(items, list):
        st.subheader(f"项目明细（{len(items)} 项）")
        st.dataframe(items, use_container_width=True, hide_index=True)

    # Raw JSON expandable
    with st.expander("查看完整 JSON"):
        st.json(cn_dict)


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
        placeholder="例如：2024年7月的发票、金额超过500、和会议相关的票据 ...",
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
        # Summary table
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

        # Detail expanders for each invoice
        st.subheader("发票详情")
        st.caption("点击下方发票展开查看详细信息和项目明细。")
        for idx, inv in enumerate(matched):
            iid = inv.get("invoice_id") or f"#{idx + 1}"
            vendor = inv.get("vendor") or "未知供应商"
            amount = inv.get("amount")
            amount_str = f"¥{amount}" if amount is not None else ""
            label = f"📄 {iid} — {vendor}  {amount_str}"
            with st.expander(label, expanded=False):
                _render_invoice_detail(inv)
