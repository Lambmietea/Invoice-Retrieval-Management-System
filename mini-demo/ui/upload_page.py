"""Upload & Parse tab — PDF invoice upload with real AI parsing (Module A: Qwen-VL)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from services.api_adapter import index_invoice, parse_invoice


def render_upload_page() -> None:
    st.header("Upload & Parse")
    st.caption(
        "上传 PDF 发票文件 → Module A（Qwen-VL 多模态大模型）自动识别并提取结构化数据"
        " → 索引到 Module B（SQLite + ChromaDB），后续可在 Search 标签页中检索到。"
    )

    uploaded = st.file_uploader(
        "选择 PDF 发票文件",
        type=["pdf"],
        help="支持中文增值税发票 PDF，系统将调用 Qwen-VL 多模态大模型进行智能解析。",
    )

    if uploaded is None:
        st.info("选择一个 PDF 文件开始，或切换到 **Search** / **Dashboard** 查看已有数据。")
        return

    col_run, _ = st.columns([1, 4])
    with col_run:
        process = st.button("解析并入库", type="primary", use_container_width=True)

    if not process:
        st.warning("点击 **解析并入库** 开始处理。")
        return

    status = st.status("正在处理…", expanded=True)
    try:
        with tempfile.TemporaryDirectory(prefix="inv_upload_") as tmp:
            safe_name = Path(uploaded.name).name
            file_path = Path(tmp) / safe_name
            file_path.write_bytes(uploaded.getvalue())

            status.write("📄 正在调用 AI 解析发票（Qwen-VL 多模态大模型，约需 5-15 秒）…")
            record, cn_dict = parse_invoice(str(file_path))

            status.write("📥 正在索引到向量数据库…")
            index_result = index_invoice(record, cn_dict=cn_dict)

        status.update(label="处理完成", state="complete", expanded=False)

        if index_result["status"] == "success":
            st.success(f"✅ {index_result['message']}")
        else:
            st.error(index_result["message"])

        # ── Display parsed Chinese fields ──
        st.subheader("发票解析结果")

        # Key fields summary
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

        # Line items
        items = cn_dict.get("项目明细") or []
        if items and isinstance(items, list):
            st.subheader(f"项目明细（{len(items)} 项）")
            st.dataframe(items, use_container_width=True, hide_index=True)

        # Raw JSON expandable
        with st.expander("查看完整 JSON"):
            st.json(cn_dict)

    except FileNotFoundError as e:
        status.update(label="错误", state="error")
        st.error(str(e))
    except Exception as e:
        status.update(label="错误", state="error")
        st.error(f"处理失败: {e}")
