"""Tasks tab — natural-language tasks via Module C (`module_c_agent.run_task`)."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from services.api_adapter import run_task

EXAMPLE_COMMANDS = [
    "按销售方统计一下各位商家的消费总金额是多少？",
    "帮我看看这批数据里有没有重复报销的发票？",
    "帮我查一下有没有金额异常或信息不全的发票",
    "钱都被哪几个公司赚走了？给我画个总消费画像",
    "帮我把这批数据做成报销台账表，按日期排好",
]


def render_tasks_page() -> None:
    st.header("Tasks")
    st.caption("使用自然语言描述任务；由 **Module C** 路由到聚合、查重、异常、画像或报销台账技能。")

    with st.expander("示例指令", expanded=False):
        for c in EXAMPLE_COMMANDS:
            st.code(c, language=None)

    command = st.text_input(
        "指令",
        placeholder=EXAMPLE_COMMANDS[0],
        help="可直接输入中文问题，或使用下方快捷按钮。",
    )

    st.markdown("**快捷按钮**")
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])
    with c1:
        if st.button("聚合统计", use_container_width=True):
            st.session_state["_task_cmd"] = EXAMPLE_COMMANDS[0]
    with c2:
        if st.button("发票查重", use_container_width=True):
            st.session_state["_task_cmd"] = EXAMPLE_COMMANDS[1]
    with c3:
        if st.button("异常检测", use_container_width=True):
            st.session_state["_task_cmd"] = EXAMPLE_COMMANDS[2]
    with c4:
        if st.button("消费画像", use_container_width=True):
            st.session_state["_task_cmd"] = EXAMPLE_COMMANDS[3]
    with c5:
        if st.button("报销台账", use_container_width=True):
            st.session_state["_task_cmd"] = EXAMPLE_COMMANDS[4]

    c_run, _ = st.columns([1, 4])
    with c_run:
        run_custom = st.button("运行指令", type="primary", use_container_width=True)

    cmd_to_run = None
    if run_custom:
        cmd_to_run = (command or EXAMPLE_COMMANDS[0]).strip()
    elif "_task_cmd" in st.session_state:
        cmd_to_run = st.session_state.pop("_task_cmd")

    if not cmd_to_run:
        st.info("输入指令并点击 **运行指令**，或使用快捷按钮。")
        return

    with st.spinner(f"Module C 处理中: {cmd_to_run!r}…"):
        try:
            result = run_task(cmd_to_run)
        except Exception as e:
            st.error(f"任务失败: {e}")
            return

    st.caption(
        f"任务类型: **{result.get('task_type', '—')}** · 状态: **{result.get('status', '—')}**"
    )

    top_status = result.get("status") or "success"
    msg = result.get("message", "")
    if top_status == "error":
        st.error(msg)
        return
    if top_status == "warning":
        st.warning(msg)
        if not msg:
            return
    else:
        if msg:
            st.markdown(msg)

    task_type = result.get("task_type", "generic")
    payload = result.get("result") or {}

    if task_type == "reimbursement_form":
        _render_reimbursement(payload)
    elif task_type == "duplicate_detection":
        _render_module_c_duplicates(payload)
    elif task_type == "anomaly_detection":
        _render_module_c_anomalies(payload)
    elif task_type in ("aggregation", "vendor_profiling"):
        _render_key_value_result(payload)
    elif task_type == "none":
        pass
    else:
        st.subheader("结构化结果")
        st.json(payload)


def _render_key_value_result(payload: dict) -> None:
    if not payload:
        return
    if isinstance(payload, dict) and all(
        not isinstance(v, (dict, list)) for v in payload.values()
    ):
        df = pd.DataFrame([{"项": k, "值": v} for k, v in payload.items()])
        st.subheader("数据摘要")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.subheader("数据摘要")
        st.json(payload)


def _render_reimbursement(payload: dict) -> None:
    csv_string = payload.get("csv_data") or ""
    if csv_string:
        st.subheader("下载报销台账")
        st.download_button(
            label="下载 CSV（Excel 可直接打开）",
            data=csv_string.encode("utf-8-sig"),
            file_name="发票报销台账_导出.csv",
            mime="text/csv",
        )
        st.caption("预览（前 10 行）")
        df = pd.read_csv(io.StringIO(csv_string))
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)


def _render_module_c_duplicates(payload: dict) -> None:
    st.subheader("查重结果（结构化）")
    if not payload:
        st.info("无查重数据。")
        return
    st.json(payload)


def _render_module_c_anomalies(payload: list | dict) -> None:
    st.subheader("异常列表")
    items = payload if isinstance(payload, list) else []
    if not items:
        st.info("未检测到异常。")
        return
    st.dataframe(
        pd.DataFrame(items),
        use_container_width=True,
        hide_index=True,
    )
