"""Dashboard tab — aggregate statistics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.api_adapter import get_all_invoices


def render_dashboard_page() -> None:
    st.header("Dashboard")
    st.caption("Summary statistics from all indexed invoices (via adapter).")

    try:
        invoices = get_all_invoices()
    except Exception as e:
        st.error(f"Could not load invoices: {e}")
        return

    if not invoices:
        st.warning("No invoices yet. Use **Upload & Parse** or rely on backend data.")
        return

    df = pd.DataFrame(invoices)
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    total_count = len(df)
    total_amount = float(df["amount"].sum()) if "amount" in df.columns else 0.0

    st.subheader("Overview")
    m1, m2 = st.columns(2)
    m1.metric("Total invoices", f"{total_count:,}")
    m2.metric("Total amount (sum)", f"{total_amount:,.2f} CNY")

    if "category" in df.columns and "amount" in df.columns:
        st.subheader("Spending by category")
        by_cat = df.groupby("category", dropna=False)["amount"].sum().sort_values(ascending=False)
        cat_df = by_cat.reset_index()
        cat_df.columns = ["category", "amount"]
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
        st.bar_chart(cat_df.set_index("category"))

    if "date" in df.columns and "amount" in df.columns:
        st.subheader("Spending trend by month")
        df_month = df.dropna(subset=["date"]).copy()
        df_month["month"] = df_month["date"].dt.to_period("M").astype(str)
        trend = df_month.groupby("month")["amount"].sum().sort_index()
        trend_df = trend.reset_index()
        trend_df.columns = ["month", "amount"]
        st.line_chart(trend_df.set_index("month"))

    if "vendor" in df.columns and "amount" in df.columns:
        st.subheader("Top vendors")
        top = (
            df.groupby("vendor", dropna=False)["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top.columns = ["vendor", "total_amount"]
        st.dataframe(top, use_container_width=True, hide_index=True)
