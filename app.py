from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from opslens.agent import AgentError, answer_question
from opslens.analytics import AnalysisError, compare_metric, metric_trend, prepare_data, profile_data


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="OpsLens", page_icon=None, layout="wide")
st.title("OpsLens")
st.caption("Evidence-backed investigation for sales and operations data")

with st.sidebar:
    st.subheader("Dataset")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    st.caption("Expected fields: order_date, order_id, region, category, product, units, unit_price, discount_pct, cost_per_unit, delivery_days, status.")
    st.caption("The sample dataset is used when no file is uploaded.")

source = uploaded if uploaded is not None else ROOT / "data" / "retail_operations.csv"
try:
    raw = pd.read_csv(source)
    frame = prepare_data(raw)
except (OSError, pd.errors.ParserError, UnicodeDecodeError, AnalysisError) as exc:
    st.error(str(exc))
    st.stop()

profile = profile_data(frame)
metrics = st.columns(4)
metrics[0].metric("Valid rows", f"{profile['valid_rows']:,}")
metrics[1].metric("Orders", f"{profile['unique_orders']:,}")
metrics[2].metric("Revenue", f"{profile['revenue']:,.2f}")
metrics[3].metric("Profit", f"{profile['profit']:,.2f}")
st.caption(f"Date coverage: {profile['date_range']['start']} to {profile['date_range']['end']}")

analysis_tab, data_tab = st.tabs(["Investigate", "Dataset"])
with analysis_tab:
    st.subheader("Ask an operational question")
    question = st.text_area(
        "Question",
        placeholder="Which product categories contributed most to the revenue decline?",
        height=90,
        label_visibility="collapsed",
    )
    if st.button("Run investigation", type="primary", disabled=not question.strip()):
        with st.spinner("Selecting analysis tools and checking the data..."):
            try:
                result = answer_question(frame, question)
                st.session_state["last_result"] = result
            except AgentError as exc:
                st.error(str(exc))
    result = st.session_state.get("last_result")
    if result:
        st.markdown(result["answer"])
        with st.expander("Calculation evidence", expanded=True):
            for item in result["evidence"]:
                st.markdown(f"**{item['tool']}**")
                st.json(item["result"])

    st.divider()
    st.subheader("Explore the numbers")
    left, right = st.columns(2)
    with left:
        metric = st.selectbox("Metric", ["revenue", "profit", "units", "orders", "avg_delivery_days", "cancellation_rate", "profit_margin"])
        grouping = st.selectbox("Compare by", ["region", "category", "product", "status"])
        try:
            comparison = compare_metric(frame, metric, grouping)
            comparison_frame = pd.DataFrame(comparison["results"])
            st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
            if not comparison_frame.empty:
                st.bar_chart(comparison_frame.set_index("group")["value"])
        except AnalysisError as exc:
            st.error(str(exc))
    with right:
        trend_metric_name = st.selectbox("Trend metric", ["revenue", "profit", "units", "orders", "avg_delivery_days", "cancellation_rate", "profit_margin"], key="trend_metric")
        granularity = st.radio("Time interval", ["month", "week"], horizontal=True)
        trend = metric_trend(frame, trend_metric_name, granularity)
        trend_frame = pd.DataFrame(trend["results"])
        st.dataframe(trend_frame, use_container_width=True, hide_index=True)
        if not trend_frame.empty:
            st.line_chart(trend_frame.set_index("period")["value"])

with data_tab:
    st.subheader("Validated data preview")
    st.dataframe(frame.drop(columns=["is_cancelled"], errors="ignore").head(100), use_container_width=True, hide_index=True)
    st.subheader("Data quality")
    st.json({"missing_values": profile["missing_values"], "status_counts": profile["status_counts"]})

if not os.getenv("GROQ_API_KEY"):
    st.info("Natural-language investigations require GROQ_API_KEY. The deterministic comparison and trend tools work without an API key.")
