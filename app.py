import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------
# Load cleaned SCF dashboard data (standalone)
# Expects: scf_dashboard_data.csv in the same folder as app.py
# Required columns: w, age, annuity, db_pension, stock_owner
# --------------------------------------------------

st.set_page_config(page_title="Annuity Puzzle Dashboard", layout="wide")

@st.cache_data
def load_data(path="scf_dashboard_data.csv"):
    return pd.read_csv(path)

d = load_data("scf_dashboard_data.csv")

required = {"w", "age", "annuity", "db_pension", "stock_owner"}
missing = required - set(d.columns)
if missing:
    st.error(f"Missing columns in scf_dashboard_data.csv: {missing}")
    st.stop()

st.title("The Annuity Puzzle: Theory vs Reality")
st.caption("Interactive dashboard using SCF data")

# ---------- helpers ----------
def wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return np.sum(x[m] * w[m]) / np.sum(w[m])

# ---------- sidebar filters ----------
st.sidebar.header("Filters")

age_min_default, age_max_default = 18, 95
age_min_data, age_max_data = int(d["age"].min()), int(d["age"].max())

age_min, age_max = st.sidebar.slider(
    "Age range",
    min_value=age_min_data,
    max_value=age_max_data,
    value=(max(age_min_default, age_min_data), min(age_max_default, age_max_data)),
)

pension_filter = st.sidebar.selectbox(
    "DB Pension",
    ["All", "No DB pension", "DB pension"]
)

stock_filter = st.sidebar.selectbox(
    "Stock ownership",
    ["All", "No stocks", "Own stocks"]
)

theory_rate = st.sidebar.slider(
    "Theoretical annuitization rate",
    min_value=0.50,
    max_value=1.00,
    value=0.90,
    step=0.01,
)

# ---------- apply filters ----------
df = d.copy()
df = df[(df["age"] >= age_min) & (df["age"] <= age_max)]

if pension_filter != "All":
    df = df[df["db_pension"] == (1 if pension_filter == "DB pension" else 0)]

if stock_filter != "All":
    df = df[df["stock_owner"] == (1 if stock_filter == "Own stocks" else 0)]

if df.empty or df["w"].sum() <= 0:
    st.warning("No observations match the selected filters. Try widening the filters.")
    st.stop()

# ---------- layout ----------
col1, col2 = st.columns([1, 1])

# ==================================================
# PANEL 1: Theory vs Reality
# ==================================================
with col1:
    actual = wmean(df["annuity"], df["w"])

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=[actual], y=[0], mode="markers+text",
        text=[f"SCF: {actual:.2%}"],
        textposition="top center",
        marker=dict(size=14)
    ))
    fig_gap.add_trace(go.Scatter(
        x=[theory_rate], y=[0], mode="markers+text",
        text=[f"Theory: {theory_rate:.0%}"],
        textposition="bottom center",
        marker=dict(size=14)
    ))
    fig_gap.add_trace(go.Scatter(
        x=[actual, theory_rate], y=[0, 0],
        mode="lines", line=dict(width=3)
    ))

    fig_gap.update_layout(
        title="Theory vs Reality",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(visible=False),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40)
    )

    st.plotly_chart(fig_gap, use_container_width=True)

# ==================================================
# PANEL 2: By Age
# ==================================================
with col2:
    bins = [18, 40, 55, 65, 75, 100]
    labels = ["18–39", "40–54", "55–64", "65–74", "75+"]

    tmp = df.copy()
    tmp["age_bin"] = pd.cut(tmp["age"], bins=bins, labels=labels, include_lowest=True)

    age_rates = (
        tmp.groupby("age_bin", observed=True)
           .apply(lambda x: wmean(x["annuity"], x["w"]))
    )

    fig_age = go.Figure()
    fig_age.add_trace(go.Scatter(
        x=age_rates.index.astype(str),
        y=age_rates.values,
        mode="lines+markers"
    ))

    fig_age.update_layout(
        title="Annuity Ownership by Age",
        yaxis=dict(tickformat=".1%", range=[0, 0.02]),
        height=300,
        margin=dict(l=50, r=20, t=50, b=40)
    )

    st.plotly_chart(fig_age, use_container_width=True)

# ==================================================
# PANEL 3: H1 – Pensions
# ==================================================
col3, col4 = st.columns(2)

with col3:
    h1 = (
        df.groupby("db_pension", observed=True)
          .apply(lambda x: wmean(x["annuity"], x["w"]))
    )

    # ensure order [0,1] if present
    h1 = h1.reindex([0, 1])

    fig_h1 = go.Figure(go.Bar(
        x=["No DB pension", "DB pension"],
        y=h1.values,
        text=[("" if pd.isna(v) else f"{v:.2%}") for v in h1.values],
        textposition="outside"
    ))

    fig_h1.update_layout(
        title="H1: Annuity Ownership by Pension Coverage",
        yaxis=dict(tickformat=".1%", range=[0, 0.02]),
        height=300,
        margin=dict(l=50, r=20, t=50, b=40)
    )

    st.plotly_chart(fig_h1, use_container_width=True)

# ==================================================
# PANEL 4: H2 – Financial Sophistication
# ==================================================
with col4:
    h2 = (
        df.groupby("stock_owner", observed=True)
          .apply(lambda x: wmean(x["annuity"], x["w"]))
    )

    # ensure order [0,1] if present
    h2 = h2.reindex([0, 1])

    fig_h2 = go.Figure(go.Bar(
        x=["No stocks", "Own stocks"],
        y=h2.values,
        text=[("" if pd.isna(v) else f"{v:.2%}") for v in h2.values],
        textposition="outside"
    ))

    fig_h2.update_layout(
        title="H2: Annuity Ownership by Financial Sophistication",
        yaxis=dict(tickformat=".1%", range=[0, 0.02]),
        height=300,
        margin=dict(l=50, r=20, t=50, b=40)
    )

    st.plotly_chart(fig_h2, use_container_width=True)

# ==================================================
# SUMMARY
# ==================================================
st.subheader("Summary (Filtered Sample)")

summary = pd.DataFrame({
    "Metric": [
        "Annuity ownership (weighted)",
        "DB pension share (weighted)",
        "Stock owner share (weighted)",
        "Observations (unweighted)",
        "Weight sum"
    ],
    "Value": [
        f"{wmean(df['annuity'], df['w']):.2%}",
        f"{wmean(df['db_pension'], df['w']):.2%}",
        f"{wmean(df['stock_owner'], df['w']):.2%}",
        f"{df.shape[0]:,}",
        f"{df['w'].sum():,.0f}"
    ]
})

st.write(summary)

st.caption("Tip: Run with `python -m streamlit run app.py`")
