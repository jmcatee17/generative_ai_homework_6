import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import s1_data as d
from core.ui import BAD, GOOD, NEUTRAL, SERIES, sidebar_settings, style_fig

sidebar_settings()
st.title("📊 Financial Dashboard")
st.caption("All figures from the S-1/A's audited annual and unaudited interim tables. $ in billions unless noted.")

Q = d.QUARTERLY / 1000
A = d.ANNUAL.copy()
OPEX = ["Compute and infrastructure", "Personnel", "Sales, marketing, and partnerships", "Other operating and administrative", "Bartz settlement"]

# ------------------------------------------------------------------ KPI row
rev = d.QUARTERLY["Revenue"]
k = st.columns(5)
k[0].metric("Q3'26 revenue", f"${rev.iloc[-1] / 1000:.2f}B", f"{rev.iloc[-1] / rev.iloc[-2] - 1:+.0%} QoQ")
k[1].metric("9M'26 revenue", f"${A.loc['9M 2026', 'Revenue'] / 1000:.1f}B", f"{A.loc['9M 2026', 'Revenue'] / A.loc['9M 2025', 'Revenue'] - 1:+.0%} YoY")
k[2].metric("9M'26 adj. net income", f"${A.loc['9M 2026', 'Adjusted net income (loss)'] / 1000:.1f}B", "first positive period", delta_color="off")
k[3].metric("9M'26 operating cash flow", f"${d.CASH_FLOWS.loc['9M 2026', 'Operating activities'] / 1000:.1f}B", "vs −$2.0B in 9M'25")
k[4].metric("Net revenue retention", "~900%", "Sep 30, 2026", delta_color="off")

tab_growth, tab_cost, tab_loss, tab_cash, tab_cust, tab_data = st.tabs(
    ["Growth", "Cost structure", "Profitability", "Cash & balance sheet", "Customers", "Data tables"]
)

# ------------------------------------------------------------------ growth
with tab_growth:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(go.Bar(x=Q.index, y=Q["Revenue"], marker_color=SERIES[0], name="Revenue",
                               text=[f"${v:.1f}B" for v in Q["Revenue"]], textposition="outside",
                               hovertemplate="%{x}: $%{y:.2f}B<extra></extra>"))
        fig.update_layout(title="Quarterly revenue")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        growth = (d.QUARTERLY["Revenue"].pct_change() * 100).dropna()
        fig = go.Figure(go.Scatter(x=growth.index, y=growth, mode="lines+markers+text", line=dict(color=SERIES[0], width=2),
                                   marker=dict(size=9), text=[f"{v:.0f}%" for v in growth], textposition="top center",
                                   hovertemplate="%{x}: %{y:.0f}% QoQ<extra></extra>", name="QoQ growth"))
        fig.update_layout(title="Sequential (QoQ) revenue growth")
        style_fig(fig, money=False).update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
    annual = A.loc[["FY2023", "FY2024", "FY2025"], "Revenue"] / 1000
    st.markdown(
        f"**Annual revenue:** FY23 ${annual.iloc[0]:.2f}B → FY24 ${annual.iloc[1]:.2f}B → FY25 ${annual.iloc[2]:.2f}B. "
        f"Q3'26 alone (${rev.iloc[-1] / 1000:.1f}B) is **{rev.iloc[-1] / d.ANNUAL.loc['FY2025', 'Revenue']:.1f}×** all of FY2025. "
        "The S-1 notes ~$600M of Q3'26 revenue was offset by price cuts on new model versions (p. 161)."
    )

# ------------------------------------------------------------------ cost structure
with tab_cost:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for i, col in enumerate(OPEX):
            fig.add_bar(x=Q.index, y=Q[col], name=col.replace(", and partnerships", ""), marker_color=SERIES[i],
                        hovertemplate=f"{col}: $%{{y:.2f}}B<extra></extra>")
        fig.add_scatter(x=Q.index, y=Q["Revenue"], name="Revenue", mode="lines+markers",
                        line=dict(color="#52514e", width=2, dash="dot"), marker=dict(size=8))
        fig.update_layout(title="Operating expenses vs. revenue", barmode="stack")
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    with c2:
        pct = d.QUARTERLY[OPEX[:4]].div(d.QUARTERLY["Revenue"], axis=0) * 100
        fig = go.Figure()
        for i, col in enumerate(OPEX[:4]):
            fig.add_scatter(x=pct.index, y=pct[col], name=col.replace(", and partnerships", ""), mode="lines+markers",
                            line=dict(color=SERIES[i], width=2), marker=dict(size=8),
                            hovertemplate=f"{col}: %{{y:.0f}}% of revenue<extra></extra>")
        fig.update_layout(title="Expense lines as % of revenue")
        style_fig(fig, height=420, money=False).update_yaxes(ticksuffix="%", type="log")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Log scale on the right: compute fell from 251% of revenue in Q1'25 to 58% in Q3'26 — the core of the operating-leverage story. "
               "The S-1 warns margins may compress as new compute comes online ahead of revenue (p. 135).")

# ------------------------------------------------------------------ profitability
with tab_loss:
    c1, c2 = st.columns(2)
    with c1:
        oi = Q["Operating income (loss)"]
        fig = go.Figure(go.Bar(x=oi.index, y=oi, marker_color=[GOOD if v >= 0 else BAD for v in oi],
                               text=[f"${v:.2f}B" for v in oi], textposition="outside",
                               hovertemplate="%{x}: $%{y:.2f}B<extra></extra>", name="Operating income"))
        fig.update_layout(title="GAAP operating income (loss) by quarter", showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        periods = d.ANNUAL_PERIODS
        fig = go.Figure()
        fig.add_bar(x=periods, y=A["Net loss"] / 1000, name="GAAP net loss", marker_color=SERIES[1],
                    hovertemplate="GAAP: $%{y:.1f}B<extra></extra>")
        fig.add_bar(x=periods, y=A["Adjusted net income (loss)"] / 1000, name="Adjusted net income (non-GAAP)",
                    marker_color=SERIES[0], hovertemplate="Adjusted: $%{y:.1f}B<extra></extra>")
        fig.update_layout(title="GAAP vs. adjusted net income", barmode="group")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Why is the 9M'26 net loss $104B? The loss bridge")
    labels = [x for x, _ in d.LOSS_BRIDGE_9M26] + ["Net loss"]
    vals = [v / 1000 for _, v in d.LOSS_BRIDGE_9M26]
    fig = go.Figure(go.Waterfall(
        x=labels, y=vals + [0], measure=["relative"] * len(vals) + ["total"],
        text=[f"${v:+.1f}B" for v in vals] + [f"${sum(vals):.1f}B"], textposition="outside",
        increasing=dict(marker=dict(color=GOOD)), decreasing=dict(marker=dict(color=BAD)),
        totals=dict(marker=dict(color=NEUTRAL)), connector=dict(line=dict(color="rgba(128,128,128,0.4)")),
    ))
    st.plotly_chart(style_fig(fig, height=400), use_container_width=True)
    st.info(
        "≈ **91%** of the 9M'26 net loss is the non-cash remeasurement of convertible notes, derivatives and SAFEs — "
        "they rise in value as Anthropic's valuation rises. Operationally, the company posted a **12% adjusted operating margin** "
        "and **$6.6B of operating cash flow** in the same period (S-1 p. 23, 159, 172)."
    )
    qm = d.QUARTERLY[["Operating margin %", "Adjusted operating margin %"]]
    fig = go.Figure()
    for i, col in enumerate(qm.columns):
        fig.add_scatter(x=qm.index, y=qm[col], name=col.replace(" %", "").replace("Adjusted", "Adjusted (non-GAAP)"),
                        mode="lines+markers+text", line=dict(color=SERIES[i], width=2), marker=dict(size=8),
                        text=[f"{v}%" if j >= len(qm) - 3 else "" for j, v in enumerate(qm[col])], textposition="top center",
                        hovertemplate=f"{col}: %{{y}}%<extra></extra>")
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.5)")
    fig.update_layout(title="Quarterly operating margin: GAAP vs. adjusted")
    style_fig(fig, money=False).update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Adjusted operating margin went from (93)% in Q3'25 to 15% in Q3'26. It peaked at 19% in Q2'26, then fell "
               "mainly because of $5.0B of short-term compute spend (S-1 p. 138, 163).")

# ------------------------------------------------------------------ cash
with tab_cash:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for i, col in enumerate(d.CASH_FLOWS.columns):
            fig.add_bar(x=d.CASH_FLOWS.index, y=d.CASH_FLOWS[col] / 1000, name=col, marker_color=SERIES[i],
                        hovertemplate=f"{col}: $%{{y:.1f}}B<extra></extra>")
        fig.update_layout(title="Cash flows by activity", barmode="group")
        st.plotly_chart(style_fig(fig, height=400), use_container_width=True)
    with c2:
        bs = d.BALANCE_SHEET / 1000
        st.markdown("**Balance sheet — Sep 30, 2026**")
        st.dataframe(bs.map(lambda v: f"${v:,.1f}B").to_frame(), use_container_width=True)
    st.warning(
        "**Commitments:** ~$895.4B of aggregate committed spend on purchase commitments and leases, "
        "substantially all compute & infrastructure (S-1 p. 7) — ~6.8× the cash balance."
    )

# ------------------------------------------------------------------ customers
with tab_cust:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=d.CUSTOMER_TIERS["Tier"], y=d.CUSTOMER_TIERS["Customers"], marker_color=SERIES[0],
                               text=[f"{v:,}+" for v in d.CUSTOMER_TIERS["Customers"]], textposition="outside",
                               hovertemplate="%{x}: %{y:,}<extra></extra>"))
        fig.update_layout(title="Customers by TTM revenue tier (Sep 30, 2026)", showlegend=False)
        st.plotly_chart(style_fig(fig, money=False), use_container_width=True)
    with c2:
        h = d.CUSTOMERS_100K_HISTORY
        fig = go.Figure(go.Scatter(x=h["Date"], y=h["Customers ($100K+ TTM)"], mode="lines+markers+text",
                                   text=[f"{v:,}" for v in h["Customers ($100K+ TTM)"]], textposition="top left",
                                   line=dict(color=SERIES[0], width=2), marker=dict(size=9)))
        fig.update_layout(title="Customers with $100K+ TTM revenue", showlegend=False)
        st.plotly_chart(style_fig(fig, money=False), use_container_width=True)
    st.markdown(
        "- **Net Revenue Retention ~900%** as of Sep 30, 2026 (p. 136)\n"
        "- **No single customer >10%** of 9M'26 revenue; ~79% of revenue came from outside the top five (p. 144)\n"
        "- Model versions released in the prior six months drove **>80%** of Q3'26 revenue (p. 144) — a rapid product cycle"
    )

# ------------------------------------------------------------------ raw tables
with tab_data:
    st.markdown("**Quarterly statement of operations ($M)** — S-1 p. 160")
    st.dataframe(d.QUARTERLY.T, use_container_width=True)
    st.markdown("**Annual & nine-month data ($M)** — S-1 p. 23")
    st.dataframe(d.ANNUAL.T, use_container_width=True)
    st.markdown("**Cash flows ($M)** — S-1 p. 172")
    st.dataframe(d.CASH_FLOWS.T, use_container_width=True)
    st.download_button("Download quarterly data (CSV)", d.QUARTERLY.to_csv(), "anthropic_quarterly.csv", "text/csv")
