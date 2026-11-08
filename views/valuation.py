import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import s1_data as d
from core.tools import market_snapshot
from core.ui import BAD, SERIES, sidebar_settings, style_fig

sidebar_settings()
st.title("🧮 Valuation Sandbox")
st.caption("An illustrative, simplified DCF and multiples view built on the S-1's numbers. "
           "Educational only — not investment advice.")

q = d.QUARTERLY
run_rate = q["Revenue"].iloc[-1] * 4 / 1000  # Q3'26 annualized, $B
cash = d.BALANCE_SHEET["Cash, cash equivalents & short-term investments"] / 1000
start_margin = d.ANNUAL.loc["9M 2026", "Adjusted operating margin %"] / 100

# ------------------------------------------------------------------ inputs
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    valuation = c1.number_input("Valuation to test ($B)", 100, 5000, 1500, step=25,
                                help="~$965B was the last private round; Nov 2026 reports cite ~$1.5T as the expected IPO "
                                     "valuation. The S-1/A leaves the price range blank.")
    g1 = c1.slider("Year-1 revenue growth", 0, 300, 120, 5, format="%d%%") / 100
    term_g = c1.slider("Terminal growth", 0.0, 5.0, 3.0, 0.5, format="%.1f%%") / 100
    term_margin = c2.slider("Steady-state operating margin (yr 6+)", 0, 60, 30, 1, format="%d%%") / 100
    reinvest = c2.slider("Reinvestment (capex, net of D&A) % of revenue", 0, 40, 10, 1, format="%d%%") / 100
    tax = c2.slider("Tax rate", 0, 35, 21, 1, format="%d%%") / 100
    wacc = c3.slider("Discount rate (WACC)", 6.0, 20.0, 11.0, 0.5, format="%.1f%%") / 100
    dilution = c3.slider("Annual SBC dilution", 0.0, 5.0, 2.0, 0.5, format="%.1f%%") / 100


def dcf(g1, term_g, term_margin, reinvest, tax, wacc, dilution, years=10):
    growth = term_g + (g1 - term_g) * np.power(0.6, np.arange(years))  # decays toward terminal growth
    revenue = run_rate * np.cumprod(1 + growth)
    margin = np.array(np.linspace(start_margin, term_margin, 5).tolist() + [term_margin] * (years - 5))
    fcf = revenue * (margin * (1 - tax) - reinvest)
    disc = np.power(1 + wacc, np.arange(1, years + 1))
    pv = fcf / disc
    tv = fcf[-1] * (1 + term_g) / (wacc - term_g) / disc[-1] if wacc > term_g else np.nan
    equity = (pv.sum() + tv + cash) * (1 - dilution) ** years
    return equity, pd.DataFrame({"Year": [f"Y{i + 1}" for i in range(years)], "Revenue": revenue, "Growth": growth,
                                 "Op margin": margin, "FCF": fcf, "PV of FCF": pv}), tv


equity, proj, tv = dcf(g1, term_g, term_margin, reinvest, tax, wacc, dilution)

k = st.columns(4)
k[0].metric("EV / run-rate revenue", f"{(valuation - cash) / run_rate:.1f}×", f"on ${run_rate:.0f}B (Q3'26 × 4)", delta_color="off")
k[1].metric("EV / next-year revenue", f"{(valuation - cash) / proj['Revenue'].iloc[0]:.1f}×", f"${proj['Revenue'].iloc[0]:.0f}B projected", delta_color="off")
k[2].metric("DCF implied equity value", f"${equity:,.0f}B", f"{equity / valuation - 1:+.0%} vs ${valuation:,}B")
k[3].metric("Terminal value share of DCF", f"{tv / (proj['PV of FCF'].sum() + tv):.0%}" if np.isfinite(tv) else "n/a")

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_bar(x=proj["Year"], y=proj["Revenue"], name="Projected revenue", marker_color=SERIES[0],
                hovertemplate="%{x}: $%{y:,.0f}B<extra></extra>")
    fig.add_bar(x=proj["Year"], y=proj["FCF"], name="Free cash flow", marker_color=SERIES[2],
                hovertemplate="%{x}: $%{y:,.0f}B<extra></extra>")
    fig.update_layout(title="Projection", barmode="group")
    st.plotly_chart(style_fig(fig), use_container_width=True)

with c2:
    wacc_grid = np.arange(8.0, 16.1, 1.0) / 100
    margin_grid = np.arange(15, 51, 5) / 100
    z = np.array([[dcf(g1, term_g, m, reinvest, tax, w, dilution)[0] / valuation - 1 for w in wacc_grid] for m in margin_grid])
    lim = float(np.nanmax(np.abs(np.clip(z, -1, 1))))
    fig = go.Figure(go.Heatmap(
        z=z * 100, x=[f"{w:.0%}" for w in wacc_grid], y=[f"{m:.0%}" for m in margin_grid], zmid=0,
        zmin=-lim * 100, zmax=lim * 100, colorscale=[[0, BAD], [0.5, "#e8e7e3"], [1, SERIES[0]]],
        text=[[f"{v:+.0%}" for v in row] for row in z], texttemplate="%{text}",
        hovertemplate="WACC %{x}, margin %{y}: %{text} vs. tested valuation<extra></extra>",
        colorbar=dict(ticksuffix="%", title=""),
    ))
    fig.update_layout(title="Upside / downside vs. tested valuation")
    style_fig(fig, money=False).update_xaxes(title="Discount rate").update_yaxes(title="Steady-state op. margin")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Multiples vs. public comparables")


@st.cache_data(ttl=900, show_spinner="Fetching comps…")
def comps() -> pd.DataFrame:
    return pd.DataFrame(market_snapshot(["NVDA", "MSFT", "GOOGL", "META", "PLTR", "SNOW"]))


try:
    cdf = comps().dropna(subset=["Price / Sales"])
    rows = pd.concat([pd.DataFrame([{"Ticker": "ANT @ tested value", "Price / Sales": valuation / run_rate,
                                     "Revenue growth (YoY)": q["Revenue"].iloc[-1] / q["Revenue"].iloc[-5] - 1}]),
                      cdf[["Ticker", "Price / Sales", "Revenue growth (YoY)"]]], ignore_index=True)
    rows = rows.sort_values("Price / Sales")
    fig = go.Figure(go.Bar(
        x=rows["Price / Sales"], y=rows["Ticker"], orientation="h",
        marker_color=[SERIES[1] if t.startswith("ANT") else SERIES[0] for t in rows["Ticker"]],
        text=[f"{v:.1f}×" for v in rows["Price / Sales"]], textposition="outside",
        customdata=rows["Revenue growth (YoY)"] * 100,
        hovertemplate="%{y}: %{x:.1f}× sales · growth %{customdata:.0f}%<extra></extra>",
    ))
    style_fig(fig, height=320, money=False).update_xaxes(ticksuffix="×")
    fig.update_layout(showlegend=False, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Anthropic's multiple uses Q3'26 annualized revenue (a forward-ish base), while comps use trailing-12-month revenue — "
               "so this flatters Anthropic relative to peers. Its YoY growth (~1,100%) is in a different league.")
except Exception as e:  # noqa: BLE001
    st.warning(f"Couldn't load comps: {e}")

with st.expander("Projection table"):
    st.dataframe(proj.set_index("Year").style.format({"Revenue": "${:,.0f}B", "FCF": "${:,.0f}B", "PV of FCF": "${:,.0f}B",
                                                       "Growth": "{:.0%}", "Op margin": "{:.0%}"}), use_container_width=True)
    st.caption("Ignores the ~$895B of compute commitments, the capital structure conversions at IPO, and all offering terms. "
               "Margin ramps linearly from the 9M'26 adjusted operating margin (12%) to the steady-state margin by year 5.")
