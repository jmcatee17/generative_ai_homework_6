import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from core import s1_data as d
from core.config import get_api_key
from core.tools import COMPS, market_snapshot
from core.ui import SERIES, require_key, sidebar_settings, style_fig

settings = sidebar_settings()
st.title("🌐 Market Pulse")
st.caption("Live data from Yahoo Finance and DuckDuckGo News, set against Anthropic's S-1 metrics.")


@st.cache_data(ttl=900, show_spinner="Fetching market data…")
def get_snapshot(tickers: tuple) -> pd.DataFrame:
    return pd.DataFrame(market_snapshot(list(tickers)))


@st.cache_data(ttl=900, show_spinner="Fetching price history…")
def get_history(tickers: tuple, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)["Close"]
    return df if isinstance(df, pd.DataFrame) else df.to_frame(tickers[0])


@st.cache_data(ttl=900, show_spinner="Searching news…")
def get_news(query: str) -> list[dict]:
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            return ddgs.news(query, max_results=12) or []
    except Exception:  # noqa: BLE001
        return []


# ------------------------------------------------------------------ comps
tickers = st.multiselect("Public comparables", list(COMPS), default=["NVDA", "MSFT", "GOOGL", "AMZN"],
                         max_selections=4, format_func=lambda t: f"{t} · {COMPS[t]}")
period = st.segmented_control("Price history", ["1mo", "3mo", "6mo", "1y", "2y"], default="6mo")

if tickers:
    try:
        hist = get_history(tuple(tickers), period or "6mo")
        indexed = hist / hist.bfill().iloc[0] * 100
        fig = go.Figure()
        for i, t in enumerate(tickers):  # colour follows the ticker's selection slot
            if t in indexed:
                fig.add_scatter(x=indexed.index, y=indexed[t], name=t, mode="lines", line=dict(color=SERIES[i], width=2),
                                hovertemplate=f"{t}: %{{y:.1f}}<extra></extra>")
        fig.add_hline(y=100, line_dash="dot", line_color="rgba(128,128,128,0.5)")
        fig.update_layout(title=f"Share price, indexed to 100 ({period or '6mo'} ago)")
        st.plotly_chart(style_fig(fig, money=False), use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Couldn't load price history: {e}")

    snap = get_snapshot(tuple(tickers))
    q = d.QUARTERLY
    ant = {
        "Ticker": "ANT (S-1)",
        "Company": "Anthropic — Q3'26 annualized",
        "Price": None,
        "Market cap ($B)": None,
        "TTM revenue ($B)": q["Revenue"].iloc[-4:].sum() / 1000,
        "Revenue growth (YoY)": q["Revenue"].iloc[-1] / q["Revenue"].iloc[-5] - 1,
        "Price / Sales": None,
        "Operating margin": q["Operating income (loss)"].iloc[-1] / q["Revenue"].iloc[-1],
    }
    table = pd.concat([pd.DataFrame([ant]), snap], ignore_index=True)
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Market cap ($B)": st.column_config.NumberColumn(format="$%.0f"),
            "TTM revenue ($B)": st.column_config.NumberColumn(format="$%.1f"),
            "Revenue growth (YoY)": st.column_config.NumberColumn(format="percent"),
            "Price / Sales": st.column_config.NumberColumn(format="%.1fx"),
            "Operating margin": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption("Anthropic row: TTM revenue = Q4'25–Q3'26 sum; growth = Q3'26 vs Q3'25; margin = Q3'26 GAAP operating margin. "
               "Head to the Valuation Sandbox to compare implied multiples.")

st.divider()

# ------------------------------------------------------------------ news
st.subheader("📰 Latest news")
query = st.text_input("News query", "Anthropic IPO")
news = get_news(query)
if not news:
    st.info("No news results (DuckDuckGo may be rate-limiting — try again shortly).")
else:
    if st.button("🧠 Summarize the news with AI") and require_key():
        prompt = PromptTemplate(
            input_variables=["items"],
            template="""You are a markets reporter. From these headlines and snippets about Anthropic,
write a 5-bullet briefing: what's new on the IPO (timing, pricing, valuation), investor sentiment,
and competitive context. Then one line: "Overall sentiment: Positive / Mixed / Negative" with a reason.
Cite outlets in parentheses.

{items}""",
        )
        items = "\n".join(f"- {n.get('title')} ({n.get('source')}, {str(n.get('date'))[:10]}): {n.get('body', '')[:300]}" for n in news)
        chain = prompt | ChatOpenAI(model=settings["model"], temperature=0.2, api_key=get_api_key()) | StrOutputParser()
        with st.spinner("Summarizing…"):
            st.success(chain.invoke({"items": items}))
    for n in news:
        url = n.get("url") or n.get("href")
        st.markdown(f"**[{n.get('title')}]({url})**  \n"
                    f"<small>{n.get('source', '')} · {str(n.get('date', ''))[:10]}</small>  \n{n.get('body', '')[:260]}…",
                    unsafe_allow_html=True)
