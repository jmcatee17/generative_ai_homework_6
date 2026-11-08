import streamlit as st

from core.agent import build_chat_agent
from core.config import S1_LABEL, S1_URL
from core.graphviz import to_dot
from core.rag import S1Index
from core.s1_data import ANNUAL, BALANCE_SHEET, QUARTERLY
from core.ui import sidebar_settings

sidebar_settings()

st.title("📈 Anthropic S-1 Explorer")
st.markdown(
    f"""An AI-powered research workbench for **Anthropic's IPO prospectus** —
[{S1_LABEL}]({S1_URL}). Proposed listing: **Nasdaq: ANT**."""
)

q = QUARTERLY["Revenue"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Q3'26 revenue", f"${q.iloc[-1] / 1000:.1f}B", f"{q.iloc[-1] / q.iloc[-5] - 1:+.0%} YoY")
c2.metric("9M'26 adj. operating margin", f"{ANNUAL.loc['9M 2026', 'Adjusted operating margin %']}%",
          f"{ANNUAL.loc['9M 2026', 'Adjusted operating margin %'] - ANNUAL.loc['9M 2025', 'Adjusted operating margin %']:+} pts YoY")
c3.metric("Cash & short-term investments", f"${BALANCE_SHEET.iloc[0] / 1000:.1f}B", "Sep 30, 2026", delta_color="off")
c4.metric("Customers $100K+ TTM", "11,900+", "from 950+ a year earlier")

st.divider()
left, right = st.columns([3, 2])
with left:
    st.subheader("What's inside")
    st.markdown(
        """
| Page | What it does |
|---|---|
| 📊 **Financial Dashboard** | Interactive charts of revenue, cost structure, margins, the GAAP-loss bridge, cash flow and customer growth — all from the S-1 tables. |
| 💬 **Chat with the S-1** | A **LangGraph** tool-calling agent. Hybrid RAG (embeddings + BM25) over every page of the prospectus, with page-level citations, plus a calculator, live web search, page fetcher and stock-market comps. |
| 📝 **Analyst Memo** | A multi-node **LangGraph** workflow: five research nodes query the S-1 in parallel with a web-news node, then a writer drafts an IPO memo and a scorer turns it into a scorecard. |
| 🌐 **Market Pulse** | Live news on the IPO and public AI comparables (price history, P/S, margins) next to Anthropic's S-1 metrics. |
| 🧮 **Valuation Sandbox** | Pick a valuation and growth path; see implied revenue multiples vs. live comps and a simple DCF. |
"""
    )
    st.caption("For education and research only — not investment advice.")

with right:
    st.subheader("Chat agent graph")
    st.graphviz_chart(to_dot(build_chat_agent()), use_container_width=True)
    st.caption("Dashed edge = conditional: the model decides whether to call tools or answer.")
    if S1Index.exists():
        st.success("RAG index ready.")
    else:
        st.info("The S-1 will be embedded for RAG the first time you open the chat or memo page.")
