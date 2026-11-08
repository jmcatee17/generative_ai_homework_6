"""Anthropic S-1 Explorer — entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Anthropic S-1 Explorer", page_icon="📈", layout="wide")

pages = st.navigation(
    [
        st.Page("views/home.py", title="Overview", icon="🏠", default=True),
        st.Page("views/dashboard.py", title="Financial Dashboard", icon="📊"),
        st.Page("views/chat.py", title="Chat with the S-1", icon="💬"),
        st.Page("views/memo.py", title="Analyst Memo (LangGraph)", icon="📝"),
        st.Page("views/market.py", title="Market Pulse", icon="🌐"),
        st.Page("views/valuation.py", title="Valuation Sandbox", icon="🧮"),
    ]
)
pages.run()
