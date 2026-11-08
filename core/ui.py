"""Shared Streamlit helpers: sidebar settings, cached RAG index, chart styling."""
from __future__ import annotations

import streamlit as st

from core.config import CHAT_MODELS, S1_LABEL, S1_URL, get_api_key, set_api_key
from core.rag import S1Index
from core.tools import set_index

# Validated categorical palette (fixed order) + neutral ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, BAD, NEUTRAL = "#1baf7a", "#e34948", "#8a8984"


def sidebar_settings() -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        if not get_api_key():
            key = st.text_input("OpenAI API key", type="password", help="Or put OPENAI_API_KEY in a .env file")
            if key:
                set_api_key(key)
        else:
            st.caption("✅ OpenAI API key loaded")
        model = st.selectbox("Chat model", CHAT_MODELS, key="model")
        st.divider()
        st.caption(f"Source: [{S1_LABEL}]({S1_URL})")
    return {"model": model}


def require_key() -> bool:
    if not get_api_key():
        st.warning("Add your OpenAI API key in the sidebar (or a `.env` file) to use this page.")
        return False
    return True


@st.cache_resource(show_spinner=False)
def _load_index() -> S1Index:
    return S1Index.load()


def get_index() -> S1Index | None:
    """Load the embedded S-1 index, building it (one-time, ~1 min, < $0.01) if needed."""
    if not S1Index.exists():
        if not require_key():
            return None
        st.info("First run: embedding the S-1 for retrieval (one-time, about a minute).")
        bar = st.progress(0.0, text="Embedding S-1 chunks…")
        S1Index.build(progress=lambda p: bar.progress(p, text=f"Embedding S-1 chunks… {p:.0%}"))
        bar.empty()
        _load_index.clear()
    index = _load_index()
    set_index(index)
    return index


def style_fig(fig, height: int = 360, money: bool = True):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None),
        hovermode="x unified",
        bargap=0.25,
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=True, zerolinecolor="rgba(128,128,128,0.4)", title=None)
    if money:
        fig.update_yaxes(tickprefix="$", ticksuffix="B")
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig
