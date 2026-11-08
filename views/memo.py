import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.graphviz import to_dot
from core.memo_graph import RESEARCH_AREAS, build_memo_graph
from core.ui import SERIES, get_index, require_key, sidebar_settings, style_fig

settings = sidebar_settings()
st.title("📝 Analyst Memo Generator")
st.caption(
    "A multi-node LangGraph workflow: five S-1 research nodes and a web-news node run **in parallel**, "
    "a writer node fans their findings in, and a scorer produces a structured scorecard."
)

if not require_key() or (index := get_index()) is None:
    st.stop()

graph = build_memo_graph(index, settings["model"])
with st.expander("View the workflow graph"):
    st.graphviz_chart(to_dot(graph), use_container_width=True)

focus = st.text_input(
    "Memo focus",
    "Is Anthropic's growth durable enough to justify a $1T+ valuation, and what could break the thesis?",
)
include_news = st.checkbox("Include live web news", value=True)

if st.button("✍️ Generate memo", type="primary"):
    labels = {k: v["title"] for k, v in RESEARCH_AREAS.items()} | {
        "web_news": "Recent news", "write_memo": "Drafting memo", "scorecard": "Scoring"}
    result = {"findings": {}, "sources": []}
    with st.status("Running research graph…", expanded=True) as status:
        for update in graph.stream({"focus": focus, "include_news": include_news, "findings": {}, "sources": []},
                                   stream_mode="updates"):
            for node, payload in update.items():
                status.write(f"✅ {labels.get(node, node)}")
                for key, val in (payload or {}).items():
                    if key == "findings":
                        result["findings"].update(val)
                    elif key == "sources":
                        result["sources"].extend(val)
                    else:
                        result[key] = val
        status.update(label="Memo ready", state="complete", expanded=False)
    st.session_state["memo_result"] = result

result = st.session_state.get("memo_result")
if result and result.get("memo"):
    memo_col, side = st.columns([3, 2])
    with memo_col:
        st.markdown(result["memo"])
        st.download_button("Download memo (.md)", result["memo"], "anthropic_ipo_memo.md")
    with side:
        card = result.get("scorecard") or {}
        if card:
            st.subheader("Scorecard")
            dims = ["growth", "profitability", "balance_sheet", "competitive_moat", "governance", "risk_level"]
            names = [x.replace("_", " ").title() for x in dims]
            vals = [card[x] for x in dims]
            fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", text=vals, textposition="outside",
                                   marker_color=[SERIES[1] if x == "risk_level" else SERIES[0] for x in dims],
                                   hovertemplate="%{y}: %{x}/10<extra></extra>"))
            style_fig(fig, height=320, money=False).update_xaxes(range=[0, 10.5])
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(showlegend=False, hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Orange = risk (higher is riskier). Blue = higher is better.")
            st.info(f"**Verdict:** {card.get('verdict', '')}")
        if result["sources"]:
            src = pd.DataFrame(result["sources"]).drop_duplicates()
            st.subheader("Pages retrieved")
            st.dataframe(src.groupby("section").size().rename("chunks").sort_values(ascending=False),
                         use_container_width=True)
    with st.expander("Raw research notes from each node"):
        for k, v in result["findings"].items():
            st.markdown(f"#### {RESEARCH_AREAS[k]['title']}\n{v}")
        if result.get("news"):
            st.markdown(f"#### Recent news\n{result['news']}")
