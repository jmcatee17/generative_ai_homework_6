import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.agent import build_chat_agent
from core.ui import get_index, require_key, sidebar_settings

settings = sidebar_settings()
with st.sidebar:
    allow_web = st.toggle("Allow web & market tools", value=True,
                          help="Off = answers come only from the S-1 (RAG + financial tables + calculator).")
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.pop("chat", None)

st.title("💬 Chat with the S-1")
st.caption("A LangGraph agent with hybrid RAG over all ~380 pages of the prospectus. Answers cite S-1 page numbers.")

if not require_key() or (index := get_index()) is None:
    st.stop()

# One agent + thread per (model, web) setting; changing settings starts a fresh chat
cfg_key = (settings["model"], allow_web)
chat = st.session_state.get("chat")
if chat is None or chat["cfg"] != cfg_key:
    chat = st.session_state["chat"] = {
        "cfg": cfg_key,
        "agent": build_chat_agent(settings["model"], allow_web=allow_web),
        "thread": str(uuid.uuid4()),
        "history": [],  # (role, text, trace)
    }

SUGGESTIONS = [
    "Summarize Anthropic's business model and how it makes money.",
    "Why is the GAAP net loss so large when adjusted operating margin is positive?",
    "What are the top 5 risk factors an investor should worry about?",
    "How does the Long-Term Benefit Trust affect shareholder control?",
    "What are Anthropic's compute commitments and who are the key partners?",
    "Compare Anthropic's revenue growth and margins to Palantir and Snowflake right now.",
    "What's the latest news on the IPO pricing and timing?",
]


def render_trace(trace):
    with st.expander(f"🔧 Agent steps ({len(trace)} tool calls)", expanded=False):
        for name, args, output in trace:
            st.markdown(f"**`{name}`** `{args}`")
            st.code(output[:1500] + ("…" if len(output) > 1500 else ""), language=None)


for role, text, trace in chat["history"]:
    with st.chat_message(role):
        if trace:
            render_trace(trace)
        st.markdown(text)

prompt = st.chat_input("Ask anything about Anthropic's S-1…") or st.session_state.pop("pending_prompt", None)

if not chat["history"] and not prompt:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sugg{i}", use_container_width=True):
            st.session_state["pending_prompt"] = s
            st.rerun()

if prompt:
    chat["history"].append(("user", prompt, []))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        trace, pending, answer = [], {}, ""
        status = st.status("Thinking…", expanded=True)
        try:
            for update in chat["agent"].stream(
                {"messages": [HumanMessage(prompt)]},
                config={"configurable": {"thread_id": chat["thread"]}, "recursion_limit": 30},
                stream_mode="updates",
            ):
                for node, payload in update.items():
                    for msg in payload["messages"]:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                pending[tc["id"]] = (tc["name"], tc["args"])
                                status.write(f"🔧 `{tc['name']}` — {', '.join(f'{k}={v!r}' for k, v in tc['args'].items())}")
                        elif isinstance(msg, ToolMessage):
                            name, args = pending.get(msg.tool_call_id, (msg.name, {}))
                            trace.append((name, args, str(msg.content)))
                        elif isinstance(msg, AIMessage):
                            answer = msg.content
            status.update(label=f"Done — {len(trace)} tool calls", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            status.update(label="Error", state="error")
            answer = f"⚠️ Something went wrong: `{e}`"
        if trace:
            render_trace(trace)
        st.markdown(answer)
    chat["history"].append(("assistant", answer, trace))
