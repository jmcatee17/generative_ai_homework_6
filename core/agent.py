"""LangGraph ReAct-style agent for chatting with the S-1.

Graph:  START -> agent --(tool calls?)--> tools -> agent ... -> END

The agent node is an OpenAI chat model with tools bound. The tools node runs
whatever the model asked for (RAG search, financial tables, calculator, web).
An in-memory checkpointer keeps the conversation per Streamlit session thread.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from core.config import S1_LABEL, get_api_key
from core.tools import ALL_TOOLS, S1_ONLY_TOOLS

SYSTEM_TEMPLATE = PromptTemplate(
    input_variables=["source", "web_policy"],
    template="""You are an equity research analyst assistant investigating Anthropic's IPO.
Your primary source is: {source}.

How to work:
- Ground every claim about Anthropic in the S-1. Call `search_s1` (try a section filter
  when it helps) and `get_s1_financial_table` for exact numbers.
- Broad or ranking questions ("top risks", "summarize the business", "bull vs. bear case",
  "what matters most") need breadth. Never answer them from a single generic search. First
  list the 5-8 distinct sub-topics an expert would check. For risks, those are compute
  commitments and capital intensity, dependence on cloud providers who are also competitors,
  customer concentration, competition and pricing pressure, net losses and convertible-note
  accounting, PBC/LTBT governance and dual-class control, regulation and safety incidents,
  and IP/copyright litigation. Issue a separate, specific `search_s1` call for each one in
  the same turn (parallel tool calls), plus `get_s1_financial_table("key_facts")`.
- When ranking, order by materiality: dollar magnitude, prominence in the Prospectus
  Summary, and how directly it threatens revenue or cash. Say briefly why each item ranks
  where it does. Don't let a narrow passage outrank a company-level issue just because it
  matched the query wording.
- Use `calculator` for any arithmetic (growth, margins, multiples).
- When explaining a figure, quantify its largest drivers with exact numbers from the tables
  (e.g. size each reconciling item and its share of the total), biggest first.
- {web_policy}
- Cite S-1 facts inline exactly like [S-1 p. 160 · MD&A]. Cite web facts with the source
  name and link. Clearly separate what the S-1 says from outside commentary.
- If the S-1 doesn't answer something, say so rather than guessing. Note that some
  offering terms (price range, share count) are blank in this amendment.
- Be concise and analytical: lead with the answer, then supporting detail. Use tables
  for numeric comparisons. You provide research, not personalized investment advice.""",
)

WEB_ON = (
    "You may use `web_search`, `fetch_webpage`, and `get_market_data` for things the S-1 "
    "can't tell you: current news, IPO pricing/timing, analyst views, and public comps."
)
WEB_OFF = "Web tools are disabled: answer only from the S-1."


def build_chat_agent(model: str = "gpt-4.1-mini", temperature: float = 0.1, allow_web: bool = True):
    tools = ALL_TOOLS if allow_web else S1_ONLY_TOOLS
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=get_api_key() or "sk-missing").bind_tools(tools)
    system = SystemMessage(SYSTEM_TEMPLATE.format(source=S1_LABEL, web_policy=WEB_ON if allow_web else WEB_OFF))

    def agent(state: MessagesState):
        return {"messages": [llm.invoke([system, *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())
