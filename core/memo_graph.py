"""Multi-node LangGraph workflow that writes an IPO analyst memo.

            ┌─> financials ─┐
            ├─> customers  ─┤
   START ───┼─> compute    ─┼──> write_memo ──> scorecard ──> END
            ├─> risks      ─┤
            ├─> governance ─┤
            └─> web_news   ─┘

Each research node runs several RAG queries against the S-1 (fan-out, in
parallel), condenses what it found into cited bullet points, and writes into a
shared `findings` dict. The writer node fans those back in and drafts the memo;
the last node turns it into a structured scorecard for charting.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core import s1_data
from core.config import get_api_key
from core.rag import S1Index, format_hits
from core.tools import web_search

RESEARCH_AREAS = {
    "financials": {
        "title": "Financial performance & profitability",
        "section": None,
        "queries": [
            "revenue growth drivers results of operations comparison nine months",
            "adjusted operating margin non-GAAP compute efficiency gross margin",
            "net loss change in fair value of convertible notes explanation",
        ],
        "tables": ["annual", "quarterly", "cash_flows"],
    },
    "customers": {
        "title": "Customers, products & go-to-market",
        "section": None,
        "queries": [
            "customers with $100,000 or more TTM revenue net revenue retention",
            "Claude Code Claude Platform applications consumption subscription revenue mix",
            "top five customers concentration cloud partners AWS Google Azure distribution",
        ],
        "tables": ["customers"],
    },
    "compute": {
        "title": "Compute strategy & commitments",
        "section": None,
        "queries": [
            "compute purchase commitments lease agreements aggregate committed spend",
            "TPUs Trainium NVIDIA GPUs multi-chip compute architecture data centers",
            "short-term compute cost capital expenditures cash used in investing",
        ],
        "tables": [],
    },
    "risks": {
        "title": "Key risks",
        "section": "Risk Factors",
        "queries": [
            "competition OpenAI Google open-weight models pricing pressure",
            "dependence on cloud service providers who are also competitors",
            "catastrophic AI incident safety misuse liability regulation",
            "copyright litigation training data lawsuits",
        ],
        "tables": [],
    },
    "governance": {
        "title": "Governance, PBC structure & capital structure",
        "section": None,
        "queries": [
            "Long-Term Benefit Trust board election rights public benefit corporation",
            "Class A Class N Class F Class T Class E common stock voting rights",
            "Google Amazon ownership investment convertible notes",
        ],
        "tables": [],
    },
}

RESEARCH_PROMPT = PromptTemplate(
    input_variables=["area", "focus", "context"],
    template="""You are an equity research associate covering Anthropic's IPO.
Research area: {area}
Analyst's focus for this memo: {focus}

Using ONLY the S-1 excerpts and tables below, write 5-8 dense bullet points of the most
decision-relevant findings for this area. Include concrete numbers. End every bullet with
its citation exactly as given, e.g. [S-1 p. 160 · MD&A]. Flag anything notable or unusual.

S-1 material:
{context}""",
)

NEWS_PROMPT = PromptTemplate(
    input_variables=["focus", "results"],
    template="""Summarize the recent news below about Anthropic's IPO in 4-6 bullets
(pricing/timing, valuation, investor reaction, competitive context). Analyst focus: {focus}.
Cite each bullet with the outlet name. Ignore irrelevant results.

{results}""",
)

MEMO_PROMPT = PromptTemplate(
    input_variables=["focus", "findings", "news"],
    template="""Write an IPO research memo on Anthropic, PBC (proposed Nasdaq: ANT).
Analyst focus: {focus}

Structure (markdown):
## Executive summary  (3-4 sentences, with a bottom-line view)
## Investment thesis — bull case  (bullets)
## Bear case & key risks  (bullets)
## Financial snapshot  (a compact markdown table of the most important figures)
## Governance & structure considerations
## What the market is saying  (from the news section; skip if empty)
## Questions to diligence before the roadshow  (5 bullets)

Keep the S-1 citations from the research notes. Be balanced and specific.
Label every figure's period precisely (e.g. Q3'26, 9M 2026, FY2025, "as of Sep 30, 2026").
Never call nine-month or quarterly figures "TTM" or "annual". This is research,
not a buy/sell recommendation to any individual.

Research notes:
{findings}

Recent news:
{news}""",
)


class Scorecard(BaseModel):
    growth: int = Field(description="Revenue growth & momentum")
    profitability: int = Field(description="Path to profitability / margins")
    balance_sheet: int = Field(description="Liquidity & funding strength")
    competitive_moat: int = Field(description="Durability of competitive position")
    governance: int = Field(description="Shareholder-friendliness of governance (10 = most friendly)")
    risk_level: int = Field(description="Overall risk (10 = riskiest)")
    verdict: str = Field(description="One-sentence bottom line")


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class MemoState(TypedDict):
    focus: str
    include_news: bool
    findings: Annotated[dict, _merge]
    sources: Annotated[list, operator.add]
    news: str
    memo: str
    scorecard: dict


def build_memo_graph(index: S1Index, model: str = "gpt-4.1-mini"):
    llm = ChatOpenAI(model=model, temperature=0.2, api_key=get_api_key() or "sk-missing")
    to_text = llm | StrOutputParser()

    def make_research_node(key: str, spec: dict):
        def node(state: MemoState):
            seen, hits = set(), []
            for q in spec["queries"]:
                for h in index.search(f"{q} {state['focus']}", k=4, section=spec["section"]):
                    if h.text not in seen:
                        seen.add(h.text)
                        hits.append(h)
            context = format_hits(hits)
            for t in spec["tables"]:
                context += f"\n\n[S-1 financial table: {t}]\n{s1_data.table_as_text(t)}"
            notes = to_text.invoke(
                RESEARCH_PROMPT.format(area=spec["title"], focus=state["focus"], context=context)
            )
            return {
                "findings": {key: notes},
                "sources": [{"area": spec["title"], "page": h.page, "section": h.section} for h in hits],
            }

        return node

    def news_node(state: MemoState):
        if not state.get("include_news", True):
            return {"news": ""}
        results = web_search.invoke({"query": "Anthropic IPO S-1", "news": True, "max_results": 8})
        return {"news": to_text.invoke(NEWS_PROMPT.format(focus=state["focus"], results=results))}

    def write_memo(state: MemoState):
        findings = "\n\n".join(
            f"### {RESEARCH_AREAS[k]['title']}\n{v}" for k, v in state["findings"].items()
        )
        memo = to_text.invoke(MEMO_PROMPT.format(focus=state["focus"], findings=findings, news=state.get("news") or "(none)"))
        return {"memo": memo}

    def scorecard(state: MemoState):
        card = llm.with_structured_output(Scorecard).invoke(
            "Score Anthropic's IPO on each dimension (1-10) based strictly on this memo:\n\n" + state["memo"]
        )
        return {"scorecard": {k: (max(1, min(10, v)) if isinstance(v, int) else v) for k, v in card.model_dump().items()}}

    g = StateGraph(MemoState)
    for key, spec in RESEARCH_AREAS.items():
        g.add_node(key, make_research_node(key, spec))
        g.add_edge(START, key)
    g.add_node("web_news", news_node)
    g.add_edge(START, "web_news")
    g.add_node("write_memo", write_memo)
    g.add_node("scorecard", scorecard)
    g.add_edge([*RESEARCH_AREAS, "web_news"], "write_memo")
    g.add_edge("write_memo", "scorecard")
    g.add_edge("scorecard", END)
    return g.compile()
