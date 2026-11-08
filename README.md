# Anthropic S-1 Explorer

A Streamlit + LangGraph research app for **Anthropic, PBC's IPO prospectus**
([Form S-1/A, filed Nov 2, 2026](https://www.sec.gov/Archives/edgar/data/2130357/000162828026068676/anthropicpbc-sx1ax1.htm), proposed Nasdaq: ANT).

## Run it

```bash
cd generative_ai_homework_6_assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your OpenAI key into .env
streamlit run app.py
```

You can also paste the key into the sidebar. The first time you open the chat or memo page, the
S-1 is embedded for RAG (~1,400 chunks with `text-embedding-3-small`: about a minute, under $0.01).
The result is cached in `data/`.

## Pages

| Page | Highlights |
|---|---|
| **Overview** | Headline KPIs and a live drawing of the chat agent's LangGraph. |
| **Financial Dashboard** | Quarterly revenue and growth, cost structure (absolute and as a % of revenue), GAAP vs. adjusted profitability, a **net-loss waterfall** (most of the $104B 9M'26 loss is non-cash fair-value remeasurement), cash flows, the balance sheet and customer tiers. |
| **Chat with the S-1** | A LangGraph tool-calling agent (see below) that shows its tool calls and cites page numbers. |
| **Analyst Memo** | A parallel multi-node LangGraph workflow that writes an IPO memo and a 1–10 scorecard. |
| **Market Pulse** | Live price history and valuation multiples for AI comps (yfinance), plus news (DuckDuckGo) with an AI news summary. |
| **Valuation Sandbox** | Interactive DCF, a WACC × margin sensitivity heatmap, and implied P/S compared with live comps. |

## Architecture

```
core/
  config.py      paths, models (gpt-4.1-mini default, same as class template), API key
  s1_data.py     hand-extracted financial tables from the S-1 (with page refs)
  rag.py         page-aware chunking -> OpenAI embeddings + BM25 -> reciprocal-rank fusion
  tools.py       search_s1, get_s1_financial_table, calculator, web_search, fetch_webpage, get_market_data
  agent.py       chat agent: StateGraph(MessagesState) agent <-> ToolNode, InMemorySaver memory
  memo_graph.py  memo workflow: 5 RAG research nodes + web_news run in parallel -> write_memo -> scorecard
views/           the Streamlit pages
scripts/prepare_s1.py   re-downloads the filing from EDGAR and re-extracts the text
```

**RAG details.** The EDGAR HTML renders each printed page as its own block, so text is extracted page by
page. Each chunk keeps its printed page number and its prospectus section, which comes from the table of
contents. Retrieval fuses dense similarity with BM25 keyword scores. That matters for exact terms like
"Bartz settlement", "Class N" or "SAFEs". The agent can also limit a search to one section, such as
Risk Factors.

**Prompt templates.** These use `PromptTemplate` and `prompt | model | StrOutputParser()` chains, as in
`streamlit/chatbot_example.py`.

*For education only. This is not investment advice.*
