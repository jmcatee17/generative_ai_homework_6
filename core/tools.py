"""LangChain tools the LangGraph agents can call."""
from __future__ import annotations

import ast
import operator as op

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from core import s1_data
from core.rag import SECTIONS, S1Index, format_hits

_INDEX: S1Index | None = None
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (S-1 research app for a class assignment)"}

# Peers the agent can pull live market data for
COMPS = {
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "PLTR": "Palantir",
    "CRM": "Salesforce",
    "SNOW": "Snowflake",
}


def set_index(index: S1Index) -> None:
    global _INDEX
    _INDEX = index


# ---------------------------------------------------------------- S-1 tools
@tool
def search_s1(query: str, section: str | None = None, k: int = 6) -> str:
    """Semantic + keyword search over Anthropic's S-1 prospectus.

    Use this for ANY question about the filing: business, risks, governance,
    compute deals, compensation, capital structure, offering terms, etc.
    Optionally restrict to one `section`, e.g. "Risk Factors", "MD&A", "Business",
    "Governance Structure", "Executive Compensation", "Description of Capital Stock",
    "Financial Statements". Returns passages with page citations.
    """
    if _INDEX is None:
        return "The S-1 index is not loaded."
    hits = _INDEX.search(query, k=max(1, min(k, 10)), section=section if section in SECTIONS else None)
    return format_hits(hits)


@tool
def get_s1_financial_table(table: str) -> str:
    """Exact figures from the S-1's financial tables (in $ millions).

    `table` is one of: "annual" (FY2023-FY2025 and 9M 2025/2026 P&L, margins,
    adjusted metrics), "quarterly" (Q1'25-Q3'26 P&L), "cash_flows",
    "balance_sheet" (Sep 30, 2026), "customers" (customer counts by TTM revenue tier),
    "key_facts" (NRR, compute commitments, listing details, etc.).
    Prefer this over search_s1 when you need precise numbers.
    """
    if table not in {*s1_data.TABLES, "key_facts"}:
        return f"Unknown table. Choose from {sorted([*s1_data.TABLES, 'key_facts'])}."
    return s1_data.table_as_text(table)


# ---------------------------------------------------------------- calculator
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos, ast.Mod: op.mod,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only arithmetic is allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "(16956 - 1382) / 1382 * 100".
    Use it for growth rates, margins, multiples — never do math in your head."""
    try:
        return str(round(_eval(ast.parse(expression, mode="eval").body), 6))
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


# ---------------------------------------------------------------- web tools
@tool
def web_search(query: str, news: bool = True, max_results: int = 6) -> str:
    """Search the live web (DuckDuckGo). Set news=True for recent news articles
    (IPO pricing, analyst commentary, market reaction) or news=False for general pages."""
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            results = (ddgs.news if news else ddgs.text)(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001
        return f"Web search failed: {e}"
    if not results:
        return "No results."
    lines = []
    for r in results:
        url = r.get("url") or r.get("href")
        meta = " · ".join(x for x in [r.get("source"), (r.get("date") or "")[:10]] if x)
        lines.append(f"- {r.get('title')} ({meta})\n  {url}\n  {r.get('body', '')[:300]}")
    return "\n".join(lines)


@tool
def fetch_webpage(url: str) -> str:
    """Download a web page and return its main text (first ~6,000 characters).
    Use after web_search when a snippet isn't enough."""
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"Could not fetch page: {e}"
    soup = BeautifulSoup(resp.text, "lxml")
    for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
        t.decompose()
    text = " ".join((soup.find("article") or soup.body or soup).get_text(" ").split())
    return text[:6000] or "Page had no readable text."


def market_snapshot(tickers: list[str]) -> list[dict]:
    """Live quote + valuation fields from Yahoo Finance (used by the tool and the UI)."""
    import yfinance as yf

    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "Ticker": t,
                "Company": info.get("shortName", COMPS.get(t, t)),
                "Price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "Market cap ($B)": (info.get("marketCap") or 0) / 1e9,
                "TTM revenue ($B)": (info.get("totalRevenue") or 0) / 1e9,
                "Revenue growth (YoY)": info.get("revenueGrowth"),
                "Price / Sales": info.get("priceToSalesTrailing12Months"),
                "Operating margin": info.get("operatingMargins"),
            }
        )
    return rows


@tool
def get_market_data(tickers: list[str]) -> str:
    """Live stock data for public comparables (price, market cap, TTM revenue,
    growth, price/sales, operating margin). Good peers: NVDA, MSFT, GOOGL, AMZN,
    META, PLTR, CRM, SNOW. Use to put Anthropic's S-1 numbers in market context."""
    rows = market_snapshot([t.upper() for t in tickers][:8])
    if not rows:
        return "No market data returned."
    pct_fields = {"Revenue growth (YoY)", "Operating margin"}  # yfinance returns these as fractions

    def fmt(k, v):
        if k in pct_fields and isinstance(v, (int, float)):
            return f"{k}: {v * 100:.1f}%"
        if k == "Price / Sales" and isinstance(v, (int, float)):
            return f"{k}: {v:.1f}x"
        return f"{k}: {v:,.2f}" if isinstance(v, float) else f"{k}: {v}"

    return "\n".join(", ".join(fmt(k, v) for k, v in r.items() if v is not None) for r in rows)


ALL_TOOLS = [search_s1, get_s1_financial_table, calculator, web_search, fetch_webpage, get_market_data]
S1_ONLY_TOOLS = [search_s1, get_s1_financial_table, calculator]
