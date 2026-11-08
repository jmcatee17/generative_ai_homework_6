"""Structured financial data hand-extracted from Anthropic's S-1/A (Nov 2, 2026).

All dollar figures are in $ millions unless noted. Page numbers are the printed
prospectus page numbers so they can be checked against the filing.
"""
import pandas as pd

# --- Annual + nine-month statement of operations (Summary Consolidated Financial Data, p. 23)
ANNUAL_PERIODS = ["FY2023", "FY2024", "FY2025", "9M 2025", "9M 2026"]
ANNUAL = pd.DataFrame(
    {
        "Revenue": [41, 386, 4585, 2545, 33242],
        "Compute and infrastructure": [433, 2522, 7326, 4381, 18808],
        "Personnel": [95, 454, 1839, 1027, 4863],
        "Bartz settlement": [0, 0, 1500, 1500, 0],
        "Sales, marketing, and partnerships": [2, 43, 568, 322, 5529],
        "Other operating and administrative": [62, 347, 1412, 539, 5302],
        "Total operating expenses": [592, 3366, 12645, 7769, 34502],
        "Operating loss": [-551, -2980, -8060, -5224, -1260],
        "Interest income": [69, 106, 431, 239, 2359],
        "Change in FV of convertible notes, derivatives & SAFEs": [-1045, -5434, -34308, -20107, -94324],
        "Other income (expense), net": [-2, 0, -8, -3, -10397],
        "Provision for income taxes": [2, 1, 20, 12, 417],
        "Net loss": [-1531, -8309, -41965, -25107, -104039],
        "Stock-based compensation": [29, 192, 1165, 591, 3255],
        "Adjusted operating income (loss)": [-522, -2788, -4854, -3133, 4081],
        "Adjusted net income (loss)": [-457, -2683, -4455, -2911, 5591],
        "Operating margin %": [-1344, -772, -176, -205, -4],
        "Adjusted operating margin %": [-1273, -722, -106, -123, 12],
    },
    index=ANNUAL_PERIODS,
)

# --- Quarterly statement of operations (Quarterly Results of Operations, p. 160)
QUARTERS = ["Q1'25", "Q2'25", "Q3'25", "Q4'25", "Q1'26", "Q2'26", "Q3'26"]
QUARTERLY = pd.DataFrame(
    {
        "Revenue": [376, 787, 1382, 2040, 4731, 11555, 16956],
        "Compute and infrastructure": [944, 1390, 2047, 2945, 3387, 5558, 9863],
        "Personnel": [218, 320, 489, 812, 1295, 1857, 1711],
        "Bartz settlement": [0, 0, 1500, 0, 0, 0, 0],
        "Sales, marketing, and partnerships": [24, 96, 202, 246, 1033, 2135, 2361],
        "Other operating and administrative": [125, 181, 233, 873, 683, 1031, 3588],
        "Total operating expenses": [1311, 1987, 4471, 4876, 6398, 10581, 17523],
        "Operating income (loss)": [-935, -1200, -3089, -2836, -1667, 974, -567],
        "Interest income": [51, 64, 124, 192, 349, 762, 1248],
        "Change in FV of convertible notes, derivatives & SAFEs": [-5701, -5241, -9165, -14201, -12791, -59597, -21936],
        "Net loss": [-6592, -6377, -12138, -16858, -14439, -66309, -23291],
        "Stock-based compensation": [105, 181, 305, 574, 924, 1300, 1100],
        # Quarterly margins from the non-GAAP reconciliation (p. 163)
        "Operating margin %": [-249, -152, -224, -139, -35, 8, -3],
        "Adjusted operating margin %": [-221, -129, -93, -84, -13, 19, 15],
    },
    index=QUARTERS,
)

# --- Cash flows (p. 172)
CASH_FLOWS = pd.DataFrame(
    {
        "Operating activities": [-1627, -5692, -2637, -2024, 6569],
        "Investing activities": [-2, -22, -3077, -2007, -36546],
        "Financing activities": [2320, 7248, 19960, 18508, 110735],
    },
    index=ANNUAL_PERIODS,
)

# --- Balance sheet as of Sep 30, 2026 (p. 25)
BALANCE_SHEET = pd.Series(
    {
        "Cash, cash equivalents & short-term investments": 131845,
        "Working capital": 129349,
        "Total assets": 154320,
        "Total liabilities": 146683,
        "Redeemable convertible preferred stock": 157510,
        "Additional paid-in capital": 6481,
        "Accumulated deficit": -156193,
        "Total stockholders' deficit": -149873,
    },
    name="As of Sep 30, 2026",
)

# --- Customer tiers (TTM revenue, as of Sep 30, 2026; p. 136)
CUSTOMER_TIERS = pd.DataFrame(
    {
        "Tier": ["$100K+ TTM", "$1M+ TTM", "$10M+ TTM"],
        "Customers": [11900, 1900, 250],
    }
)
CUSTOMERS_100K_HISTORY = pd.DataFrame(
    {"Date": ["Mar 31, 2025", "Sep 30, 2025", "Sep 30, 2026"], "Customers ($100K+ TTM)": [400, 950, 11900]}
)

# --- Headline facts with citations, used by the dashboard and the agent's tool
KEY_FACTS = {
    "Q3 2026 revenue": ("$17.0B (three months ended Sep 30, 2026)", "p. 5, p. 160"),
    "Net Revenue Retention": ("~900% as of Sep 30, 2026", "p. 136"),
    "Customers with $100K+ TTM revenue": ("11,900+ (vs. 950+ a year earlier)", "p. 136"),
    "Customers with $1M+ TTM revenue": ("1,900+", "p. 136"),
    "Customers with $10M+ TTM revenue": ("~250", "p. 136"),
    "Customer concentration": ("No single customer >10% of 9M 2026 revenue", "p. 144"),
    "Compute purchase commitments": ("~$895.4B aggregate committed spend, substantially all compute/infrastructure", "p. 7"),
    "Q3'25→Q3'26 price reductions": ("~$600M revenue offset in Q3'26 from price cuts/discounts", "p. 161"),
    "Unrecognized SBC (RSUs)": ("$24.1B over ~2.0 years, plus $2.1B other awards", "p. 24"),
    "Listing": ("Class A common stock, Nasdaq Global Select Market, ticker 'ANT'", "cover"),
    "Corporate form": ("Delaware public benefit corporation with a Long-Term Benefit Trust", "p. 246"),
}

# Bridge from operating loss to net loss, 9M 2026 (p. 23) — shows how much is non-cash
LOSS_BRIDGE_9M26 = [
    ("Operating loss", -1260),
    ("Interest income", 2359),
    ("FV change: notes, derivatives, SAFEs", -94324),
    ("Other income (expense), net", -10397),
    ("Income taxes", -417),
]


TABLES = {
    "annual": ANNUAL,
    "quarterly": QUARTERLY,
    "cash_flows": CASH_FLOWS,
    "balance_sheet": BALANCE_SHEET.to_frame(),
    "customers": CUSTOMER_TIERS.set_index("Tier"),
}


TABLE_CITES = {
    "annual": "[S-1 p. 23 · Summary Financial Data] (non-GAAP: [S-1 p. 25])",
    "quarterly": "[S-1 p. 160 · MD&A] (quarterly margins: [S-1 p. 163 · MD&A])",
    "cash_flows": "[S-1 p. 172 · MD&A]",
    "balance_sheet": "[S-1 p. 25 · Summary Financial Data]",
    "customers": "[S-1 p. 136 · MD&A]",
}


def table_as_text(name: str) -> str:
    """CSV-ish text of a table for the LLM tool."""
    if name == "key_facts":
        return "\n".join(f"- {k}: {v} (S-1 {p})" for k, (v, p) in KEY_FACTS.items())
    df = TABLES[name]
    return f"Source: {TABLE_CITES[name]}. Cite it like that. (in $ millions unless noted)\n{df.to_csv()}"
