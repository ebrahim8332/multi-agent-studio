"""
Agent functions for the Stock Analyser module (M02).

Nine agents:
  1. Resolver              — no LLM, validates the ticker via yfinance
  2. Data Agent             — no LLM, pulls yfinance + Tavily data, runs the
                               completeness gate, computes the data quality score
  3. Fundamentals Analyst   — LLM, revenue/margin/valuation trends
  4. Business Quality Analyst — LLM, moat/brand via Exa qualitative search
  5. Risk Analyst           — LLM, five risk categories from tiered news + macro
  6. Fact Checker           — LLM extracts claims, Python verifies them against
                               data_bundle — not another model's opinion
  7. Bull Advocate          — LLM, strongest case to own the stock
  8. Bear Advocate          — LLM, strongest case against owning it
  9. Synthesizer            — LLM, weighs Bull vs Bear, issues rating + confidence

Agents 1 and 2 have no LLM calls — they are pure data lookups and never
fabricate a number. If yfinance is missing a required field, Agent 2 halts
the pipeline before any LLM agent runs (see run_data_agent / REQUIRED_FIELDS).

Agent 6 (Fact Checker) exists because of a specific failure mode: Fundamentals
and Risk Analyst are the two agents most likely to quote a specific number
badly (transposed digit, wrong direction, wrong period), and any such error
propagates uncorrected into Bull, Bear, and the Synthesizer, which all treat
the analyst text as ground truth. Unlike a typical LLM fact-checker (which
checks claims against fuzzy search snippets), this one checks claims against
data_bundle directly — the same numbers already fetched from yfinance — so
verification is an exact numeric comparison in Python, not another model's
guess about whether something sounds right.

A note on sector names: yfinance's `.info["sector"]` uses Yahoo Finance's own
taxonomy, not GICS. Confirmed against live tickers before writing this file:
  JPM -> "Financial Services"   (not "Financials")
  AMZN -> "Consumer Cyclical"   (not "Consumer Discretionary")
  PG -> "Consumer Defensive"    (not "Consumer Staples")
  LIN -> "Basic Materials"      (not "Materials")
SECTOR_PEERS and SECTOR_MACRO_QUERIES below are keyed on yfinance's actual
strings so lookups work. The other seven sectors (Technology, Healthcare,
Energy, Industrials, Utilities, Communication Services, Real Estate) match
GICS naming exactly and needed no change.
"""

import math
import json
from datetime import datetime
from urllib.parse import urlparse

import yfinance as yf

from utils.search_client import search_tavily_only, search_exa_only

# ── Sector lookup tables ──────────────────────────────────────────────────────

# 2-5 large, liquid peers per sector. Sector-level, not industry-level — a
# simple hardcoded pool as the spec calls for, not a perfect industry match.
SECTOR_PEERS = {
    "Technology":              ["AAPL", "MSFT", "GOOGL", "NVDA", "ORCL"],
    "Financial Services":      ["JPM", "BAC", "WFC", "GS", "MS"],
    "Healthcare":              ["UNH", "JNJ", "PFE", "ABBV", "MRK"],
    "Energy":                  ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Consumer Cyclical":       ["AMZN", "HD", "MCD", "NKE", "SBUX"],
    "Consumer Defensive":      ["PG", "KO", "PEP", "WMT", "COST"],
    "Industrials":             ["CAT", "BA", "HON", "GE", "UPS"],
    "Utilities":               ["NEE", "DUK", "SO", "D", "AEP"],
    "Basic Materials":         ["LIN", "SHW", "APD", "FCX", "NEM"],
    "Communication Services":  ["META", "GOOGL", "T", "VZ", "DIS"],
    "Real Estate":             ["PLD", "AMT", "EQIX", "SPG", "O"],
}

# Tier 1 = regulatory filings + major financial press. Tier 2 = industry and
# business press. Anything not listed defaults to Tier 3 (blogs, press
# releases, aggregators) in _news_tier() below.
NEWS_SOURCE_TIERS = {
    "sec.gov": 1, "reuters.com": 1, "bloomberg.com": 1, "ft.com": 1,
    "wsj.com": 1, "apnews.com": 1, "finance.yahoo.com": 1,
    "cnbc.com": 2, "marketwatch.com": 2, "barrons.com": 2,
    "investopedia.com": 2, "seekingalpha.com": 2, "morningstar.com": 2,
}

SECTOR_MACRO_QUERIES = {
    "Technology":             "AI spending enterprise software market outlook 2026",
    "Financial Services":     "interest rate outlook banking sector 2026",
    "Healthcare":              "FDA approvals drug pricing regulatory outlook 2026",
    "Energy":                  "oil price outlook energy transition investment 2026",
    "Consumer Cyclical":       "consumer spending outlook retail 2026",
    "Consumer Defensive":      "consumer staples inflation pricing power 2026",
    "Industrials":             "manufacturing capex infrastructure spending 2026",
    "Utilities":               "utility regulation rate case energy grid investment 2026",
    "Basic Materials":         "commodity prices materials supply chain 2026",
    "Communication Services":  "digital advertising streaming competition 2026",
    "Real Estate":             "commercial real estate interest rates REIT outlook 2026",
}

TREND_YEARS = 5
MAX_EARNINGS_QUARTERS = 8
MAX_PEERS = 3

# max_tokens on every chain.complete() call below is chosen against two
# constraints measured directly against the live APIs, not guessed:
#   1. Realistic content size. Each analyst's output (headed sections or a
#      bounded Bull/Bear format) runs roughly 900-2500 tokens in practice.
#      These ceilings give 2-4x headroom above that so depth instructions in
#      the prompts aren't quietly cut off mid-sentence.
#   2. Groq's free-tier per-model tokens-per-minute limit. Tested directly:
#      qwen/qwen3-32b (the tightest model in the chain) starts returning
#      HTTP 413 "Request too large... TPM" around 6000-7000 tokens
#      (input + max_tokens combined) — and that budget is shared across
#      whatever else lands on the same model in the same minute, which
#      matters here because 3 agents (or 2) can all fall through to Groq
#      at once in this module's fan-out pattern. 4000 stayed clean across
#      all 5 Groq tiers in that test; going materially higher risks a
#      fallback that itself immediately errors instead of just running
#      short. The Synthesizer gets more headroom (6000) than the free-text
#      analysts because its output is schema-enforced JSON — truncation
#      there doesn't just cut a sentence short, it breaks json.loads()
#      entirely and the whole call is wasted.

# Writing rules injected into every prompt that produces reader-facing text.
STYLE_RULES = """
Writing rules — follow exactly:
- Short sentences. One idea per sentence. Under 20 words.
- Business formal. Direct. No hedging.
- No em dashes. Use a comma, colon, or period instead.
- No banned words: leverage, seamlessly, transformative, delve, empower, foster,
  ecosystem, paramount, unlock, thought leadership, actionable insights, cutting-edge,
  unparalleled, "it is worth noting", "in today's rapidly evolving landscape".
- Do not open with broad scene-setting. Get to the point in the first sentence.
- No motivational closing paragraph. End when the content is done.
"""


def _is_missing(value) -> bool:
    """True for None and for NaN (yfinance/pandas return NaN for unavailable fields)."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value):
    """Returns a plain float, or None if the value is missing/unusable."""
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _news_tier(url: str) -> int:
    """Tags a news URL Tier 1/2/3 using NEWS_SOURCE_TIERS. Default Tier 3."""
    try:
        domain = urlparse(url).netloc.replace("www.", "")
    except Exception:
        domain = ""
    return NEWS_SOURCE_TIERS.get(domain, 3)


# ── Agent 1: Resolver ─────────────────────────────────────────────────────────

def run_resolver(raw_input: str) -> dict:
    """
    Validates a ticker or company name against yfinance. No LLM — this is a
    direct lookup. Tries the input as typed first; if that fails and it looks
    like it could be a non-US ticker, retries once with common suffixes.

    Returns:
        {"halted": False, "ticker": str, "company_name": str}
        or
        {"halted": True, "error": str}
    """
    candidate = raw_input.strip()
    if not candidate:
        return {"halted": True, "error": "Enter a ticker or company name to analyse."}

    def _try(symbol: str):
        try:
            info = yf.Ticker(symbol).info
        except Exception:
            return None
        if not info or not (info.get("longName") or info.get("shortName")):
            return None
        return info

    info = _try(candidate)
    resolved_symbol = candidate.upper()

    if info is None:
        # One retry with common non-US suffixes.
        for suffix in (".TO", ".L", ".HK", ".AX"):
            trial_symbol = candidate.upper() + suffix
            info = _try(trial_symbol)
            if info is not None:
                resolved_symbol = trial_symbol
                break

    if info is None:
        return {
            "halted": True,
            "error": f"Could not resolve '{raw_input}' to a valid ticker. Check the symbol and try again.",
        }

    ticker = info.get("symbol") or resolved_symbol
    company_name = info.get("longName") or info.get("shortName") or ticker
    return {"halted": False, "ticker": ticker, "company_name": company_name}


# ── Agent 2: Data Agent ───────────────────────────────────────────────────────

def _fetch_trend_data(t: "yf.Ticker") -> list[dict]:
    """
    Pulls up to TREND_YEARS annual periods from .financials / .cashflow /
    .balance_sheet and computes revenue, margins, FCF, ROE, and debt-to-equity
    per year. Returns oldest-to-newest, skipping any year missing revenue.
    """
    try:
        fin = t.financials
        cf = t.cashflow
        bs = t.balance_sheet
    except Exception:
        return []
    if fin is None or fin.empty:
        return []

    trend = []
    for col in list(fin.columns)[:TREND_YEARS]:
        revenue = fin.loc["Total Revenue", col] if "Total Revenue" in fin.index else None
        if _is_missing(revenue):
            continue  # e.g. the trailing NaN column yfinance sometimes includes

        gross_profit = fin.loc["Gross Profit", col] if "Gross Profit" in fin.index else None
        operating_income = fin.loc["Operating Income", col] if "Operating Income" in fin.index else None
        net_income = fin.loc["Net Income", col] if "Net Income" in fin.index else None

        fcf = None
        if cf is not None and col in cf.columns and "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow", col]

        total_debt = None
        equity = None
        if bs is not None and col in bs.columns:
            if "Total Debt" in bs.index:
                total_debt = bs.loc["Total Debt", col]
            if "Stockholders Equity" in bs.index:
                equity = bs.loc["Stockholders Equity", col]

        revenue_f = _safe_float(revenue)
        gross_profit_f = _safe_float(gross_profit)
        operating_income_f = _safe_float(operating_income)
        net_income_f = _safe_float(net_income)
        equity_f = _safe_float(equity)
        total_debt_f = _safe_float(total_debt)

        year_label = col.strftime("FY%Y") if hasattr(col, "strftime") else str(col)
        trend.append({
            "year": year_label,
            "revenue": revenue_f,
            "gross_margin": (gross_profit_f / revenue_f) if gross_profit_f is not None and revenue_f else None,
            "operating_margin": (operating_income_f / revenue_f) if operating_income_f is not None and revenue_f else None,
            "net_income": net_income_f,
            "fcf": _safe_float(fcf),
            "roe": (net_income_f / equity_f) if net_income_f is not None and equity_f else None,
            "debt_to_equity": (total_debt_f / equity_f) if total_debt_f is not None and equity_f else None,
        })

    trend.reverse()
    return trend


def _fetch_earnings_history(t: "yf.Ticker") -> tuple[list[dict], str | None]:
    """
    Pulls up to MAX_EARNINGS_QUARTERS reported quarters from .earnings_dates.
    Returns (history, most_recent_report_date_iso). yfinance's quarterly
    history only carries EPS estimate/actual/surprise — no revenue surprise
    series is available, so that field is intentionally left out rather than
    invented. Returns oldest-to-newest.
    """
    try:
        ed = t.earnings_dates
    except Exception:
        return [], None
    if ed is None or ed.empty:
        return [], None

    history = []
    most_recent_date = None
    for idx, row in ed.iterrows():
        actual = row.get("Reported EPS")
        if _is_missing(actual):
            continue  # future / not-yet-reported quarter
        estimate = row.get("EPS Estimate")
        surprise_pct = row.get("Surprise(%)")

        quarter_label = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        if most_recent_date is None:
            most_recent_date = quarter_label

        history.append({
            "quarter": quarter_label,
            "eps_estimate": _safe_float(estimate),
            "eps_actual": _safe_float(actual),
            "eps_surprise_pct": _safe_float(surprise_pct),
        })
        if len(history) >= MAX_EARNINGS_QUARTERS:
            break

    history.reverse()
    return history, most_recent_date


def _fetch_peer_data(sector: str, ticker: str) -> list[dict]:
    """
    Fetches up to MAX_PEERS peers from SECTOR_PEERS, skipping the subject
    ticker and any candidate too incomplete to be useful for comparison.
    """
    candidates = [p for p in SECTOR_PEERS.get(sector, []) if p.upper() != ticker.upper()]
    peers = []
    for candidate in candidates:
        if len(peers) >= MAX_PEERS:
            break
        try:
            info = yf.Ticker(candidate).info
        except Exception:
            continue
        pe = info.get("trailingPE") if not _is_missing(info.get("trailingPE")) else info.get("forwardPE")
        gross_margin = info.get("grossMargins")
        if _is_missing(pe) and _is_missing(gross_margin):
            continue  # too incomplete to be a useful comparison row
        peers.append({
            "ticker": candidate,
            "pe": _safe_float(pe),
            "gross_margin": _safe_float(gross_margin),
            "operating_margin": _safe_float(info.get("operatingMargins")),
            "revenue_growth": _safe_float(info.get("revenueGrowth")),
            "market_cap": _safe_float(info.get("marketCap")),
        })
    return peers


def _fetch_analyst_distribution(t: "yf.Ticker") -> dict | None:
    """
    Returns the most recent Buy/Hold/Sell analyst distribution from
    .recommendations, or None if that data is unavailable.
    """
    try:
        rec_df = t.recommendations
    except Exception:
        return None
    if rec_df is None or rec_df.empty:
        return None
    row = rec_df.iloc[0]
    return {
        "strong_buy": int(row.get("strongBuy", 0) or 0),
        "buy": int(row.get("buy", 0) or 0),
        "hold": int(row.get("hold", 0) or 0),
        "sell": int(row.get("sell", 0) or 0),
        "strong_sell": int(row.get("strongSell", 0) or 0),
    }


def _compute_data_quality_score(fields_found: int, fields_required: int, analyst_count: int,
                                 filing_age_days: int | None, news_count_30d: int,
                                 peer_count: int) -> tuple[int, str, list[str]]:
    """
    0-100 score per the Data Quality Score table in the spec:
    required fields (40) + analyst coverage (20) + filing age (20)
    + news freshness (10) + peer availability (10).

    Also returns a `breakdown` list of plain-English notes on which
    specific components fell short of full marks. Without this, the
    Synthesizer only ever sees the final number — it can say "data
    quality is low" but never say *why*. The breakdown lets its
    confidence_explanation cite the actual driver (e.g. "only 2 analysts
    cover this stock") instead of a generic restatement of the score.
    """
    score = 40 * (fields_found / fields_required) if fields_required else 0
    breakdown = []

    if analyst_count >= 10:
        score += 20
    elif analyst_count >= 3:
        score += 10
        breakdown.append(f"analyst coverage is moderate ({analyst_count} analysts, not yet 10+)")
    else:
        breakdown.append(f"analyst coverage is thin ({analyst_count} analyst(s), fewer than 3)")

    if filing_age_days is not None:
        if filing_age_days < 90:
            score += 20
        elif filing_age_days <= 180:
            score += 10
            breakdown.append(f"most recent earnings report is {filing_age_days} days old (90-180 day range)")
        else:
            breakdown.append(f"most recent earnings report is {filing_age_days} days old (over 180 days)")
    else:
        breakdown.append("no earnings report date available to assess data freshness")

    if news_count_30d >= 5:
        score += 10
    elif news_count_30d >= 1:
        score += 5
        breakdown.append(f"only {news_count_30d} recent news article(s) found (fewer than 5)")
    else:
        breakdown.append("no recent news articles found in the last 30 days")

    if peer_count >= 2:
        score += 10
    elif peer_count == 1:
        score += 5
        breakdown.append("only 1 peer found for comparison (fewer than 2)")
    else:
        breakdown.append("no peer data available for comparison")

    score = round(score)
    if score >= 80:
        label = "High"
    elif score >= 60:
        label = "Adequate"
    else:
        label = "Thin"
    return score, label, breakdown


def run_data_agent(ticker: str, company_name: str) -> dict:
    """
    Pulls every required, supplementary, trend, and earnings-history field
    from yfinance, plus news/catalysts/macro context from Tavily. Runs the
    completeness gate before any LLM agent is allowed to run.

    Returns either:
        {"halted": True, "error": str}
    or a fully populated data_bundle dict with "halted": False.
    """
    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception:
        info = {}

    if not info:
        return {"halted": True, "error": f"yfinance returned no data for '{ticker}'. Try a different ticker."}

    sector = info.get("sector")
    industry = info.get("industry")
    current_price = info.get("currentPrice") if not _is_missing(info.get("currentPrice")) else info.get("regularMarketPrice")
    market_cap = info.get("marketCap")

    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    pe_used, pe_value = ("trailing", trailing_pe) if not _is_missing(trailing_pe) else ("forward", forward_pe)

    analyst_dist = _fetch_analyst_distribution(t)
    consensus_key = info.get("recommendationKey")
    analyst_count = info.get("numberOfAnalystOpinions")
    if _is_missing(analyst_count) and analyst_dist:
        analyst_count = sum(analyst_dist.values())
    analyst_count = int(analyst_count) if not _is_missing(analyst_count) else 0

    # ── Trend data, fetched early ────────────────────────────────────────────
    # Needed before the required-fields check below because debt_to_equity is
    # sourced from it, not from yfinance's info["debtToEquity"] — see note.
    trend_data = _fetch_trend_data(t)

    # yfinance's info["debtToEquity"] uses an internal Yahoo scaling that does
    # not match the standard ratio convention (confirmed directly against the
    # API: AAPL returns 79.5 from this field, while Total Debt / Stockholders
    # Equity from the balance sheet is 1.34 -- a >50x difference, not rounding
    # noise). Preferring the balance-sheet-computed ratio here keeps this one
    # number internally consistent with the identical calculation already
    # used for every year in trend_data, so the model is never shown two
    # different "debt-to-equity" figures for the same company in the same
    # prompt. Found by the Fact Checker flagging a "wrong" analyst claim that
    # was actually correct against the trend table and only wrong against
    # this raw field -- falls back to the raw field only if the balance
    # sheet computation is unavailable.
    debt_to_equity = trend_data[-1]["debt_to_equity"] if trend_data else None
    if debt_to_equity is None:
        debt_to_equity = _safe_float(info.get("debtToEquity"))

    # ── Required fields — first missing one halts the pipeline ──────────────
    required_checks = [
        ("company_name", "Company name", company_name),
        ("sector", "Sector", sector),
        ("industry", "Industry", industry),
        ("current_price", "Current price", current_price),
        ("market_cap", "Market cap", market_cap),
        ("pe_ratio", "Trailing or forward P/E ratio", pe_value),
        ("revenue_ttm", "Revenue (trailing twelve months)", info.get("totalRevenue")),
        ("revenue_growth", "Revenue growth year-over-year", info.get("revenueGrowth")),
        ("gross_margin", "Gross margin", info.get("grossMargins")),
        ("operating_margin", "Operating margin", info.get("operatingMargins")),
        ("net_margin", "Net margin", info.get("profitMargins")),
        ("free_cash_flow", "Free cash flow", info.get("freeCashflow")),
        ("debt_to_equity", "Debt-to-equity ratio", debt_to_equity),
        ("return_on_equity", "Return on equity", info.get("returnOnEquity")),
        ("fifty_two_week_high", "52-week high", info.get("fiftyTwoWeekHigh")),
        ("fifty_two_week_low", "52-week low", info.get("fiftyTwoWeekLow")),
        ("analyst_consensus", "Analyst consensus rating", analyst_dist or consensus_key),
        ("analyst_mean_target", "Analyst mean price target", info.get("targetMeanPrice")),
    ]

    fields_found = 0
    for key, label, value in required_checks:
        if _is_missing(value):
            return {
                "halted": True,
                "error": (
                    f"Data incomplete — cannot proceed.\n\n"
                    f"Missing: {label}\n\n"
                    "yfinance does not have sufficient data for this ticker. "
                    "This can occur for small-cap stocks, OTC securities, "
                    "recently listed companies, or non-US equities.\n\n"
                    "Try a different ticker or check that the symbol is correct."
                ),
            }
        fields_found += 1

    thin_coverage = analyst_count < 3

    # ── Supplementary fields (absence noted, never halts) ───────────────────
    supplementary = {
        "eps_trailing": _safe_float(info.get("trailingEps")),
        "price_to_book": _safe_float(info.get("priceToBook")),
        "dividend_yield": _safe_float(info.get("dividendYield")),
        "beta": _safe_float(info.get("beta")),
        "short_percent_of_float": _safe_float(info.get("shortPercentOfFloat")),
        "institutional_ownership_pct": _safe_float(info.get("heldPercentInstitutions")),
    }

    try:
        insider_df = t.insider_transactions
        insider_transactions = insider_df.head(10).to_dict("records") if insider_df is not None else []
    except Exception:
        insider_transactions = []

    try:
        holders_df = t.institutional_holders
        institutional_holders = holders_df.head(5).to_dict("records") if holders_df is not None else []
    except Exception:
        institutional_holders = []

    # ── Earnings history (trend_data was already fetched above) ─────────────
    earnings_history, most_recent_earnings_date = _fetch_earnings_history(t)

    filing_age_days = None
    if most_recent_earnings_date:
        try:
            filing_age_days = (datetime.now().date() - datetime.fromisoformat(most_recent_earnings_date).date()).days
        except Exception:
            filing_age_days = None

    # ── Peers ─────────────────────────────────────────────────────────────────
    peers = _fetch_peer_data(sector, ticker)

    # ── News, catalysts, macro (Tavily) ──────────────────────────────────────
    raw_news = search_tavily_only(f"{company_name} stock news", max_results=10, days=30)
    news_items = [
        {**item, "tier": _news_tier(item.get("url", ""))}
        for item in raw_news
    ]

    catalyst_items = search_tavily_only(
        f"{company_name} earnings date upcoming catalyst product launch investor day",
        max_results=5, days=90,
    )

    macro_context = None
    macro_query = SECTOR_MACRO_QUERIES.get(sector)
    if macro_query:
        macro_hits = search_tavily_only(macro_query, max_results=1)
        if macro_hits:
            macro_context = macro_hits[0].get("content", "")[:500]

    data_quality_score, data_quality_label, data_quality_breakdown = _compute_data_quality_score(
        fields_found=fields_found,
        fields_required=len(required_checks),
        analyst_count=analyst_count,
        filing_age_days=filing_age_days,
        news_count_30d=len(news_items),
        peer_count=len(peers),
    )

    return {
        "halted": False,
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "current_price": _safe_float(current_price),
        "market_cap": _safe_float(market_cap),
        "pe_used": pe_used,
        "pe_value": _safe_float(pe_value),
        "revenue_ttm": _safe_float(info.get("totalRevenue")),
        "revenue_growth": _safe_float(info.get("revenueGrowth")),
        "gross_margin": _safe_float(info.get("grossMargins")),
        "operating_margin": _safe_float(info.get("operatingMargins")),
        "net_margin": _safe_float(info.get("profitMargins")),
        "free_cash_flow": _safe_float(info.get("freeCashflow")),
        "debt_to_equity": debt_to_equity,
        "return_on_equity": _safe_float(info.get("returnOnEquity")),
        "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
        "analyst_distribution": analyst_dist,
        "analyst_consensus_key": consensus_key,
        "analyst_count": analyst_count,
        "thin_analyst_coverage": thin_coverage,
        "analyst_mean_target": _safe_float(info.get("targetMeanPrice")),
        "supplementary": supplementary,
        "insider_transactions": insider_transactions,
        "institutional_holders": institutional_holders,
        "trend_data": trend_data,
        "earnings_history": earnings_history,
        "peers": peers,
        "news_items": news_items,
        "catalyst_items": catalyst_items,
        "macro_context": macro_context,
        "data_quality_score": data_quality_score,
        "data_quality_label": data_quality_label,
        "data_quality_breakdown": data_quality_breakdown,
    }


# ── Shared formatting helpers for LLM prompts ────────────────────────────────

def _format_peer_table(peers: list[dict]) -> str:
    if not peers:
        return "No peer data available."
    lines = ["Ticker | P/E | Gross Margin | Operating Margin | Revenue Growth | Market Cap"]
    for p in peers:
        pe = f"{p['pe']:.1f}" if p.get("pe") is not None else "n/a"
        gm = f"{p['gross_margin']*100:.1f}%" if p.get("gross_margin") is not None else "n/a"
        om = f"{p['operating_margin']*100:.1f}%" if p.get("operating_margin") is not None else "n/a"
        rg = f"{p['revenue_growth']*100:.1f}%" if p.get("revenue_growth") is not None else "n/a"
        mc = f"${p['market_cap']/1e9:.1f}B" if p.get("market_cap") is not None else "n/a"
        lines.append(f"{p['ticker']} | {pe} | {gm} | {om} | {rg} | {mc}")
    return "\n".join(lines)


def _format_trend_data(trend: list[dict]) -> str:
    if not trend:
        return "Insufficient trend data — fewer than 2 years available."
    lines = []
    for row in trend:
        rev = f"${row['revenue']/1e9:.2f}B" if row.get("revenue") is not None else "n/a"
        gm = f"{row['gross_margin']*100:.1f}%" if row.get("gross_margin") is not None else "n/a"
        om = f"{row['operating_margin']*100:.1f}%" if row.get("operating_margin") is not None else "n/a"
        fcf = f"${row['fcf']/1e9:.2f}B" if row.get("fcf") is not None else "n/a"
        roe = f"{row['roe']*100:.1f}%" if row.get("roe") is not None else "n/a"
        dte = f"{row['debt_to_equity']:.2f}" if row.get("debt_to_equity") is not None else "n/a"
        lines.append(f"{row['year']}: revenue {rev}, gross margin {gm}, operating margin {om}, FCF {fcf}, ROE {roe}, debt-to-equity {dte}")
    return "\n".join(lines)


def _format_earnings_history(history: list[dict]) -> str:
    if not history:
        return "No earnings history available."
    lines = []
    for row in history:
        est = f"{row['eps_estimate']:.2f}" if row.get("eps_estimate") is not None else "n/a"
        act = f"{row['eps_actual']:.2f}" if row.get("eps_actual") is not None else "n/a"
        surprise = f"{row['eps_surprise_pct']:+.1f}%" if row.get("eps_surprise_pct") is not None else "n/a"
        lines.append(f"{row['quarter']}: EPS estimate {est}, actual {act}, surprise {surprise}")
    lines.append("(Revenue estimate-vs-actual is not available from yfinance's quarterly history — EPS only.)")
    return "\n".join(lines)


def _format_news_items(news_items: list[dict]) -> str:
    if not news_items:
        return "No recent news found."
    lines = []
    for item in news_items:
        lines.append(f"[Tier {item.get('tier', 3)}] {item.get('title', '')}: {item.get('content', '')[:550]}")
    return "\n".join(lines)


def _format_catalyst_items(catalyst_items: list[dict]) -> str:
    if not catalyst_items:
        return "No upcoming catalysts identified."
    return "\n".join(f"- {item.get('title', '')}: {item.get('content', '')[:400]}" for item in catalyst_items)


def _word_count_check(text: str, min_words: int) -> tuple[int, bool]:
    """
    Objective word count, not a model's self-report — same lesson Module 1
    learned: LLMs cannot reliably judge their own output length, so Python
    counts instead of trusting the model. Returns (word_count, thin) where
    thin=True means the output fell below the expected floor for its
    format and should be flagged in the UI rather than silently accepted
    as a complete, full-quality analysis.
    """
    count = len(text.split())
    return count, count < min_words


# ── Agent 3: Fundamentals Analyst ─────────────────────────────────────────────

def run_fundamentals_analyst(state: dict, chain) -> dict:
    """
    Analyses revenue growth, profitability trends, margins, earnings quality,
    and peer valuation. No qualitative claims beyond what the data shows.
    Returns: fundamentals_analysis (str), model_used (str), prompt_sent (list)
    """
    db = state["data_bundle"]
    time_horizon = state.get("time_horizon", "Medium-term (1-3 years)")
    supp = db.get("supplementary", {})
    eps_trailing = supp.get("eps_trailing")
    price_to_book = supp.get("price_to_book")
    dividend_yield = supp.get("dividend_yield")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fundamentals analyst. Analyse only what the data shows. "
                "Do not infer what is not present. For each trend metric (revenue, margins, "
                "FCF, ROE), describe the direction over the available periods: improving, "
                "stable, or deteriorating. Do not just report point-in-time numbers. "
                "For earnings history, note the pattern of beats or misses over the last "
                "4-8 quarters. A consistent beat pattern is a positive signal; consistent "
                "misses are a negative signal. State this explicitly. "
                "State when a metric is above, at, or below sector peers — do not just "
                "report the number. Flag any metric that is deteriorating. Where trailing "
                "EPS or price-to-book is provided, use it to sharpen the Valuation vs Peers "
                "section, not just P/E. Where the current price sits close to the 52-week "
                "high or low, say so explicitly and what it implies for this time horizon. "
                "Where a dividend yield is provided, note whether it is meaningful for this "
                "stock's total-return case or negligible.\n\n"
                "Time horizon affects framing: short-term weights recent momentum and "
                "earnings surprises; long-term weights structural margin trends and FCF "
                "trajectory.\n\n"
                "Output format: structured findings under exactly five headings: "
                "Revenue Trend, Profitability Trend, Earnings Quality, Valuation vs Peers, "
                "Red Flags (write 'None identified' if there are none). "
                "Length is a hard floor, not a target: each heading needs a minimum of 80 "
                "words of substantive analysis grounded in specific numbers from the data "
                "below. Go deeper than the floor whenever the data supports it — do not stop "
                "at the minimum by default. A one-line heading is a failure, not an answer. "
                "Before finishing, check every heading against this floor and expand any "
                "that fall short. Plain English. No jargon.\n\n"
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {db['company_name']} ({db['ticker']}) — {db['sector']} / {db['industry']}\n"
                f"Time horizon: {time_horizon}\n\n"
                f"Current price: ${db['current_price']}   |   52-week range: ${db['fifty_two_week_low']} - ${db['fifty_two_week_high']}\n"
                f"Current P/E ({db['pe_used']}): {db['pe_value']}\n"
                + (f"Trailing EPS: {eps_trailing:.2f}\n" if eps_trailing is not None else "")
                + (f"Price-to-book: {price_to_book:.2f}\n" if price_to_book is not None else "")
                + (f"Dividend yield: {dividend_yield*100:.2f}%\n" if dividend_yield is not None else "Dividend yield: none / not paid\n")
                + f"Revenue (TTM): ${db['revenue_ttm']:,.0f}\n"
                f"Revenue growth YoY: {db['revenue_growth']*100:.1f}%\n"
                f"Gross margin: {db['gross_margin']*100:.1f}%\n"
                f"Operating margin: {db['operating_margin']*100:.1f}%\n"
                f"Net margin: {db['net_margin']*100:.1f}%\n"
                f"Free cash flow: ${db['free_cash_flow']:,.0f}\n"
                f"Return on equity: {db['return_on_equity']*100:.1f}%\n"
                f"Debt-to-equity: {db['debt_to_equity']:.2f}\n\n"
                f"Annual trend (oldest to newest):\n{_format_trend_data(db['trend_data'])}\n\n"
                f"Earnings history (oldest to newest):\n{_format_earnings_history(db['earnings_history'])}\n\n"
                f"Peer comparison:\n{_format_peer_table(db['peers'])}\n\n"
                "Produce the analysis under the five required headings."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=120, max_tokens=4000, agent_label="Fundamentals Analyst")
    word_count, thin_output = _word_count_check(response, min_words=400)
    return {
        "fundamentals_analysis": response, "model_used": model, "prompt_sent": messages,
        "word_count": word_count, "thin_output": thin_output,
    }


# ── Agent 4: Business Quality Analyst ─────────────────────────────────────────

def run_quality_analyst(state: dict, chain) -> dict:
    """
    Analyses competitive moat, brand strength, management signals, and
    long-term growth prospects. Uses two Exa searches for qualitative grounding.
    Returns: quality_analysis (str), model_used (str), prompt_sent (list)
    """
    db = state["data_bundle"]
    time_horizon = state.get("time_horizon", "Medium-term (1-3 years)")
    company_name = db["company_name"]

    moat_hits = search_exa_only(f"{company_name} competitive moat brand strength", max_results=4)
    growth_hits = search_exa_only(f"{company_name} long-term growth prospects industry position", max_results=4)

    def _format_hits(hits):
        # search_exa_only() already fetches up to 2000 chars per hit (see
        # search_client.py) — truncating much lower than that here would
        # throw away most of what was just fetched for no reason.
        if not hits:
            return "No relevant results found."
        return "\n".join(f"- {h.get('title', '')}: {h.get('content', '')[:1800]}" for h in hits)

    supp = db.get("supplementary", {})
    insider_lines = "\n".join(
        f"- {row.get('Insider', 'Unknown')} ({row.get('Position', '')}): {row.get('Transaction', '')} {row.get('Shares', '')} shares on {row.get('Start Date', '')}"
        for row in db.get("insider_transactions", [])[:5]
    ) or "No recent insider transactions found."

    holder_lines = "\n".join(
        f"- {row.get('Holder', 'Unknown')}: {row.get('Value', 'n/a')}"
        for row in db.get("institutional_holders", [])[:5]
    ) or "No institutional holder data found."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a business quality analyst. Ground qualitative claims in the "
                "search results provided. If no relevant results are found, say so "
                "explicitly. Do not speculate beyond the evidence.\n\n"
                "Insider buying is a mild positive signal. Insider selling is ambiguous — "
                "it could reflect a liquidity need rather than a loss of conviction. "
                "State this framing explicitly when discussing insider activity.\n\n"
                "Output format: structured findings under exactly four headings: "
                "Competitive Moat, Brand and Market Position, Management Signals, "
                "Long-Term Prospects. "
                "Length is a hard floor, not a target: each heading needs a minimum of 80 "
                "words of substantive analysis grounded in the research provided. Go deeper "
                "than the floor whenever the research supports it — do not stop at the "
                "minimum by default. A one-line heading is a failure, not an answer. Before "
                "finishing, check every heading against this floor and expand any that fall "
                "short. Plain English.\n\n"
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {company_name} ({db['ticker']}) — {db['sector']} / {db['industry']}\n"
                f"Time horizon: {time_horizon}\n\n"
                f"Institutional ownership: {supp.get('institutional_ownership_pct', 'n/a')}\n\n"
                f"Recent insider transactions:\n{insider_lines}\n\n"
                f"Top institutional holders:\n{holder_lines}\n\n"
                f"Competitive moat / brand research (Exa):\n{_format_hits(moat_hits)}\n\n"
                f"Long-term growth prospects research (Exa):\n{_format_hits(growth_hits)}\n\n"
                "Produce the analysis under the four required headings."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=120, max_tokens=4000, agent_label="Business Quality Analyst")
    word_count, thin_output = _word_count_check(response, min_words=320)
    return {
        "quality_analysis": response, "model_used": model, "prompt_sent": messages,
        "word_count": word_count, "thin_output": thin_output,
    }


# ── Agent 5: Risk Analyst ─────────────────────────────────────────────────────

def run_risk_analyst(state: dict, chain) -> dict:
    """
    Rates five risk categories (Valuation, Economic, Competition, Regulatory,
    Business Dependency) using tiered news evidence, macro context, and
    upcoming catalysts. Unrated categories are marked Unknown, never guessed.
    Returns: risk_analysis (str), model_used (str), prompt_sent (list)
    """
    db = state["data_bundle"]
    time_horizon = state.get("time_horizon", "Medium-term (1-3 years)")
    supp = db.get("supplementary", {})
    beta = supp.get("beta")
    short_pct = supp.get("short_percent_of_float")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a risk analyst. For each of the five risk categories, give a "
                "severity rating: Low / Medium / High. Cite specific data or news "
                "evidence for each rating. If there is no evidence, rate the category "
                "Unknown — do not assign a rating from general knowledge.\n\n"
                "Weight news evidence by source tier. Tier 1 sources (SEC filings, "
                "Reuters, Bloomberg, FT, WSJ, AP, Yahoo Finance) carry more weight than "
                "Tier 2 (industry publications) or Tier 3 (blogs, press releases). Each "
                "news item below is labelled with its tier.\n\n"
                "Where beta is provided, use it as evidence for Economic Risk — a beta "
                "above 1.3 means the stock swings harder than the market in a downturn; "
                "below 0.8 means it is more defensive. Where short interest is provided, "
                "use it as evidence for Competition or Valuation Risk — elevated short "
                "interest signals the market sees a specific bear thesis, not just noise.\n\n"
                "Incorporate the macro context: note whether the current macro "
                "environment is a tailwind, headwind, or neutral for this sector. Scan "
                "catalyst items for upcoming events that could affect risk.\n\n"
                "Output format: one paragraph per risk category, beginning with the "
                "severity rating in brackets. Example: '[High] Valuation risk: ...'. "
                "Length is a hard floor, not a target: each risk paragraph needs a minimum "
                "of 70 words citing specific evidence — go deeper than the floor whenever "
                "the news and data support it. A one-line paragraph is a failure, not an "
                "answer, even for a category rated Unknown — explain what evidence you "
                "looked for and did not find. "
                "Categories, in order: Valuation Risk, Economic Risk, Competition Risk, "
                "Regulatory Risk, Business Dependency Risk. "
                "Add a final paragraph titled 'Macro Context' and a final paragraph "
                "titled 'Upcoming Catalysts' (or state 'No catalysts identified' if none). "
                "Before finishing, check every paragraph against the 70-word floor and "
                "expand any that fall short.\n\n"
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {db['company_name']} ({db['ticker']}) — {db['sector']} / {db['industry']}\n"
                f"Time horizon: {time_horizon}\n"
                f"Current P/E ({db['pe_used']}): {db['pe_value']}\n"
                + (f"Beta: {beta:.2f}\n" if beta is not None else "")
                + (f"Short interest (% of float): {short_pct*100:.1f}%\n" if short_pct is not None else "")
                + f"Peer comparison:\n{_format_peer_table(db['peers'])}\n\n"
                f"Recent news (tiered):\n{_format_news_items(db['news_items'])}\n\n"
                f"Macro context for this sector:\n{db.get('macro_context') or 'No sector macro context available.'}\n\n"
                f"Upcoming catalysts:\n{_format_catalyst_items(db['catalyst_items'])}\n\n"
                "Produce the analysis under the required format."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=120, max_tokens=4000, agent_label="Risk Analyst")
    word_count, thin_output = _word_count_check(response, min_words=500)
    return {
        "risk_analysis": response, "model_used": model, "prompt_sent": messages,
        "word_count": word_count, "thin_output": thin_output,
    }


# ── Agent 6: Fact Checker ──────────────────────────────────────────────────────

# Metrics with a structured ground-truth value in data_bundle worth checking.
# is_percent=True means the true value is stored as a decimal fraction (0.44)
# but analysts write it as a percentage (44%) — comparison scales accordingly.
# Scoped to current/TTM values only, not historical trend years or specific
# peer figures — those would need a claim schema disambiguating which year or
# which peer, and current-period numbers are what a reader is most likely to
# open a live finance site and check right now. Trend/peer verification would
# be a reasonable v2, not attempted here.
VERIFIABLE_METRICS = {
    "gross_margin":           (True,  "Gross margin"),
    "operating_margin":       (True,  "Operating margin"),
    "net_margin":              (True,  "Net margin"),
    "revenue_growth":          (True,  "Revenue growth year-over-year"),
    "return_on_equity":        (True,  "Return on equity (ROE)"),
    "short_percent_of_float":  (True,  "Short interest as % of float"),
    "dividend_yield":          (True,  "Dividend yield"),
    "debt_to_equity":          (False, "Debt-to-equity ratio"),
    "pe_value":                (False, "P/E ratio (trailing or forward)"),
    "beta":                    (False, "Beta"),
    "price_to_book":           (False, "Price-to-book ratio"),
    "eps_trailing":            (False, "Trailing EPS"),
    "free_cash_flow":          (False, "Free cash flow (in dollars)"),
    "revenue_ttm":             (False, "Revenue, trailing twelve months (in dollars)"),
    "market_cap":              (False, "Market capitalization (in dollars)"),
    "current_price":           (False, "Current share price"),
    "fifty_two_week_high":     (False, "52-week high price"),
    "fifty_two_week_low":      (False, "52-week low price"),
    "analyst_mean_target":     (False, "Analyst mean price target"),
}

FACT_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_agent":  {"type": "string", "enum": ["Fundamentals Analyst", "Risk Analyst"]},
                    "claim_text":    {"type": "string"},
                    "metric":        {"type": "string", "enum": list(VERIFIABLE_METRICS.keys())},
                    "claimed_value": {"type": "string"},
                },
                "required": ["source_agent", "claim_text", "metric", "claimed_value"],
            },
        },
    },
    "required": ["claims"],
}


def _get_true_value(db: dict, metric: str):
    """Ground-truth lookup for a verifiable metric, checking both the top
    level of data_bundle and the supplementary dict."""
    if metric in db and db[metric] is not None:
        return db[metric]
    return db.get("supplementary", {}).get(metric)


def _parse_claimed_number(raw: str) -> float | None:
    """Parses a claim like '44%', '$451.4B', '$451.44 billion', '36.7', or
    '-0.18' into a plain float. Returns None if it cannot be parsed — that
    becomes 'Could not verify', not a false mismatch."""
    if not raw:
        return None
    cleaned = raw.strip().replace(",", "").replace("$", "").replace("%", "")
    lowered = cleaned.lower()
    multiplier = 1.0
    for word, mult in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3)):
        if lowered.endswith(word):
            multiplier, cleaned = mult, cleaned[: -len(word)].strip()
            break
    else:
        if cleaned.upper().endswith("T"):
            multiplier, cleaned = 1e12, cleaned[:-1]
        elif cleaned.upper().endswith("B"):
            multiplier, cleaned = 1e9, cleaned[:-1]
        elif cleaned.upper().endswith("M"):
            multiplier, cleaned = 1e6, cleaned[:-1]
        elif cleaned.upper().endswith("K"):
            multiplier, cleaned = 1e3, cleaned[:-1]
    try:
        return float(cleaned) * multiplier
    except (TypeError, ValueError):
        return None


def _verify_claim(db: dict, metric: str, claimed_value_str: str) -> dict:
    """
    Compares one claimed numeric value against the real data_bundle value.
    The LLM never sees the true value before extraction — this is a real
    independent check, not the model grading its own homework.

    Tolerance is deliberately generous: 1.5 percentage points for
    percent-type metrics, or 5% relative (minimum 0.5 absolute) for raw
    values — enough room for reasonable rounding ("$451.4B" -> "$450B" is
    not a factual error), tight enough to catch a genuinely wrong number.
    """
    if metric not in VERIFIABLE_METRICS:
        return {"verdict": "Could not verify", "true_value": None, "reason": "not a recognised metric"}

    is_percent, _ = VERIFIABLE_METRICS[metric]
    true_raw = _get_true_value(db, metric)
    if true_raw is None:
        return {"verdict": "Could not verify", "true_value": None, "reason": "no source value available"}

    claimed = _parse_claimed_number(claimed_value_str)
    if claimed is None:
        return {"verdict": "Could not verify", "true_value": true_raw, "reason": "could not parse claimed value"}

    true_value = true_raw * 100 if is_percent else true_raw
    tolerance = 1.5 if is_percent else max(abs(true_value) * 0.05, 0.5)
    diff = abs(claimed - true_value)

    if diff <= tolerance:
        return {"verdict": "Confirmed", "true_value": true_value, "reason": ""}
    unit = "%" if is_percent else ""
    return {
        "verdict": "Mismatch", "true_value": true_value,
        "reason": f"claimed {claimed_value_str}, actual is {true_value:.2f}{unit}",
    }


def run_fact_checker(state: dict, chain) -> dict:
    """
    Extracts specific numeric claims from the Fundamentals and Risk Analysts'
    text, then verifies each one in Python against data_bundle — the same
    numbers already fetched from yfinance — instead of asking another model
    whether the claim sounds plausible. Business Quality Analyst is excluded:
    its claims are qualitative, grounded in Exa search text, not a structured
    number with a ground truth to check against.

    Returns: fact_check_claims (list), fact_check_summary (str),
    fact_check_flagged (bool — True if any claim came back Mismatch),
    fact_check_error (bool — True if the LLM call/parse itself failed, a
    real failure distinct from "zero claims found"), model_used, prompt_sent
    """
    db = state["data_bundle"]
    fundamentals_text = state.get("fundamentals_analysis", "")
    risk_text = state.get("risk_analysis", "")

    metric_list = "\n".join(f"- {key}: {label}" for key, (_, label) in VERIFIABLE_METRICS.items())

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fact extraction assistant. You will be given two analyst "
                "reports. Extract every specific numeric claim about the CURRENT or "
                "most recent value of one of the metrics listed below. Do not extract "
                "claims about metrics not on this list — those cannot be automatically "
                "verified. Do not extract a trend description with no specific number "
                "('margins improved') — there is nothing to check in that case.\n\n"
                "Critical: only current-period values can be verified here, not "
                "historical ones. When a sentence cites a change over time (e.g. "
                "'debt-to-equity decreased from 2.61 to 1.34 since FY2022', 'ROE was "
                "197% two years ago, now 141%'), extract ONLY the current/most recent "
                "number (1.34, 141%) as the claim. Do not extract the older historical "
                "figure — there is no current-period ground truth to check it against, "
                "and flagging it as wrong would be a false alarm, not a real error.\n\n"
                f"Recognised metrics:\n{metric_list}\n\n"
                "For each claim found, report:\n"
                "source_agent: 'Fundamentals Analyst' or 'Risk Analyst'\n"
                "claim_text: the exact sentence or phrase making the claim\n"
                "metric: the recognised metric key it refers to, exactly as listed above\n"
                "claimed_value: the number as stated, with sign and unit if given "
                "(e.g. '44.3%', '36.7', '$451.4B', '-0.18')\n\n"
                "Extract every instance, including repeated mentions of the same "
                "metric. Do not invent a claim that is not in the text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Fundamentals Analyst report:\n{fundamentals_text}\n\n"
                f"Risk Analyst report:\n{risk_text}\n\n"
                "Extract every verifiable numeric claim."
            ),
        },
    ]

    model = ""
    raw_claims = None
    try:
        response, model = chain.complete(
            messages, timeout=90, max_tokens=3000, agent_label="Fact Checker", schema=FACT_CHECK_SCHEMA
        )
        raw_claims = json.loads(response).get("claims", [])
    except Exception:
        raw_claims = None

    if raw_claims is None:
        return {
            "fact_check_claims": [],
            "fact_check_summary": "Fact Checker failed to run. Claims were not verified.",
            "fact_check_flagged": True,
            "fact_check_error": True,
            "model_used": model,
            "prompt_sent": messages,
        }

    claims = []
    for raw in raw_claims:
        metric = str(raw.get("metric", ""))
        claimed_value = str(raw.get("claimed_value", ""))
        verdict = _verify_claim(db, metric, claimed_value)
        claims.append({
            "source_agent": str(raw.get("source_agent", "Unknown")),
            "claim_text": str(raw.get("claim_text", "")),
            "metric": metric,
            "claimed_value": claimed_value,
            **verdict,
        })

    confirmed = sum(1 for c in claims if c["verdict"] == "Confirmed")
    mismatched = sum(1 for c in claims if c["verdict"] == "Mismatch")
    unverifiable = sum(1 for c in claims if c["verdict"] == "Could not verify")
    total = len(claims)

    if total == 0:
        summary = "No specific numeric claims were found to verify against source data."
    elif mismatched == 0:
        summary = f"Verified {confirmed} of {total} numeric claim(s) against source data — all confirmed."
        if unverifiable:
            summary += f" ({unverifiable} could not be automatically checked.)"
    else:
        summary = f"Checked {total} numeric claim(s): {confirmed} confirmed, {mismatched} mismatch(es) found."

    return {
        "fact_check_claims": claims,
        "fact_check_summary": summary,
        "fact_check_flagged": mismatched > 0,
        "fact_check_error": False,
        "model_used": model,
        "prompt_sent": messages,
    }


# ── Agent 7: Bull Advocate ─────────────────────────────────────────────────────

def run_bull_advocate(state: dict, chain) -> dict:
    """
    Builds the strongest possible investment case using only the three
    analyst outputs and data_bundle context — no new facts.
    Returns: bull_case (str), model_used (str), prompt_sent (list)
    """
    db = state["data_bundle"]
    time_horizon = state.get("time_horizon", "Medium-term (1-3 years)")

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Bull Advocate. Build the strongest possible investment case "
                "for this stock over the stated time horizon. Argue from the evidence "
                "provided — do not introduce facts not in the inputs. Explicitly "
                "acknowledge the most significant risk, but explain why it does not "
                "undermine the thesis. Do not be generically optimistic — ground every "
                "positive claim in a specific data point or finding from the analyst outputs.\n\n"
                "Output format: Investment thesis (3-5 full sentences, not clipped to the "
                "minimum), three strongest supporting points (bulleted, 1-2 sentences of "
                "substantiation each, not just a bare claim), one key risk acknowledged and "
                "countered in 2+ sentences. Minimum 150 words total — this is a floor, not "
                "a target.\n\n"
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {db['company_name']} ({db['ticker']})\n"
                f"Time horizon: {time_horizon}\n"
                f"Analyst mean price target: ${db['analyst_mean_target']}\n"
                f"Analyst consensus: {db.get('analyst_distribution') or db.get('analyst_consensus_key')}\n\n"
                f"Fundamentals Analyst findings:\n{state['fundamentals_analysis']}\n\n"
                f"Business Quality Analyst findings:\n{state['quality_analysis']}\n\n"
                f"Risk Analyst findings:\n{state['risk_analysis']}\n\n"
                "Build the bull case in the required format."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=90, max_tokens=2000, agent_label="Bull Advocate")
    word_count, thin_output = _word_count_check(response, min_words=150)
    return {
        "bull_case": response, "model_used": model, "prompt_sent": messages,
        "word_count": word_count, "thin_output": thin_output,
    }


# ── Agent 8: Bear Advocate ─────────────────────────────────────────────────────

def run_bear_advocate(state: dict, chain) -> dict:
    """
    Builds the strongest possible case against owning the stock. Instructed
    to reason from first principles, not simply invert the Bull case.
    Returns: bear_case (str), model_used (str), prompt_sent (list)
    """
    db = state["data_bundle"]
    time_horizon = state.get("time_horizon", "Medium-term (1-3 years)")

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Bear Advocate. Build the strongest possible case against "
                "owning this stock over the stated time horizon. Argue from first "
                "principles — do not simply invert the Bull case. Find the structural "
                "weaknesses in the business or the valuation that an optimistic read "
                "glosses over. Ground every negative claim in a specific data point or "
                "finding from the analyst outputs. Explicitly acknowledge the most "
                "significant bullish factor, but explain why it does not overcome the "
                "thesis. Do not be generically pessimistic — a weak bear case is as bad "
                "as a weak bull case.\n\n"
                "Output format: Investment thesis (3-5 full sentences, not clipped to the "
                "minimum), three strongest opposing points (bulleted, 1-2 sentences of "
                "substantiation each, not just a bare claim), one key strength acknowledged "
                "and countered in 2+ sentences. Minimum 150 words total — this is a floor, "
                "not a target.\n\n"
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {db['company_name']} ({db['ticker']})\n"
                f"Time horizon: {time_horizon}\n"
                f"Analyst mean price target: ${db['analyst_mean_target']}\n"
                f"Analyst consensus: {db.get('analyst_distribution') or db.get('analyst_consensus_key')}\n\n"
                f"Fundamentals Analyst findings:\n{state['fundamentals_analysis']}\n\n"
                f"Business Quality Analyst findings:\n{state['quality_analysis']}\n\n"
                f"Risk Analyst findings:\n{state['risk_analysis']}\n\n"
                "Build the bear case in the required format."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=90, max_tokens=2000, agent_label="Bear Advocate")
    word_count, thin_output = _word_count_check(response, min_words=150)
    return {
        "bear_case": response, "model_used": model, "prompt_sent": messages,
        "word_count": word_count, "thin_output": thin_output,
    }


# ── Agent 9: Synthesizer ──────────────────────────────────────────────────────

SYNTHESIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "rating":                   {"type": "string", "enum": ["Buy", "Hold", "Sell"]},
        "confidence":               {"type": "string", "enum": ["High", "Medium", "Low"]},
        "confidence_explanation":   {"type": "string"},
        "investment_thesis":        {"type": "string"},
        "fundamentals_bullets":     {"type": "array", "items": {"type": "string"}},
        "quality_bullets":          {"type": "array", "items": {"type": "string"}},
        "key_risks":                {"type": "array", "items": {"type": "string"}},
        "debate_synthesis":         {"type": "string"},
        "street_comparison":        {"type": "string"},
        "investor_type":            {"type": "string", "enum": ["growth", "value", "income", "speculative"]},
        "trend_summary":            {"type": "string"},
        "earnings_quality_summary": {"type": "string"},
        "evidence_summary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sign": {"type": "string", "enum": ["+", "-"]},
                    "text": {"type": "string"},
                },
                "required": ["sign", "text"],
            },
        },
    },
    "required": [
        "rating", "confidence", "confidence_explanation", "investment_thesis",
        "fundamentals_bullets", "quality_bullets", "key_risks", "debate_synthesis",
        "street_comparison", "investor_type", "trend_summary",
        "earnings_quality_summary", "evidence_summary",
    ],
}


def _street_consensus_text(db: dict) -> str:
    dist = db.get("analyst_distribution")
    if dist:
        buy = dist["strong_buy"] + dist["buy"]
        hold = dist["hold"]
        sell = dist["sell"] + dist["strong_sell"]
        return f"{buy} Buy, {hold} Hold, {sell} Sell ({db['analyst_count']} analysts)"
    key = db.get("analyst_consensus_key")
    return f"Consensus rating: {key} ({db['analyst_count']} analysts, distribution unavailable)"


def run_synthesizer(state: dict, chain) -> dict:
    """
    Weighs the Bull and Bear cases and issues the structured research note.
    The LLM returns structured fields via SYNTHESIZER_SCHEMA — the final
    research_note text is assembled deterministically in Python from those
    fields plus data_bundle numbers, the same lesson Module 1 learned about
    not trusting an LLM to hit an exact template or count things itself.

    Data quality override: if data_quality_score < 60, confidence is forced
    to Low regardless of what the model returns, and the note states why.

    Returns: research_note (str), evidence_summary (list), rating (str),
    confidence (str), model_used (str), prompt_sent (list)
    """
    db = state["data_bundle"]
    time_horizon = state.get("time_horizon", "Medium-term (1-3 years)")
    quality_score = db["data_quality_score"]
    quality_label = db["data_quality_label"]
    quality_breakdown = db.get("data_quality_breakdown", [])
    # Give the Synthesizer the specific drivers, not just the final number —
    # otherwise confidence_explanation can only ever restate the score
    # generically ("data quality is low") instead of citing why.
    breakdown_text = ("; ".join(quality_breakdown) + ".") if quality_breakdown else "no specific gaps — all components scored fully."

    quality_instruction = (
        f"data_quality_score = {quality_score}/100 ({quality_label}). Specific drivers: {breakdown_text} "
        + (
            "The score is below 60. You MUST set confidence to Low regardless of how "
            "strong the analytical case appears. confidence_explanation MUST name the "
            "specific driver(s) above, not just restate the score."
            if quality_score < 60 else
            "Confidence can reflect the analysis on its merits, but if any driver above "
            "is listed, name the specific one that matters most in confidence_explanation "
            "rather than a generic restatement of the score."
        )
    )

    fact_check_flagged = state.get("fact_check_flagged", False)
    if fact_check_flagged:
        fact_check_instruction = (
            "\n\nThe Fact Checker found at least one numeric claim in the Fundamentals "
            "or Risk Analyst reports that did not match the source data, and the user "
            "chose to proceed anyway. You MUST set confidence to Low regardless of how "
            "strong the analytical case appears, and confidence_explanation MUST state "
            "that a fact-check mismatch was found and overridden by the user."
        )
    else:
        fact_check_instruction = ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Synthesizer. Weigh the Bull and Bear cases and issue a "
                "structured research note via the schema provided.\n\n"
                f"{quality_instruction}"
                f"{fact_check_instruction}\n\n"
                "If the Bull and Bear cases are closely matched, default to Hold — do "
                "not manufacture a conviction the evidence does not support.\n\n"
                "Compare your rating to the Street consensus. street_comparison must "
                "begin with either 'Agree.' or 'Diverges:' followed by one sentence "
                "explaining why.\n\n"
                "investor_type must be your honest assessment of which investor this "
                "thesis suits given the time horizon — this is educational framing, not "
                "personal advice.\n\n"
                "evidence_summary must contain 8-12 items, each tagged + (positive) or "
                "- (negative), each grounded in a specific data point or finding — no "
                "generic statements.\n\n"
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {db['company_name']} ({db['ticker']}) — {db['sector']} / {db['industry']}\n"
                f"Time horizon: {time_horizon}\n\n"
                f"Street consensus: {_street_consensus_text(db)}\n"
                f"Mean price target: ${db['analyst_mean_target']}  |  Current price: ${db['current_price']}\n\n"
                f"Fundamentals Analyst findings:\n{state['fundamentals_analysis']}\n\n"
                f"Business Quality Analyst findings:\n{state['quality_analysis']}\n\n"
                f"Risk Analyst findings:\n{state['risk_analysis']}\n\n"
                f"Bull case:\n{state['bull_case']}\n\n"
                f"Bear case:\n{state['bear_case']}\n\n"
                f"Annual trend data:\n{_format_trend_data(db['trend_data'])}\n\n"
                f"Earnings history:\n{_format_earnings_history(db['earnings_history'])}\n\n"
                "Issue the structured research note via the schema."
            ),
        },
    ]

    response, model = chain.complete(
        messages, timeout=140, max_tokens=6000, agent_label="Synthesizer", schema=SYNTHESIZER_SCHEMA
    )

    parse_error = False
    try:
        data = json.loads(response)
    except Exception:
        # This is a real failure, not a legitimate low-confidence result --
        # the model's response was not valid JSON at all. Flagging it as
        # parse_error=True lets the UI show this as a failed agent (red,
        # "Try Start Over") instead of a green "Complete" that just happens
        # to say Hold/Low. Silently smoothing this into a normal-looking
        # result is exactly the confident-looking-wrong-answer failure mode
        # this module's own design is supposed to prevent.
        parse_error = True
        data = {
            "rating": "Hold", "confidence": "Low",
            "confidence_explanation": "Synthesizer response could not be parsed. Human review required.",
            "investment_thesis": "Unable to generate a thesis due to a synthesis error.",
            "fundamentals_bullets": [], "quality_bullets": [], "key_risks": [],
            "debate_synthesis": "", "street_comparison": "", "investor_type": "value",
            "trend_summary": "", "earnings_quality_summary": "", "evidence_summary": [],
        }

    # Data quality override — enforced in code, not trusted to the model.
    confidence = data.get("confidence", "Medium")
    if quality_score < 60:
        confidence = "Low"
        if "data quality" not in data.get("confidence_explanation", "").lower():
            data["confidence_explanation"] = (
                f"Data quality score is {quality_score}/100 ({quality_label}), below the "
                f"threshold for higher confidence. {data.get('confidence_explanation', '')}"
            ).strip()

    # Fact-check override — same principle: enforced in code, not trusted to
    # the model to remember to apply it.
    if fact_check_flagged:
        confidence = "Low"
        if "fact-check" not in data.get("confidence_explanation", "").lower() and "fact check" not in data.get("confidence_explanation", "").lower():
            data["confidence_explanation"] = (
                "A fact-check mismatch was found in an earlier analyst's numeric claim "
                f"and overridden by the user. {data.get('confidence_explanation', '')}"
            ).strip()

    research_note = _build_research_note(db, state, data, confidence, time_horizon)

    return {
        "research_note": research_note,
        "evidence_summary": data.get("evidence_summary", []),
        "rating": data.get("rating", "Hold"),
        "confidence": confidence,
        "model_used": model,
        "prompt_sent": messages,
        "synthesizer_data": data,
        "parse_error": parse_error,
    }


def _build_research_note(db: dict, state: dict, data: dict, confidence: str, time_horizon: str) -> str:
    """Assembles the exact Research Note Output Format from structured fields."""
    today = datetime.now().strftime("%B %d, %Y")
    rule = "━" * 40

    upside = None
    if db.get("analyst_mean_target") and db.get("current_price"):
        upside = (db["analyst_mean_target"] - db["current_price"]) / db["current_price"] * 100

    fundamentals_bullets = "\n".join(f"- {b}" for b in data.get("fundamentals_bullets", [])) or "- No findings returned."
    quality_bullets = "\n".join(f"- {b}" for b in data.get("quality_bullets", [])) or "- No findings returned."
    key_risks = "\n".join(f"- {r}" for r in data.get("key_risks", [])) or "- No risks returned."

    sections = f"""EQUITY RESEARCH NOTE — EDUCATIONAL PURPOSE ONLY
{db['company_name']} ({db['ticker']}) | {db['sector']}
Time Horizon: {time_horizon}
Analysis Date: {today}

{rule}

RATING: {data.get('rating', 'Hold')}
CONFIDENCE: {confidence} — {data.get('confidence_explanation', '')}

{rule}

INVESTMENT THESIS
{data.get('investment_thesis', '')}

{rule}

FUNDAMENTALS SUMMARY
{fundamentals_bullets}

{rule}

BUSINESS QUALITY SUMMARY
{quality_bullets}

{rule}

KEY RISKS TO THE THESIS
{key_risks}

{rule}

THE DEBATE

Bull Case:
{state['bull_case']}

Bear Case:
{state['bear_case']}

Synthesis:
{data.get('debate_synthesis', '')}

{rule}

VALUATION CONTEXT

Current Price: ${db.get('current_price')}   |   52-Week Range: ${db.get('fifty_two_week_low')} - ${db.get('fifty_two_week_high')}
Trailing P/E: {db.get('pe_value') if db.get('pe_used') == 'trailing' else 'n/a'}     |   Forward P/E: {db.get('pe_value') if db.get('pe_used') == 'forward' else 'n/a'}
Gross Margin: {db['gross_margin']*100:.1f}%    |   Operating Margin: {db['operating_margin']*100:.1f}%

Peer Comparison:
{_format_peer_table(db['peers'])}

Street Consensus: {_street_consensus_text(db)}
Mean Price Target: ${db.get('analyst_mean_target')}   |   Implied Upside/Downside: {f'{upside:+.1f}%' if upside is not None else 'n/a'}
This Analysis vs Street: {data.get('street_comparison', '')}

Data Quality Score: {db['data_quality_score']} / 100 — {db['data_quality_label']}

TREND SUMMARY
{data.get('trend_summary', '')}

EARNINGS QUALITY
{data.get('earnings_quality_summary', '')}

UPCOMING CATALYSTS
{_format_catalyst_items(db['catalyst_items'])}

MACRO CONTEXT
{db.get('macro_context') or 'No sector macro context available.'}

INVESTOR TYPE
This thesis suits a {data.get('investor_type', 'value')} investor with a {time_horizon} horizon.

{rule}

WHAT THIS ANALYSIS CANNOT KNOW
- Private management guidance not disclosed publicly
- Undisclosed material events (pending litigation, M&A, regulatory actions)
- Real-time order flow and institutional positioning
- Your personal financial situation, risk tolerance, and investment goals
- Tax implications of any transaction

{rule}

DISCLAIMER
This analysis was produced by an AI system for educational and personal learning
purposes only. It is not investment advice. It does not constitute a recommendation
to buy, sell, or hold any security. Past performance is not indicative of future
results. The analysis relies on publicly available data and is subject to the
limitations described in the "What This Analysis Cannot Know" section above.
Do not make investment decisions based on this output."""
    return sections
