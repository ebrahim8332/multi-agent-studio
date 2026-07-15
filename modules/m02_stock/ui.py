"""
Streamlit UI for the Stock Analyser module.

Phase state machine (m02_phase in session_state):
  idle              — disclaimer + form shown, nothing run yet
  loading           — Resolver then Data Agent run inline (fast, no LLM calls)
  error             — Resolver or Data Agent halted; plain-English error shown
  data_checkpoint   — Checkpoint 1: data summary shown, user clicks Continue
  analysts_running  — Fundamentals + Business Quality + Risk run in parallel
  fact_checking     — Fact Checker verifies Fundamentals/Risk claims against
                       data_bundle; auto-proceeds if clean, checkpoint if not
  advocates_running — Bull + Bear run in parallel
  synthesizing      — Synthesizer runs
  complete          — Checkpoint 2: charts, evidence panel, full note, download

The spec's phase diagram lists resolver_error and data_error separately, and
lists "running" as its own phase before the Researcher-equivalent step. Both
are collapsed here: Resolver and Data Agent are both fast, non-LLM lookups,
so they run back-to-back inside one "loading" phase instead of two separate
reruns, and both halt types land on the same "error" phase since the UI
treatment (plain-English message, Start Over) is identical either way. Every
required checkpoint and halt behaviour from the spec is preserved — this only
collapses phases that would otherwise look identical on screen.

All session state keys are prefixed "m02_" to stay isolated from other
modules. Token usage is accumulated in "m02_call_log" by the model chain.
"""

import re
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor

from utils.model_client import get_chain, APPROX_PRICING, SESSION_LOCK_KEY
from utils.doc_builder import build_stock_research_doc
from archive_helper import save_report, notify_archived
from modules.m02_stock.agents import (
    run_resolver, run_data_agent, run_fundamentals_analyst, run_quality_analyst,
    run_risk_analyst, run_fact_checker, run_bull_advocate, run_bear_advocate, run_synthesizer,
    format_metric_value,
)

TIME_HORIZON_OPTIONS = [
    "Short-term (< 1 year)",
    "Medium-term (1-3 years)",
    "Long-term (3+ years)",
]

STATUS_WAITING  = "⬜ Waiting"
STATUS_RUNNING  = "🔄 Running..."
STATUS_COMPLETE = "✅ Complete"
STATUS_FLAGGED  = "⚠️ Flagged"
STATUS_FAILED   = "❌ Failed"

AGENTS = [
    ("resolver",     "Agent 1: Resolver",               "Validates the ticker against yfinance"),
    ("data_agent",   "Agent 2: Data Agent",              "Pulls financials, peers, and news — no AI, pure data lookup"),
    ("fundamentals", "Agent 3: Fundamentals Analyst",    "Revenue, margins, and valuation vs peers"),
    ("quality",      "Agent 4: Business Quality Analyst","Moat, brand, and management signals"),
    ("risk",         "Agent 5: Risk Analyst",            "Five risk categories from tiered news and macro context"),
    ("fact_checker", "Agent 6: Fact Checker",            "Verifies analyst claims against source data — not another model's opinion"),
    ("bull",         "Agent 7: Bull Advocate",           "Strongest case to own the stock"),
    ("bear",         "Agent 8: Bear Advocate",           "Strongest case against owning it"),
    ("synthesizer",  "Agent 9: Synthesizer",             "Weighs the debate and issues the rating"),
]

_STATE_KEYS = [
    "m02_phase", "m02_ticker", "m02_company_name", "m02_time_horizon",
    "m02_data_bundle", "m02_pipeline_state", "m02_agent_outputs", "m02_error",
    "m02_call_log", "locked_provider_index", "locked_model_name",
    # Set by FallbackChain.complete() in model_client.py, not by this file --
    # without it here, a rate-limit/fallback event from ticker A's run stays
    # in session state and silently carries into ticker B's run after
    # Start Over.
    "_fallback_errors",
    "m02_archive_url",
]

RISK_CATEGORY_KEYWORDS = {
    "Valuation Risk":           "valuation",
    "Economic Risk":            "economic",
    "Competition Risk":         "competition",
    "Regulatory Risk":          "regulatory",
    "Business Dependency Risk": "business dependency",
}
SEVERITY_VALUES = {"Low": 1, "Unknown": 2, "Medium": 3, "High": 4}

M02_CALL_LOG_KEY = "m02_call_log"


# ── Panel renderer ────────────────────────────────────────────────────────────

_FAN_OUT_AGENT_LABELS = (
    "Fundamentals Analyst", "Business Quality Analyst", "Risk Analyst",
    "Bull Advocate", "Bear Advocate",
)


def _agent_panel(placeholder, label: str, description: str, status: str,
                  output: str = "", model: str = "", expanded: bool = False,
                  running: bool = False, prompt: list = None, bordered: bool = None,
                  thin_warning: str = "") -> None:
    """Renders a single agent panel into a placeholder. Same conventions as
    Module 1's _agent_panel(): status line, running caption with the locked
    model name, collapsible output, collapsible prompt viewer.

    bordered: when True, wraps the header in a colored box — blue while
    waiting/running, green when complete, red if failed — the same visual
    Module 1 uses for the Tavily/Exa panels and Writer A/B. Auto-detected
    from the label for the five agents that run as part of a fan-out
    (Fundamentals, Business Quality, and Risk Analysts; Bull and Bear
    Advocates), so every call site gets it without passing the flag.

    thin_warning: when set, shows a caption warning that the output came in
    under the expected word floor — an objective Python word count, not the
    model's own judgment (same lesson Module 1 learned: LLMs can't reliably
    self-assess length). A short-but-complete run should look different from
    a properly substantive one, not both show the same green checkmark."""
    if bordered is None:
        bordered = any(keyword in label for keyword in _FAN_OUT_AGENT_LABELS)
    with placeholder.container():
        if bordered:
            with st.container(border=True):
                header = f"**{label}**  \n{description}  \n{status}"
                if status == STATUS_COMPLETE:
                    st.success(header)
                elif status == STATUS_FAILED:
                    st.error(header)
                elif status == STATUS_FLAGGED:
                    st.warning(header)
                else:
                    st.info(header)
                if running:
                    locked_model = st.session_state.get("locked_model_name", "")
                    st.caption(f"⏳ Working... · {locked_model}" if locked_model else "⏳ Working...")
                if thin_warning:
                    st.caption(f"⚠️ {thin_warning}")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{label}**  \n{description}")
            with col2:
                st.markdown(status)
            if running:
                locked_model = st.session_state.get("locked_model_name", "")
                st.caption(f"⏳ Working... · {locked_model}" if locked_model else "⏳ Working...")
            if thin_warning:
                st.caption(f"⚠️ {thin_warning}")
        if output:
            with st.expander("View output", expanded=expanded):
                st.markdown(output)
                if model:
                    st.caption(f"Model: {model}")
        if prompt:
            with st.expander("🔍 View prompt sent to AI", expanded=False):
                for msg in prompt:
                    role = msg.get("role", "").upper()
                    content = msg.get("content", "")
                    st.caption(f"── {role} ──")
                    display = content if len(content) <= 3000 else content[:3000] + "\n\n… [truncated]"
                    st.code(display, language=None)
        st.divider()


# ── Parallel fan-out helper ───────────────────────────────────────────────────

def _run_parallel(base_session_state: dict, jobs: list) -> tuple[list, list]:
    """
    Runs each (fn, args) job concurrently, each in its own isolated
    session-state copy so concurrent chain.complete() calls never race on
    shared state — same isolation Module 1 uses for Writer A/B. `fn` must
    accept `chain` as its final positional argument.

    Merges every job's call log back into base_session_state, and adopts
    job 0's model lock as the new shared lock (an arbitrary but consistent
    choice, matching Module 1's convention).

    Returns (results, errors) — errors[i] is None on success, otherwise a
    non-empty failure description. Callers must check `results[i] is None`
    (or `errors[i] is not None`) to detect a failed job — never rely on the
    truthiness of the error string itself: some exceptions (notably
    concurrent.futures.TimeoutError raised with no message) stringify to
    "", which is falsy and would let a real failure slip past `any(errors)`.
    """
    isolated = []
    for _ in jobs:
        s = dict(base_session_state)
        s[M02_CALL_LOG_KEY] = []
        isolated.append(s)

    chains = [get_chain(s, call_log_key=M02_CALL_LOG_KEY) for s in isolated]
    results: list = [None] * len(jobs)
    errors: list = [None] * len(jobs)

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(fn, *args, chains[i]): i
            for i, (fn, args) in enumerate(jobs)
        }
        for future, i in futures.items():
            try:
                results[i] = future.result(timeout=180)
            except Exception as e:
                errors[i] = str(e) or repr(e) or f"{type(e).__name__} (no message)"

    merged_log = list(base_session_state.get(M02_CALL_LOG_KEY, []))
    for s in isolated:
        merged_log += s.get(M02_CALL_LOG_KEY, [])
    base_session_state[M02_CALL_LOG_KEY] = merged_log

    if isolated and isolated[0].get(SESSION_LOCK_KEY) is not None:
        base_session_state[SESSION_LOCK_KEY] = isolated[0][SESSION_LOCK_KEY]
        base_session_state["locked_model_name"] = isolated[0].get("locked_model_name", "")

    return results, errors


# ── Chart builders ─────────────────────────────────────────────────────────────

def _build_trend_chart(trend_data: list, ticker: str):
    """Dual-axis line chart: revenue on the left axis, gross/operating margin
    on the right. Returns None if fewer than 2 years are available."""
    if len(trend_data) < 2:
        return None
    years = [t["year"] for t in trend_data]
    revenue = [t["revenue"] / 1e9 if t.get("revenue") is not None else None for t in trend_data]
    gross_margin = [t["gross_margin"] * 100 if t.get("gross_margin") is not None else None for t in trend_data]
    operating_margin = [t["operating_margin"] * 100 if t.get("operating_margin") is not None else None for t in trend_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=revenue, name="Revenue ($B)", mode="lines+markers",
                              line=dict(color="#1E3A5F", width=3)))
    fig.add_trace(go.Scatter(x=years, y=gross_margin, name="Gross Margin %", mode="lines+markers",
                              line=dict(color="#4CAF50"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=years, y=operating_margin, name="Operating Margin %", mode="lines+markers",
                              line=dict(color="#C8860A"), yaxis="y2"))
    fig.update_layout(
        title=f"{ticker} — Revenue and Margin Trend",
        yaxis=dict(title="Revenue ($B)"),
        yaxis2=dict(title="Margin %", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=70),
    )
    return fig


def _build_peer_chart(peers: list, ticker: str, subject_pe, subject_gm, subject_om):
    """Three side-by-side horizontal bar panels (P/E, Gross Margin, Operating
    Margin) — one set of bars per company, subject ticker highlighted navy.
    Metrics are shown as separate panels rather than one shared axis because
    P/E and margin percentages are on incompatible scales."""
    if not peers:
        return None
    companies = [ticker] + [p["ticker"] for p in peers]
    colors = ["#1E3A5F"] + ["#9E9E9E"] * len(peers)

    pe_vals = [subject_pe] + [p.get("pe") for p in peers]
    gm_vals = [(subject_gm * 100 if subject_gm is not None else None)] + \
              [(p["gross_margin"] * 100 if p.get("gross_margin") is not None else None) for p in peers]
    om_vals = [(subject_om * 100 if subject_om is not None else None)] + \
              [(p["operating_margin"] * 100 if p.get("operating_margin") is not None else None) for p in peers]

    fig = make_subplots(rows=1, cols=3, subplot_titles=("P/E Ratio", "Gross Margin %", "Operating Margin %"))
    fig.add_trace(go.Bar(y=companies, x=pe_vals, orientation="h", marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(y=companies, x=gm_vals, orientation="h", marker_color=colors, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(y=companies, x=om_vals, orientation="h", marker_color=colors, showlegend=False), row=1, col=3)
    fig.update_layout(title=f"{ticker} vs Peers", height=350, margin=dict(t=90))
    return fig


def _parse_risk_severities(risk_analysis: str) -> dict:
    """
    Keyword-matches the Risk Analyst's '[Severity] Category risk: ...'
    paragraphs to the five required categories. Any category not clearly
    tagged stays "Unknown" — this is a legitimate radar value, not a failure.
    """
    severities = {cat: "Unknown" for cat in RISK_CATEGORY_KEYWORDS}
    paragraphs = re.split(r"\n\s*\n", risk_analysis or "")
    for para in paragraphs:
        match = re.match(r"\s*\[(Low|Medium|High|Unknown)\]", para, re.IGNORECASE)
        if not match:
            continue
        severity = match.group(1).capitalize()
        lower = para.lower()
        for category, keyword in RISK_CATEGORY_KEYWORDS.items():
            if keyword in lower and severities[category] == "Unknown":
                severities[category] = severity
    return severities


def _build_risk_radar(risk_analysis: str, ticker: str):
    """Five-axis radar chart. Always shown — the Risk Analyst always runs
    by the time this chart is rendered."""
    severities = _parse_risk_severities(risk_analysis)
    categories = list(severities.keys())
    values = [SEVERITY_VALUES[severities[c]] for c in categories]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed, fill="toself",
        line_color="#C62828", name=ticker,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(
            visible=True, range=[0, 4], tickvals=[1, 2, 3, 4],
            ticktext=["Low", "Unknown", "Medium", "High"],
        )),
        title=f"Risk Profile — {ticker}",
        showlegend=False,
    )
    return fig, severities


# ── Run summary (token tracking) ──────────────────────────────────────────────

def _show_run_summary() -> None:
    log = st.session_state.get(M02_CALL_LOG_KEY, [])
    if not log:
        return
    total_input = sum(e["input_tokens"] for e in log)
    total_output = sum(e["output_tokens"] for e in log)
    total_cost = 0.0
    for entry in log:
        price = APPROX_PRICING.get(entry["model"])
        if price:
            in_price, out_price = price
            total_cost += (entry["input_tokens"] / 1_000_000) * in_price
            total_cost += (entry["output_tokens"] / 1_000_000) * out_price

    with st.expander("💰 Session usage", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LLM calls", len(log))
        col2.metric("Input tokens", f"{total_input:,}")
        col3.metric("Output tokens", f"{total_output:,}")
        col4.metric("Est. cost", f"${total_cost:.4f}")
        st.caption("Cost is estimated from published pricing, not a billing figure.")
        st.table([{"Agent": e["agent"], "Model": e["model"],
                   "Input": e["input_tokens"], "Output": e["output_tokens"]} for e in log])


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("📊 Stock Analyser")
    st.caption("Nine agents research, verify, debate, and rate a stock from real financial data.")
    st.markdown(
        "This module analyses a real ticker using **yfinance** for financials, **Tavily** "
        "for news and events, and **Exa** for qualitative research. If required data is "
        "missing, the pipeline stops and tells you exactly what's missing — it never "
        "guesses. Three new patterns beyond Module 1:\n\n"
        "- **Fan-out / merge** — three analysts run at the same time from the same data, "
        "then a fourth stage runs two more agents arguing opposite sides\n"
        "- **Independent fact-checking** — a dedicated agent re-checks specific numeric "
        "claims against the source data itself, not another model's opinion, before the "
        "debate stage ever sees them\n"
        "- **Evidence-grounded confidence** — a data quality score computed from objective "
        "signals directly controls how confident the final rating is allowed to sound"
    )

    with st.expander("ℹ️ How this is different from asking a chat AI", expanded=False):
        st.markdown(
            "A chat AI answers a stock question from memory, with no live data, no source "
            "verification, and no way to say 'I don't know.' This pipeline pulls real "
            "financials right now, halts outright if the data is too thin to analyse "
            "honestly, independently re-checks specific numbers before they can influence "
            "the debate, and has two agents argue opposite sides before a third weighs the "
            "debate. The confidence rating is capped by an objective data quality score, "
            "not just how convincing the argument sounds."
        )

    with st.expander("🔍 Architecture note: Independent fact-checking", expanded=False):
        st.markdown(
            "The Fact Checker does not ask another model whether the Fundamentals and Risk "
            "Analysts' numbers 'sound right' — that would just be one model's opinion "
            "checking another's. Instead, it extracts each specific numeric claim into a "
            "structured record, then Python compares it directly against `data_bundle` — "
            "the exact numbers already pulled from yfinance:\n\n"
            "```python\n"
            "# The LLM only extracts WHAT was claimed. It never sees the true value.\n"
            "claimed = extract_claims(fundamentals_text, risk_text)\n\n"
            "# Python does the actual verification — an exact numeric comparison,\n"
            "# not a model's guess about whether something looks plausible.\n"
            "for claim in claimed:\n"
            "    true_value = data_bundle[claim.metric]\n"
            "    verdict = 'Confirmed' if abs(claim.value - true_value) <= tolerance else 'Mismatch'\n"
            "```\n\n"
            "If nothing is wrong, this becomes a visible confirmation message — proof the "
            "numbers were actually checked, not just a claim that they were. If something "
            "doesn't match, the pipeline stops and shows you exactly which claim, what was "
            "claimed, and what the data actually says, before Bull, Bear, or the Synthesizer "
            "ever treat it as ground truth."
        )

    with st.expander("🧩 Architecture note: Fan-out / merge", expanded=False):
        st.markdown(
            "Three analysts (Fundamentals, Business Quality, Risk) all read the same "
            "`data_bundle` and have no dependency on each other. They run **simultaneously** "
            "in separate threads, each writing to its own slot in shared state:\n\n"
            "```python\n"
            "with ThreadPoolExecutor(max_workers=3) as executor:\n"
            "    f = executor.submit(run_fundamentals_analyst, state, chain)\n"
            "    q = executor.submit(run_quality_analyst, state, chain)\n"
            "    r = executor.submit(run_risk_analyst, state, chain)\n"
            "    fundamentals, quality, risk = f.result(), q.result(), r.result()\n"
            "```\n\n"
            "The same pattern runs Bull and Bear at the same time. Use this whenever tasks "
            "are independent and each one takes meaningful time — running them one after "
            "another would just add up their wait times for no benefit."
        )

    with st.expander("🎓 About this module — what it teaches", expanded=False):
        st.markdown(
            "1. **Fan-out / merge** — independent agents running at once, merged into shared state.\n"
            "2. **Adversarial synthesis** — Bull and Bear argue opposing cases; a Synthesizer weighs both.\n"
            "3. **Conditional routing** — the pipeline halts on an invalid ticker or incomplete data "
            "instead of producing a confident-sounding wrong answer.\n"
            "4. **Independent verification, not another opinion** — the Fact Checker verifies specific "
            "numbers against source data in Python, not by asking a model if a claim sounds right. "
            "Checks and balances in a multi-agent system need a real ground truth to check against, "
            "not just a second AI's agreement.\n"
            "5. **Structured vs unstructured data** — the Data Agent is deterministic API calls; the "
            "analyst agents are LLM reasoning. Keeping these separate is a design choice, not a shortcut.\n"
            "6. **Epistemological honesty** — 'What This Analysis Cannot Know' states the system's real "
            "limits, not a legal disclaimer.\n"
            "7. **Evidence-grounded confidence** — the data quality score is computed from objective "
            "signals and caps what confidence level the Synthesizer is allowed to issue. A fact-check "
            "mismatch a human overrides caps it the same way.\n"
            "8. **Trend over point estimates** — one year's revenue number means little next to three "
            "years of direction. The Fundamentals Analyst is told to describe direction, not just numbers."
        )

    st.markdown("---")

    if "m02_form_key" not in st.session_state:
        st.session_state["m02_form_key"] = 0
    fk = st.session_state["m02_form_key"]

    phase = st.session_state.get("m02_phase", "idle")
    locked = phase not in ("idle", "error")

    # ── Disclaimer gate ────────────────────────────────────────────────────────
    st.warning(
        "**⚠️ Educational Use Only**\n\n"
        "This tool is for personal learning and educational exploration of multi-agent "
        "AI systems. The output is not investment advice. It does not account for your "
        "personal financial situation. Do not use this analysis to make investment decisions."
    )
    acknowledged = st.checkbox(
        "I understand this is for educational purposes only and is not investment advice.",
        key=f"m02_ack_{fk}", disabled=locked,
    )

    # ── Input form ─────────────────────────────────────────────────────────────
    raw_input = st.text_input(
        "Ticker or company name", placeholder="e.g. AAPL or Apple",
        key=f"m02_input_{fk}", disabled=locked,
        help="The Resolver validates this against yfinance before anything else runs.",
    )
    time_horizon = st.selectbox(
        "Time horizon", TIME_HORIZON_OPTIONS, index=1,
        key=f"m02_horizon_{fk}", disabled=locked,
        help="Shapes how every analyst frames its findings — short-term weights recent momentum, long-term weights structural trends.",
    )

    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        run_clicked = st.button(
            "Run Analysis", type="primary",
            disabled=not (acknowledged and raw_input.strip()) or locked,
            key="m02_run_btn",
        )
    with col_clear:
        clear_clicked = st.button("Start Over", key="m02_clear_btn")

    if clear_clicked:
        st.session_state["m02_form_key"] += 1
        for key in _STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("---")

    # ── Placeholders ───────────────────────────────────────────────────────────
    # error_ph is created first, before Agent 1's own placeholder, so a halt
    # message renders at the top of the page instead of below all nine agent
    # panels -- previously the error phase used plain st.error() calls, which
    # execute after every st.empty() below has already claimed its position,
    # landing the message at the very bottom of a tall page, easy to miss
    # without scrolling.
    error_ph      = st.empty()
    resolver_ph   = st.empty()
    data_ph       = st.empty()
    checkpoint_ph = st.empty()
    _analyst_cols = st.columns(3)
    fundamentals_ph = _analyst_cols[0].empty()
    quality_ph      = _analyst_cols[1].empty()
    risk_ph         = _analyst_cols[2].empty()
    fact_checker_ph  = st.empty()
    fact_check_gate_ph = st.empty()
    _advocate_cols = st.columns(2)
    bull_ph = _advocate_cols[0].empty()
    bear_ph = _advocate_cols[1].empty()
    synth_ph      = st.empty()
    results_ph    = st.container()

    ph = {
        "resolver": resolver_ph, "data_agent": data_ph, "fundamentals": fundamentals_ph,
        "quality": quality_ph, "risk": risk_ph, "fact_checker": fact_checker_ph,
        "bull": bull_ph, "bear": bear_ph, "synthesizer": synth_ph,
    }

    if run_clicked and acknowledged and raw_input.strip():
        for key in _STATE_KEYS:
            st.session_state.pop(key, None)
        st.session_state["m02_time_horizon"] = time_horizon
        st.session_state["m02_phase"] = "loading"
        st.session_state["_m02_pending_input"] = raw_input.strip()
        st.rerun()

    phase = st.session_state.get("m02_phase", "idle")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: idle
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "idle":
        for name, label, desc in AGENTS:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: loading  (Resolver, then Data Agent — both fast, no LLM)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "loading":
        pending_input = st.session_state.pop("_m02_pending_input", raw_input.strip())
        _agent_panel(resolver_ph, "Agent 1: Resolver", "Validating ticker...", STATUS_RUNNING, running=True)
        for name, label, desc in AGENTS[1:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)

        # Guards on each step so a rerun mid-flight (network stall, browser
        # resize, refresh) resumes from whichever step already finished
        # instead of re-running yfinance/Tavily calls from scratch.
        ticker = st.session_state.get("m02_ticker")
        company_name = st.session_state.get("m02_company_name")
        if not ticker:
            with st.spinner(f"Resolving '{pending_input}'..."):
                resolved = run_resolver(pending_input)

            if resolved["halted"]:
                st.session_state["m02_error"] = resolved["error"]
                st.session_state["m02_phase"] = "error"
                st.rerun()
                return

            ticker = resolved["ticker"]
            company_name = resolved["company_name"]
            st.session_state["m02_ticker"] = ticker
            st.session_state["m02_company_name"] = company_name

        _agent_panel(resolver_ph, "Agent 1: Resolver", "Validated the ticker",
                     STATUS_COMPLETE, output=f"{company_name} ({ticker})")

        _agent_panel(data_ph, "Agent 2: Data Agent", "Pulling financials, peers, and news...",
                     STATUS_RUNNING, running=True)
        data_bundle = st.session_state.get("m02_data_bundle")
        if not data_bundle:
            with st.spinner("Pulling financials, peers, and news (yfinance + Tavily)..."):
                data_bundle = run_data_agent(ticker, company_name)

            if data_bundle["halted"]:
                st.session_state["m02_error"] = data_bundle["error"]
                st.session_state["m02_phase"] = "error"
                st.rerun()
                return

            st.session_state["m02_data_bundle"] = data_bundle

        st.session_state["m02_phase"] = "data_checkpoint"
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: error
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "error":
        error_msg = st.session_state.get("m02_error", "Something went wrong.")
        with error_ph.container():
            for line in error_msg.split("\n\n"):
                if line.strip():
                    st.error(line) if error_msg.split("\n\n")[0] == line else st.markdown(line)
        for name, label, desc in AGENTS:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)
        return

    # From here on, every phase needs the ticker, company name, and data bundle.
    ticker = st.session_state.get("m02_ticker", "")
    company_name = st.session_state.get("m02_company_name", "")
    db = st.session_state.get("m02_data_bundle", {})
    time_horizon = st.session_state.get("m02_time_horizon", TIME_HORIZON_OPTIONS[1])

    _agent_panel(resolver_ph, "Agent 1: Resolver", "Validated the ticker",
                 STATUS_COMPLETE, output=f"{company_name} ({ticker})")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: data_checkpoint  (Checkpoint 1)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "data_checkpoint":
        data_summary = (
            f"**{company_name} ({ticker})** — {db.get('sector')} / {db.get('industry')}\n\n"
            f"Current price: ${db.get('current_price')}  |  Market cap: ${db.get('market_cap', 0)/1e9:.1f}B\n\n"
            f"Data quality: **{db['data_quality_score']} / 100 — {db['data_quality_label']}**\n\n"
            f"Peers found: {', '.join(p['ticker'] for p in db.get('peers', [])) or 'None'}"
            + ("" if len(db.get("peers", [])) >= 2 else " — fewer than 2 peers found; comparison will be limited.") + "\n\n"
            f"News articles retrieved: {len(db.get('news_items', []))}\n\n"
            f"Time horizon: {time_horizon}"
        )
        # expanded left at its default (False) here -- the checkpoint box just
        # below shows this exact same summary, open and interactive with the
        # Continue/Start Over buttons. Auto-expanding this panel's "View
        # output" too meant the identical text appeared twice in a row.
        _agent_panel(data_ph, "Agent 2: Data Agent", "Data pulled and validated",
                     STATUS_COMPLETE, output=data_summary)

        with checkpoint_ph.container():
            st.info(data_summary)
            if db.get("thin_analyst_coverage"):
                st.caption(f"⚠️ Thin analyst coverage: only {db.get('analyst_count')} analyst(s).")
            breakdown = db.get("data_quality_breakdown", [])
            if breakdown:
                with st.expander("Why isn't the data quality score 100?", expanded=False):
                    for note in breakdown:
                        st.caption(f"• {note.capitalize()}")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Continue to Analysis →", type="primary", key="m02_continue_btn"):
                    st.session_state["m02_phase"] = "analysts_running"
                    st.rerun()
            with col2:
                if st.button("Start Over", key="m02_checkpoint_stop_btn"):
                    st.session_state["m02_form_key"] += 1
                    for key in _STATE_KEYS:
                        st.session_state.pop(key, None)
                    st.rerun()

        for name, label, desc in AGENTS[2:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)
        return

    # From here on, Checkpoint 1 has passed.
    data_summary_short = f"{company_name} ({ticker}) — data quality {db['data_quality_score']}/100"
    _agent_panel(data_ph, "Agent 2: Data Agent", "Data pulled and validated",
                 STATUS_COMPLETE, output=data_summary_short)
    checkpoint_ph.empty()

    pipeline_state = st.session_state.get("m02_pipeline_state", {})
    agent_outputs = st.session_state.get("m02_agent_outputs", {})

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: analysts_running  (fan-out / merge, 3 workers)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "analysts_running":
        for placeholder, (label, desc) in (
            (fundamentals_ph, ("Agent 3: Fundamentals Analyst", "Analysing revenue, margins, and peer valuation...")),
            (quality_ph,      ("Agent 4: Business Quality Analyst", "Researching moat, brand, and management signals...")),
            (risk_ph,         ("Agent 5: Risk Analyst", "Assessing five risk categories...")),
        ):
            _agent_panel(placeholder, label, desc, STATUS_RUNNING, running=True)
        for name, label, desc in AGENTS[5:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)

        # Only run once. This is the most expensive step in the pipeline (3
        # concurrent LLM calls) and it can take up to 180s -- without this
        # guard, any rerun that fires while _run_parallel is still in flight
        # (browser resize, flaky reconnect, refresh) would re-enter this block
        # and fire a second full fan-out, doubling the API cost and racing two
        # writes to pipeline_state. Same guard pattern as fact_checking below.
        if "fundamentals_analysis" not in pipeline_state:
            state_for_agents = {"data_bundle": db, "time_horizon": time_horizon}
            with st.spinner("Running Fundamentals, Business Quality, and Risk analysts simultaneously..."):
                results, errors = _run_parallel(st.session_state, [
                    (run_fundamentals_analyst, (state_for_agents,)),
                    (run_quality_analyst,      (state_for_agents,)),
                    (run_risk_analyst,         (state_for_agents,)),
                ])

            if any(r is None for r in results):
                failed = [AGENTS[2 + i][1] for i, r in enumerate(results) if r is None]
                st.error(f"Analyst agent(s) failed: {', '.join(failed)}. Try Start Over.")
                for name, label, desc in AGENTS[2:5]:
                    _agent_panel(ph[name], label, desc, STATUS_FAILED)
                return

            pipeline_state["fundamentals_analysis"] = results[0]["fundamentals_analysis"]
            pipeline_state["quality_analysis"] = results[1]["quality_analysis"]
            pipeline_state["risk_analysis"] = results[2]["risk_analysis"]
            agent_outputs["fundamentals"] = {"output": results[0]["fundamentals_analysis"], "model": results[0]["model_used"], "prompt": results[0]["prompt_sent"], "thin_output": results[0].get("thin_output"), "word_count": results[0].get("word_count")}
            agent_outputs["quality"]      = {"output": results[1]["quality_analysis"],      "model": results[1]["model_used"], "prompt": results[1]["prompt_sent"], "thin_output": results[1].get("thin_output"), "word_count": results[1].get("word_count")}
            agent_outputs["risk"]         = {"output": results[2]["risk_analysis"],         "model": results[2]["model_used"], "prompt": results[2]["prompt_sent"], "thin_output": results[2].get("thin_output"), "word_count": results[2].get("word_count")}

            st.session_state["m02_pipeline_state"] = pipeline_state
            st.session_state["m02_agent_outputs"] = agent_outputs

        st.session_state["m02_phase"] = "fact_checking"
        st.rerun()
        return

    # From here on, the three analysts have completed.
    for key, placeholder, (label, desc) in (
        ("fundamentals", fundamentals_ph, ("Agent 3: Fundamentals Analyst", "Revenue, margins, and valuation vs peers")),
        ("quality",      quality_ph,      ("Agent 4: Business Quality Analyst", "Moat, brand, and management signals")),
        ("risk",         risk_ph,         ("Agent 5: Risk Analyst", "Five risk categories from tiered news and macro context")),
    ):
        out = agent_outputs.get(key, {})
        thin_warning = (
            f"Output ran short: {out.get('word_count')} words, under the expected floor for this format."
            if out.get("thin_output") else ""
        )
        _agent_panel(placeholder, label, desc, STATUS_COMPLETE, output=out.get("output", ""),
                     model=out.get("model", ""), prompt=out.get("prompt", []), thin_warning=thin_warning)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: fact_checking
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "fact_checking":
        # Only run once. If a mismatch is found, the checkpoint below keeps
        # phase="fact_checking" across reruns while the user decides — this
        # guard stops the Fact Checker (and its LLM call) from re-running on
        # every one of those reruns.
        if "fact_check_summary" not in pipeline_state:
            _agent_panel(fact_checker_ph, "Agent 6: Fact Checker",
                         "Verifying analyst claims against source data...", STATUS_RUNNING, running=True)
            for name, label, desc in AGENTS[6:]:
                _agent_panel(ph[name], label, desc, STATUS_WAITING)

            state_for_fc = {"data_bundle": db, **pipeline_state}
            chain = get_chain(st.session_state, call_log_key=M02_CALL_LOG_KEY)
            with st.spinner("Cross-checking numeric claims against yfinance data..."):
                try:
                    fc_result = run_fact_checker(state_for_fc, chain)
                except Exception as e:
                    st.error(f"Fact Checker failed to run: {e}. Try Start Over.")
                    _agent_panel(fact_checker_ph, "Agent 6: Fact Checker", "", STATUS_FAILED)
                    return

            pipeline_state["fact_check_claims"] = fc_result["fact_check_claims"]
            pipeline_state["fact_check_summary"] = fc_result["fact_check_summary"]
            pipeline_state["fact_check_flagged"] = fc_result["fact_check_flagged"]
            agent_outputs["fact_checker"] = {
                "output": fc_result["fact_check_summary"], "model": fc_result["model_used"],
                "prompt": fc_result["prompt_sent"],
            }
            st.session_state["m02_pipeline_state"] = pipeline_state
            st.session_state["m02_agent_outputs"] = agent_outputs

            if fc_result["fact_check_error"]:
                detail = fc_result.get("fact_check_error_detail", "")
                error_msg = (
                    "Fact Checker failed to run — claims from Fundamentals and Risk were not "
                    "verified. This is a model/formatting failure, not a clean pass. Try Start Over."
                )
                if detail:
                    error_msg += f"\n\nError detail: {detail}"
                st.error(error_msg)
                _agent_panel(fact_checker_ph, "Agent 6: Fact Checker", "Failed to run", STATUS_FAILED,
                             prompt=fc_result["prompt_sent"])
                return

            if not fc_result["fact_check_flagged"]:
                # Clean pass — the confidence-building signal: proof the
                # numbers were actually checked, not just asserted.
                _agent_panel(fact_checker_ph, "Agent 6: Fact Checker", "Verified analyst claims against source data",
                             STATUS_COMPLETE, output=fc_result["fact_check_summary"], model=fc_result["model_used"],
                             prompt=fc_result["prompt_sent"], expanded=True)
                st.success(f"✅ {fc_result['fact_check_summary']}")
                st.session_state["m02_phase"] = "advocates_running"
                st.rerun()
                return

            # Flagged — render straight from the result just computed rather
            # than forcing another rerun to pick up what we already have.
            claims = fc_result["fact_check_claims"]
            summary = fc_result["fact_check_summary"]
            model_used = fc_result["model_used"]
            prompt_sent = fc_result["prompt_sent"]
        else:
            claims = pipeline_state.get("fact_check_claims", [])
            summary = pipeline_state.get("fact_check_summary", "")
            out = agent_outputs.get("fact_checker", {})
            model_used = out.get("model", "")
            prompt_sent = out.get("prompt", [])

        _agent_panel(fact_checker_ph, "Agent 6: Fact Checker", "Mismatch found — awaiting your decision",
                     STATUS_FLAGGED, output=summary, model=model_used, prompt=prompt_sent, expanded=True)

        with fact_check_gate_ph.container():
            st.warning(f"⚠️ {summary}")
            for c in claims:
                if c["verdict"] != "Mismatch":
                    continue
                true_str = format_metric_value(c.get("metric", ""), c.get("true_value"))
                st.markdown(
                    f"**{c['source_agent']}** claimed **{c['metric']}** = `{c['claimed_value']}` — "
                    f"actual value is `{true_str}`.\n\n> {c['claim_text']}"
                )
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Proceed despite mismatch(es) →", type="primary", key="m02_factcheck_proceed_btn"):
                    fact_check_gate_ph.empty()
                    st.session_state["m02_phase"] = "advocates_running"
                    st.rerun()
            with col2:
                if st.button("Start Over", key="m02_factcheck_stop_btn"):
                    st.session_state["m02_form_key"] += 1
                    for key in _STATE_KEYS:
                        st.session_state.pop(key, None)
                    st.rerun()
            st.caption(
                "Proceeding forces the final rating's confidence to Low — a fact-check "
                "mismatch overridden by a human is not a High-confidence result."
            )

        for name, label, desc in AGENTS[6:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)
        return

    # From here on, the Fact Checker has completed (clean or overridden).
    fc_out = agent_outputs.get("fact_checker", {})
    fact_checker_label = (
        "Mismatch found — proceeded anyway" if pipeline_state.get("fact_check_flagged")
        else "Verified analyst claims against source data"
    )
    _agent_panel(fact_checker_ph, "Agent 6: Fact Checker", fact_checker_label,
                 STATUS_FLAGGED if pipeline_state.get("fact_check_flagged") else STATUS_COMPLETE,
                 output=fc_out.get("output", ""), model=fc_out.get("model", ""), prompt=fc_out.get("prompt", []))
    fact_check_gate_ph.empty()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: advocates_running  (fan-out / merge, 2 workers)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "advocates_running":
        _agent_panel(bull_ph, "Agent 7: Bull Advocate", "Building the strongest case to own the stock...", STATUS_RUNNING, running=True)
        _agent_panel(bear_ph, "Agent 8: Bear Advocate", "Building the strongest case against owning it...", STATUS_RUNNING, running=True)
        _agent_panel(synth_ph, "Agent 9: Synthesizer", "Weighs the debate and issues the rating", STATUS_WAITING)

        # Same re-entrancy guard as analysts_running above -- without it, a
        # rerun mid-flight re-fires both advocate LLM calls a second time.
        if "bull_case" not in pipeline_state:
            state_for_advocates = {"data_bundle": db, "time_horizon": time_horizon, **pipeline_state}
            with st.spinner("Running Bull and Bear advocates simultaneously..."):
                results, errors = _run_parallel(st.session_state, [
                    (run_bull_advocate, (state_for_advocates,)),
                    (run_bear_advocate, (state_for_advocates,)),
                ])

            if any(r is None for r in results):
                st.error("Bull or Bear advocate failed. Try Start Over.")
                _agent_panel(bull_ph, "Agent 7: Bull Advocate", "", STATUS_FAILED)
                _agent_panel(bear_ph, "Agent 8: Bear Advocate", "", STATUS_FAILED)
                return

            pipeline_state["bull_case"] = results[0]["bull_case"]
            pipeline_state["bear_case"] = results[1]["bear_case"]
            agent_outputs["bull"] = {"output": results[0]["bull_case"], "model": results[0]["model_used"], "prompt": results[0]["prompt_sent"], "thin_output": results[0].get("thin_output"), "word_count": results[0].get("word_count")}
            agent_outputs["bear"] = {"output": results[1]["bear_case"], "model": results[1]["model_used"], "prompt": results[1]["prompt_sent"], "thin_output": results[1].get("thin_output"), "word_count": results[1].get("word_count")}

            st.session_state["m02_pipeline_state"] = pipeline_state
            st.session_state["m02_agent_outputs"] = agent_outputs

        st.session_state["m02_phase"] = "synthesizing"
        st.rerun()
        return

    # From here on, Bull and Bear have completed.
    for key, placeholder, (label, desc) in (
        ("bull", bull_ph, ("Agent 7: Bull Advocate", "Strongest case to own the stock")),
        ("bear", bear_ph, ("Agent 8: Bear Advocate", "Strongest case against owning it")),
    ):
        out = agent_outputs.get(key, {})
        thin_warning = (
            f"Output ran short: {out.get('word_count')} words, under the expected floor for this format."
            if out.get("thin_output") else ""
        )
        _agent_panel(placeholder, label, desc, STATUS_COMPLETE, output=out.get("output", ""),
                     model=out.get("model", ""), prompt=out.get("prompt", []), thin_warning=thin_warning)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: synthesizing
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "synthesizing":
        _agent_panel(synth_ph, "Agent 9: Synthesizer", "Weighing the debate and issuing a rating...", STATUS_RUNNING, running=True)

        state_for_synth = {"data_bundle": db, "time_horizon": time_horizon, **pipeline_state}
        chain = get_chain(st.session_state, call_log_key=M02_CALL_LOG_KEY)
        try:
            with st.spinner("Weighing the debate..."):
                result = run_synthesizer(state_for_synth, chain)
        except Exception as e:
            st.error(f"Synthesizer failed: {e}")
            _agent_panel(synth_ph, "Agent 9: Synthesizer", "", STATUS_FAILED)
            return

        if result.get("parse_error"):
            # The model's response was not valid JSON at all — a real
            # failure, not a legitimate Hold/Low result. Show it as failed
            # rather than letting it through looking like a normal answer.
            st.error(
                "Synthesizer response could not be parsed into a valid result. "
                "This is a model/formatting failure, not a genuine low-confidence "
                "rating. Try Start Over — a retry often succeeds."
            )
            _agent_panel(synth_ph, "Agent 9: Synthesizer", "Response failed to parse", STATUS_FAILED,
                         prompt=result.get("prompt_sent", []))
            return

        pipeline_state.update(result)
        agent_outputs["synthesizer"] = {"output": result["research_note"], "model": result["model_used"], "prompt": result["prompt_sent"]}

        st.session_state["m02_pipeline_state"] = pipeline_state
        st.session_state["m02_agent_outputs"] = agent_outputs
        st.session_state["m02_phase"] = "complete"
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: complete  (Checkpoint 2)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "complete":
        synth_out = agent_outputs.get("synthesizer", {})
        _agent_panel(synth_ph, "Agent 9: Synthesizer", "Weighs the debate and issues the rating",
                     STATUS_COMPLETE, output=synth_out.get("output", ""), model=synth_out.get("model", ""),
                     prompt=synth_out.get("prompt", []))

        with results_ph:
            st.markdown("---")
            rating = pipeline_state.get("rating", "Hold")
            confidence = pipeline_state.get("confidence", "Medium")
            rating_icon = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}.get(rating, "⚪")
            st.subheader(f"{rating_icon} Rating: {rating}  ·  Confidence: {confidence}")

            trend_chart = _build_trend_chart(db.get("trend_data", []), ticker)
            if trend_chart:
                st.plotly_chart(trend_chart, width="stretch")
            else:
                st.info("Insufficient trend data for chart.")

            peer_chart = _build_peer_chart(db.get("peers", []), ticker, db.get("pe_value"),
                                            db.get("gross_margin"), db.get("operating_margin"))
            if peer_chart:
                st.plotly_chart(peer_chart, width="stretch")

            radar_chart, severities = _build_risk_radar(pipeline_state.get("risk_analysis", ""), ticker)
            st.plotly_chart(radar_chart, width="stretch")

            with st.expander("🔍 Why did the AI reach this conclusion?", expanded=True):
                evidence = pipeline_state.get("evidence_summary", [])
                positives = [e for e in evidence if e.get("sign") == "+"]
                negatives = [e for e in evidence if e.get("sign") == "-"]
                st.caption(f"{len(positives)} positive signals, {len(negatives)} negative signals.")
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.markdown("**Positive**")
                    for e in positives:
                        st.success(e.get("text", ""), icon="➕")
                with col_neg:
                    st.markdown("**Negative**")
                    for e in negatives:
                        st.error(e.get("text", ""), icon="➖")

            with st.expander("📄 Full research note", expanded=False):
                st.text(pipeline_state.get("research_note", ""))

            _show_run_summary()

            st.markdown("⚠️ *AI-generated output. Review before use. Not investment advice.*")

            doc_state = {**pipeline_state, "data_bundle": db, "ticker": ticker,
                         "company_name": company_name, "time_horizon": time_horizon}
            today_slug = ticker.replace(".", "-")
            if not pipeline_state.get("research_note", "").strip():
                st.warning(
                    "The research note is empty — the Synthesizer did not produce output. "
                    "Check the error messages above and run again."
                )
            else:
                doc_bytes = build_stock_research_doc(doc_state)
                file_name = f"{today_slug}-research-note.docx"

                interim_summary = {
                    "ticker": ticker,
                    "time_horizon": time_horizon,
                    "data_quality_score": db.get("data_quality_score"),
                    "data_quality_label": db.get("data_quality_label"),
                    "fact_check_summary": pipeline_state.get("fact_check_summary"),
                    "fact_check_flagged": pipeline_state.get("fact_check_flagged"),
                    "model_used": pipeline_state.get("model_used"),
                }
                summary_text = f"{company_name} ({ticker}) — {rating} rating, {confidence} confidence"

                # Archive once per completed run, not once per rerun. Streamlit
                # reruns the whole script on things as small as a browser
                # reconnect — with no guard here, a rerun re-uploaded the same
                # report under the same deterministic (ticker-based) filename,
                # which Supabase rejects as a duplicate, producing a false
                # "backup failed" warning even after a real upload succeeded.
                # Same bug, same fix as m01_research_assistant/ui.py.
                if "m02_archive_url" not in st.session_state:
                    with st.spinner("Archiving report to permanent storage..."):
                        archive_url = save_report(
                            app_name="multi-agent-studio",
                            module_name="m02-stock",
                            file_bytes=doc_bytes,
                            file_name=file_name,
                            file_type="docx",
                            user_prompt=f"{ticker} ({company_name})",
                            interim_steps=interim_summary,
                            final_output_summary=summary_text,
                        )
                    st.session_state["m02_archive_url"] = archive_url
                    if archive_url is None:
                        st.warning("⚠️ Report ready below. Cloud backup failed — download still works.")
                    else:
                        notify_archived()
                else:
                    # Re-displayed on a later rerun — no new upload, no repeat chime.
                    if st.session_state["m02_archive_url"] is None:
                        st.warning("⚠️ Report ready below. Cloud backup failed — download still works.")
                    else:
                        st.caption("☁️ Archived to permanent storage")

                st.download_button(
                    "⬇️ Download research note (.docx)", data=doc_bytes,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="m02_download_btn",
                )

            if st.button("Start Over", key="m02_complete_stop_btn"):
                st.session_state["m02_form_key"] += 1
                for key in _STATE_KEYS:
                    st.session_state.pop(key, None)
                st.rerun()
        return
