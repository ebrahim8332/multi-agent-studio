"""
LangGraph pipeline for the Stock Analyser module.

Same convention as Module 1: this file defines and compiles the graph so the
pipeline shape is explicit and inspectable, but the UI layer (ui.py) does
not call graph.invoke() directly. It calls the agent functions one phase at a
time so it can pause at checkpoints (Data Checkpoint, Complete) and run the
fan-out stages with ThreadPoolExecutor instead of LangGraph's own execution
model. The graph below documents the same flow the UI actually drives.

Graph structure:
    resolver -> data_agent -> [fundamentals, quality, risk] (parallel)
             -> [bull, bear] (parallel) -> synthesizer -> END

Conditional routing happens at two points, both enforced in agents.py, not
here: the Resolver halts on an invalid ticker, and the Data Agent halts on
missing required fields. Either halt stops the graph before it reaches the
fan-out stage — no LLM agent ever runs on unresolved or incomplete data.
"""

from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END

from modules.m02_stock.agents import (
    run_resolver,
    run_data_agent,
    run_fundamentals_analyst,
    run_quality_analyst,
    run_risk_analyst,
    run_bull_advocate,
    run_bear_advocate,
    run_synthesizer,
)


class StockState(TypedDict):
    raw_input: str          # user's typed ticker or company name
    time_horizon: str        # selected by user — shapes every analyst prompt
    ticker: str              # canonical ticker, filled by Resolver
    company_name: str        # filled by Resolver
    halted: bool             # True if Resolver or Data Agent stopped the pipeline
    halt_error: str          # plain-English error shown to the user when halted
    data_bundle: dict        # filled by Data Agent — every field described in the spec
    fundamentals_analysis: str
    quality_analysis: str
    risk_analysis: str
    bull_case: str
    bear_case: str
    research_note: str
    evidence_summary: list
    rating: str
    confidence: str
    model_used: str


def build_graph(chain):
    """
    Builds and compiles the Stock Analyser pipeline. `chain` is the
    FallbackChain from model_client.py, captured in closures so LangGraph
    nodes keep the required (state) -> dict signature.

    This graph is provided for documentation and for anyone who wants to run
    the pipeline outside the Streamlit UI (e.g. a test script). The UI itself
    drives the same steps directly — see the "New Architectural Pattern:
    Fan-Out / Merge" note in m02_ui.py for why.
    """

    def resolver_node(state: StockState) -> dict:
        result = run_resolver(state["raw_input"])
        if result["halted"]:
            return {"halted": True, "halt_error": result["error"]}
        return {"halted": False, "ticker": result["ticker"], "company_name": result["company_name"]}

    def data_agent_node(state: StockState) -> dict:
        if state.get("halted"):
            return {}
        result = run_data_agent(state["ticker"], state["company_name"])
        if result["halted"]:
            return {"halted": True, "halt_error": result["error"]}
        return {"halted": False, "data_bundle": result}

    def fan_out_analysts_node(state: StockState) -> dict:
        """Fan-out / merge: three independent analysts read the same
        data_bundle and write to separate state keys. No dependency between
        them, so they run concurrently via ThreadPoolExecutor."""
        if state.get("halted"):
            return {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_future = executor.submit(run_fundamentals_analyst, state, chain)
            q_future = executor.submit(run_quality_analyst, state, chain)
            r_future = executor.submit(run_risk_analyst, state, chain)
            f_result = f_future.result()
            q_result = q_future.result()
            r_result = r_future.result()
        return {
            "fundamentals_analysis": f_result["fundamentals_analysis"],
            "quality_analysis":      q_result["quality_analysis"],
            "risk_analysis":         r_result["risk_analysis"],
        }

    def fan_out_advocates_node(state: StockState) -> dict:
        """Same fan-out / merge pattern, two workers this time."""
        if state.get("halted"):
            return {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            bull_future = executor.submit(run_bull_advocate, state, chain)
            bear_future = executor.submit(run_bear_advocate, state, chain)
            bull_result = bull_future.result()
            bear_result = bear_future.result()
        return {
            "bull_case": bull_result["bull_case"],
            "bear_case": bear_result["bear_case"],
        }

    def synthesizer_node(state: StockState) -> dict:
        if state.get("halted"):
            return {}
        result = run_synthesizer(state, chain)
        return {
            "research_note":    result["research_note"],
            "evidence_summary": result["evidence_summary"],
            "rating":           result["rating"],
            "confidence":       result["confidence"],
            "model_used":       result["model_used"],
        }

    graph = StateGraph(StockState)
    graph.add_node("resolver",         resolver_node)
    graph.add_node("data_agent",       data_agent_node)
    graph.add_node("analysts",         fan_out_analysts_node)
    graph.add_node("advocates",        fan_out_advocates_node)
    graph.add_node("synthesizer",      synthesizer_node)

    graph.set_entry_point("resolver")
    graph.add_edge("resolver",    "data_agent")
    graph.add_edge("data_agent",  "analysts")
    graph.add_edge("analysts",    "advocates")
    graph.add_edge("advocates",   "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def get_initial_state(raw_input: str, time_horizon: str = "Medium-term (1-3 years)") -> StockState:
    """Returns a clean starting state for a new pipeline run."""
    return StockState(
        raw_input=raw_input,
        time_horizon=time_horizon,
        ticker="",
        company_name="",
        halted=False,
        halt_error="",
        data_bundle={},
        fundamentals_analysis="",
        quality_analysis="",
        risk_analysis="",
        bull_case="",
        bear_case="",
        research_note="",
        evidence_summary=[],
        rating="",
        confidence="",
        model_used="",
    )
