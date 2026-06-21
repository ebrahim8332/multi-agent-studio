"""
LangGraph pipeline for the Research Assistant module.

How it works:
- ResearchState is a TypedDict — a typed dictionary that defines every field the pipeline uses.
- StateGraph takes that type and builds a directed graph.
- Each node is a Python function. We wrap the agent functions in closures so they
  can access the shared chain without it being part of the state.
- .compile() turns the graph definition into an executable object.
- .stream() runs the graph and yields {node_name: partial_state} after each node completes.

Graph structure (linear — no branching in this module):
    planner → researcher → critic → writer → editor → END
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from modules.m01_research_assistant.agents import (
    run_planner,
    run_researcher,
    run_critic,
    run_writer,
    run_editor,
)


# ── State definition ──────────────────────────────────────────────────────────
#
# TypedDict tells Python (and LangGraph) exactly what fields the state holds
# and what type each field is. Every node reads from and writes to this dict.
# Fields start empty and are filled in as agents run.

class ResearchState(TypedDict):
    topic: str          # user input — set before the pipeline starts
    audience: str       # selected by user — injected into Writer and Editor prompts
    questions: list     # filled by Planner
    research: dict      # filled by Researcher: {question: [search result dicts]}
    critique: str       # filled by Critic
    draft: str          # filled by Writer
    final: str          # filled by Editor
    model_used: str     # updated by each LLM agent — holds the last model name
    sources: list       # filled by Researcher: list of URLs


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_graph(chain):
    """
    Builds and compiles the research pipeline.

    `chain` is the FallbackChain from model_client.py. It is passed in from
    the UI layer (which has access to st.session_state) so the chain can lock
    to a model for the full pipeline run.

    We wrap each LLM agent in a closure so it receives the chain. LangGraph
    nodes must have the signature (state) -> dict — they cannot accept extra
    arguments directly.
    """

    # Closures: each function captures `chain` from the outer scope.
    # LangGraph calls them with just `state` — the chain is already baked in.

    def planner_node(state: ResearchState) -> dict:
        return run_planner(state, chain)

    def researcher_node(state: ResearchState) -> dict:
        return run_researcher(state)

    def critic_node(state: ResearchState) -> dict:
        return run_critic(state, chain)

    def writer_node(state: ResearchState) -> dict:
        return run_writer(state, chain)

    def editor_node(state: ResearchState) -> dict:
        return run_editor(state, chain)

    # Build the graph
    graph = StateGraph(ResearchState)

    # Add each node — first argument is the name used in streaming output
    graph.add_node("planner",    planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic",     critic_node)
    graph.add_node("writer",     writer_node)
    graph.add_node("editor",     editor_node)

    # Draw the edges — linear flow, no branching
    graph.set_entry_point("planner")
    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_edge("critic",     "writer")
    graph.add_edge("writer",     "editor")
    graph.add_edge("editor",     END)

    # Compile turns the graph definition into an executable object
    return graph.compile()


def get_initial_state(topic: str, audience: str = "General business audience") -> ResearchState:
    """Returns a clean starting state for a new pipeline run."""
    return ResearchState(
        topic=topic,
        audience=audience,
        questions=[],
        research={},
        critique="",
        draft="",
        final="",
        model_used="",
        sources=[],
    )
