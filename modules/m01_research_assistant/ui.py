"""
Streamlit UI for the Research Assistant module.

UI flow:
1. User enters a topic and clicks Run.
2. Five agent panels appear — each starts as "Waiting".
3. As each LangGraph node completes, its panel updates to show output.
4. When all agents finish, a download button appears.

State is stored in st.session_state under keys prefixed with "m01_".
This keeps it separate from other modules.
"""

import streamlit as st
from utils.model_client import get_chain
from modules.m01_research_assistant.pipeline import build_graph, get_initial_state
from utils.doc_builder import build_research_doc

# Agent display config: internal name → display label
AGENTS = [
    ("planner",    "Agent 1: Planner",    "Breaks the topic into focused research questions"),
    ("researcher", "Agent 2: Researcher", "Searches the web for evidence on each question"),
    ("critic",     "Agent 3: Critic",     "Assesses source quality and flags gaps"),
    ("writer",     "Agent 4: Writer",     "Drafts the full research paper"),
    ("editor",     "Agent 5: Editor",     "Polishes the draft and removes weak language"),
]

STATUS_WAITING  = "⬜ Waiting"
STATUS_RUNNING  = "🔄 Running..."
STATUS_COMPLETE = "✅ Complete"
STATUS_FAILED   = "❌ Failed"


def _agent_panel(placeholder, label: str, description: str, status: str,
                 output: str = "", model: str = "", expanded: bool = False) -> None:
    """Renders a single agent panel into a placeholder."""
    with placeholder.container():
        # Status line
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{label}**  \n{description}")
        with col2:
            st.markdown(status)

        # Output expander — only shown when there is content
        if output:
            with st.expander("View output", expanded=expanded):
                st.markdown(output)
                if model:
                    st.caption(f"Model: {model}")

        st.divider()


def render() -> None:
    st.title("📝 Research Assistant")
    st.caption("Five agents research, critique, write, and edit a structured paper.")
    st.markdown(
        "> **How this works:** Enter a topic. The Planner breaks it into research questions. "
        "The Researcher searches the web. The Critic reviews the sources. "
        "The Writer drafts a full paper. The Editor polishes it."
    )
    st.markdown("---")

    # ── Input ────────────────────────────────────────────────────────────────
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Impact of small modular reactors on grid reliability",
        help="Enter any topic. Be specific — a focused topic produces a better paper.",
        key="m01_topic_input",
    )

    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        run_clicked = st.button("Run Research", type="primary", disabled=not topic.strip())
    with col_clear:
        if st.session_state.get("m01_final"):
            if st.button("Clear / New topic"):
                for key in list(st.session_state.keys()):
                    if key.startswith("m01_"):
                        del st.session_state[key]
                st.rerun()

    st.markdown("---")
    st.markdown("⚠️ *AI-generated output. Review before use.*")
    st.markdown("---")

    # ── Agent panels (shown whether pipeline is running or results exist) ────
    placeholders = {}
    for name, label, description in AGENTS:
        placeholders[name] = st.empty()

    # Show existing results if a run has already completed
    if st.session_state.get("m01_final") and not run_clicked:
        saved = st.session_state.get("m01_agent_outputs", {})
        for name, label, description in AGENTS:
            output = saved.get(name, {}).get("output", "")
            model  = saved.get(name, {}).get("model", "")
            _agent_panel(placeholders[name], label, description, STATUS_COMPLETE,
                         output=output, model=model, expanded=False)
        _show_download()
        return

    # Initialize all panels as Waiting before the run starts
    for name, label, description in AGENTS:
        _agent_panel(placeholders[name], label, description, STATUS_WAITING)

    if not run_clicked:
        return

    # ── Pipeline run ─────────────────────────────────────────────────────────
    chain = get_chain(st.session_state)
    app   = build_graph(chain)
    state = get_initial_state(topic.strip())

    agent_outputs = {}  # saves outputs for display after run completes
    full_state    = dict(state)  # accumulates all state fields across chunks
    current_index = 0  # tracks which agent is running

    # Mark the first agent as Running
    first_name, first_label, first_desc = AGENTS[0]
    _agent_panel(placeholders[first_name], first_label, first_desc, STATUS_RUNNING)

    try:
        for chunk in app.stream(state):
            node_name = list(chunk.keys())[0]
            updated   = chunk[node_name]
            full_state.update(updated)

            # Find this agent's display config
            agent_cfg = next((a for a in AGENTS if a[0] == node_name), None)
            if not agent_cfg:
                continue
            _, label, description = agent_cfg

            # Build the output text for this agent's panel
            output = _format_agent_output(node_name, updated, full_state)
            model  = updated.get("model_used", "")
            agent_outputs[node_name] = {"output": output, "model": model}

            # Mark this agent complete
            _agent_panel(placeholders[node_name], label, description, STATUS_COMPLETE,
                         output=output, model=model, expanded=True)

            # Mark the next agent as Running (if there is one)
            current_index += 1
            if current_index < len(AGENTS):
                next_name, next_label, next_desc = AGENTS[current_index]
                _agent_panel(placeholders[next_name], next_label, next_desc, STATUS_RUNNING)

    except Exception as e:
        # Mark any remaining agents as Failed and show the error
        for i in range(current_index, len(AGENTS)):
            name, label, description = AGENTS[i]
            _agent_panel(placeholders[name], label, description, STATUS_FAILED)
        st.error(f"Pipeline stopped: {e}")
        return

    # ── Save results and show download ───────────────────────────────────────
    st.session_state["m01_final"]         = full_state.get("final", "")
    st.session_state["m01_full_state"]    = full_state
    st.session_state["m01_agent_outputs"] = agent_outputs

    _show_download()


def _format_agent_output(node_name: str, updated: dict, full_state: dict) -> str:
    """Returns a readable summary of what each agent produced."""
    if node_name == "planner":
        questions = updated.get("questions", [])
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    elif node_name == "researcher":
        research = updated.get("research", {})
        sources  = updated.get("sources", [])
        lines = [f"**{len(sources)} sources collected across {len(research)} questions**\n"]
        for q, hits in research.items():
            lines.append(f"- *{q[:70]}...* — {len(hits)} results")
        return "\n".join(lines)

    elif node_name == "critic":
        return updated.get("critique", "")

    elif node_name == "writer":
        draft = updated.get("draft", "")
        return f"*Draft: {len(draft):,} characters*\n\n" + draft[:600] + "..."

    elif node_name == "editor":
        final = updated.get("final", "")
        return final  # show the full final paper

    return ""


def _show_download() -> None:
    """Renders the download section after a successful run."""
    full_state = st.session_state.get("m01_full_state", {})
    topic      = full_state.get("topic", "research")
    model      = full_state.get("model_used", "unknown")

    st.markdown("---")
    st.success("Research complete.")
    st.caption(f"Final model: {model}")

    # Build filename: research-[slug]-v1.docx
    slug = topic.lower()[:40].replace(" ", "-").replace("/", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    filename = f"research-{slug}-v1.docx"

    doc_bytes = build_research_doc(full_state)
    st.download_button(
        label="Download Word document",
        data=doc_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
