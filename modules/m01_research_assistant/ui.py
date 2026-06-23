"""
Streamlit UI for the Research Assistant module.

UI flow:
1. User enters a topic, selects audience/format/length, and clicks Run.
2. Planner runs and its panel updates to Complete.
3. Approval checkpoint: user reviews the research questions and either:
   a. Approves — downstream agents run (Researcher → Critic → Writer → Editor)
   b. Edits and replans — Planner runs again with the user's edits as context.
      Attempt counter increments. Loop continues until user approves.
4. After approval, the remaining four agents run in sequence.
5. Download button appears when all agents complete.

Phase-based state machine (stored in st.session_state["m01_phase"]):
  idle          — form shown, nothing run yet
  planner_done  — Planner complete, approval UI visible, downstream blocked
  running       — approved, downstream pipeline executing
  complete      — all agents done, download available

All session state keys are prefixed with "m01_" to stay isolated from other modules.
"""

import streamlit as st
from utils.model_client import get_chain
from modules.m01_research_assistant.agents import run_planner
from modules.m01_research_assistant.pipeline import build_downstream_graph, get_initial_state
from utils.doc_builder import build_research_doc


# Agent display config: (internal_name, display_label, description)
AGENTS = [
    ("planner",    "Agent 1: Planner",    "Breaks the topic into focused research questions"),
    ("researcher", "Agent 2: Researcher", "Searches the web for evidence on each question"),
    ("critic",     "Agent 3: Critic",     "Assesses source quality and flags gaps"),
    ("writer",     "Agent 4: Writer",     "Drafts the full research paper"),
    ("editor",     "Agent 5: Editor",     "Polishes the draft and removes weak language"),
]

# Downstream agents only — used after the Planner approval checkpoint
DOWNSTREAM_AGENTS = AGENTS[1:]

AUDIENCE_OPTIONS = [
    "General business audience",
    "Board / Executive team",
    "Technical team",
    "External / Public",
    "Academic / Research audience",
]

FORMAT_OPTIONS = [
    "White Paper / Analytical",
    "Harvard Business Review",
    "Academic / Research paper",
    "Government / Policy brief",
    "McKinsey / Bain",
    "Consulting one-pager",
]

FORMAT_HINTS = {
    "White Paper / Analytical": "Analytical narrative. Explains a topic in depth. No recommendations. Best for research questions.",
    "Harvard Business Review":  "Analytical with real-world examples. Ends with practical takeaways. Best for business topics.",
    "Academic / Research paper": "Formal. Abstract, findings, discussion, references. Best for evidence-based research.",
    "Government / Policy brief": "Neutral. Issue, findings, policy options. Best for regulatory or public-policy topics.",
    "McKinsey / Bain":          "Consulting deliverable. Opens with a recommendation. Every section ends with action items.",
    "Consulting one-pager":     "Compressed executive summary. Bullet-point sections. Best for a quick briefing.",
}

LENGTH_OPTIONS = [
    "Short brief (~800 words, 1-2 pages)",
    "Standard length (~2,000 words, 4-5 pages)",
    "Full report (~4,500 words, 9-11 pages)",
]

STATUS_WAITING  = "⬜ Waiting"
STATUS_RUNNING  = "🔄 Running..."
STATUS_COMPLETE = "✅ Complete"
STATUS_FAILED   = "❌ Failed"

# Keys to clear on reset
_STATE_KEYS = [
    "m01_final", "m01_full_state", "m01_agent_outputs",
    "m01_pending_state", "m01_planner_attempt", "m01_editing",
    "m01_planner_model", "m01_inputs", "m01_phase",
]


def _agent_panel(placeholder, label: str, description: str, status: str,
                 output: str = "", model: str = "", expanded: bool = False,
                 running: bool = False) -> None:
    """Renders a single agent panel into a placeholder."""
    with placeholder.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{label}**  \n{description}")
        with col2:
            st.markdown(status)

        if running:
            st.spinner("Working...")

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
        "Five specialized AI agents run in sequence — an **agent pipeline** where each agent "
        "builds on the work of the one before it.\n\n"
        "- **Planner agent** — breaks your topic into focused research questions\n"
        "- **Researcher agent** — searches the live web for evidence on each question\n"
        "- **Critic agent** — evaluates source quality and flags gaps\n"
        "- **Writer agent** — drafts a full structured paper from the evidence\n"
        "- **Editor agent** — polishes the language and confirms the format"
    )
    st.markdown("---")

    # ── Form key — incremented by Clear to force all widgets to reset ─────────
    if "m01_form_key" not in st.session_state:
        st.session_state["m01_form_key"] = 0
    fk = st.session_state["m01_form_key"]

    phase = st.session_state.get("m01_phase", "idle")

    # ── Input form ────────────────────────────────────────────────────────────
    # Lock form while the downstream pipeline is actively running
    locked = (phase == "running")

    topic = st.text_area(
        "Research topic",
        placeholder=(
            "e.g. Impact of small modular reactors on grid reliability\n\n"
            "Add extra context here if you have it. A focused topic produces a better paper."
        ),
        help="Enter any topic. Add context if helpful. More specific = better output.",
        height=120,
        key=f"m01_topic_{fk}",
        disabled=locked,
    )

    angle = st.text_input(
        "Specific angle or focus (optional)",
        placeholder="e.g. regulatory risk, investor perspective, implementation challenges",
        help="Narrows the Planner's questions. Leave blank for a broad overview.",
        key=f"m01_angle_{fk}",
        disabled=locked,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        audience = st.selectbox(
            "Audience",
            AUDIENCE_OPTIONS,
            index=0,
            help="The paper will be written for this audience.",
            key=f"m01_audience_{fk}",
            disabled=locked,
        )
    with col_right:
        format_style = st.selectbox(
            "Format",
            FORMAT_OPTIONS,
            index=0,
            help="The structure and style of the output paper.",
            key=f"m01_format_{fk}",
            disabled=locked,
        )
        st.caption(FORMAT_HINTS.get(format_style, ""))

    length = st.selectbox(
        "Length",
        LENGTH_OPTIONS,
        index=1,
        help="Target length of the final paper.",
        key=f"m01_length_{fk}",
        disabled=locked,
    )

    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        run_clicked = st.button(
            "Run Research",
            type="primary",
            disabled=not topic.strip() or locked,
        )
    with col_clear:
        clear_clicked = st.button("Clear / New topic")

    if clear_clicked:
        st.session_state["m01_form_key"] += 1
        for key in _STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("---")
    st.markdown("⚠️ *AI-generated output. Review before use.*")
    st.markdown("---")

    # ── Agent panel placeholders ───────────────────────────────────────────────
    # The approval checkpoint placeholder sits between Planner and Researcher.
    # It is shown only in the planner_done phase and cleared otherwise.
    planner_ph   = st.empty()
    approval_ph  = st.empty()
    downstream_ph = {name: st.empty() for name, _, _ in DOWNSTREAM_AGENTS}

    all_ph = {"planner": planner_ph, **downstream_ph}

    # ── If Run is clicked, always start fresh from the Planner ────────────────
    if run_clicked and topic.strip():
        for key in _STATE_KEYS:
            st.session_state.pop(key, None)
        _start_planner(topic, angle, audience, format_style, length, planner_ph)
        # _start_planner ends with st.rerun() — nothing below executes

    # Re-read phase after potential clear (run_clicked path already returned)
    phase = st.session_state.get("m01_phase", "idle")

    # ── PHASE: idle ───────────────────────────────────────────────────────────
    if phase == "idle":
        for name, label, desc in AGENTS:
            _agent_panel(all_ph[name], label, desc, STATUS_WAITING)
        return

    # ── PHASE: planner_done ───────────────────────────────────────────────────
    if phase == "planner_done":
        pending  = st.session_state.get("m01_pending_state", {})
        questions = pending.get("questions", [])
        p_model  = st.session_state.get("m01_planner_model", "")
        attempt  = st.session_state.get("m01_planner_attempt", 1)
        editing  = st.session_state.get("m01_editing", False)

        # Show Planner panel as Complete with questions expanded so user can read them
        attempt_note = f" · attempt {attempt}" if attempt > 1 else ""
        planner_out  = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        _agent_panel(
            planner_ph, "Agent 1: Planner",
            "Breaks the topic into focused research questions",
            STATUS_COMPLETE, output=planner_out,
            model=p_model + attempt_note, expanded=True,
        )

        # Approval checkpoint
        with approval_ph.container():
            if not editing:
                st.info(
                    "Review the research questions above. "
                    "Approve to continue, or edit them to redirect the pipeline."
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Approve and continue →", type="primary"):
                        st.session_state["m01_phase"] = "running"
                        st.rerun()
                with col2:
                    if st.button("Edit questions"):
                        st.session_state["m01_editing"] = True
                        st.rerun()
                st.caption("The Researcher will not start until you approve.")
            else:
                # Edit mode — free text box, one question per line
                st.markdown("**Edit the questions below.** One per line.")
                st.caption(
                    "Reword, delete, add, or redirect questions. "
                    "You can also write a plain note about the direction you want. "
                    "The Planner will use your input to regenerate."
                )
                st.text_area(
                    "Research questions",
                    value="\n".join(questions),
                    height=200,
                    key="m01_edit_area",
                    label_visibility="collapsed",
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Replan with my edits →", type="primary"):
                        user_edits = st.session_state.get("m01_edit_area", "\n".join(questions))
                        new_attempt = attempt + 1
                        st.session_state["m01_planner_attempt"] = new_attempt
                        st.session_state["m01_editing"] = False
                        inputs = st.session_state.get("m01_inputs", {})
                        chain  = get_chain(st.session_state)
                        state  = get_initial_state(
                            inputs.get("topic", ""),
                            inputs.get("audience", "General business audience"),
                            inputs.get("format_style", "McKinsey / Bain"),
                            inputs.get("length", "Standard length (~2,000 words, 4-5 pages)"),
                            inputs.get("angle", ""),
                        )
                        result = run_planner(state, chain, user_edits=user_edits)
                        pending.update(result)
                        st.session_state["m01_pending_state"] = pending
                        st.session_state["m01_planner_model"]  = result.get("model_used", "")
                        st.session_state["m01_phase"] = "planner_done"
                        st.rerun()
                with col2:
                    if st.button("Cancel"):
                        st.session_state["m01_editing"] = False
                        st.rerun()

        # Downstream agents shown as Waiting while approval is pending
        for name, label, desc in DOWNSTREAM_AGENTS:
            _agent_panel(downstream_ph[name], label, desc, STATUS_WAITING)
        return

    # ── PHASE: running ────────────────────────────────────────────────────────
    if phase == "running":
        pending  = st.session_state.get("m01_pending_state", {})
        questions = pending.get("questions", [])
        p_model  = st.session_state.get("m01_planner_model", "")
        attempt  = st.session_state.get("m01_planner_attempt", 1)

        # Planner panel locked as Complete
        attempt_note = f" · attempt {attempt}" if attempt > 1 else ""
        planner_out  = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        _agent_panel(
            planner_ph, "Agent 1: Planner",
            "Breaks the topic into focused research questions",
            STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note,
        )

        # Clear the approval area — pipeline is now running
        approval_ph.empty()

        # Initialize downstream panels as Waiting
        for name, label, desc in DOWNSTREAM_AGENTS:
            _agent_panel(downstream_ph[name], label, desc, STATUS_WAITING)

        # Mark first downstream agent as Running
        first_name, first_label, first_desc = DOWNSTREAM_AGENTS[0]
        _agent_panel(downstream_ph[first_name], first_label, first_desc,
                     STATUS_RUNNING, running=True)

        chain         = get_chain(st.session_state)
        app           = build_downstream_graph(chain)
        full_state    = dict(pending)
        agent_outputs = {"planner": {"output": planner_out, "model": p_model}}
        current_index = 0

        try:
            for chunk in app.stream(full_state):
                node_name = list(chunk.keys())[0]
                updated   = chunk[node_name]
                full_state.update(updated)

                agent_cfg = next((a for a in DOWNSTREAM_AGENTS if a[0] == node_name), None)
                if not agent_cfg:
                    continue
                _, label, desc = agent_cfg

                output = _format_agent_output(node_name, updated, full_state)
                model  = updated.get("model_used", "")
                agent_outputs[node_name] = {"output": output, "model": model}

                _agent_panel(downstream_ph[node_name], label, desc,
                             STATUS_COMPLETE, output=output, model=model, expanded=True)

                current_index += 1
                if current_index < len(DOWNSTREAM_AGENTS):
                    next_name, next_label, next_desc = DOWNSTREAM_AGENTS[current_index]
                    _agent_panel(downstream_ph[next_name], next_label, next_desc,
                                 STATUS_RUNNING, running=True)

        except Exception as e:
            for i in range(current_index, len(DOWNSTREAM_AGENTS)):
                name, label, desc = DOWNSTREAM_AGENTS[i]
                _agent_panel(downstream_ph[name], label, desc, STATUS_FAILED)
            st.error(f"Pipeline stopped: {e}")
            return

        # Save results and transition to complete
        st.session_state["m01_final"]         = full_state.get("final", "")
        st.session_state["m01_full_state"]    = full_state
        st.session_state["m01_agent_outputs"] = agent_outputs
        st.session_state["m01_phase"]         = "complete"

        _show_sources()
        _show_download()
        return

    # ── PHASE: complete — restore saved results on re-render ─────────────────
    if phase == "complete":
        saved = st.session_state.get("m01_agent_outputs", {})
        for name, label, desc in AGENTS:
            out   = saved.get(name, {}).get("output", "")
            model = saved.get(name, {}).get("model", "")
            _agent_panel(all_ph[name], label, desc, STATUS_COMPLETE,
                         output=out, model=model)
        _show_sources()
        _show_download()
        return


def _start_planner(topic, angle, audience, format_style, length, planner_ph) -> None:
    """
    Runs the Planner agent for the first time and stores results in session_state.
    Called when Run is clicked. Ends with st.rerun() to enter the planner_done phase.
    """
    _, label, desc = AGENTS[0]
    _agent_panel(planner_ph, label, desc, STATUS_RUNNING, running=True)

    chain = get_chain(st.session_state)
    state = get_initial_state(topic, audience, format_style, length, angle)

    result = run_planner(state, chain)

    # Store the user's inputs so replan calls can reconstruct state
    st.session_state["m01_inputs"] = {
        "topic": topic, "angle": angle, "audience": audience,
        "format_style": format_style, "length": length,
    }

    # pending_state carries all fields through the approval checkpoint into the downstream graph
    pending = dict(state)
    pending.update(result)

    st.session_state["m01_pending_state"]    = pending
    st.session_state["m01_planner_model"]    = result.get("model_used", "")
    st.session_state["m01_planner_attempt"]  = 1
    st.session_state["m01_editing"]          = False
    st.session_state["m01_agent_outputs"]    = {}
    st.session_state["m01_phase"]            = "planner_done"

    st.rerun()


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
        return updated.get("final", "")

    return ""


def _show_sources() -> None:
    """Renders a collapsible sources section — collapsed by default."""
    full_state = st.session_state.get("m01_full_state", {})
    sources    = full_state.get("sources", [])
    if not sources:
        return
    st.markdown("---")
    with st.expander(f"Sources ({len(sources)} URLs)", expanded=False):
        for i, url in enumerate(sources, 1):
            st.markdown(f"{i}. {url}")


def _show_download() -> None:
    """Renders the download section after a successful run."""
    full_state = st.session_state.get("m01_full_state", {})
    topic      = full_state.get("topic", "research")
    model      = full_state.get("model_used", "unknown")

    st.markdown("---")
    st.success("Research complete.")
    st.caption(f"Final model: {model}")

    slug     = topic.lower()[:40].replace(" ", "-").replace("/", "-")
    slug     = "".join(c for c in slug if c.isalnum() or c == "-")
    filename = f"research-{slug}-v1.docx"

    doc_bytes = build_research_doc(full_state)
    st.download_button(
        label="Download Word document",
        data=doc_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
