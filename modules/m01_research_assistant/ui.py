"""
Streamlit UI for the Research Assistant module.

Phase-based state machine (stored in st.session_state["m01_phase"]):
  idle          — form shown, nothing run yet
  planner_done  — Planner complete, approval UI visible, downstream blocked
  running       — approved; runs Researcher + retry loop inline
  quality_gate  — retry cap reached, still weak sources; user decides to proceed or stop
  writing       — runs Critic → Writer → Editor (entered from quality_gate approval)
  complete      — all agents done, download available

Researcher quality gate — two-pass check:
  Pass 1 (objective): flag_weak_questions() — zero results or all low-authority domains.
  Pass 2 (LLM):      flag_irrelevant_questions() — binary YES/NO per source for questions
                     that passed Pass 1. Flags if fewer than half the results answer YES.
  Combined flagged list drives the retry loop (>2 = retry, max 2 retries).
  If still >2 after retries, pause at quality_gate for user decision.

All session state keys are prefixed with "m01_" to stay isolated from other modules.
"""

import streamlit as st
from utils.model_client import get_chain
from modules.m01_research_assistant.agents import (
    run_planner, run_researcher, run_critic, run_writer, run_editor,
    flag_weak_questions, flag_irrelevant_questions,
)
from modules.m01_research_assistant.pipeline import get_initial_state
from utils.doc_builder import build_research_doc


AGENTS = [
    ("planner",    "Agent 1: Planner",    "Breaks the topic into focused research questions"),
    ("researcher", "Agent 2: Researcher", "Searches the web for evidence on each question"),
    ("critic",     "Agent 3: Critic",     "Assesses source quality and flags gaps"),
    ("writer",     "Agent 4: Writer",     "Drafts the full research paper"),
    ("editor",     "Agent 5: Editor",     "Polishes the draft and removes weak language"),
]

DOWNSTREAM_AGENTS = AGENTS[1:]   # researcher through editor
WRITING_AGENTS    = AGENTS[2:]   # critic, writer, editor — used in writing phase

MAX_RESEARCHER_RETRIES = 2   # max re-search attempts after the initial run

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

_STATE_KEYS = [
    "m01_final", "m01_full_state", "m01_agent_outputs",
    "m01_pending_state", "m01_planner_attempt", "m01_editing",
    "m01_planner_model", "m01_inputs", "m01_phase",
    "m01_flagged_questions", "m01_researcher_attempt",
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

    if "m01_form_key" not in st.session_state:
        st.session_state["m01_form_key"] = 0
    fk = st.session_state["m01_form_key"]

    phase  = st.session_state.get("m01_phase", "idle")
    locked = phase in ("running", "writing")

    # ── Input form ────────────────────────────────────────────────────────────
    topic = st.text_area(
        "Research topic",
        placeholder=(
            "e.g. Impact of small modular reactors on grid reliability\n\n"
            "Add extra context here if you have it. A focused topic produces a better paper."
        ),
        height=120,
        key=f"m01_topic_{fk}",
        disabled=locked,
    )

    angle = st.text_input(
        "Specific angle or focus (optional)",
        placeholder="e.g. regulatory risk, investor perspective, implementation challenges",
        key=f"m01_angle_{fk}",
        disabled=locked,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        audience = st.selectbox(
            "Audience", AUDIENCE_OPTIONS, index=0,
            key=f"m01_audience_{fk}", disabled=locked,
        )
    with col_right:
        format_style = st.selectbox(
            "Format", FORMAT_OPTIONS, index=0,
            key=f"m01_format_{fk}", disabled=locked,
        )
        st.caption(FORMAT_HINTS.get(format_style, ""))

    length = st.selectbox(
        "Length", LENGTH_OPTIONS, index=1,
        key=f"m01_length_{fk}", disabled=locked,
    )

    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        run_clicked = st.button(
            "Run Research", type="primary",
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

    # ── Placeholders ─────────────────────────────────────────────────────────
    # Layout order:
    #   planner panel
    #   approval_ph     ← Planner approval checkpoint
    #   researcher panel
    #   quality_gate_ph ← Researcher quality gate (shown only when needed)
    #   critic panel
    #   writer panel
    #   editor panel
    planner_ph      = st.empty()
    approval_ph     = st.empty()
    researcher_ph   = st.empty()
    quality_gate_ph = st.empty()
    writing_ph      = {name: st.empty() for name, _, _ in WRITING_AGENTS}

    all_ph = {
        "planner":    planner_ph,
        "researcher": researcher_ph,
        **writing_ph,
    }

    # ── Run clicked — always start fresh ─────────────────────────────────────
    if run_clicked and topic.strip():
        for key in _STATE_KEYS:
            st.session_state.pop(key, None)
        _start_planner(topic, angle, audience, format_style, length, planner_ph)

    phase = st.session_state.get("m01_phase", "idle")

    # ── PHASE: idle ───────────────────────────────────────────────────────────
    if phase == "idle":
        for name, label, desc in AGENTS:
            _agent_panel(all_ph[name], label, desc, STATUS_WAITING)
        return

    # ── PHASE: planner_done ───────────────────────────────────────────────────
    if phase == "planner_done":
        pending   = st.session_state.get("m01_pending_state", {})
        questions = pending.get("questions", [])
        p_model   = st.session_state.get("m01_planner_model", "")
        attempt   = st.session_state.get("m01_planner_attempt", 1)
        editing   = st.session_state.get("m01_editing", False)

        attempt_note = f" · attempt {attempt}" if attempt > 1 else ""
        planner_out  = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        _agent_panel(
            planner_ph, "Agent 1: Planner",
            "Breaks the topic into focused research questions",
            STATUS_COMPLETE, output=planner_out,
            model=p_model + attempt_note, expanded=True,
        )

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
                        user_edits  = st.session_state.get("m01_edit_area", "\n".join(questions))
                        new_attempt = attempt + 1
                        st.session_state["m01_planner_attempt"] = new_attempt
                        st.session_state["m01_editing"] = False
                        inputs = st.session_state.get("m01_inputs", {})
                        chain  = get_chain(st.session_state)
                        state  = get_initial_state(
                            inputs.get("topic", ""),
                            inputs.get("audience", "General business audience"),
                            inputs.get("format_style", "White Paper / Analytical"),
                            inputs.get("length", "Standard length (~2,000 words, 4-5 pages)"),
                            inputs.get("angle", ""),
                        )
                        result = run_planner(state, chain, user_edits=user_edits)
                        pending.update(result)
                        st.session_state["m01_pending_state"] = pending
                        st.session_state["m01_planner_model"] = result.get("model_used", "")
                        st.session_state["m01_phase"] = "planner_done"
                        st.rerun()
                with col2:
                    if st.button("Cancel"):
                        st.session_state["m01_editing"] = False
                        st.rerun()

        for name, label, desc in DOWNSTREAM_AGENTS:
            _agent_panel(all_ph[name], label, desc, STATUS_WAITING)
        return

    # ── PHASE: running ────────────────────────────────────────────────────────
    if phase == "running":
        pending       = st.session_state.get("m01_pending_state", {})
        questions     = pending.get("questions", [])
        p_model       = st.session_state.get("m01_planner_model", "")
        planner_attempt = st.session_state.get("m01_planner_attempt", 1)

        attempt_note = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out  = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        _agent_panel(
            planner_ph, "Agent 1: Planner",
            "Breaks the topic into focused research questions",
            STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note,
        )
        approval_ph.empty()

        for name, label, desc in DOWNSTREAM_AGENTS:
            _agent_panel(all_ph[name], label, desc, STATUS_WAITING)

        full_state    = dict(pending)
        agent_outputs = {"planner": {"output": planner_out, "model": p_model}}
        chain         = get_chain(st.session_state)

        # ── Step 1: Researcher (first pass) ───────────────────────────────────
        _agent_panel(
            researcher_ph, "Agent 2: Researcher",
            "Searching the web for evidence on each question",
            STATUS_RUNNING, running=True,
        )
        try:
            result = run_researcher(full_state)
            full_state.update(result)
        except Exception as e:
            _agent_panel(researcher_ph, "Agent 2: Researcher", "", STATUS_FAILED)
            for name, label, desc in WRITING_AGENTS:
                _agent_panel(writing_ph[name], label, desc, STATUS_FAILED)
            st.error(f"Researcher failed: {e}")
            return

        researcher_out = _format_researcher_output(full_state)
        researcher_model = full_state.get("model_used", "")
        agent_outputs["researcher"] = {"output": researcher_out, "model": researcher_model}
        _agent_panel(
            researcher_ph, "Agent 2: Researcher",
            "Searches the web for evidence on each question",
            STATUS_COMPLETE, output=researcher_out, model=researcher_model, expanded=True,
        )

        # ── Step 2: Quality check and retry loop ──────────────────────────────
        flagged = _combined_flag_check(full_state, chain, researcher_ph, quality_gate_ph,
                                       researcher_out, researcher_model)
        researcher_attempt = 1

        while len(flagged) > 2 and researcher_attempt <= MAX_RESEARCHER_RETRIES:
            researcher_attempt += 1
            _agent_panel(
                researcher_ph, "Agent 2: Researcher",
                f"Re-searching {len(flagged)} weak question(s) — attempt {researcher_attempt} of {MAX_RESEARCHER_RETRIES + 1}",
                STATUS_RUNNING, running=True,
            )
            try:
                result = run_researcher(full_state, target_questions=flagged)
                full_state.update(result)
            except Exception as e:
                break  # if retry fails, proceed with what we have

            researcher_out = _format_researcher_output(full_state)
            agent_outputs["researcher"] = {"output": researcher_out, "model": researcher_model}
            _agent_panel(
                researcher_ph, "Agent 2: Researcher",
                "Searches the web for evidence on each question",
                STATUS_COMPLETE, output=researcher_out,
                model=f"{researcher_model} · {researcher_attempt} attempts", expanded=True,
            )
            flagged = _combined_flag_check(full_state, chain, researcher_ph, quality_gate_ph,
                                           researcher_out,
                                           f"{researcher_model} · {researcher_attempt} attempts")

        # ── Step 3: Quality gate if still weak after retries ──────────────────
        if len(flagged) > 2:
            st.session_state["m01_pending_state"]    = full_state
            st.session_state["m01_flagged_questions"] = flagged
            st.session_state["m01_agent_outputs"]    = agent_outputs
            st.session_state["m01_researcher_attempt"] = researcher_attempt
            st.session_state["m01_phase"] = "quality_gate"
            st.rerun()
            return

        # ── Step 4: Run Critic → Writer → Editor ──────────────────────────────
        _run_writing_agents(
            full_state, agent_outputs, chain,
            planner_ph, researcher_ph, quality_gate_ph, writing_ph,
            planner_out, p_model, attempt_note, researcher_out, researcher_model,
        )
        return

    # ── PHASE: quality_gate ───────────────────────────────────────────────────
    if phase == "quality_gate":
        pending           = st.session_state.get("m01_pending_state", {})
        flagged           = st.session_state.get("m01_flagged_questions", [])
        questions         = pending.get("questions", [])
        p_model           = st.session_state.get("m01_planner_model", "")
        planner_attempt   = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs     = st.session_state.get("m01_agent_outputs", {})
        r_attempt         = st.session_state.get("m01_researcher_attempt", 1)

        attempt_note = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out  = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        _agent_panel(
            planner_ph, "Agent 1: Planner",
            "Breaks the topic into focused research questions",
            STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note,
        )
        approval_ph.empty()

        researcher_out   = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model = agent_outputs.get("researcher", {}).get("model", "")
        _agent_panel(
            researcher_ph, "Agent 2: Researcher",
            f"Completed {r_attempt + 1} search attempt(s)",
            STATUS_COMPLETE, output=researcher_out,
            model=f"{researcher_model} · {r_attempt + 1} attempts",
        )

        with quality_gate_ph.container():
            st.warning(
                f"After {r_attempt + 1} search attempts, **{len(flagged)} question(s)** still have "
                "weak or no sources. The Writer will note these gaps explicitly in the paper."
            )
            st.markdown("**Questions with weak sources:**")
            for q in flagged:
                st.markdown(f"- {q[:100]}")
            st.markdown("")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Proceed to Writer →", type="primary"):
                    st.session_state["m01_phase"] = "writing"
                    st.rerun()
            with col2:
                if st.button("Stop here"):
                    for key in _STATE_KEYS:
                        st.session_state.pop(key, None)
                    st.session_state["m01_form_key"] = st.session_state.get("m01_form_key", 0)
                    st.rerun()

        for name, label, desc in WRITING_AGENTS:
            _agent_panel(writing_ph[name], label, desc, STATUS_WAITING)
        return

    # ── PHASE: writing ────────────────────────────────────────────────────────
    if phase == "writing":
        pending           = st.session_state.get("m01_pending_state", {})
        questions         = pending.get("questions", [])
        p_model           = st.session_state.get("m01_planner_model", "")
        planner_attempt   = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs     = st.session_state.get("m01_agent_outputs", {})
        r_attempt         = st.session_state.get("m01_researcher_attempt", 1)

        attempt_note     = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out      = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        researcher_out   = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model = agent_outputs.get("researcher", {}).get("model", "")
        full_state       = dict(pending)
        chain            = get_chain(st.session_state)

        _agent_panel(
            planner_ph, "Agent 1: Planner",
            "Breaks the topic into focused research questions",
            STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note,
        )
        approval_ph.empty()
        _agent_panel(
            researcher_ph, "Agent 2: Researcher",
            f"Completed {r_attempt + 1} search attempt(s)",
            STATUS_COMPLETE, output=researcher_out,
            model=f"{researcher_model} · {r_attempt + 1} attempts",
        )
        quality_gate_ph.empty()

        for name, label, desc in WRITING_AGENTS:
            _agent_panel(writing_ph[name], label, desc, STATUS_WAITING)

        _run_writing_agents(
            full_state, agent_outputs, chain,
            planner_ph, researcher_ph, quality_gate_ph, writing_ph,
            planner_out, p_model, attempt_note, researcher_out, researcher_model,
        )
        return

    # ── PHASE: complete ───────────────────────────────────────────────────────
    if phase == "complete":
        saved = st.session_state.get("m01_agent_outputs", {})
        for name, label, desc in AGENTS:
            out   = saved.get(name, {}).get("output", "")
            model = saved.get(name, {}).get("model", "")
            _agent_panel(all_ph[name], label, desc, STATUS_COMPLETE, output=out, model=model)
        _show_sources()
        _show_download()
        return


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combined_flag_check(full_state: dict, chain, researcher_ph, quality_gate_ph,
                         researcher_out: str, researcher_model_label: str) -> list[str]:
    """
    Two-pass quality check on research results.

    Pass 1 — objective domain check (flag_weak_questions):
      Flags questions with zero results or all results from low-authority domains.
      Fast, free, no LLM involved.

    Pass 2 — LLM relevance check (flag_irrelevant_questions):
      ONE LLM call for all questions (batched). 20-second timeout — skipped if slow.
      Only checks questions that passed Pass 1.
      Shows a visible status in quality_gate_ph while running.

    Returns the combined list of flagged question strings.
    quality_gate_ph is cleared before returning so the caller can reuse it.
    """
    research = full_state.get("research", {})

    # Pass 1: objective — instant
    domain_flagged = flag_weak_questions(research)

    # Show visible status in the space between Researcher and Critic
    with quality_gate_ph.container():
        st.info("🔍 Checking source relevance... (this takes a few seconds)")

    # Pass 2: LLM relevance — one batched call, 20s timeout
    llm_flagged = flag_irrelevant_questions(research, chain, skip=domain_flagged)

    # Clear the status — caller will populate quality_gate_ph only if gate triggers
    quality_gate_ph.empty()

    combined = domain_flagged + [q for q in llm_flagged if q not in domain_flagged]
    return combined


def _start_planner(topic, angle, audience, format_style, length, planner_ph) -> None:
    """Runs the Planner and stores results. Ends with st.rerun() into planner_done."""
    _, label, desc = AGENTS[0]
    _agent_panel(planner_ph, label, desc, STATUS_RUNNING, running=True)

    chain = get_chain(st.session_state)
    state = get_initial_state(topic, audience, format_style, length, angle)
    result = run_planner(state, chain)

    st.session_state["m01_inputs"] = {
        "topic": topic, "angle": angle, "audience": audience,
        "format_style": format_style, "length": length,
    }
    pending = dict(state)
    pending.update(result)

    st.session_state["m01_pending_state"]   = pending
    st.session_state["m01_planner_model"]   = result.get("model_used", "")
    st.session_state["m01_planner_attempt"] = 1
    st.session_state["m01_editing"]         = False
    st.session_state["m01_agent_outputs"]   = {}
    st.session_state["m01_phase"]           = "planner_done"
    st.rerun()


def _run_writing_agents(
    full_state, agent_outputs, chain,
    planner_ph, researcher_ph, quality_gate_ph, writing_ph,
    planner_out, p_model, attempt_note, researcher_out, researcher_model,
) -> None:
    """
    Runs Critic → Writer → Editor in sequence, updating panels after each.
    Called from both the running phase (direct path) and writing phase (after quality gate).
    Saves results and sets phase to complete when done.
    """
    quality_gate_ph.empty()

    _agent_panel(
        writing_ph["critic"], "Agent 3: Critic",
        "Assesses source quality and flags gaps",
        STATUS_RUNNING, running=True,
    )

    agents_to_run = [
        ("critic",  lambda: run_critic(full_state, chain),  "Agent 3: Critic",  "Assesses source quality and flags gaps"),
        ("writer",  lambda: run_writer(full_state, chain),  "Agent 4: Writer",  "Drafts the full research paper"),
        ("editor",  lambda: run_editor(full_state, chain),  "Agent 5: Editor",  "Polishes the draft and removes weak language"),
    ]

    current_index = 0
    try:
        for agent_name, agent_fn, label, desc in agents_to_run:
            result = agent_fn()
            full_state.update(result)

            output = _format_agent_output(agent_name, result, full_state)
            model  = result.get("model_used", "")
            agent_outputs[agent_name] = {"output": output, "model": model}

            _agent_panel(writing_ph[agent_name], label, desc,
                         STATUS_COMPLETE, output=output, model=model, expanded=True)

            current_index += 1
            if current_index < len(agents_to_run):
                next_name, _, next_label, next_desc = agents_to_run[current_index]
                _agent_panel(writing_ph[next_name], next_label, next_desc,
                             STATUS_RUNNING, running=True)

    except Exception as e:
        for i in range(current_index, len(agents_to_run)):
            fail_name, _, fail_label, fail_desc = agents_to_run[i]
            _agent_panel(writing_ph[fail_name], fail_label, fail_desc, STATUS_FAILED)
        st.error(f"Pipeline stopped: {e}")
        return

    st.session_state["m01_final"]         = full_state.get("final", "")
    st.session_state["m01_full_state"]    = full_state
    st.session_state["m01_agent_outputs"] = agent_outputs
    st.session_state["m01_phase"]         = "complete"

    _show_sources()
    _show_download()


def _format_researcher_output(full_state: dict) -> str:
    """Builds the Researcher panel summary text."""
    research = full_state.get("research", {})
    sources  = full_state.get("sources", [])
    lines = [f"**{len(sources)} sources collected across {len(research)} questions**\n"]
    for q, hits in research.items():
        lines.append(f"- *{q[:70]}...* — {len(hits)} results")
    return "\n".join(lines)


def _format_agent_output(node_name: str, updated: dict, full_state: dict) -> str:
    """Returns a readable summary of what each agent produced."""
    if node_name == "researcher":
        return _format_researcher_output(full_state)
    elif node_name == "critic":
        return updated.get("critique", "")
    elif node_name == "writer":
        draft = updated.get("draft", "")
        return f"*Draft: {len(draft):,} characters*\n\n" + draft[:600] + "..."
    elif node_name == "editor":
        return updated.get("final", "")
    return ""


def _show_sources() -> None:
    full_state = st.session_state.get("m01_full_state", {})
    sources    = full_state.get("sources", [])
    if not sources:
        return
    st.markdown("---")
    with st.expander(f"Sources ({len(sources)} URLs)", expanded=False):
        for i, url in enumerate(sources, 1):
            st.markdown(f"{i}. {url}")


def _show_download() -> None:
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
