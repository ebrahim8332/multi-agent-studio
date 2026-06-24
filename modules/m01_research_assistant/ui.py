"""
Streamlit UI for the Research Assistant module.

Phase state machine (m01_phase in session_state):
  idle           — form shown, nothing run yet
  planner_done   — Planner complete, approval checkpoint
  running        — Researcher + quality gate running
  quality_gate   — retry cap reached, still weak; user decides to proceed or stop
  critic_running — Critic agent running (entered from running or quality_gate approval)
  critic_done    — Critic complete, approval checkpoint before Writer
  writing        — Writer + Editor running
  complete       — all done, download available

Researcher quality gate (two-pass, in running phase):
  Pass 1: flag_weak_questions() — zero results or all low-authority domains (no LLM)
  Pass 2: flag_irrelevant_questions() — one batched LLM call, YES/NO per question
  Combined > 2 flagged → retry (max 2x). Still > 2 → quality_gate checkpoint.

All session state keys are prefixed "m01_" to stay isolated from other modules.
Token usage is accumulated in "m01_call_log" by the model chain.
"""

import re
import streamlit as st
import streamlit.components.v1 as components
from utils.model_client import get_chain, APPROX_PRICING
from modules.m01_research_assistant.agents import (
    run_planner, run_researcher, run_critic, run_writer, run_judge, run_editor,
    flag_weak_questions, flag_irrelevant_questions,
)
from modules.m01_research_assistant.pipeline import get_initial_state
from utils.doc_builder import build_research_doc


AGENTS = [
    ("planner",    "Agent 1: Planner",    "Breaks the topic into focused research questions"),
    ("researcher", "Agent 2: Researcher", "Searches the web for evidence on each question"),
    ("critic",     "Agent 3: Critic",     "Assesses source quality and flags gaps"),
    ("writer",     "Agent 4: Writer",     "Drafts the full research paper"),
    ("judge",      "Agent 5: Judge",      "Evaluates draft quality before the Editor runs"),
    ("editor",     "Agent 6: Editor",     "Polishes the draft and removes weak language"),
]

MAX_RESEARCHER_RETRIES = 2
MAX_WRITER_RETRIES     = 2

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
    "White Paper / Analytical":  "Analytical narrative. Explains a topic in depth. No recommendations. Best for research questions.",
    "Harvard Business Review":   "Analytical with real-world examples. Ends with practical takeaways. Best for business topics.",
    "Academic / Research paper": "Formal. Abstract, findings, discussion, references. Best for evidence-based research.",
    "Government / Policy brief": "Neutral. Issue, findings, policy options. Best for regulatory or public-policy topics.",
    "McKinsey / Bain":           "Consulting deliverable. Opens with a recommendation. Every section ends with action items.",
    "Consulting one-pager":      "Compressed executive summary. Bullet-point sections. Best for a quick briefing.",
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
    "m01_planner_model", "m01_planner_prompt", "m01_inputs", "m01_phase",
    "m01_flagged_questions", "m01_researcher_attempt",
    "m01_call_log",
    "m01_writer_attempt", "m01_writer_feedback", "m01_judge_editing",
    "m01_judge_result",
]


# ── Panel renderer ────────────────────────────────────────────────────────────

def _agent_panel(placeholder, label: str, description: str, status: str,
                 output: str = "", model: str = "", expanded: bool = False,
                 running: bool = False, prompt: list = None) -> None:
    """Renders a single agent panel into a placeholder.

    prompt: list of message dicts (system/user). When provided, shows a
    'View prompt sent to AI' expander so the user can inspect what the
    model actually received.
    """
    with placeholder.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{label}**  \n{description}")
        with col2:
            st.markdown(status)
        if running:
            st.caption("⏳ Working...")
            components.html(
                """
                <script>
                    // Scroll the parent page to this iframe's position.
                    // Iterates iframes and matches by contentWindow — no cross-origin access needed.
                    var frames = window.parent.document.querySelectorAll('iframe');
                    for (var i = 0; i < frames.length; i++) {
                        try {
                            if (frames[i].contentWindow === window) {
                                frames[i].scrollIntoView({behavior: 'smooth', block: 'center'});
                                break;
                            }
                        } catch(e) {}
                    }
                </script>
                """,
                height=0,
            )
        if output:
            with st.expander("View output", expanded=expanded):
                st.markdown(output)
                if model:
                    st.caption(f"Model: {model}")
        if prompt:
            with st.expander("🔍 View prompt sent to AI", expanded=False):
                for msg in prompt:
                    role    = msg.get("role", "").upper()
                    content = msg.get("content", "")
                    st.caption(f"── {role} ──")
                    display = (
                        content if len(content) <= 3000
                        else content[:3000] + "\n\n… [truncated — full prompt is longer]"
                    )
                    st.code(display, language=None)
        st.divider()


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.title("📝 Research Assistant")
    st.caption("Six agents research, critique, write, judge, and edit a structured paper.")
    st.markdown(
        "Six specialized AI agents run in sequence — an **agent pipeline** where each agent "
        "builds on the work of the one before it. "
        "✋ marks a human checkpoint where you review and approve before the next agent runs.\n\n"
        "- **Planner agent** — breaks your topic into focused research questions ✋\n"
        "- **Researcher agent** — searches the live web for evidence on each question\n"
        "- **Critic agent** — evaluates source quality and flags gaps ✋\n"
        "- **Writer agent** — drafts a full structured paper from the evidence\n"
        "- **Judge agent** — scores the draft on four quality dimensions ✋\n"
        "- **Editor agent** — polishes the language and confirms the format"
    )
    st.markdown("---")

    st.markdown("""
<style>
div[data-baseweb="select"] { cursor: pointer; }
div[data-baseweb="select"] * { cursor: pointer; }

/* Normalize heading sizes in paper output so h1/h2/h3 are not dramatically different */
.stMarkdown h1 { font-size: 1.35rem !important; font-weight: 700 !important; margin-top: 1.2rem !important; }
.stMarkdown h2 { font-size: 1.20rem !important; font-weight: 700 !important; margin-top: 1.0rem !important; }
.stMarkdown h3 { font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

    if "m01_form_key" not in st.session_state:
        st.session_state["m01_form_key"] = 0
    fk = st.session_state["m01_form_key"]

    phase  = st.session_state.get("m01_phase", "idle")
    locked = phase in ("running", "critic_running", "writing", "judge_running", "editor_running")

    # ── Input form ────────────────────────────────────────────────────────────
    topic = st.text_area(
        "Research topic",
        placeholder=(
            "e.g. Impact of small modular reactors on grid reliability\n\n"
            "Add extra context here if you have it. A focused topic produces a better paper."
        ),
        height=120, key=f"m01_topic_{fk}", disabled=locked,
    )
    angle = st.text_input(
        "Specific angle or focus (optional)",
        placeholder="e.g. regulatory risk, investor perspective, implementation challenges",
        key=f"m01_angle_{fk}", disabled=locked,
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

    # ── Placeholders (rendered top to bottom in layout order) ─────────────────
    planner_ph      = st.empty()
    approval_ph     = st.empty()
    researcher_ph   = st.empty()
    quality_gate_ph = st.empty()
    critic_ph       = st.empty()
    critic_gate_ph  = st.empty()   # checkpoint between Critic and Writer
    writer_ph       = st.empty()
    judge_ph        = st.empty()
    judge_gate_ph   = st.empty()   # checkpoint between Judge and Editor
    editor_ph       = st.empty()

    ph = {
        "planner":    planner_ph,
        "researcher": researcher_ph,
        "critic":     critic_ph,
        "writer":     writer_ph,
        "judge":      judge_ph,
        "editor":     editor_ph,
    }

    # ── Run clicked ───────────────────────────────────────────────────────────
    if run_clicked and topic.strip():
        for key in _STATE_KEYS:
            st.session_state.pop(key, None)
        _start_planner(topic, angle, audience, format_style, length, planner_ph)

    phase = st.session_state.get("m01_phase", "idle")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: idle
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "idle":
        for name, label, desc in AGENTS:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: planner_done
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "planner_done":
        pending   = st.session_state.get("m01_pending_state", {})
        questions = pending.get("questions", [])
        p_model   = st.session_state.get("m01_planner_model", "")
        p_prompt  = st.session_state.get("m01_planner_prompt", [])
        attempt   = st.session_state.get("m01_planner_attempt", 1)
        editing   = st.session_state.get("m01_editing", False)

        attempt_note = f" · attempt {attempt}" if attempt > 1 else ""
        planner_out  = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        _agent_panel(
            planner_ph, "Agent 1: Planner",
            "Breaks the topic into focused research questions",
            STATUS_COMPLETE, output=planner_out,
            model=p_model + attempt_note, expanded=True, prompt=p_prompt,
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
                    "Research questions", value="\n".join(questions),
                    height=200, key="m01_edit_area", label_visibility="collapsed",
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
                        st.session_state["m01_pending_state"]   = pending
                        st.session_state["m01_planner_model"]   = result.get("model_used", "")
                        st.session_state["m01_planner_prompt"]  = result.get("prompt_sent", [])
                        st.session_state["m01_phase"] = "planner_done"
                        st.rerun()
                with col2:
                    if st.button("Cancel"):
                        st.session_state["m01_editing"] = False
                        st.rerun()

        for name, label, desc in AGENTS[1:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: running  (Researcher + quality gate)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "running":
        pending  = st.session_state.get("m01_pending_state", {})
        questions = pending.get("questions", [])
        p_model   = st.session_state.get("m01_planner_model", "")
        p_prompt  = st.session_state.get("m01_planner_prompt", [])
        planner_attempt = st.session_state.get("m01_planner_attempt", 1)
        attempt_note = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out  = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

        _agent_panel(
            planner_ph, "Agent 1: Planner",
            "Breaks the topic into focused research questions",
            STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note, prompt=p_prompt,
        )
        approval_ph.empty()

        # Show all downstream panels as Waiting
        for name, label, desc in AGENTS[1:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)

        full_state    = dict(pending)
        agent_outputs = st.session_state.get("m01_agent_outputs", {"planner": {"output": planner_out, "model": p_model, "prompt": p_prompt}})
        chain         = get_chain(st.session_state)

        # ── Researcher ────────────────────────────────────────────────────────
        _agent_panel(
            researcher_ph, "Agent 2: Researcher",
            "Searching the web for evidence on each question",
            STATUS_RUNNING, running=True,
        )
        try:
            with st.spinner("Searching the web... (this takes 10–20 seconds)"):
                result = run_researcher(full_state)
            full_state.update(result)
        except Exception as e:
            _agent_panel(researcher_ph, "Agent 2: Researcher", "", STATUS_FAILED)
            for name, label, desc in AGENTS[2:]:
                _agent_panel(ph[name], label, desc, STATUS_FAILED)
            st.error(f"Researcher failed: {e}")
            return

        researcher_out    = _format_researcher_output(full_state)
        researcher_model  = full_state.get("model_used", "")
        researcher_prompt = result.get("prompt_sent", [])
        agent_outputs["researcher"] = {"output": researcher_out, "model": researcher_model, "prompt": researcher_prompt}
        _agent_panel(
            researcher_ph, "Agent 2: Researcher",
            "Searches the web for evidence on each question",
            STATUS_COMPLETE, output=researcher_out, model=researcher_model,
            expanded=True, prompt=researcher_prompt,
        )

        # ── Quality gate ──────────────────────────────────────────────────────
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
                with st.spinner(f"Re-searching {len(flagged)} question(s)..."):
                    result = run_researcher(full_state, target_questions=flagged)
                full_state.update(result)
            except Exception:
                break

            researcher_out    = _format_researcher_output(full_state)
            researcher_prompt = result.get("prompt_sent", [])
            agent_outputs["researcher"] = {"output": researcher_out, "model": researcher_model, "prompt": researcher_prompt}
            _agent_panel(
                researcher_ph, "Agent 2: Researcher",
                "Searches the web for evidence on each question",
                STATUS_COMPLETE, output=researcher_out,
                model=f"{researcher_model} · {researcher_attempt} attempts",
                expanded=True, prompt=researcher_prompt,
            )
            flagged = _combined_flag_check(
                full_state, chain, researcher_ph, quality_gate_ph,
                researcher_out, f"{researcher_model} · {researcher_attempt} attempts",
            )

        # ── Transition ────────────────────────────────────────────────────────
        st.session_state["m01_pending_state"]      = full_state
        st.session_state["m01_agent_outputs"]      = agent_outputs
        st.session_state["m01_researcher_attempt"] = researcher_attempt

        if len(flagged) > 2:
            st.session_state["m01_flagged_questions"] = flagged
            st.session_state["m01_phase"] = "quality_gate"
        else:
            quality_gate_ph.empty()
            st.session_state["m01_phase"] = "critic_running"

        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: quality_gate
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "quality_gate":
        pending         = st.session_state.get("m01_pending_state", {})
        flagged         = st.session_state.get("m01_flagged_questions", [])
        questions       = pending.get("questions", [])
        p_model         = st.session_state.get("m01_planner_model", "")
        p_prompt        = st.session_state.get("m01_planner_prompt", [])
        planner_attempt = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs   = st.session_state.get("m01_agent_outputs", {})
        r_attempt       = st.session_state.get("m01_researcher_attempt", 1)

        attempt_note     = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out      = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        researcher_out    = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model  = agent_outputs.get("researcher", {}).get("model", "")
        researcher_prompt = agent_outputs.get("researcher", {}).get("prompt", [])

        _agent_panel(planner_ph, "Agent 1: Planner",
                     "Breaks the topic into focused research questions",
                     STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note, prompt=p_prompt)
        approval_ph.empty()
        _agent_panel(researcher_ph, "Agent 2: Researcher",
                     f"Completed {r_attempt + 1} search attempt(s)",
                     STATUS_COMPLETE, output=researcher_out,
                     model=f"{researcher_model} · {r_attempt + 1} attempts",
                     prompt=researcher_prompt)

        with quality_gate_ph.container():
            st.warning(
                f"After {r_attempt + 1} search attempt(s), **{len(flagged)} question(s)** still have "
                "weak or no sources. The Writer will note these gaps explicitly in the paper."
            )
            st.markdown("**Questions with weak sources:**")
            for q in flagged:
                st.markdown(f"- {q[:120]}")
            st.markdown("")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Proceed to Critic →", type="primary"):
                    quality_gate_ph.empty()
                    st.session_state["m01_phase"] = "critic_running"
                    st.rerun()
            with col2:
                if st.button("Stop here"):
                    for key in _STATE_KEYS:
                        st.session_state.pop(key, None)
                    st.session_state["m01_form_key"] = st.session_state.get("m01_form_key", 0)
                    st.rerun()

        for name, label, desc in AGENTS[2:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: critic_running
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "critic_running":
        pending         = st.session_state.get("m01_pending_state", {})
        questions       = pending.get("questions", [])
        p_model         = st.session_state.get("m01_planner_model", "")
        p_prompt        = st.session_state.get("m01_planner_prompt", [])
        planner_attempt = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs   = st.session_state.get("m01_agent_outputs", {})
        r_attempt       = st.session_state.get("m01_researcher_attempt", 1)

        attempt_note     = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out      = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        researcher_out    = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model  = agent_outputs.get("researcher", {}).get("model", "")
        researcher_prompt = agent_outputs.get("researcher", {}).get("prompt", [])

        _agent_panel(planner_ph, "Agent 1: Planner",
                     "Breaks the topic into focused research questions",
                     STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note, prompt=p_prompt)
        approval_ph.empty()
        _agent_panel(researcher_ph, "Agent 2: Researcher",
                     "Searches the web for evidence on each question",
                     STATUS_COMPLETE, output=researcher_out, model=researcher_model,
                     prompt=researcher_prompt)
        quality_gate_ph.empty()
        _agent_panel(critic_ph, "Agent 3: Critic",
                     "Assessing source quality and flagging gaps",
                     STATUS_RUNNING, running=True)
        for name, label, desc in AGENTS[3:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)

        full_state = dict(pending)
        chain = get_chain(st.session_state)

        try:
            result = run_critic(full_state, chain)
            full_state.update(result)
        except Exception as e:
            _agent_panel(critic_ph, "Agent 3: Critic", "", STATUS_FAILED)
            for name, label, desc in AGENTS[3:]:
                _agent_panel(ph[name], label, desc, STATUS_FAILED)
            st.error(f"Critic failed: {e}")
            return

        critic_out    = result.get("critique", "")
        critic_model  = result.get("model_used", "")
        critic_prompt = result.get("prompt_sent", [])

        # Build verdict summary — stored at top of output so it's visible in every dropdown
        _summary      = _parse_critic_summary(critic_out, full_state.get("questions", [])  )
        _ratings      = [e["rating"] for e in _summary]
        _strong       = sum(1 for r in _ratings if r == "Strong")
        _adequate     = sum(1 for r in _ratings if r == "Adequate")
        _weak         = sum(1 for r in _ratings if r == "Weak")
        _total        = len(_ratings)
        if _weak == 0:
            _verdict = (
                f"✅ **Verdict: Good to proceed.** "
                f"All {_total} questions have relevant sources ({_strong} strong, {_adequate} adequate). "
                "The Writer has enough evidence. It will flag thin areas rather than invent facts."
            )
        elif _weak <= _total // 2:
            _verdict = (
                f"⚠️ **Verdict: Proceed with caution.** "
                f"{_weak} of {_total} questions have weak sources. "
                "The paper will have gaps in those areas."
            )
        else:
            _verdict = (
                f"❌ **Verdict: Consider stopping.** "
                f"{_weak} of {_total} questions have weak sources. "
                "Most of the paper will lack solid evidence."
            )
        critic_display = _verdict + "\n\n---\n\n" + _format_critic_output(critic_out)
        agent_outputs["critic"] = {"output": critic_display, "model": critic_model, "prompt": critic_prompt}

        st.session_state["m01_pending_state"]  = full_state
        st.session_state["m01_agent_outputs"]  = agent_outputs
        st.session_state["m01_phase"] = "critic_done"
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: critic_done  (Critic checkpoint)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "critic_done":
        pending         = st.session_state.get("m01_pending_state", {})
        questions       = pending.get("questions", [])
        p_model         = st.session_state.get("m01_planner_model", "")
        p_prompt        = st.session_state.get("m01_planner_prompt", [])
        planner_attempt = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs   = st.session_state.get("m01_agent_outputs", {})

        attempt_note     = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out      = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        researcher_out    = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model  = agent_outputs.get("researcher", {}).get("model", "")
        researcher_prompt = agent_outputs.get("researcher", {}).get("prompt", [])
        critic_out    = agent_outputs.get("critic", {}).get("output", "")
        critic_model  = agent_outputs.get("critic", {}).get("model", "")
        critic_prompt = agent_outputs.get("critic", {}).get("prompt", [])

        _agent_panel(planner_ph, "Agent 1: Planner",
                     "Breaks the topic into focused research questions",
                     STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note, prompt=p_prompt)
        approval_ph.empty()
        _agent_panel(researcher_ph, "Agent 2: Researcher",
                     "Searches the web for evidence on each question",
                     STATUS_COMPLETE, output=researcher_out, model=researcher_model,
                     prompt=researcher_prompt)
        quality_gate_ph.empty()
        _agent_panel(critic_ph, "Agent 3: Critic",
                     "Assesses source quality and flags gaps",
                     STATUS_COMPLETE, output=critic_out, model=critic_model,
                     expanded=True, prompt=critic_prompt)

        # ── Smart Critic checkpoint ───────────────────────────────────────────
        summary    = _parse_critic_summary(critic_out, questions)
        ratings    = [e["rating"] for e in summary]
        all_strong = all(r == "Strong" for r in ratings)
        weak_count = sum(1 for r in ratings if r == "Weak")

        if all_strong:
            # No human decision needed — all sources rated Strong, proceed automatically
            st.session_state["m01_phase"] = "writing"
            st.rerun()
            return

        # One or more questions are Adequate or Weak — show the checkpoint
        with critic_gate_ph.container():
            strong_count   = sum(1 for r in ratings if r == "Strong")
            adequate_count = sum(1 for r in ratings if r == "Adequate")
            total          = len(ratings)

            # Single verdict box — situation + recommendation in one message
            if weak_count == 0:
                st.success(
                    f"✅ **Verdict: Good to proceed.** "
                    f"All {total} questions have relevant sources ({strong_count} strong, {adequate_count} adequate). "
                    "The Writer has enough evidence. It will flag any thin areas rather than invent facts."
                )
            elif weak_count <= total // 2:
                st.warning(
                    f"⚠️ **Verdict: Proceed with caution.** "
                    f"{weak_count} of {total} question(s) have weak sources. "
                    "The paper will have gaps in those areas. "
                    "Fine to continue if those gaps are not central to your topic."
                )
            else:
                st.error(
                    f"❌ **Verdict: Consider stopping.** "
                    f"{weak_count} of {total} questions have weak sources. "
                    "Proceeding will produce a paper that is mostly caveats. "
                    "Consider stopping and refining the topic or questions."
                )

            # Per-question table
            st.markdown("**Source quality by question:**")
            for entry in summary:
                rating   = entry["rating"]
                icon     = "🟢" if rating == "Strong" else ("🟡" if rating == "Adequate" else "🔴")
                q_text   = entry["question"][:85] + ("..." if len(entry["question"]) > 85 else "")
                gap      = entry["gap"]
                best     = entry["source"]
                gap_text = (
                    f"Best source: {best}" if rating == "Strong"
                    else (f"Gap: {gap}" if gap and gap.lower() not in ("none", "none identified", "none.")
                          else "No specific gap identified")
                )

                col_icon, col_rating, col_q, col_gap = st.columns([0.3, 1, 2.5, 3])
                with col_icon:
                    st.markdown(icon)
                with col_rating:
                    st.caption(f"**{rating}**")
                with col_q:
                    st.caption(q_text)
                with col_gap:
                    st.caption(gap_text)

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
            st.caption("The Writer will not start until you approve.")

        for name, label, desc in AGENTS[3:]:
            _agent_panel(ph[name], label, desc, STATUS_WAITING)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: writing  (Writer only — then transitions to judge_running)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "writing":
        pending          = st.session_state.get("m01_pending_state", {})
        questions        = pending.get("questions", [])
        p_model          = st.session_state.get("m01_planner_model", "")
        p_prompt         = st.session_state.get("m01_planner_prompt", [])
        planner_attempt  = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs    = st.session_state.get("m01_agent_outputs", {})
        writer_feedback  = st.session_state.get("m01_writer_feedback", "")
        writer_attempt   = st.session_state.get("m01_writer_attempt", 1)

        attempt_note      = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        w_attempt_note    = f" · re-draft {writer_attempt}" if writer_attempt > 1 else ""
        planner_out       = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        researcher_out    = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model  = agent_outputs.get("researcher", {}).get("model", "")
        researcher_prompt = agent_outputs.get("researcher", {}).get("prompt", [])
        critic_out    = agent_outputs.get("critic", {}).get("output", "")
        critic_model  = agent_outputs.get("critic", {}).get("model", "")
        critic_prompt = agent_outputs.get("critic", {}).get("prompt", [])
        full_state = dict(pending)
        chain = get_chain(st.session_state)

        _agent_panel(planner_ph, "Agent 1: Planner",
                     "Breaks the topic into focused research questions",
                     STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note, prompt=p_prompt)
        approval_ph.empty()
        _agent_panel(researcher_ph, "Agent 2: Researcher",
                     "Searches the web for evidence on each question",
                     STATUS_COMPLETE, output=researcher_out, model=researcher_model,
                     prompt=researcher_prompt)
        quality_gate_ph.empty()
        _agent_panel(critic_ph, "Agent 3: Critic",
                     "Assesses source quality and flags gaps",
                     STATUS_COMPLETE, output=critic_out, model=critic_model, prompt=critic_prompt)
        critic_gate_ph.empty()
        _agent_panel(writer_ph, "Agent 4: Writer",
                     f"Drafting the research paper{w_attempt_note}",
                     STATUS_RUNNING, running=True)
        _agent_panel(judge_ph,  "Agent 5: Judge",
                     "Evaluates draft quality before the Editor runs", STATUS_WAITING)
        _agent_panel(editor_ph, "Agent 6: Editor",
                     "Polishes the draft and removes weak language",   STATUS_WAITING)

        try:
            with st.spinner("Writing the research paper... (this can take 30–60 seconds)"):
                result = run_writer(full_state, chain, user_feedback=writer_feedback)
            full_state.update(result)
        except Exception as e:
            _agent_panel(writer_ph, "Agent 4: Writer", "", STATUS_FAILED)
            _agent_panel(judge_ph,  "Agent 5: Judge",  "", STATUS_FAILED)
            _agent_panel(editor_ph, "Agent 6: Editor", "", STATUS_FAILED)
            st.error(f"Writer failed: {e}")
            return

        writer_out    = _format_agent_output("writer", result, full_state)
        writer_model  = result.get("model_used", "")
        writer_prompt = result.get("prompt_sent", [])
        agent_outputs["writer"] = {"output": writer_out, "model": writer_model + w_attempt_note,
                                   "prompt": writer_prompt}

        st.session_state["m01_pending_state"]   = full_state
        st.session_state["m01_agent_outputs"]   = agent_outputs
        st.session_state["m01_writer_feedback"] = ""   # clear after use
        st.session_state["m01_phase"] = "judge_running"
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: judge_running
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "judge_running":
        pending          = st.session_state.get("m01_pending_state", {})
        questions        = pending.get("questions", [])
        p_model          = st.session_state.get("m01_planner_model", "")
        p_prompt         = st.session_state.get("m01_planner_prompt", [])
        planner_attempt  = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs    = st.session_state.get("m01_agent_outputs", {})
        writer_attempt   = st.session_state.get("m01_writer_attempt", 1)

        attempt_note     = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out      = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        researcher_out   = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model = agent_outputs.get("researcher", {}).get("model", "")
        researcher_prompt = agent_outputs.get("researcher", {}).get("prompt", [])
        critic_out    = agent_outputs.get("critic", {}).get("output", "")
        critic_model  = agent_outputs.get("critic", {}).get("model", "")
        critic_prompt = agent_outputs.get("critic", {}).get("prompt", [])
        writer_out    = agent_outputs.get("writer", {}).get("output", "")
        writer_model  = agent_outputs.get("writer", {}).get("model", "")
        writer_prompt = agent_outputs.get("writer", {}).get("prompt", [])
        full_state = dict(pending)
        chain = get_chain(st.session_state)

        _agent_panel(planner_ph, "Agent 1: Planner",
                     "Breaks the topic into focused research questions",
                     STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note, prompt=p_prompt)
        approval_ph.empty()
        _agent_panel(researcher_ph, "Agent 2: Researcher",
                     "Searches the web for evidence on each question",
                     STATUS_COMPLETE, output=researcher_out, model=researcher_model,
                     prompt=researcher_prompt)
        quality_gate_ph.empty()
        _agent_panel(critic_ph, "Agent 3: Critic",
                     "Assesses source quality and flags gaps",
                     STATUS_COMPLETE, output=critic_out, model=critic_model, prompt=critic_prompt)
        critic_gate_ph.empty()
        _agent_panel(writer_ph, "Agent 4: Writer",
                     "Drafts the full research paper",
                     STATUS_COMPLETE, output=writer_out, model=writer_model, prompt=writer_prompt)
        _agent_panel(judge_ph, "Agent 5: Judge",
                     "Evaluating draft quality...", STATUS_RUNNING, running=True)
        _agent_panel(editor_ph, "Agent 6: Editor",
                     "Polishes the draft and removes weak language", STATUS_WAITING)

        try:
            with st.spinner("Evaluating draft quality..."):
                result = run_judge(full_state, chain)
        except Exception as e:
            _agent_panel(judge_ph,  "Agent 5: Judge",  "", STATUS_FAILED)
            _agent_panel(editor_ph, "Agent 6: Editor", "", STATUS_FAILED)
            st.error(f"Judge failed: {e}")
            return

        judge_out    = _format_judge_output(result)
        judge_model  = result.get("model_used", "")
        judge_prompt = result.get("prompt_sent", [])
        agent_outputs["judge"] = {"output": judge_out, "model": judge_model, "prompt": judge_prompt}

        st.session_state["m01_judge_result"]   = result
        st.session_state["m01_agent_outputs"]  = agent_outputs
        st.session_state["m01_phase"] = "judge_done"
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: judge_done  (Judge checkpoint)
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "judge_done":
        pending          = st.session_state.get("m01_pending_state", {})
        questions        = pending.get("questions", [])
        p_model          = st.session_state.get("m01_planner_model", "")
        p_prompt         = st.session_state.get("m01_planner_prompt", [])
        planner_attempt  = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs    = st.session_state.get("m01_agent_outputs", {})
        judge_result     = st.session_state.get("m01_judge_result", {})
        writer_attempt   = st.session_state.get("m01_writer_attempt", 1)
        judge_editing    = st.session_state.get("m01_judge_editing", False)

        attempt_note     = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out      = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        researcher_out   = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model = agent_outputs.get("researcher", {}).get("model", "")
        researcher_prompt = agent_outputs.get("researcher", {}).get("prompt", [])
        critic_out    = agent_outputs.get("critic",   {}).get("output", "")
        critic_model  = agent_outputs.get("critic",   {}).get("model",  "")
        critic_prompt = agent_outputs.get("critic",   {}).get("prompt", [])
        writer_out    = agent_outputs.get("writer",   {}).get("output", "")
        writer_model  = agent_outputs.get("writer",   {}).get("model",  "")
        writer_prompt = agent_outputs.get("writer",   {}).get("prompt", [])
        judge_out     = agent_outputs.get("judge",    {}).get("output", "")
        judge_model   = agent_outputs.get("judge",    {}).get("model",  "")
        judge_prompt  = agent_outputs.get("judge",    {}).get("prompt", [])
        flagged       = judge_result.get("flagged", False)

        _agent_panel(planner_ph, "Agent 1: Planner",
                     "Breaks the topic into focused research questions",
                     STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note, prompt=p_prompt)
        approval_ph.empty()
        _agent_panel(researcher_ph, "Agent 2: Researcher",
                     "Searches the web for evidence on each question",
                     STATUS_COMPLETE, output=researcher_out, model=researcher_model,
                     prompt=researcher_prompt)
        quality_gate_ph.empty()
        _agent_panel(critic_ph, "Agent 3: Critic",
                     "Assesses source quality and flags gaps",
                     STATUS_COMPLETE, output=critic_out, model=critic_model, prompt=critic_prompt)
        critic_gate_ph.empty()
        _agent_panel(writer_ph, "Agent 4: Writer",
                     "Drafts the full research paper",
                     STATUS_COMPLETE, output=writer_out, model=writer_model, prompt=writer_prompt)
        _agent_panel(judge_ph, "Agent 5: Judge",
                     "Evaluates draft quality before the Editor runs",
                     STATUS_COMPLETE, output=judge_out, model=judge_model,
                     expanded=True, prompt=judge_prompt)

        # ── Smart Judge checkpoint ────────────────────────────────────────────
        scores     = judge_result.get("scores", {})
        rule       = judge_result.get("rule_check", {})
        all_pass   = (
            rule.get("word_count_ok", False)
            and rule.get("sections_ok", False)
            and all(v.get("score", 0) >= 4 for v in scores.values())
        )

        with judge_gate_ph.container():
            can_redraft = writer_attempt <= MAX_WRITER_RETRIES

            low_dims  = [(k, v) for k, v in scores.items() if v.get("score", 5) < 4]
            rule_fail = not rule.get("word_count_ok", True) or not rule.get("sections_ok", True)
            dim_names = {
                "completeness": "Completeness", "argument_quality": "Argument quality",
                "source_integration": "Source integration", "format_adherence": "Format adherence",
            }

            # Verdict box — one clear message regardless of outcome
            if all_pass:
                st.success(
                    "✅ **Verdict: Proceed.** "
                    "The draft passed all quality checks. "
                    "The Editor will polish the language and confirm the format."
                )
            elif rule_fail or low_dims:
                issues = []
                if not rule.get("word_count_ok", True):
                    issues.append(f"word count short ({rule.get('word_count', 0):,} vs {rule.get('word_count_target', 0):,} target)")
                if not rule.get("sections_ok", True):
                    issues.append(f"too few sections ({rule.get('section_count', 0)} vs {rule.get('min_sections', 0)} minimum)")
                for k, v in low_dims:
                    issues.append(f"{dim_names.get(k, k)} scored {v.get('score', 0)}/5")
                st.warning(
                    "⚠️ **Verdict: Issues found — ** " + ", ".join(issues) + ". "
                    "You can proceed anyway or re-draft with specific feedback."
                )

            # Scorecard
            _show_judge_scorecard(judge_result)

            # Specific notes for any dimension below 4
            if low_dims:
                st.markdown("**What the Judge said about the low-scoring dimensions:**")
                for k, v in low_dims:
                    score = v.get("score", 0)
                    note  = v.get("note", "No note provided.")
                    icon  = "🔴" if score < 3 else "🟡"
                    st.caption(f"{icon} **{dim_names.get(k, k)} ({score}/5):** {note}")

            if not judge_editing:
                if all_pass:
                    # Clean run — one button only
                    if st.button("Continue to Editor →", type="primary"):
                        judge_gate_ph.empty()
                        st.session_state["m01_phase"] = "editor_running"
                        st.rerun()
                else:
                    # Issues found — three options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Proceed to Editor →", type="primary"):
                            judge_gate_ph.empty()
                            st.session_state["m01_phase"] = "editor_running"
                            st.rerun()
                    with col2:
                        redraft_label = "Re-draft with feedback" if can_redraft else f"Re-draft (max {MAX_WRITER_RETRIES} reached)"
                        if st.button(redraft_label, disabled=not can_redraft):
                            st.session_state["m01_judge_editing"] = True
                            st.rerun()
                    with col3:
                        if st.button("Stop here"):
                            for key in _STATE_KEYS:
                                st.session_state.pop(key, None)
                            st.session_state["m01_form_key"] = st.session_state.get("m01_form_key", 0)
                            st.rerun()
                st.caption("The Editor will not start until you approve.")
            else:
                suggestion = _build_redraft_suggestion(judge_result)
                st.markdown("**What should the Writer fix?**")
                st.caption("Pre-filled from the Judge's findings — edit or use as-is.")
                st.text_area(
                    "Feedback for re-draft", height=160,
                    value=suggestion,
                    key="m01_writer_feedback_input",
                    label_visibility="collapsed",
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Submit and re-draft →", type="primary"):
                        feedback = st.session_state.get("m01_writer_feedback_input", "")
                        st.session_state["m01_writer_feedback"] = feedback
                        st.session_state["m01_writer_attempt"]  = writer_attempt + 1
                        st.session_state["m01_judge_editing"]   = False
                        st.session_state["m01_phase"] = "writing"
                        st.rerun()
                with col2:
                    if st.button("Cancel"):
                        st.session_state["m01_judge_editing"] = False
                        st.rerun()

        _agent_panel(editor_ph, "Agent 6: Editor",
                     "Polishes the draft and removes weak language", STATUS_WAITING)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: editor_running
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "editor_running":
        pending          = st.session_state.get("m01_pending_state", {})
        questions        = pending.get("questions", [])
        p_model          = st.session_state.get("m01_planner_model", "")
        p_prompt         = st.session_state.get("m01_planner_prompt", [])
        planner_attempt  = st.session_state.get("m01_planner_attempt", 1)
        agent_outputs    = st.session_state.get("m01_agent_outputs", {})

        attempt_note     = f" · attempt {planner_attempt}" if planner_attempt > 1 else ""
        planner_out      = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        researcher_out   = agent_outputs.get("researcher", {}).get("output", "")
        researcher_model = agent_outputs.get("researcher", {}).get("model", "")
        researcher_prompt = agent_outputs.get("researcher", {}).get("prompt", [])
        critic_out    = agent_outputs.get("critic", {}).get("output", "")
        critic_model  = agent_outputs.get("critic", {}).get("model",  "")
        critic_prompt = agent_outputs.get("critic", {}).get("prompt", [])
        writer_out    = agent_outputs.get("writer", {}).get("output", "")
        writer_model  = agent_outputs.get("writer", {}).get("model",  "")
        writer_prompt = agent_outputs.get("writer", {}).get("prompt", [])
        judge_out     = agent_outputs.get("judge",  {}).get("output", "")
        judge_model   = agent_outputs.get("judge",  {}).get("model",  "")
        judge_prompt  = agent_outputs.get("judge",  {}).get("prompt", [])
        full_state = dict(pending)
        chain = get_chain(st.session_state)

        _agent_panel(planner_ph, "Agent 1: Planner",
                     "Breaks the topic into focused research questions",
                     STATUS_COMPLETE, output=planner_out, model=p_model + attempt_note, prompt=p_prompt)
        approval_ph.empty()
        _agent_panel(researcher_ph, "Agent 2: Researcher",
                     "Searches the web for evidence on each question",
                     STATUS_COMPLETE, output=researcher_out, model=researcher_model,
                     prompt=researcher_prompt)
        quality_gate_ph.empty()
        _agent_panel(critic_ph, "Agent 3: Critic",
                     "Assesses source quality and flags gaps",
                     STATUS_COMPLETE, output=critic_out, model=critic_model, prompt=critic_prompt)
        critic_gate_ph.empty()
        _agent_panel(writer_ph, "Agent 4: Writer",
                     "Drafts the full research paper",
                     STATUS_COMPLETE, output=writer_out, model=writer_model, prompt=writer_prompt)
        judge_gate_ph.empty()
        _agent_panel(judge_ph,  "Agent 5: Judge",
                     "Evaluates draft quality before the Editor runs",
                     STATUS_COMPLETE, output=judge_out, model=judge_model, prompt=judge_prompt)
        _agent_panel(editor_ph, "Agent 6: Editor",
                     "Polishing the draft", STATUS_RUNNING, running=True)

        try:
            with st.spinner("Polishing the draft..."):
                result = run_editor(full_state, chain)
            full_state.update(result)
        except Exception as e:
            _agent_panel(editor_ph, "Agent 6: Editor", "", STATUS_FAILED)
            st.error(f"Editor failed: {e}")
            return

        editor_out    = _format_agent_output("editor", result, full_state)
        editor_model  = result.get("model_used", "")
        editor_prompt = result.get("prompt_sent", [])
        agent_outputs["editor"] = {"output": editor_out, "model": editor_model, "prompt": editor_prompt}
        _agent_panel(editor_ph, "Agent 6: Editor",
                     "Polishes the draft and removes weak language",
                     STATUS_COMPLETE, output=editor_out, model=editor_model,
                     expanded=True, prompt=editor_prompt)

        st.session_state["m01_final"]         = full_state.get("final", "")
        st.session_state["m01_full_state"]    = full_state
        st.session_state["m01_agent_outputs"] = agent_outputs
        st.session_state["m01_phase"]         = "complete"

        _show_sources()
        _show_run_summary()
        _show_download()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE: complete
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "complete":
        saved = st.session_state.get("m01_agent_outputs", {})
        for name, label, desc in AGENTS:
            out    = saved.get(name, {}).get("output", "")
            model  = saved.get(name, {}).get("model", "")
            prompt = saved.get(name, {}).get("prompt", [])
            _agent_panel(ph[name], label, desc, STATUS_COMPLETE,
                         output=out, model=model, prompt=prompt)
        _show_sources()
        _show_run_summary()
        _show_download()
        return


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combined_flag_check(full_state: dict, chain, researcher_ph, quality_gate_ph,
                         researcher_out: str, researcher_model_label: str) -> list[str]:
    """Two-pass quality check. Shows validation progress on the Researcher panel."""
    research = full_state.get("research", {})

    # Pass 1: objective domain check — instant
    domain_flagged = flag_weak_questions(research)

    # Pass 2: LLM relevance check — show Researcher as still active during the call
    _agent_panel(researcher_ph, "Agent 2: Researcher",
                 "Validating source relevance...",
                 STATUS_RUNNING, running=True)

    llm_flagged = flag_irrelevant_questions(research, chain, skip=domain_flagged)

    # Restore Researcher to complete
    _agent_panel(researcher_ph, "Agent 2: Researcher",
                 "Searches the web for evidence on each question",
                 STATUS_COMPLETE, output=researcher_out, model=researcher_model_label)

    combined = domain_flagged + [q for q in llm_flagged if q not in domain_flagged]
    return combined


def _start_planner(topic, angle, audience, format_style, length, planner_ph) -> None:
    """Runs the Planner and stores results, then reruns into planner_done."""
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
    st.session_state["m01_planner_prompt"]  = result.get("prompt_sent", [])
    st.session_state["m01_planner_attempt"] = 1
    st.session_state["m01_editing"]         = False
    st.session_state["m01_agent_outputs"]   = {}
    st.session_state["m01_writer_attempt"]  = 1
    st.session_state["m01_writer_feedback"] = ""
    st.session_state["m01_judge_editing"]   = False
    st.session_state["m01_judge_result"]    = {}
    st.session_state["m01_phase"]           = "planner_done"
    st.rerun()


def _format_researcher_output(full_state: dict) -> str:
    research = full_state.get("research", {})
    sources  = full_state.get("sources", [])
    lines = [f"**{len(sources)} sources collected across {len(research)} questions**\n"]
    for q, hits in research.items():
        lines.append(f"- *{q[:70]}...* — {len(hits)} results")
    return "\n".join(lines)


def _format_agent_output(node_name: str, result: dict, full_state: dict) -> str:
    if node_name == "researcher":
        return _format_researcher_output(full_state)
    elif node_name == "critic":
        return result.get("critique", "")
    elif node_name == "writer":
        draft = result.get("draft", "")
        return f"*Draft: {len(draft):,} characters*\n\n" + draft[:600] + "..."
    elif node_name == "editor":
        return result.get("final", "")
    return ""


def _format_judge_output(result: dict) -> str:
    """Plain-text summary of Judge results for the agent panel output expander."""
    rule   = result.get("rule_check", {})
    scores = result.get("scores", {})

    wc_ok  = "✅" if rule.get("word_count_ok") else "❌"
    sec_ok = "✅" if rule.get("sections_ok") else "❌"

    dim_labels = {
        "completeness":       "Completeness",
        "argument_quality":   "Argument quality",
        "source_integration": "Source integration",
        "format_adherence":   "Format adherence",
    }

    parts = [
        "**Rule check**",
        f"{wc_ok} Word count: {rule.get('word_count', 0):,} (target {rule.get('word_count_target', 0):,})",
        f"{sec_ok} Sections: {rule.get('section_count', 0)} (min {rule.get('min_sections', 0)})",
        "",
        "**Quality scores**",
    ]
    for key, label in dim_labels.items():
        s     = scores.get(key, {})
        score = s.get("score", 0)
        note  = s.get("note", "")
        icon  = "🟢" if score >= 4 else ("🟡" if score == 3 else "🔴")
        parts.append(f"{icon} {label}: {score}/5 — {note}")

    flagged = result.get("flagged", False)
    parts.append("")
    parts.append("⚠️ Flagged for review." if flagged else "✅ Draft passed quality check.")

    return "\n\n".join(p for p in parts)


def _show_judge_scorecard(result: dict) -> None:
    """Renders the Judge scorecard using st.progress bars."""
    rule   = result.get("rule_check", {})
    scores = result.get("scores", {})

    st.markdown("**Rule check**")
    wc_ok  = rule.get("word_count_ok", True)
    sec_ok = rule.get("sections_ok", True)
    col1, col2 = st.columns(2)
    with col1:
        icon = "✅" if wc_ok else "❌"
        st.caption(f"{icon} Words: {rule.get('word_count', 0):,} / {rule.get('word_count_target', 0):,} target")
    with col2:
        icon = "✅" if sec_ok else "❌"
        st.caption(f"{icon} Sections: {rule.get('section_count', 0)} / {rule.get('min_sections', 0)} minimum")

    st.markdown("**Quality scores** (1 = poor · 3 = acceptable · 5 = excellent)")
    dim_labels = {
        "completeness":       "Completeness",
        "argument_quality":   "Argument quality",
        "source_integration": "Source integration",
        "format_adherence":   "Format adherence",
    }
    for key, label in dim_labels.items():
        s     = scores.get(key, {})
        score = s.get("score", 3)
        note  = s.get("note", "")
        icon  = "🟢" if score >= 4 else ("🟡" if score == 3 else "🔴")
        col_label, col_bar = st.columns([1, 2])
        with col_label:
            st.caption(f"{icon} {label}: **{score}/5**")
        with col_bar:
            st.progress(score / 5)
        if note:
            st.caption(f"   {note}")


def _build_redraft_suggestion(judge_result: dict) -> str:
    """
    Builds a pre-filled re-draft note from the Judge's findings.
    Covers word count, section count, and any dimension scored below 4.
    The Writer receives this as its correction instruction.
    """
    rule   = judge_result.get("rule_check", {})
    scores = judge_result.get("scores", {})
    dim_labels = {
        "completeness":       "Completeness",
        "argument_quality":   "Argument quality",
        "source_integration": "Source integration",
        "format_adherence":   "Format adherence",
    }
    notes = []

    if not rule.get("word_count_ok", True):
        actual = rule.get("word_count", 0)
        target = rule.get("word_count_target", 0)
        notes.append(
            f"The paper is {actual:,} words — well below the {target:,} word target. "
            "Expand every section with more depth, evidence, and analysis. Do not stop early."
        )

    if not rule.get("sections_ok", True):
        actual  = rule.get("section_count", 0)
        minimum = rule.get("min_sections", 0)
        notes.append(
            f"The paper has only {actual} section heading(s) — the minimum is {minimum}. "
            "Add the missing sections."
        )

    for key, label in dim_labels.items():
        s     = scores.get(key, {})
        score = s.get("score", 5)
        note  = s.get("note", "")
        if score < 4 and note:
            notes.append(f"{label} ({score}/5): {note}")

    return "\n\n".join(notes)


def _format_critic_output(critique: str) -> str:
    """
    Reformats Critic output so each field is on its own line, with icon and bold.
    The LLM often runs Question / Rating / Strongest source / Gap on one line.
    Step 1: insert line breaks before each field label when mid-line.
    Step 2: apply icon and bold formatting.
    """
    result = critique

    # Step 1 — break fields onto their own lines when they appear mid-sentence.
    # Markdown requires two trailing spaces + \n for a <br>, or \n\n for a paragraph break.
    # We use \n\n so each field is clearly separated regardless of renderer.
    result = re.sub(r"([^\n])\s+(Rating:)",           r"\1\n\n\2",  result, flags=re.IGNORECASE)
    result = re.sub(r"([^\n])\s+(Strongest source:)", r"\1\n\n\2",  result, flags=re.IGNORECASE)
    result = re.sub(r"([^\n])\s+(Gap:)\s*",           r"\1\n\n\2 ", result, flags=re.IGNORECASE)

    # Step 2 — Rating with colour icon and bold.
    # The LLM sometimes wraps the value in its own **bold** markers (e.g. Rating: **Strong**).
    # \** matches zero or more literal asterisks, so this handles both forms.
    result = re.sub(r"Rating:\s*\**(Strong)\**",   "**🟢 Rating: Strong**",   result, flags=re.IGNORECASE)
    result = re.sub(r"Rating:\s*\**(Adequate)\**", "**🟡 Rating: Adequate**", result, flags=re.IGNORECASE)
    result = re.sub(r"Rating:\s*\**(Weak)\**",     "**🔴 Rating: Weak**",     result, flags=re.IGNORECASE)

    # Bold remaining field labels
    result = re.sub(r"Strongest source:", "**Strongest source:**", result)
    result = re.sub(r"\bGap:",            "**Gap:**",              result)

    # Add a divider before each Question block (except the very first one)
    result = re.sub(r"\n\n((?:\*\*)?Question)", r"\n\n---\n\n\1", result)

    # Bold Overall Assessment heading
    result = re.sub(r"(Overall Assessment:?)", r"**\1**", result, flags=re.IGNORECASE)

    return result


def _parse_critic_summary(critique: str, questions: list) -> list:
    """
    Parses Critic output into a list of dicts: {question, rating, source, gap}.
    One dict per question, in order. Falls back gracefully if any field is missing.
    """
    results = []
    blocks = re.split(r"\nQuestion:", "\n" + critique)
    blocks = [b.strip() for b in blocks if b.strip()]

    for i, question in enumerate(questions):
        block    = blocks[i] if i < len(blocks) else ""
        rating_m = re.search(r"Rating:\s*(Strong|Adequate|Weak)", block, re.IGNORECASE)
        source_m = re.search(r"Strongest source:\s*(.+)", block)
        gap_m    = re.search(r"Gap:\s*(.+)", block)
        results.append({
            "question": question,
            "rating":   rating_m.group(1).capitalize() if rating_m else "Adequate",
            "source":   source_m.group(1).strip() if source_m else "Not specified",
            "gap":      gap_m.group(1).strip() if gap_m else "None identified",
        })

    return results


def _show_sources() -> None:
    full_state = st.session_state.get("m01_full_state", {})
    sources    = full_state.get("sources", [])
    if not sources:
        return
    st.markdown("---")
    with st.expander(f"Sources ({len(sources)} URLs)", expanded=False):
        for i, url in enumerate(sources, 1):
            st.markdown(f"{i}. {url}")


def _show_run_summary() -> None:
    """Shows token usage, model breakdown, and estimated cost for this run."""
    log = st.session_state.get("m01_call_log", [])
    if not log:
        return

    total_input  = sum(e["input_tokens"]  for e in log)
    total_output = sum(e["output_tokens"] for e in log)
    total_tokens = total_input + total_output

    total_cost = 0.0
    for entry in log:
        price = APPROX_PRICING.get(entry["model"])
        if price:
            in_p, out_p = price
            total_cost += (entry["input_tokens"]  / 1_000_000) * in_p
            total_cost += (entry["output_tokens"] / 1_000_000) * out_p

    cost_str = f"~${total_cost:.4f}" if total_cost > 0 else "N/A"

    st.markdown("---")
    with st.expander(
        f"📊 Run summary — {len(log)} LLM call(s) · {total_tokens:,} tokens · {cost_str}",
        expanded=False,
    ):
        writer_attempt  = st.session_state.get("m01_writer_attempt", 1)
        planner_attempt = st.session_state.get("m01_planner_attempt", 1)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LLM calls", len(log))
        with col2:
            st.metric("Total tokens", f"{total_tokens:,}")
        with col3:
            st.metric("Est. cost (USD)", cost_str)
        notes = []
        if planner_attempt > 1:
            notes.append(f"Planner replanned {planner_attempt - 1}× (user edits)")
        if writer_attempt > 1:
            notes.append(f"Writer re-drafted {writer_attempt - 1}× (Judge feedback)")
        if notes:
            st.caption(" · ".join(notes))

        st.caption(f"Input: {total_input:,} tokens · Output: {total_output:,} tokens")
        st.caption(
            "Input tokens are what you send to the model — your topic, instructions, and all "
            "source text. Output tokens are the model's response. Input is typically 3–10× "
            "larger than output, which is why input price matters even though it costs less per token."
        )
        st.markdown("")
        st.markdown("**Call detail**")
        for i, entry in enumerate(log, 1):
            in_t   = entry["input_tokens"]
            out_t  = entry["output_tokens"]
            model  = entry["model"]
            agent  = entry.get("agent", "")
            label  = f"**{agent}** | {model}" if agent else model
            price  = APPROX_PRICING.get(model)
            if price:
                call_cost = f" · ~${((in_t/1_000_000)*price[0] + (out_t/1_000_000)*price[1]):.5f}"
            else:
                call_cost = ""
            st.caption(f"Call {i}: {label} — {in_t:,} in · {out_t:,} out{call_cost}")

        st.markdown("")
        st.caption(
            "⚠️ Cost estimates are approximate. Pricing is public list rates as of June 2026. "
            "Actual billing depends on your API plan."
        )


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
