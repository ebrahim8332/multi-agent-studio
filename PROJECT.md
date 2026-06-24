# PROJECT.md — multi-agent-studio

## Status: Module 1 complete — Session 7 smart checkpoints + agent context improvements. Next module TBD.
## Last updated: 2026-06-24

---

## Module Status

| # | Module | Status |
|---|--------|--------|
| 1 | Research Assistant | Live |
| 2 | Competitive Intelligence Monitor | Not Started |
| 3 | Document Interrogator | Not Started |
| 4 | Meeting Prep Agent | Not Started |
| 5 | Regulatory Watch Agent | Not Started |
| 6 | Investment Diligence Agent | Not Started |
| 7 | Contract Risk Reviewer | Not Started |
| 8 | Earnings Call Analyzer | Not Started |

---

### Completed
- [x] CLAUDE.md spec written
- [x] PROJECT.md created

### Phase 1: Platform Shell
- [x] Full folder structure created
- [x] requirements.txt
- [x] utils/model_client.py built and tested (gemini-2.5-flash responded, 14 providers)
- [x] app.py with sidebar routing and Coming Soon placeholders
- [x] Deployed to Streamlit Cloud: https://multi-agent-studio.streamlit.app
- [x] GitHub repo created: ebrahim8332/multi-agent-studio
- [x] Feedback repo: multi-agent-studio-feedback → pushed to GitHub

### Phase 2: Module 1 — Research Assistant
- [x] m01 agents.py (5 agent functions)
- [x] m01 pipeline.py (LangGraph StateGraph)
- [x] utils/doc_builder.py
- [x] m01 ui.py (live agent panels)
- [x] Wired into app.py
- [x] End-to-end test passed (topic: AI governance frameworks for enterprise adoption)
- [x] Deployed — pushed to GitHub, Streamlit Cloud auto-deploys

### Phase 3: Next Module — TBD
- [ ] Alnoor deciding on next module. Original spec was Competitive Intelligence but reconsidering.
- [ ] Key discussion: next module should teach something architecturally new (branching, parallel agents, loops, conditional routing) — not just a repeat of the linear 5-agent pattern from Module 1.
- [ ] 22 ideas explored across enterprise, agentic, and decision-support use cases. None confirmed yet.

### Phase 4: Module 3 — Document Interrogator
- [ ] Not started

### Phase 5: Module 4 — Meeting Prep Agent
- [ ] Not started

### Phase 6: Module 5 — Regulatory Watch
- [ ] Not started

### Phase 7: Module 6 — Investment Diligence
- [ ] Not started

### Phase 8: Module 7 — Contract Risk Reviewer
- [ ] Not started

### Phase 9: Module 8 — Earnings Call Analyzer
- [ ] Not started

---

### Known Issues
- None

---

## Session Notes

### Session 7 — 2026-06-24 (smart checkpoints + agent context improvements)
**Agent-to-agent context pass**
- Critic ratings now annotated inline in Writer evidence blocks: each evidence block ends with `[Critic source rating: Strong/Adequate/Weak]`
- Judge now receives research questions list and angle — used to evaluate completeness against specific questions, not just the paper generally
- Editor now receives paper title — with explicit instruction to preserve it exactly and not add it as a heading

**Critic agent calibration**
- Concrete definitions added for Strong / Adequate / Weak ratings
- Scepticism bias added: "when in doubt between Strong and Adequate, rate Adequate; when in doubt between Adequate and Weak, rate Weak"
- Prevents artificially inflated ratings that mislead Writer and user

**Judge agent calibration**
- Scoring anchors added for all 4 dimensions (1=Poor → 5=Excellent)
- Dimension definitions added: Completeness, Argument_Quality, Source_Integration, Format_Adherence
- Scepticism bias: "when in doubt between two scores, use the lower one"
- Judge now receives research questions list + angle for accurate completeness evaluation

**Smart checkpoints**
- Critic checkpoint (critic_done): auto-proceeds silently if ALL questions rated Strong; otherwise shows verdict box (✅/⚠️/❌) + per-question table with icon, rating, and gap summary
- Judge checkpoint (judge_done): always shows verdict; single green button if all pass; otherwise shows 3 buttons (Proceed / Re-draft / Stop) with Judge's specific note for each failing dimension
- Pre-filled re-draft text box: when Judge flags issues, the re-draft input is pre-populated with specific corrective instructions built from Judge findings (word count shortfall, section count, low-scoring dimensions with notes)
- Removed duplicate "What the Judge said" section — notes already appear under each score bar

**Critic output formatting**
- Rating lines get colour icons: 🟢 Strong / 🟡 Adequate / 🔴 Weak
- Fields (Rating, Strongest source, Gap) forced onto separate lines with bold labels
- Dividers (---) between each question block for readability
- Verdict prepended at top of Critic output expander — visible at any time including after pipeline completes

**CSS fix**
- Heading sizes normalized in st.markdown: h1=1.35rem, h2=1.20rem, h3=1.05rem
- Prevents giant H1 paper title from clashing visually with H2/H3 section headings

**Writer heading rule**
- Writer instructed to use ## for top-level sections and ### for subheadings; never use # (single hash); do not add paper title as a heading

### Session 6 — 2026-06-23 (continued — three enterprise features)
**Feature 1: Token tracking and run summary**
- utils/base.py: complete() now returns tuple[str, int, int] (text, input_tokens, output_tokens)
- utils/gemini_provider.py: reads response.usage_metadata for token counts
- utils/groq_provider.py: reads response.usage for token counts
- utils/model_client.py: FallbackChain.complete() accumulates all calls into session_state["m01_call_log"]; APPROX_PRICING dict maps 14 model names to (input_price, output_price) per 1M tokens; usage_summary property computes totals
- ui.py: _show_run_summary() reads call log after run completes; shows total LLM calls, total tokens (in/out), estimated USD cost, per-call model breakdown

**Feature 2: Critic checkpoint**
- Phase state machine extended to 8 phases: idle / planner_done / running / quality_gate / critic_running / critic_done / writing / complete
- Critic now runs in critic_running phase, transitions to critic_done before starting Writer
- critic_done phase: shows full critique in expanded panel; Approve → writing; Stop → clears state
- This makes the cost of a bad Critic run explicit — user can stop before Writer burns tokens

**Feature 3: Prompt viewer**
- All LLM agents (Planner, Critic, Writer, Editor) return prompt_sent in result dict
- Stored in session_state["m01_planner_prompt"] and agent_outputs[name]["prompt"]
- _agent_panel() accepts optional prompt parameter
- Each panel with a prompt shows "🔍 View prompt sent to AI" expander; system and user sections shown separately; prompts >3000 chars truncated with notice

**Quality gate update (Session 6 earlier)**
- Researcher quality gate upgraded to two-pass: flag_weak_questions() (no LLM) + flag_irrelevant_questions() (one batched LLM call, YES/NO per question)
- Visible status ("🔍 Checking source relevance...") shown in quality_gate_ph placeholder during LLM call
- Batched design: all questions in one call using Q1/Q2/Q3 labels — fixes 60+ second silent hang from N sequential calls
- Backward compat: questions already domain-flagged are skipped in the LLM pass

### Session 5 — 2026-06-23
- Planner prompt fixed: no longer reinterprets topic through audience lens. Added explicit instruction to stay faithful to topic as stated.
- Planner now accepts user_edits parameter. When replanning, the user's edited text is passed back to the LLM as a correction signal.
- Pipeline split into two graphs: Planner runs standalone first; build_downstream_graph() runs Researcher → Critic → Writer → Editor after approval.
- UI restructured as a phase-based state machine (idle / planner_done / running / complete).
- Approval checkpoint added: Planner pauses after completing. User sees questions and chooses Approve or Edit.
- Edit mode: free text area with current questions. User can reword, delete, add, or write a plain note. Replan reruns Planner with edits as context. Attempt counter shows from attempt 2 onwards. No cap on attempts.
- Format list overhauled: White Paper / Analytical added as new default. McKinsey/Bain moved to bottom. Format hints added under dropdown.
- Default audience changed from Board / Executive team to General business audience.
- doc_builder.py redesigned: navy/blue/amber heading color hierarchy, blue horizontal rule under title, proper List Bullet style, paragraph spacing, sources in small grey text.

### Session 4 — 2026-06-22
- Model chain updated to 14-provider reference order per Multi-Agent Studio Model Chain Reference doc
- Added gemini-3-flash-preview (pos 1), gemini-3.1-flash-lite (pos 2), openai/gpt-oss-20b (last Groq slot)
- gemini-flash-latest remains last — unstable alias, appended after all Groq models
- utils/groq_provider.py and utils/model_client.py updated; pushed to GitHub via feedback repo

### Session 3 — 2026-06-22
- Search fallback chain built: Tavily → Exa → Serper (utils/search_client.py). All three tested and confirmed working.
- 6 new Gemini models added to chain. Total: 13 providers. gemini-flash-latest moved to last position (experimental alias, unpredictable output limits).
- max_tokens=8000 threaded through base.py → gemini_provider.py → groq_provider.py → FallbackChain.complete(). Writer and Editor both use it.
- Five quality improvements to agents.py:
  1. Researcher: max_results 3 → 5
  2. Critic: reads domain + 300-char snippet per source
  3. Writer: downweights Critic-flagged weak sources
  4. Writer: names gaps where no sources found
  5. Editor: receives critique, softens claims built on weak evidence
- Clear button fixed: counter key pattern (m01_form_key)
- How-it-works rewritten: agent pipeline framing, bulleted
- Output truncation fixed: gemini-flash-latest demoted to last position
- exa-py and google-search-results added to requirements.txt
- All changes pushed to GitHub

### Session 2 — 2026-06-21
- Batch 1 enhancements: larger text area, audience dropdown, spinner on active agents, sources section (collapsed), audience injected into Writer/Editor prompts
- Batch 2 enhancements: format style dropdown (5 options), paper length dropdown (3 options), both injected into Writer/Editor prompts
- Writer now generates a clean TITLE: on first line — parsed out and used in Word doc heading instead of raw user input
- Clear button: fixed using counter key pattern — increments m01_form_key, all widget keys change, form resets to defaults
- How-it-works section rewritten: agent pipeline framing, bullet points per agent
- Counter key pattern documented in style/claude-code-working-style.md Pattern Library

### Session 1 — 2026-06-21
- Full platform scoped in Claude.ai chat
- 8 modules defined and specced in CLAUDE.md
- Phase 1 complete: folder structure, model chain, app.py shell, deployed
- Phase 2 complete: Module 1 Research Assistant built and deployed
- Gemini-first fallback chain: 6 Gemini models, then 5 Groq models
- LangGraph StateGraph confirmed working with st.empty() live panel updates
- Tavily search working: search_depth=advanced, max_results=3
- Word doc download working: doc_builder.py handles markdown → .docx conversion
- Two-repo deploy confirmed: dev in multi-agent-studio/, feedback pushed to GitHub
