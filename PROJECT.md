# PROJECT.md — multi-agent-studio

## Status: Module 1 complete and polished. Next module TBD — Alnoor thinking through ideas.
## Last updated: 2026-06-21

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
- [x] utils/model_client.py built and tested (gemini-2.5-flash responded, 11 providers)
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
