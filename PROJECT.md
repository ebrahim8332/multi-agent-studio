# PROJECT.md — multi-agent-studio

## Status: Module 1 live — claim-level citation system added. Module 2 (Stock Analyser) live — now 9 agents after adding a Fact Checker (Agent 6) following a human-in-the-loop reliability review.
## Last updated: 2026-07-05

---

## Module Status

| # | Module | Status |
|---|--------|--------|
| 1 | Research Assistant | Live |
| 2 | Stock Analyser (Equity Research) | Live |
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

### Phase 3: Module 2 — Stock Analyser
- [x] Spec confirmed (v2.0): replaces Competitive Intelligence Monitor in the module slot
- [x] Prototyped yfinance field availability against live tickers before writing agents.py — caught two spec mismatches early: yfinance's sector names don't match GICS (spec used "Financials", yfinance returns "Financial Services", etc.), and there's no quarterly revenue-surprise history, only EPS
- [x] modules/m02_stock/agents.py — Resolver, Data Agent (completeness gate, data quality score, peer comparison, trend data, earnings history, news tiering, macro query), Fundamentals/Business Quality/Risk analysts, Bull/Bear advocates, Synthesizer (schema-enforced JSON, research note assembled in Python)
- [x] modules/m02_stock/pipeline.py — LangGraph StateGraph (documents the flow; UI drives agents directly for checkpoint control, same convention as Module 1)
- [x] utils/doc_builder.py — added build_stock_research_doc()
- [x] utils/search_client.py — added search_tavily_only() / search_exa_only() for direct single-provider access (Tavily for news/catalysts/macro, Exa for qualitative research — not a fallback chain between them)
- [x] utils/model_client.py — added optional call_log_key param to FallbackChain/get_chain() so each module's token usage tracks independently (found and fixed during testing — Module 2's run summary was silently empty because it read from a different key than the chain wrote to)
- [x] modules/m02_stock/ui.py — phase state machine, fan-out panels (3-way then 2-way), 3 Plotly charts, explainability panel, run summary, download
- [x] Wired into app.py as "📊 Equity Research", replacing the Competitive Intelligence placeholder
- [x] End-to-end test passed: AAPL (full run, Buy/Hold/Sell rating produced, all 8 agent panels, 3 charts, evidence panel, docx download) and MSFT (second full run via script, confirmed rating="Buy")
- [x] Halt path tested: invalid ticker, and two real thin-data tickers (BBIG halts on missing ROE, GNS halts on missing revenue growth) — both halt cleanly with the exact spec error format, no LLM agent runs
- [x] Pushed to GitHub (ebrahim8332/multi-agent-studio) 2026-07-02 — Streamlit Cloud auto-deploys

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

- **2026-07-03 — yfinance failing on Streamlit Cloud specifically, likely a Yahoo IP block. UPDATE same day: appears resolved.** Module 2's Resolver started failing on every ticker, including AAPL, on the live Streamlit Cloud deploy ("Could not resolve 'AAPL' to a valid ticker"). Confirmed the same tickers resolve instantly from a local machine at the exact same time, ruling out a Yahoo-wide outage or a code bug (resolver logic tested directly, works correctly). Root cause is almost certainly Yahoo Finance rate-limiting or blocking Streamlit Community Cloud's shared server IP pool — yfinance has no official API and scrapes Yahoo's site directly, and a shared IP used by many unrelated Streamlit apps is a known target for this. Decision at the time was to wait and re-test on 2026-07-04. **Re-tested sooner, same day (2026-07-03), directly on the live URL (multi-agent-studio.streamlit.app): AAPL resolved and ran a full successful pipeline (price $308.63, market cap $4533.0B, data quality 100/100, 3 peers, 8 news articles), and VTI correctly triggered the ETF halt message rather than an error.** This confirms the block was temporary, as hypothesized, and lifted on its own well within the wait window. Status changed from "investigating" to "monitoring for recurrence" — not closed outright, since IP-based blocks like this can return if Yahoo re-flags the shared IP pool later. No code change made. If it recurs, the real options, in rough order of effort, are:
  1. Add retry-with-backoff in the Resolver/Data Agent — cheap, but won't help if the IP is fully blocked rather than just rate-limited.
  2. Try a browser-impersonation workaround (`curl_cffi`, used by some yfinance/Streamlit Cloud users) — not guaranteed, can turn into an arms race as Yahoo adjusts detection.
  3. Move off yfinance to a real, supported financial data API. Researched free-tier options on 2026-07-03 (verified via web search, not assumed from training data):
     - **Finnhub — 60 calls/min free.** Strongest candidate: covers real-time-ish quotes and basic fundamentals, and its rate limit comfortably absorbs this module's real usage pattern (a single run looks up the main ticker plus 2-3 peers plus trend history, 5+ calls per run).
     - **Financial Modeling Prep (FMP) — 250 calls/day free.** Deepest fundamentals coverage (financial statements, ratios, SEC filings) — closest match to what this module's required fields actually need — but the daily cap (not per-minute) works out to roughly 50 full analyses/day shared across all users, fine for a personal/educational tool, tight for real traffic.
     - **Twelve Data — 800 calls/day free**, but weighted toward quotes/technical indicators, thinner on deep fundamentals than this module needs.
     - **Alpha Vantage — ruled out.** Free tier has been tightened to ~25 requests/day, wouldn't survive a single full analysis run.
     - Recommendation if this path is taken: Finnhub as primary, FMP to backfill any fundamentals fields Finnhub's free tier doesn't cover. **Caveat: this is a real rebuild, not a swap** — every field name and response shape in `agents.py` (e.g. `info.get("debtToEquity")`) is built around yfinance's specific structure, so moving providers means rewriting the data-fetching layer, not changing one line.
  4. Host this module somewhere with a dedicated IP instead of Streamlit Cloud's shared pool — the code already works, this just gives it a stable identity Yahoo hasn't flagged.

---

### Ideas Backlog

- **LangGraph deep-dive module.** After learning that Modules 1 and 2 define LangGraph graphs but never actually execute them (the live UI drives everything by hand for Streamlit-rerun reasons), Alnoor wants to explore building something where LangGraph's real runtime features (checkpointing/persistence, native cycles, async human-in-the-loop pauses) are actually load-bearing, not decorative — specifically as a way to learn the framework properly rather than just install it. Candidate shapes worth considering next time this comes up: (1) an overnight/scheduled batch mode that must survive a restart mid-run — needs real persisted state, not Streamlit `session_state`; (2) a genuine debate loop (e.g., Bull/Bear go back and forth multiple rounds until a convergence condition, not a single fixed pass) — needs native cycle support; (3) an approval step that can be resolved asynchronously hours later (e.g., via email or Slack) rather than a same-session button click — needs a durable checkpoint outside the browser session.

---

## Session Notes

### Session 14 — 2026-07-05 (Module 1 claim-level citation system)

Alnoor's ask, in two parts: (1) can the final paper show which citation backs a specific number or claim, since 20+ undifferentiated sources at the bottom made it impossible to fact-check anything, and (2) confirm the user's topic/angle input is fully passed to the Planner and not silently curtailed downstream.

**Part 2 answered first (no code change needed):** confirmed by reading the actual code, not assumed. The topic `st.text_area` and angle field have no `max_chars` cap, and `run_planner()` interpolates both directly into the prompt with no truncation. Critic, Writer, Writer B, and Editor all receive the raw topic/angle too, not just the Planner. The only thing the Researcher receives is the Planner's derived question list (not the raw topic) — intentional, since search needs concrete queries, not a paragraph. No curtailment bug found.

**Part 1 — built the citation system (Option B: numbered citations, not fuzzy claim-matching):**
- `build_source_registry()` added to `utils/search_client.py` — assigns stable `S1, S2, ...` IDs to every unique source across `research`, deterministic on repeated calls (no new state field needed).
- Writer A / Writer B tag every specific factual claim with `[S#]` per new `CITATION_RULES` in agents.py — numbers, dates, named studies, quotes, attributed claims. Not required for the Writer's own analysis.
- Fact Checker extended: reports `matched_id` (its own independent judgment of the supporting source) and `cited_tag` (what the Writer actually wrote in the draft) per claim. `_resolve_claim_sources()` flags `citation_mismatch` when they differ — exact ID comparison, not fuzzy text matching.
- Editor instructed to preserve `[S#]` tags exactly; `_count_citation_tags()` verifies the count before/after in Python, not trusted to the model's self-report.
- `doc_builder.py`: citations render as superscript in the downloaded paper; the old bare-URL "Sources" list replaced with a numbered "References" section (ID + title + URL).
- Expanded the "angle" field from a single-line `st.text_input` to a `st.text_area` per Alnoor's follow-up request.

**Real bug found and fixed during testing, unrelated to the citation feature itself but exposed by it:** `GeminiProvider._call()` in `utils/gemini_provider.py` never checked `finish_reason` — a response that stopped early (token cap, safety filter, or a quota edge case) was returned as if it succeeded, producing JSON truncated mid-string that `json.loads()` then failed on with no diagnostic. Groq's provider already had this exact fix (`finish_reason == "length"` → `FallbackTrigger`) from Session 13; Gemini never got the equivalent. Added the same guard, keyed on `finish_reason != STOP` rather than a specific value, since Gemini's failure modes (safety, recitation, max tokens) are broader than Groq's single "length" signal. Also added `error_detail` to the Fact Checker's exception handler — it was silently swallowing the real exception message before this, same silent-failure class already fixed in m02.

**Verified end-to-end against real data (not synthetic fixtures):** ran the full pipeline via script — Planner → Researcher → Critic → Writer A/B → Debate Judge → Fact Checker → Editor → docx — on the topic "CEO succession planning best practices and statistics." Confirmed: 24 of 27 number-bearing sentences carried a citation tag, the Editor pass didn't lose any tags (24 before, 25 after), the References section rendered with real titles/URLs, and the Fact Checker caught one genuine mis-citation (Writer wrote `[S4]`, the actual supporting source was `[S5]`) — exactly the failure mode the feature exists to catch.

Not yet pushed to GitHub — dev source only, awaiting Alnoor's go-ahead to copy to the feedback repo and deploy.

### Session 12 — 2026-07-03 (manual testing found 4 real UI issues, all fixed)

Alnoor manually tested the live module end to end for the first time and reported issues as he found them, one at a time, holding off on fixes until the full list was consolidated. All four turned out to be real, diagnosed against the actual code (not guessed), and fixed:

1. **Data Agent summary rendered twice** — once in its own "View output" expander (forced open with `expanded=True`), once again in the checkpoint's blue info box directly below it, identical text both times. Fixed by leaving the expander at its normal collapsed default; the interactive checkpoint box is the one that should be visible by default.
2. **"Start Over" needed two clicks** — 3 of 4 Start Over buttons (Data Checkpoint, Fact Check gate, Complete) cleared pipeline state but never incremented `m02_form_key`, the counter that forces the ticker input widget to reset. Only the very first Start Over button (shown before any run starts) had that line. Added it to the other three.
3. **Halt/error message appeared at the bottom of the page** — initially reported as "VTI just sits there," which looked like a hang. Root cause: all nine agent-panel placeholders are created via `st.empty()` near the top of the page before the error phase ever runs; the error text used plain `st.error()` calls with no placeholder, so it rendered after all nine placeholders had already claimed their position, landing at the very bottom of a tall page. Fixed by adding an `error_ph` placeholder created before Agent 1's own placeholder, and writing the halt message into it.
4. **ETF/mutual fund tickers gave a generic, misleading error** — correctly halted (missing sector/margins/revenue, which funds don't have), but the message only listed stock-specific reasons (small-cap, OTC, recently listed, non-US) with no mention of funds. Added a precise, separate message keyed on yfinance's `quoteType` field (`ETF` or `MUTUALFUND`) explaining directly that the ticker is a fund, not an individual company.

Verified live in Chrome for all four: VTI shows the ETF message immediately at the top; AAPL's Data Checkpoint shows the summary once; Start Over from that checkpoint clears the form in a single click.

Pushed as commit 24e03d1.

### Session 11 — 2026-07-02 (architecture and learning guide written)

Alnoor asked for an educational architecture/learning guide covering the whole Module 2 pipeline, ticker-in to output-out, for his own learning. Written as `docs/m02-stock-analyser-architecture-and-learning-guide.md`: full agent-by-agent walkthrough with an ASCII flow diagram, six named architectural patterns (fan-out/merge, phase state machine, checkpoints, independent verification, code-enforced guardrails, structured output, idempotency guards), a real worked example using today's actual NVDA run (15 claims, 3 mismatches, forced Low confidence), a case study on the debt_to_equity fallback bug as a general lesson about fallbacks reintroducing fixed bugs, a glossary, and a set of concrete extension ideas (peer/historical fact-checking, a Devil's Advocate agent, override-rate tracking, after-the-fact accuracy tracking, auditing Module 1 for the same idempotency gap). Markdown, per Alnoor's request, so he can convert to PDF himself.

### Session 10 — 2026-07-02 (fresh adversarial review, same day as build)

Alnoor asked for a second, from-scratch adversarial review of Module 2 — explicitly with no prior build context carried over, to catch what a build-and-fix cycle can miss. Ran 4 independent review agents in parallel (agents.py data/logic, ui.py state machine, cross-output consistency, security/robustness), then personally verified every finding against the actual source code before fixing anything (2 findings from the agents turned out to already be non-issues on inspection; everything else below was confirmed real).

Fixed:
1. **debt_to_equity fallback reintroduced the exact bug fixed earlier that same day** — if `trend_data` came back empty, the code fell back to `info.get("debtToEquity")`, the field already proven to be on the wrong scale (79.5 vs. 1.34 for AAPL). Removed the fallback; now halts with a clear missing-field message instead of substituting a value already known to be wrong.
2. **No re-entrancy guard on the two parallel fan-out phases** — `fact_checking` had a guard against double-running its LLM call on a mid-flight rerun; `analysts_running` (3-way) and `advocates_running` (2-way) did not. A rerun while either was in flight could re-fire the whole fan-out, doubling cost and racing writes. Added the same guard pattern to both.
3. **No re-entrancy guard on the loading phase, and no Tavily timeout** — added an idempotency check on the resolver/data-agent steps, and an explicit 20s timeout on the Tavily calls (previously relying on the SDK's own 60s default, times 3 sequential calls).
4. **NaN from yfinance crashed the analyst-distribution parser** — `int(row.get("strongBuy", 0) or 0)` raises when yfinance returns NaN (NaN is truthy in Python). Fixed with an explicit missing-value guard.
5. **Fact Checker's P/E metric label was ambiguous** ("trailing or forward") when only one convention is ever given to the model. Now dynamically labeled per-run to match whichever one was actually fetched.
6. **Downloaded Word doc had no Fact Check section and dropped the data quality breakdown** — the only trace of a fact-check override was one buried sentence. A colleague reading only the doc (not the live app) had no way to see what was flagged. Added a dedicated Fact Check Results section (mirrors the on-screen checkpoint) and the quality-score breakdown list.
7. Two smaller fixes: the raw pre-override Synthesizer confidence value stayed latent in `synthesizer_data` (now overwritten so it can't diverge from the enforced value downstream), and Gemini's catch-all exception handler leaked raw SDK error text into the UI (now matches the safer pattern already used for its other exception branches).

Verified live: real AAPL/NVDA data via script, and a full Chrome run of the entire 9-agent pipeline end to end, including a genuine fact-check mismatch flowing correctly through to the downloaded doc's new section.

Follow-up same session: Alnoor asked for the raw-number display to be fixed. Added `format_metric_value()` in `agents.py` — percent metrics now show `74.14%` instead of `74.14`, dollar metrics show `$46.34 billion` instead of `46335873024.00` (with million/billion/trillion tiers), shared by the checkpoint gate, the Word doc's new Fact Check table, and the mismatch reason string so all three always agree. Pushed as commit 9e251aa.

### Session 9 — 2026-07-02 (Module 2 built, tested, and hardened same day)

Built Module 2 (Stock Analyser) end to end: 8 agents per spec, tested live against real tickers in Chrome. Then, prompted by Alnoor's own questions (echoing Module 1's history of silent failures and thin token budgets), ran three follow-up hardening passes in the same session:
1. Boxed the fan-out agent panels (matching M01's Tavily/Exa treatment), fixed a real Exa API bug (`search_and_contents`/`use_autoprompt` both removed from the installed exa-py version — had been silently returning zero results on every run), corrected token ceilings after testing Groq's actual per-model rate limits directly.
2. Holistic quality audit: surfaced Synthesizer JSON-parse failures as real failures instead of fake-looking successes, added an objective word-count check to flag thin agent output, strengthened depth instructions from soft "2-3 sentences" to hard word floors (roughly doubled real output depth, measured before/after), closed two data-utilization gaps (52-week range, dividend yield never reaching any prompt; data-quality score breakdown never reaching the Synthesizer).
3. Added **Agent 6: Fact Checker** after a direct human-in-the-loop reliability assessment — verifies Fundamentals/Risk numeric claims against `data_bundle` in Python, not another model's opinion. Found two real bugs by testing against live data: yfinance's `debtToEquity` field uses inconsistent internal scaling vs. the balance-sheet-computed ratio (the app was showing two different numbers for the same metric), and the extractor was flagging legitimate historical trend citations as false mismatches. Verified live: a real TSLA run caught a stale price citation, halted for a human decision, and correctly forced confidence to Low after override.

Full detail in the memory file (`project_multi_agent_studio.md`) and this project's `CLAUDE.md` Module 2 section.

### Session 8 — 2026-06-25 (parallel search, bug fixes, quality improvements, learning capture)

**Model chain update**
- Removed `meta-llama/llama-4-scout-17b-16e-instruct` — deprecated by Groq June 2026, decommission July 17 2026
- Added `qwen/qwen3.6-27b` at TIER2 — Groq's own recommended replacement
- qwen3.6-27b defaults to thinking mode; `GroqProvider._call()` strips `<think>` blocks (handles complete and truncated)
- gemini-2.5-pro moved from position 0 to position 12 — never responds on free tier (silent hang, not 429); moving it to near-last avoids 120-second timeout on every run
- After chain locks, model name stored in `session_state["locked_model_name"]` — used by running caption in `_agent_panel()`

**Parallel search**
- Researcher now calls `search_parallel()` instead of `search_multi()`
- Tavily and Exa run simultaneously via `ThreadPoolExecutor(max_workers=2)`; results merged and deduplicated by URL
- Serper used as fallback only if both Tavily and Exa return zero results for a query
- `provider_stats` stored in `agent_outputs["researcher"]["stats"]`; UI shows side-by-side Tavily | Exa columns with per-provider result counts

**Bug: app resetting on button click**
- Root cause: Streamlit ghost click bug — buttons without explicit `key=` fire incorrectly when layout changes between reruns
- Fix: added `key=` to all 16 buttons in ui.py

**Bug: Writer taking 3+ minutes**
- Root cause: gemini-2.5-pro at chain position 0 accepts connections but never responds (no error, full 120-second timeout before chain falls through)
- Fix: moved gemini-2.5-pro to chain position 12 (near-last)

**Writer word count shortfall**
- Observed: full reports coming in at 970-1,154 words against 2,000-4,500 targets
- Root cause: "approximately" is treated as advisory by models
- Fix: changed to hard floor in Writer prompt — "minimum X words. This is a hard floor, not a suggestion." Added section depth instructions and self-check rule.

**Judge word count contradiction fixed**
- Observed: rule check showed ✅ 2,236 words but LLM scored Format adherence 3/5 saying it was short
- Root cause: LLMs cannot count words accurately; Judge LLM was estimating from token count
- Fix: inject actual Python word count and pass/fail status into Judge LLM prompt — model cannot override objective count

**Active model display**
- Running agent caption now shows which model is currently locked: "⏳ Working... · gemini-3-flash-preview"
- First agent shows plain "⏳ Working..." (no model locked yet). Educational purpose: shows what provider is responding.

**Key learnings captured this session**
- Chat AI vs pipeline distinction: a chat AI answers from memory with no live search, no source verification, and no quality gates. This pipeline searches the live web first, evaluates source quality, and flags gaps explicitly. Added as "How it works" expander in ui.py.
- LLMs cannot count words. Python counting is authoritative. Injecting objective counts into LLM prompts prevents hallucinated quality assessments.
- Silent API hangs are harder to debug than 429 errors. gemini-2.5-pro never returned a 429 — it just never responded. Chain position matters when failures are silent.
- Streamlit button ghost clicks: any layout change between reruns can fire the wrong button. Explicit `key=` on every button is mandatory, not optional.
- "Approximately" in prompts means optional. Hard floors with consequences work.

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
