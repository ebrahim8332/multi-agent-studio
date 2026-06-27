# Multi-Agent Studio: Architecture and Learning Guide

**Module 1 — Research Assistant**
*How the app works, from first click to final Word document*

---

## Who This Document Is For

This document is for two audiences.

**Learners building their first AI app.** It walks through every layer of the system — what each file does, who calls whom, and why each decision was made.

**Non-technical readers.** Plain English is used throughout. Technical terms are explained the first time they appear. Diagrams show the flow visually. You do not need to know how to code to understand the architecture.

---

## What This App Does in One Sentence

You type a research topic, click a button, and nine AI agents work in sequence and in parallel to research the web, critique sources, write two competing drafts, debate which is stronger, fact-check the winner, score it, and produce a polished paper — which you download as a Word document.

---

## The Big Picture

```
USER BROWSER
     │
     │  types topic, selects options, clicks Run
     ▼
STREAMLIT (ui.py)
     │
     │  runs Planner standalone
     ▼
Agent 1: Planner  ──▶  produces research questions
     │
     │  PAUSE — user reviews questions
     │
     ├── Approve ──────────────────────────────────────────────┐
     │                                                         │
     └── Edit + Replan ──▶ Planner reruns with user's edits   │
          (loop until approved)                                │
                                                              ▼
                                                  Agent 2: Researcher ──▶ Tavily + Exa in parallel
                                                              │             Serper fallback if both empty
                                                              │             Tavily Extract enrichment (top 2 per query)
                                                  ┌─── Quality Gate (two-pass check) ───┐
                                                  │  Pass 1: domain check (no LLM)      │
                                                  │  Pass 2: LLM relevance check        │
                                                  │  > 2 flagged? → retry (max 2x)      │
                                                  │  still > 2 flagged? → PAUSE         │
                                                  └─────────────────────────────────────┘
                                                              │
                                                  ┌── Proceed ──┐   ┌── Stop ──┐
                                                  │             │   │ (returns  │
                                                  ▼             │   │ to form)  │
                                            Agent 3: Critic ──▶ assesses source quality
                                                              │
                                                  PAUSE — user reviews critique
                                                              │
                                                  ├── Approve ──────────────────────────────┐
                                                  └── Stop here                             │
                                                                    ┌────────────────────────┘
                                                                    │  (parallel execution)
                                                       ┌────────────┴────────────┐
                                                       ▼                         ▼
                                               Agent 4: Writer A         Agent 5: Writer B
                                               (primary angle)           (contrarian angle)
                                                       │                         │
                                                       └──────────┬──────────────┘
                                                                  ▼
                                                  Agent 6: Debate Judge ──▶ picks stronger draft
                                                                  │           returns: winner, reasoning,
                                                                  │           what to incorporate, synthesis
                                                  PAUSE — user reviews Debate verdict
                                                                  │
                                                  ├── Proceed ──────────────────────────────┐
                                                  └── Stop here                             │
                                                                                            ▼
                                                                              Agent 7: Fact Checker
                                                                              cross-checks 6-10 claims
                                                                              against source evidence
                                                                                            │
                                                                  PAUSE if issues found (auto-proceed if clean)
                                                                                            │
                                                                  ├── Proceed ──────────────┐
                                                                  ├── Re-draft with feedback─┤ (Writers rerun, skipping
                                                                  └── Stop here             │  Debate + Fact Check)
                                                                                            ▼
                                                                              Agent 8: Judge
                                                                              rule check + 4-dimension LLM eval
                                                                                            │
                                                                  PAUSE — scorecard shown
                                                                                            │
                                                                  ├── Proceed ──────────────┐
                                                                  ├── Re-draft with feedback─┤ (Writers rerun, skipping
                                                                  └── Stop here             │  Debate + Fact Check)
                                                                                            ▼
                                                                              Agent 9: Editor
                                                                              polishes the winning draft
                                                                                            │
                                                                                            ▼
                                                                               DOC BUILDER (doc_builder.py)
                                                                                            │
                                                                                            ▼
                                                                               Word document → download button
```

The pipeline has six human-in-the-loop checkpoints.

**Checkpoint 1 — Planner approval.** The Planner generates research questions, then stops. The user reviews the questions and approves or edits them before any web searches run. This is the cheapest possible point to catch a wrong direction.

**Checkpoint 2 — Researcher quality gate.** After the Researcher finishes, a two-pass check runs automatically. If more than two questions return weak or irrelevant results, the Researcher retries those questions automatically (up to two times). If results are still weak after retries, the pipeline pauses and the user decides whether to proceed or stop.

**Checkpoint 3 — Critic smart checkpoint.** After the Critic rates every research question (Strong / Adequate / Weak), the pipeline applies a smart rule. If all questions are rated Strong, it proceeds automatically — no human decision needed. Otherwise it pauses and shows a verdict box (✅ Good to proceed / ⚠️ Proceed with caution / ❌ Consider stopping) and a per-question table with ratings and gap notes. The user approves or stops. This prevents the Writers — the most expensive agents — from running on a flawed evidence base.

**Checkpoint 4 — Debate Judge checkpoint.** After Writers A and B produce their competing drafts, the Debate Judge picks the stronger one. This is a human checkpoint: the user reviews the verdict (which draft won, why, and what to take from the loser) before the Fact Checker runs on the winning draft.

**Checkpoint 5 — Fact Checker smart checkpoint.** After the Fact Checker cross-checks 6-10 factual claims against source evidence, it reports which are supported, weakly supported, or unsupported. If all claims are supported and this is the first attempt, the pipeline proceeds automatically. Otherwise the user sees the verdict table and chooses to re-draft or proceed. On re-draft, the Writers rerun but Debate Judge and Fact Checker are skipped (re-draft shortcut).

**Checkpoint 6 — Judge smart checkpoint.** The Judge evaluates the winning draft on four dimensions (Completeness, Argument Quality, Source Integration, Format Adherence) each scored 1–5. It also runs a rule check (word count, section count). If all dimensions score 4+ and all rules pass, the pipeline shows a green verdict and a single "Continue to Editor" button. If any dimension or rule fails, the user sees a warning with the Judge's specific note for each issue and three options: proceed anyway, re-draft with feedback, or stop.

This is a core pattern in enterprise agentic systems: catch errors at the cheapest possible point before wasting downstream compute. Each checkpoint is positioned right before a more expensive step.

---

## The File Structure

```
multi-agent-studio/
├── app.py                              ← front door: sidebar + module routing
├── modules/
│   └── m01_research_assistant/
│       ├── ui.py                       ← user interface: form, 9 panels, checkpoints, download
│       ├── pipeline.py                 ← LangGraph graph: wires agents together
│       └── agents.py                  ← all nine agent functions
└── utils/
    ├── model_client.py                 ← AI model fallback chain (14 providers, Gemini-first)
    ├── search_client.py                ← Tavily + Exa parallel search, Serper fallback, Tavily Extract enrichment
    ├── doc_builder.py                  ← Word document builder
    ├── base.py                         ← base class: complete() returns (text, input_tokens, output_tokens)
    ├── gemini_provider.py              ← Google Gemini API wrapper with token tracking
    └── groq_provider.py                ← Groq API wrapper with token tracking
```

---

## Layer 1: app.py — The Front Door

When you open the app, `app.py` is the first file that runs. It does two things only:

1. **Renders the sidebar** — a radio button list of all eight modules
2. **Routes to the selected module** — loads and runs `ui.py` for whichever module is selected

```python
MODULES = {
    "🏠 Welcome":              None,
    "📝 Research Assistant":   "m01_research_assistant",   ← built
    "🔍 Competitive Intel":    None,                        ← coming soon
    ...
}

# When user selects Research Assistant:
mod = importlib.import_module("modules.m01_research_assistant.ui")
mod.render()
```

**What importlib does:** Python normally imports files at the top of a script. `importlib.import_module()` imports a file at runtime — when the user makes a selection — instead of at startup. This means only the selected module is loaded. It also means adding new modules later does not require changing how routing works.

**Modules marked `None`** show a "Coming Soon" placeholder automatically. No extra code needed.

---

## Layer 2: ui.py — The User Interface

`ui.py` is where everything the user sees and interacts with lives. It has four jobs:

1. Render the input form
2. Collect the user's choices
3. Run the pipeline and show live updates
4. Show the download button when done

### 2a. The Input Form

The user fills in five things:

| Input | Type | Passed to |
|---|---|---|
| Research topic | Text area | Planner, Writer, doc builder |
| Specific angle (optional) | Text input | Planner |
| Audience | Dropdown | Planner, Writer, Editor |
| Format | Dropdown | Writer, Editor |
| Length | Dropdown | Planner, Researcher, Writer, Editor |

**The counter key pattern.** Every widget in the form has a key like `m01_topic_0`. The `0` is a counter. When the user clicks Clear, the counter becomes `1`, making all keys `m01_topic_1`, `m01_angle_1`, etc. Streamlit has no memory of these new keys — so all fields reset to their defaults. This is the only reliable way to reset a form in Streamlit. Directly clearing session state after a widget has been rendered does not work.

### 2b. The Phase State Machine

`ui.py` is built as a state machine. The current phase is stored in `st.session_state["m01_phase"]`. On every render, the code reads the phase and decides what to show.

| Phase | What the user sees |
|---|---|
| `idle` | Form, all nine panels showing Waiting |
| `planner_done` | Planner panel Complete + approval UI. Agents 2-9 showing Waiting. |
| `running` | Planner locked. Researcher running, then quality gate checks. Transitions to `quality_gate` or `critic_running`. |
| `quality_gate` | Researcher Complete with warning. List of weak questions. Proceed or Stop buttons. |
| `critic_running` | Researcher locked. Critic running. |
| `critic_done` | Critic complete. Smart checkpoint: auto-proceeds if all Strong; otherwise shows verdict box + per-question table. |
| `writing_parallel` | Writers A+B running simultaneously in side-by-side bordered panels. Then Debate Judge runs. Transitions to `debate_done`. |
| `debate_done` | Writers A+B and Debate Judge complete. Human checkpoint: user sees verdict (winner, reasoning, what to incorporate). Approves before Fact Checker runs. |
| `fact_check_running` | Fact Checker running on the winning draft. |
| `fact_check_done` | Smart checkpoint: auto-proceeds to `judge_running` if all claims supported and first attempt; otherwise shows verdict table + re-draft option. |
| `judge_running` | Judge runs rule check + LLM quality eval on winning draft. |
| `judge_done` | Judge complete. Smart checkpoint: single green button if all pass; otherwise 3 buttons with issue notes + pre-filled re-draft text. |
| `editor_running` | Editor runs on winning, fact-checked, judge-approved draft. Transitions to `complete`. |
| `complete` | All nine panels Complete. Run summary, sources, and download button visible. |

This pattern — storing a phase in session state and branching on it — is the standard way to build multi-step flows in Streamlit. Adding new phases (like `quality_gate`) does not require restructuring the code — each phase is a self-contained block that reads state, renders UI, and either acts or waits.

### 2c. The Approval Checkpoint

After the Planner runs, the UI pauses at an approval area placed between the Planner panel and the Researcher panel:

```
[ Agent 1: Planner     ] ✅ Complete  (questions shown expanded)
[ Approval area        ] Approve and continue →   Edit questions
[ Agent 2: Researcher  ] ⬜ Waiting
...
```

If the user clicks **Approve**, the phase changes to `running` and the downstream pipeline starts.

If the user clicks **Edit questions**, a text area appears with the current questions. The user can reword, delete, add, or write a plain note redirecting the focus. Clicking **Replan** calls `run_planner()` again, passing the user's text as a `user_edits` parameter. The Planner incorporates those edits and generates a new set of questions. The attempt counter increments. This loop repeats until the user approves.

**Why this matters:** catching a wrong direction after the Planner is cheap. Catching it after the Researcher has run 5 web searches, the Critic has assessed them, and the Writer has drafted 2,000 words is expensive. The checkpoint stops the pipeline at the cheapest possible point.

### 2d. The Nine Agent Panels

Before the pipeline runs, `ui.py` creates placeholder panels — one per agent plus one for each approval area. A placeholder in Streamlit is a reserved space on the page that can be updated at any time without re-rendering everything around it.

Writers A and B are rendered side by side using `st.columns(2)` with bordered panels. The border color changes from blue (running) to green (complete). This visual treatment makes the parallel execution obvious — the user sees both writers working and then the Debate Judge choosing between them.

As each agent completes, its placeholder is immediately updated to show the output. The user watches the paper being built in real time.

### 2e. Streaming the Downstream Pipeline

`ui.py` runs the downstream LangGraph pipeline using `.stream()`, not `.invoke()`.

- `.invoke()` runs the whole pipeline and returns when everything is done. The user sees nothing until the end.
- `.stream()` yields a result after each agent completes. The UI updates after every step.

```python
for chunk in app.stream(full_state):
    node_name = list(chunk.keys())[0]   # which agent just finished
    updated   = chunk[node_name]         # what it returned
    # update the panel for this agent immediately
```

### 2f. Session State

Streamlit re-renders the entire page on any user interaction. Without session state, everything would be lost on every scroll or click. Key state entries:

```python
st.session_state["m01_phase"]              # current phase of the state machine
st.session_state["m01_pending_state"]      # full state dict carried through checkpoints
st.session_state["m01_planner_attempt"]    # how many times the Planner has run
st.session_state["m01_final"]             # final paper text — triggers complete phase on re-render
st.session_state["m01_full_state"]        # full pipeline state — used by download button
st.session_state["m01_agent_outputs"]     # per-agent outputs — used to restore panels on re-render
st.session_state["m01_flagged_questions"]  # questions still weak after retries — shown at quality gate
st.session_state["m01_researcher_attempt"] # how many Researcher attempts have run (initial + retries)
st.session_state["m01_planner_prompt"]     # list of message dicts sent to Planner — shown in prompt viewer
st.session_state["m01_call_log"]           # list of {model, input_tokens, output_tokens} per LLM call
```

---

## Layer 3: pipeline.py — The LangGraph Wiring

**What is LangGraph?**

LangGraph is an open-source library for building agent pipelines. It lets you define a graph — a map of which agents run in which order. Each agent is a node. Arrows between nodes are edges. The graph is then compiled into an executable object that can be run or streamed.

This module now has two graphs:

**`build_downstream_graph(chain)`** — the graph that runs after the approval checkpoint:
```
researcher → critic → writer → editor → END
```

**`build_graph(chain)`** — the original full linear graph (kept for reference):
```
planner → researcher → critic → writer → editor → END
```

The Planner no longer runs inside a LangGraph graph. It runs directly via `run_planner(state, chain)` so that the UI can pause and show the approval checkpoint before the expensive downstream agents begin.

### The State Dictionary

The state dictionary is the backbone of the pipeline. Every agent reads from it and writes back to it. Think of it as a shared whiteboard that every agent can see.

Here is every field, when it is filled, and what happens if it is empty:

| Field | Type | Filled by | What it contains | If empty |
|---|---|---|---|---|
| `topic` | string | User (before pipeline) | Raw research topic text | Pipeline cannot start |
| `angle` | string | User (before pipeline) | Optional focus area | Planner runs without it — broad coverage |
| `audience` | string | User (before pipeline) | Selected audience label | Falls back to "General business audience" |
| `format_style` | string | User (before pipeline) | Selected format label | Falls back to "McKinsey / Bain" |
| `length` | string | User (before pipeline) | Selected length label | Falls back to "Standard length" |
| `questions` | list | Agent 1: Planner | 2-6 focused research questions | Researcher has nothing to search |
| `research` | dict | Agent 2: Researcher | `{question: [result dicts]}` | Critic and Writer have no evidence |
| `sources` | list | Agent 2: Researcher | Deduplicated list of all URLs | Sources section in doc is empty |
| `critique` | string | Agent 3: Critic | Quality rating per question | Writer and Editor proceed without guidance |
| `title` | string | Agent 4: Writer | Clean paper title (8 words max) | Doc builder falls back to raw topic |
| `draft` | string | Agent 4: Writer | Full paper text in markdown | Editor has nothing to polish |
| `final` | string | Agent 5: Editor | Polished paper text | Download button shows draft instead |
| `model_used` | string | Each LLM agent | Name of last model that responded | Footer shows "unknown" |

### How the Graph is Built

```python
graph = StateGraph(ResearchState)

graph.add_node("planner",    planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("critic",     critic_node)
graph.add_node("writer",     writer_node)
graph.add_node("editor",     editor_node)

graph.set_entry_point("planner")
graph.add_edge("planner",    "researcher")
graph.add_edge("researcher", "critic")
graph.add_edge("critic",     "writer")
graph.add_edge("writer",     "editor")
graph.add_edge("editor",     END)

app = graph.compile()
```

**Why closures?** LangGraph requires every node function to accept only one argument: the state. But the Writer, Planner, Critic, and Editor also need the AI model chain. The solution is a closure — a function that wraps another function and "remembers" the chain from the outer scope.

```python
def writer_node(state: ResearchState) -> dict:
    return run_writer(state, chain)   # chain is captured from outer scope
```

From LangGraph's perspective, `writer_node` takes only `state`. The `chain` is invisible to it.

---

## Layer 4: agents.py — The Nine Agents

This is where the actual AI work happens. Each agent is a Python function that reads from the state, sends a prompt to the AI, and returns a dict with its output.

---

### Agent 1: Planner

**Job:** Break the topic into focused research questions.

**Reads from state:** `topic`, `angle`, `audience`, `length`

**Writes to state:** `questions`, `model_used`

**How question count is scaled to length:**

| Length selected | Questions generated |
|---|---|
| Short brief | 2 to 3 |
| Standard length | 4 to 5 |
| Full report | 5 to 6 |

**System prompt sent to the AI:**

```
You are a research planner. Your job is to decompose a broad topic
into focused, specific research questions that together give complete coverage.
Return ONLY a numbered list of questions, one per line. No preamble, no explanation.
```

**User prompt sent to the AI** (assembled from the inputs):

```
Topic: [user's topic]
Audience: [selected audience]
[If angle was provided: Focus the questions specifically on this angle: [angle]]

Generate [N] to [N] focused research questions that together cover this topic.
Frame each question for a [audience] — use vocabulary and concerns that matter to that audience.
Each question should be specific enough to search for directly.
Number each question (1. 2. 3. etc). One question per line. Nothing else.
```

**What the AI returns:** A numbered list like:
```
1. What are the current regulatory frameworks governing AI in healthcare?
2. How are hospital systems implementing AI for clinical decision support?
3. What are the measurable outcomes from AI-assisted diagnostics?
```

**How the Planner parses it:** A regex strips the number and punctuation from each line (`1. `, `2) `, `1 - `) and stores the clean question text in a list.

---

### Agent 2: Researcher

**Job:** Search the web for evidence on each question.

**Reads from state:** `questions`, `length`

**Writes to state:** `research`, `sources`

**Does not call the AI directly.** This is the only agent that does not use the language model for its primary output. It uses the search chain instead. (Tavily Extract enrichment is also an API call, but to a document fetching service, not an LLM.)

**Parallel search architecture.**
`search_parallel()` fires all (N questions × 2 providers) simultaneously using `ThreadPoolExecutor`. For 5 questions, that is 10 concurrent threads — 5 for Tavily, 5 for Exa. The wall-clock time for search is roughly equal to the slowest single query, not the sum of all queries.

- **TOP_N_PER_PROVIDER = 3** — maximum 3 results per provider per question; up to 6 sources per question total
- **Serper** used only when both Tavily and Exa return zero results for a question — safety net, not a parallel participant
- Results are merged by URL and deduplicated before being stored

**Source enrichment.**
After search, `enrich_top_sources()` calls Tavily Extract on the top 2 sources per query. This replaces the 400-character search snippet with up to 2,000 characters of full article text. The Writer receives far more evidence per source than a snippet alone provides. Low-authority domains (social media, forums) are excluded from enrichment.

**How results per question scaled to length (prior to parallel architecture):** This is now superseded by `TOP_N_PER_PROVIDER = 3` per provider, giving up to 6 sources per question regardless of length. Length now affects how deeply the Writer draws on those sources, not how many are collected.

**What the search chain returns for each query:**

```python
[
    {
        "title":   "Article headline",
        "url":     "https://source.com/article",
        "content": "~400 character snippet from the article"
    },
    ...
]
```

**What gets stored in `research`:**

```python
{
    "What are the current regulatory frameworks...": [
        {"title": "...", "url": "...", "content": "..."},
        {"title": "...", "url": "...", "content": "..."},
    ],
    "How are hospital systems implementing AI...": [
        ...
    ],
}
```

**What gets stored in `sources`:** A flat deduplicated list of every URL found, used for the sources section of the Word document.

### The Researcher Quality Gate

After the Researcher runs, a two-pass quality check determines whether the results are good enough to pass to the Writer.

**Pass 1 — Objective domain check (`flag_weak_questions`)**

This is pure Python. No AI model is involved. It checks two conditions:

1. **Zero results** — the search returned nothing for that question.
2. **All low-authority domains** — every result for that question comes from a list of domains that are never useful research sources: YouTube, LinkedIn, Reddit, Quora, Twitter/X, Instagram, TikTok, Pinterest, Medium, Facebook.

A question is flagged if either condition is true. This check is fast, free, and produces the same result every time for the same input.

**Pass 2 — LLM relevance check (`flag_irrelevant_questions`)**

This runs only on questions that *passed* Pass 1 (good domains, results found). It catches the subtler failure: a result from a credible source that is about the wrong thing.

For each question, one LLM call is made. The prompt sends the question and its result snippets. The AI is asked:

```
For each source, answer YES if the content directly addresses the question,
NO if it does not. One answer per line. No other text.
```

The response is parsed by counting YES and NO answers. If fewer than half the results answer YES, the question is flagged.

**Why binary YES/NO instead of ratings?** Ratings (Strong / Adequate / Weak) introduce subjectivity — the same content might get different ratings on different runs. YES/NO for a specific question reduces that variance. The LLM is being asked a narrower question.

**Why not use the Critic for this?** The Critic's job is to assess quality for the Writer — it produces narrative feedback about gaps and source authority. The quality gate needs a machine-readable signal, not prose. These are different jobs for different points in the pipeline.

**The combined check and retry loop**

The two flagged lists are merged. If more than two questions are flagged in total, the Researcher retries — but only for the flagged questions. Good results are preserved. The retry loop runs up to two times automatically. After each retry, both checks run again on the updated results.

If more than two questions are still flagged after all retries, the pipeline pauses at the `quality_gate` checkpoint. The user sees which questions are weak and chooses to proceed to the Writer or stop.

**What the Writer does with weak questions.** Whether the user proceeds from the quality gate or the results pass cleanly, the Writer is instructed to name gaps explicitly. Where a question had no sources, the paper says so plainly rather than skipping the section or inventing facts.

---

### Agent 3: Critic

**Job:** Assess the quality of the evidence found for each question. Rate each one: Strong, Adequate, or Weak. Flag gaps.

**Reads from state:** `questions`, `research`

**Writes to state:** `critique`, `model_used`

**What the Critic receives for each source:**

For every URL found, the Critic is given the article title, the domain name (extracted from the URL — e.g. `reuters.com`, `nih.gov`), and the first 300 characters of the snippet. The domain is a credibility signal — `.gov` and `.edu` sources are treated differently from blogs or forums.

Example of what one question's evidence looks like to the Critic:

```
Question: What are the current regulatory frameworks governing AI in healthcare?
  - FDA Guidance on AI in Medical Devices [fda.gov]: The FDA has issued guidance on...
  - AI Healthcare Regulation 2025 [healthcareit.com]: New regulations require...
  - AI in Medicine: What Doctors Need to Know [medium.com]: A quick overview of...
```

**System prompt sent to the AI:**

```
You are a research critic. Your job is to assess the quality of sources
found for each research question and flag where evidence is weak or missing.
Be concise and direct. Rate each question: Strong / Adequate / Weak.
Flag any significant gaps the writer should note.
```

**User prompt sent to the AI:**

```
Research summary:

[full source summary, one block per question]

For each question, provide: rating (Strong / Adequate / Weak) and one sentence
on what is missing or confirmed. Then a brief overall assessment in 2-3 sentences.
```

**What the Critic returns** — an example:

```
Question 1 — Strong: Multiple authoritative government sources confirm the regulatory landscape.
Question 2 — Adequate: Good practitioner sources but missing peer-reviewed clinical data.
Question 3 — Weak: Only two sources found, both from vendor websites. Independent research is absent.

Overall: Evidence is sufficient for an overview but thin on clinical outcomes data.
Question 3 should be flagged as a gap in the paper.
```

**What happens after the Critic.** The pipeline pauses at the Critic smart checkpoint (`critic_done` phase). If every question is rated Strong, the pipeline proceeds automatically — no user action needed. Otherwise it shows a verdict box and a per-question table with the rating, strongest source, and gap for each question. The user approves or stops. The Critic output expander always shows a verdict line at the top — visible even after the pipeline completes.

**How the Critic is calibrated.** The Critic prompt includes concrete definitions: Strong means multiple authoritative sources with specific data; Adequate means at least one relevant source with partial coverage; Weak means no sources directly address the question, or all sources are low-authority. A scepticism bias is built in: when in doubt between Strong and Adequate, rate Adequate. This prevents inflated ratings that mislead the Writer and the user.

**What the Writer does with Critic ratings.** Each evidence block the Writer receives is annotated with the Critic's rating: `[Critic source rating: Weak]`. The Writer uses this to downweight thin evidence, name gaps explicitly, and avoid building strong claims on weak sources.

**Prompt viewer.** The Critic's prompt (the full source summary and instructions) is stored in session state and shown in a "View prompt sent to AI" expander. The user can see exactly what evidence the Critic evaluated.

---

### Agent 4: Writer

**Job:** Write the full research paper using the evidence and critique.

**Reads from state:** `topic`, `questions`, `research`, `critique`, `audience`, `format_style`, `length`

**Writes to state:** `draft`, `title`, `model_used`

**The evidence block.** Before the prompt is sent, the Writer builds a structured evidence block from the research dict — one section per question, each with title and snippet per source:

```
Question: What are the regulatory frameworks governing AI in healthcare?
  - FDA Guidance on AI in Medical Devices: The FDA has issued guidance...
  - AI Healthcare Regulation 2025: New regulations require...

Question: How are hospital systems implementing AI...
  - [sources]
```

**System prompt sent to the AI:**

```
You are a research writer. Write structured, well-sourced papers
for senior business and technical audiences.

Writing rules — follow exactly:
- Short sentences. One idea per sentence. Under 20 words.
- Business formal. Direct. No hedging.
- No em dashes. Use a comma, colon, or period instead.
- No banned words: leverage, seamlessly, transformative, delve, empower, foster,
  ecosystem, paramount, unlock, thought leadership, actionable insights, cutting-edge,
  unparalleled, "it is worth noting", "in today's rapidly evolving landscape".
- Do not open with broad scene-setting. Get to the point in the first sentence.
- No motivational closing paragraph. End when the content is done.
```

**User prompt sent to the AI:**

```
Topic: [topic]
Audience: [audience]
Format: [format_style]
Target length: [length]

Format instructions:
[structure rules for the selected format, e.g. McKinsey SCR flow, or HBR case-example style]

Evidence gathered:
[structured evidence block]

Source quality assessment:
[critique from Agent 3]

Begin your response with a single line in this exact format:
TITLE: [a short, professional title — 8 words or fewer]

Then write the complete paper following the format instructions above exactly.
You must write ALL sections from start to finish without stopping early.
Do not stop mid-section or mid-sentence.
End only after you have written the Conclusions section.
Hit the target length. Calibrate vocabulary and detail for the stated audience.
Where the Critic rated a source as Weak, treat it as background context only —
do not build a key argument on it.
Where a research question had no sources found, name that gap explicitly in the
paper rather than skipping it.
Where evidence is weak or absent, say so plainly. Do not invent facts.
```

**The six format styles.** The Writer receives a different set of structure instructions depending on the format selected:

| Format | Structure instruction summary |
|---|---|
| White Paper / Analytical *(default)* | Analytical narrative. Executive Summary, then evidence-driven sections. No recommendations. Conclusions synthesise findings only. Best for research questions. |
| Harvard Business Review | Open with a real-world example. Weave in case examples. Practical takeaways at the end. |
| Academic / Research paper | Abstract, Introduction, Literature Context, Methodology, Findings, Discussion, Conclusions, References. |
| Government / Policy brief | Issue Statement, Background, Policy Landscape, Findings, Policy Options, Recommended Option, Implementation. |
| McKinsey / Bain | Open with the single most important recommendation. Use SCR flow. Numbered recommendations per section. |
| Consulting one-pager | One Executive Summary paragraph, then 3-5 bulleted sections. Max compression. Senior executive reads it in 3 minutes. |

**Why White Paper / Analytical is the default.** The original default was McKinsey / Bain. In testing, that format caused the Writer to reframe research questions as business strategy problems and open every paper with a recommendation — regardless of what the user actually asked. White Paper / Analytical explicitly instructs the Writer to describe and analyse, not prescribe. The format hint shown under the dropdown tells the user what each style produces before they choose.

**Title extraction.** The Writer is instructed to put `TITLE: [title]` on the very first line. The code strips this line, stores the clean title separately, and removes it from the draft body. The title is used as the heading in the Word document.

**max_tokens = 8,000.** The Writer requests up to 8,000 output tokens. This is the maximum response length the AI is allowed to produce. Without this, some models default to much shorter outputs and truncate long papers mid-sentence.

---

### Agent 5: Writer B

**Job:** Write a complete alternative draft of the paper using the same evidence as Writer A but from a contrarian or different angle.

**Reads from state:** `topic`, `questions`, `research`, `critique`, `audience`, `format_style`, `length`

**Writes to state:** `draft_b`, `title_b`, `model_used_b`

**Why two writers?** A single writer produces one perspective on the evidence. That perspective may be coherent but not necessarily the strongest framing. Writer B is instructed to challenge assumptions, lead with different implications, or structure the argument differently. The Debate Judge then picks the stronger draft — preserving the best thinking from both and discarding the weaker interpretation.

Writer A and Writer B run **simultaneously** via `ThreadPoolExecutor`. The user sees them working side by side in the UI. Total time for two drafts is approximately equal to one draft.

---

### Agent 6: Debate Judge

**Job:** Compare Writer A and Writer B drafts. Pick the stronger one. Identify what should be incorporated from the losing draft.

**Reads from state:** `draft`, `draft_b`, `title`, `title_b`, `critique`

**Writes to state:** `debate_result` (dict with keys: `winner`, `reasoning`, `incorporate`, `synthesis`, `model_used`, `prompt_sent`)

**What the Debate Judge evaluates:**
- Argument quality — is the core claim supported by the evidence?
- Evidence use — does the draft cite specific sources rather than making general claims?
- Structure — does the narrative flow logically from question to conclusion?
- Audience alignment — is the depth and vocabulary right for the stated audience?

**The `incorporate` field** names specific ideas, framings, or evidence uses from the losing draft that the winner should absorb. This prevents the debate from discarding good thinking along with the weaker overall structure.

**This is a human checkpoint.** The `debate_done` phase pauses the pipeline. The user reads the Debate Judge verdict before approving the move to the Fact Checker. If the winning draft is not what the user expected, they can stop here rather than running Fact Check and Judge on a draft they will not use.

---

### Agent 7: Fact Checker

**Job:** Cross-check 6-10 factual claims in the winning draft against the source evidence the Researcher collected.

**Reads from state:** `draft` (the winning draft), `research`, `sources`

**Writes to state:** `fact_check_result` (dict with keys: `claims`, `summary`, `unsupported_count`, `weak_count`, `flagged`, `model_used`, `prompt_sent`)

**What it checks:** The Fact Checker reads the winning draft and identifies specific factual claims — numbers, named entities, cause-effect statements, regulatory references, comparisons. For each claim, it checks the source evidence. It does not search the web — it checks only what the Researcher already found.

**Verdict per claim:**
- **Supported** — claim appears in source evidence
- **Weakly Supported** — source mentions the topic but does not directly confirm the claim
- **Unsupported** — no source evidence found for this claim

**Smart checkpoint.** If all claims are supported and this is the first attempt, the pipeline proceeds automatically — no human decision needed. Otherwise, the user sees a verdict table (icon | verdict | claim | source) and chooses to re-draft or proceed.

**Re-draft shortcut.** If the user chooses re-draft at either the Fact Checker or Judge checkpoint, the Writers rerun with user feedback — but Debate Judge and Fact Checker are skipped entirely on the second attempt. The rationale: re-drafted output already incorporates explicit corrective instructions; re-running debate and fact check on a re-draft adds cost without improving quality.

---

### Agent 8: Judge

**Job:** Evaluate the winning draft on four dimensions. Produce a scorecard. Decide whether the draft is ready for the Editor.

**Reads from state:** `draft` (winning draft), `critique`, `questions`, `length`

**Writes to state:** `judge_result` (rule check dict + scores dict + `flagged` bool)

**Two-pass evaluation:**

Pass 1 — Rule check (Python, no LLM, instant):
- Word count vs `LENGTH_WORD_TARGETS` minimum threshold
- Section heading count vs `LENGTH_SECTION_TARGETS` minimum
- These checks are objective and cannot be influenced by the model's estimates

Pass 2 — LLM evaluation (one call, 4 dimensions):

| Dimension | What it measures |
|---|---|
| Completeness | Does the paper address every research question? |
| Argument Quality | Is the core argument supported step by step? |
| Source Integration | Are specific sources cited rather than general claims? |
| Format Adherence | Does the structure match the selected format? |

Each dimension is scored 1–5. The Judge provides a specific note for any dimension below 3. Scepticism bias is built in — when between scores, use the lower one.

**Why inject Python word count into the Judge prompt?** LLMs cannot count words accurately — they estimate from token patterns. The rule check computes the exact word count in Python and injects it into the Judge's LLM prompt as a stated fact. The model can then interpret the count but cannot override it.

---

### Agent 9: Editor

**Job:** Polish the winning, fact-checked, judge-approved draft. Remove banned words, break long sentences, confirm structure, calibrate tone. Soften any claims that outrun the evidence.

**Reads from state:** `draft`, `critique`, `audience`, `format_style`, `length`

**Writes to state:** `final`, `model_used`

**System prompt sent to the AI:**

```
You are a senior editor. Your job is to polish research papers.
You do not change substance — only language quality and structure compliance.

[same writing rules as Writer]
```

**User prompt sent to the AI:**

```
Audience: [audience]
Format: [format_style]
Target length: [length]

Source quality assessment from the Critic:
[critique]

Edit the following paper:

[full draft from Agent 4]

Tasks:
1. Remove any banned words (replace with direct alternatives)
2. Break sentences over 20 words into shorter ones
3. Remove padding, filler phrases, and repetition
4. Confirm the structure matches the stated format exactly
5. Confirm tone and vocabulary suit the stated audience
6. Trim or expand to hit the target length
7. If any claim uses language stronger than the evidence supports, soften it —
   cross-reference the Critic's ratings to identify these

Return the complete edited paper. Preserve all headings and structure.
You must return ALL sections from start to finish. Do not stop mid-section.
```

**Why the Editor also gets the critique.** The Writer knows to downweight weak sources. But the Writer may still use strong language when building an argument on thin evidence. The Editor is the safety net — it reads the critique and softens any overclaims the Writer missed.

---

## Layer 5: The Model Chain

**What it is.** A list of 14 AI model providers, tried in order. The first one that responds successfully is locked for the rest of the session. If a locked model hits a rate limit mid-session, the chain continues from that point.

**Why a chain at all?** All models in this app are used on free API tiers. Free tiers have rate limits — a maximum number of requests per minute (TPM) and per day (TPD). If the top model is rate-limited, the app would crash without a fallback. With 14 providers, the probability of all failing simultaneously is extremely low.

**The full order (June 2026):**

| # | Model | Provider | Why this position |
|---|---|---|---|
| 0 | gemini-3-flash-preview | Gemini | GA model, matches 2.5 Pro quality, 80K output, fastest |
| 1 | gemini-3.1-flash-lite | Gemini | Outperforms 2.5 Flash on benchmarks, 381 t/s |
| 2 | gemini-2.5-flash | Gemini | Strong hybrid reasoning, 65K output |
| 3 | gemini-2.5-flash-lite | Gemini | Lighter 2.5, better than 2.0 |
| 4 | gemini-2.0-flash | Gemini | Deprecated June 2026, 8K output cap |
| 5 | gemini-2.0-flash-lite | Gemini | Deprecated June 2026, 8K output cap |
| 6 | llama-3.3-70b-versatile | Groq | Best Groq all-rounder, 86% MMLU |
| 7 | qwen/qwen3.6-27b | Groq | Replaced llama-4-scout-17b (deprecated June 2026) |
| 8 | qwen/qwen3-32b | Groq | Competitive coding/reasoning, 85.7% MMLU |
| 9 | openai/gpt-oss-120b | Groq | Large reasoning model |
| 10 | llama-3.1-8b-instant | Groq | Smallest model, high RPD |
| 11 | openai/gpt-oss-20b | Groq | Smaller/faster sibling to 120B, last Groq resort |
| 12 | gemini-2.5-pro | Gemini | Moved near-end: silent hang on free tier (accepts connections, never responds, causes 120s timeout) |
| 13 | gemini-flash-latest | Gemini | Unresolved alias — last resort only |

**Why gemini-flash-latest is last.** It is not a real model ID. It is an alias — a pointer Google updates to whatever model they are currently testing. The output limit is unpredictable. It has caused papers to be cut off mid-sentence. It stays in the chain as a final safety net but should never fire under normal operation.

**How the chain locks.** The chain index is stored in Streamlit session state under `locked_provider_index`. When model 0 succeeds, `locked_provider_index = 0`. All subsequent calls in that session start at index 0 — no retry overhead. When a locked model fails, the index advances and re-locks.

---

## Layer 6: The Search Chain

**What it is.** Two primary providers (Tavily and Exa) that run in parallel, plus Serper as a last-resort fallback. Unlike the model chain, the search chain does not lock — both primary providers are called on every run.

**Provider roles:**

| Provider | Role | Free tier | Notes |
|---|---|---|---|
| Tavily | Primary | 1,000 searches/month | AI-optimised, best snippet quality |
| Exa | Primary (parallel) | 1,000 searches/month | AI-native semantic search |
| Serper | Fallback only | 2,500 searches/month | Used only when both Tavily and Exa return zero results |

**Why parallel, not sequential?** Running Tavily and Exa one after the other means waiting for each to finish before starting the next. Running them simultaneously via `ThreadPoolExecutor` cuts total search time roughly in half. Network latency dominates search time; thread overhead is negligible.

**TOP_N_PER_PROVIDER = 3.** Each provider returns up to 3 results per question. After merging and deduplicating by URL, the Researcher has up to 6 sources per question. Serper is activated only if both Tavily and Exa return zero for a specific question.

**Source enrichment.** After the parallel search completes, `enrich_top_sources()` calls Tavily Extract on the top 2 sources per query. This fetches the full article text (up to 2,000 characters) to replace the 400-character search snippet. Low-authority domains (social media, video platforms, forums) are excluded from enrichment — they are not worth fetching in full.

**Normalised format.** Each provider returns results in a different format. The search chain normalises everything to the same structure:

```python
{"title": "...", "url": "...", "content": "..."}
```

This means the Researcher does not know or care which provider responded. It always receives the same format.

**Graceful degradation.** If all providers fail (missing API key, network error, rate limit), the search chain returns an empty list. It never raises an exception. The Critic will flag that question as having no sources. The Writer will name the gap explicitly rather than inventing facts.

---

## Layer 7: doc_builder.py — The Word Document

**What it does.** Takes the completed state dict and produces a `.docx` file as bytes. The bytes are passed directly to Streamlit's download button — no file is ever written to disk.

**What goes into the document:**

1. **Title** — from `state["title"]`, generated by the Writer. Falls back to raw topic if empty.
2. **Subtitle** — "Research Assistant | [today's date]"
3. **Main content** — from `state["final"]`, the Editor's polished output
4. **Sources** — numbered list of all URLs from `state["sources"]`
5. **Footer** — model name, date, disclaimer

**Markdown conversion.** The Writer and Editor produce text with markdown formatting — `## Heading`, `### Sub-heading`, `**bold text**`. The doc builder converts these to proper Word styles:

| Markdown | Word style |
|---|---|
| `## Heading` | Heading 1 |
| `### Sub-heading` | Heading 2 |
| `**bold text**` | Bold run |
| Blank line | Paragraph break |

**Why bytes, not a file?** Streamlit Cloud (where the app is hosted) does not give apps a persistent file system. Writing a file to disk would either fail or leave orphaned files. Returning bytes directly to the download button is the correct pattern for cloud-hosted apps.

---

## The Full Flow: Step by Step

Here is the complete journey from button click to Word document.

```
1.  User types topic, selects options, clicks "Run Research"

2.  ui.py collects all inputs and calls get_initial_state()
    → produces ResearchState with topic, angle, audience, format_style, length
    → all other fields are empty at this point

3.  ui.py calls get_chain(st.session_state)
    → builds the 14-provider model chain
    → chain is ready but no model has been called yet

4.  AGENT 1 — Planner (LLM call)
    Reads:  topic, angle, audience, length
    Sends:  system prompt + user prompt to model chain
    Model chain tries gemini-2.5-pro first
    Returns: numbered list of research questions
    Writes: questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    ui.py updates the Planner panel to ✅ Complete
    → phase changes to "planner_done"

    CHECKPOINT 1 — User sees questions and chooses:
    ├── Approve → phase changes to "running"
    └── Edit → free text area, user rewrites or redirects
                Planner reruns with user_edits parameter
                Loop repeats until approved

5.  AGENT 2 — Researcher (search calls + enrichment, no LLM)
    Reads:  questions, length
    Calls:  search_parallel(questions) — fires all (N questions × 2 providers) simultaneously
            ThreadPoolExecutor: 5 questions = 10 parallel threads (5 Tavily, 5 Exa)
            TOP_N_PER_PROVIDER = 3 per provider per question; max 6 sources per question
            Serper fires only when both Tavily and Exa return zero for a specific question
    Then:   enrich_top_sources() — Tavily Extract on top 2 sources per query (up to 2,000 chars)
            Low-authority domains excluded from enrichment
    Returns: {question: [results]} and [urls] and provider_stats
    Writes: research = {Q1: [...], Q2: [...], ...}, sources = [url1, url2, ...]
            enriched_count = N (how many sources got full text vs snippet)
    ui.py updates the Researcher panel to ✅ Complete

6.  QUALITY GATE — Pass 1: domain check (no LLM)
    flag_weak_questions(research) runs
    Checks each question: zero results? All social-media domains?
    Produces: domain_flagged = ["Q3", "Q5"]  (example)

7.  QUALITY GATE — Pass 2: LLM relevance check
    flag_irrelevant_questions(research, chain, skip=domain_flagged) runs
    ALL questions not in domain_flagged are sent in ONE batched LLM call:
      Labels questions Q1, Q2, Q3...
      Asks: for each question, is the content relevant? Answer Q1: YES or NO
      Parses response: counts YES/NO per question
      Flags any question that answers NO
    Produces: llm_flagged = ["Q2"]  (example)
    Note: one call for all questions — not N sequential calls. Avoids long hangs.
    Combined: flagged = ["Q2", "Q3", "Q5"]

    If len(flagged) > 2:
    → Researcher retries flagged questions only (up to 2 times)
    → Researcher panel shows "Re-searching N questions — attempt X of 3"
    → Both checks run again after each retry

    If still len(flagged) > 2 after retries:
    → phase changes to "quality_gate"

    CHECKPOINT 2 — User sees list of weak questions and chooses:
    ├── Proceed to Writer → phase changes to "writing"
    └── Stop here → form resets to idle

    If len(flagged) ≤ 2 (or zero):
    → no pause, proceeds directly to Critic

8.  AGENT 3 — Critic (LLM call)
    Reads:  questions, research
    Builds: source summary with title + domain + 300-char snippet per source
    Sends:  system prompt + source summary to model chain
    Returns: quality ratings and gap assessment
    Writes: critique = "Q1 — Strong: ... Q2 — Weak: ..."
    ui.py updates the Critic panel to ✅ Complete
    → phase changes to "critic_done"

    CHECKPOINT 3 — User reads the critique and chooses:
    ├── Approve → phase changes to "writing"
    └── Stop here → form resets to idle

9.  AGENTS 4+5 — Writers A and B (parallel LLM calls)
    → phase changes to "writing_parallel"
    Both writers receive same inputs: topic, questions, research, critique, audience, format_style, length
    Writer A: primary angle, standard approach
    Writer B: contrarian angle, alternative framing of the same evidence
    ThreadPoolExecutor runs both simultaneously
    Returns: draft + title (Writer A), draft_b + title_b (Writer B)
    ui.py shows side-by-side bordered panels — blue while running, green when complete

10. AGENT 6 — Debate Judge (LLM call)
    Reads:  draft, draft_b, title, title_b, critique
    Evaluates: argument quality, evidence use, structure, audience alignment
    Returns: debate_result dict: {winner, reasoning, incorporate, synthesis}
    → phase changes to "debate_done"

    CHECKPOINT 4 — User sees Debate verdict and chooses:
    ├── Approve → phase changes to "fact_check_running"
    └── Stop here → form resets to idle

11. AGENT 7 — Fact Checker (LLM call)
    → phase changes to "fact_check_running"
    Reads:  winning draft (draft or draft_b per debate_result winner), research, sources
    Identifies 6-10 specific factual claims in the draft
    Cross-checks each against source evidence (does not search the web — uses what Researcher found)
    Returns: fact_check_result dict: {claims, summary, unsupported_count, weak_count, flagged}
    → phase changes to "fact_check_done"

    CHECKPOINT 5 — Smart:
    ├── All supported + first attempt → auto-proceeds to "judge_running"
    ├── Issues found → shows verdict table → user: Re-draft or Proceed
    └── Re-draft → Writers rerun (Debate + Fact Check SKIPPED on re-draft)

12. AGENT 8 — Judge (rule check + LLM call)
    → phase changes to "judge_running"
    Reads:  winning draft, critique, questions, length
    Pass 1: Python rule check — word count vs LENGTH_WORD_TARGETS, section count vs LENGTH_SECTION_TARGETS
    Pass 2: LLM eval — scores 4 dimensions 1-5 (completeness, argument_quality, source_integration, format_adherence)
    Word count from Python injected into LLM prompt explicitly (LLMs cannot count accurately)
    Returns: {rule_check, scores, flagged, model_used}
    → phase changes to "judge_done"

    CHECKPOINT 6 — Smart:
    ├── All dimensions ≥4, all rules pass → single green "Continue to Editor" button
    ├── Issues found → 3 buttons + notes + pre-filled re-draft text
    │   Re-draft → Writers rerun (Debate + Fact Check SKIPPED on re-draft)
    └── Stop here → form resets to idle

13. AGENT 9 — Editor (LLM call)
    → phase changes to "editor_running"
    Reads:  winning draft, critique, audience, format_style, length
    Sends:  full draft + critique + editing instructions
    Returns: polished final paper
    Writes: final = "[polished paper]"
    ui.py updates the Editor panel to ✅ Complete
    → phase changes to "complete"

11. ui.py saves full state to session state
    → st.session_state["m01_final"] = final
    → st.session_state["m01_full_state"] = full_state

12. ui.py calls build_research_doc(full_state)
    → converts markdown to Word styles
    → adds title, metadata, sources, footer
    → returns bytes

13. ui.py calls _show_run_summary()
    Reads st.session_state["m01_call_log"]
    Shows: total LLM calls, total tokens (in/out), estimated USD cost
    Shows: per-call breakdown — model, input tokens, output tokens, estimated cost per call
    Note: cost is approximate, based on public list pricing as of June 2026

14. ui.py renders st.download_button with the bytes
    User clicks → Word document downloads to their computer
```

---

## Key Design Decisions

**Why LangGraph?** The pipeline could have been written as five consecutive function calls. LangGraph was chosen because it makes the agent structure explicit and visible — you can see the graph. It also provides the streaming mechanism that enables the live panel updates. For future modules with branching or parallel agents, LangGraph handles that natively.

**Why a shared state dict?** Each agent only knows about the fields it needs. But each agent also has access to everything upstream. The Writer can read the original topic, the research, and the critique — not just the research. This is by design: the more context each agent has, the better its output.

**Why not pass results directly from agent to agent?** Each agent returns only the fields it produced. LangGraph merges that partial result into the full state before passing to the next node. This means a bug in the Writer cannot accidentally overwrite the Researcher's sources.

**Why closures for the model chain?** LangGraph node functions must accept only the state argument. The model chain is not part of the state — it is a live object with connection state. Closures let us give each agent access to the chain without putting it in the state.

**Why max_tokens=8000 on Writer and Editor?** Some AI models default to short outputs. Without explicitly requesting a high token limit, long papers get cut off mid-sentence. 8,000 tokens is approximately 6,000 words — more than enough for the longest output this app produces.

**Why two repos?** Streamlit Cloud deploys directly from a GitHub repo. The dev source (`multi-agent-studio/`) is never pushed to GitHub — it is the working copy. The feedback repo (`multi-agent-studio-feedback/`) is the lean deploy copy that is pushed to GitHub. This separation means in-progress work never accidentally goes live.

**Why combine objective and LLM checks in the quality gate?** Each method catches a different failure mode. The domain check catches obvious junk fast and free — social media links, zero results. The LLM check catches the subtler problem: a credible source that answers a different question. Using LLM alone risks inconsistency; using domain check alone misses relevance failures. The two-pass design gives the benefits of both without the weaknesses of either.

**Why retry only flagged questions?** Re-running the Researcher on all questions would discard good results along with bad ones. Targeted re-search updates only the weak questions and merges the new results back in. This preserves everything that already worked and costs only the minimum additional searches needed.

**Why cap retries at two?** The same query to the same search providers will mostly return the same results. After two retries, further attempts rarely improve quality — they just cost time and API quota. The quality gate checkpoint exists precisely for the case where automated retries cannot fix the problem. The user decides whether to proceed with imperfect evidence or stop and rephrase the topic.

**Why add a Critic checkpoint?** The Writer is the most expensive agent in the pipeline. It produces 2,000 to 4,500 words using a high max_tokens setting. If the Critic produces a poor critique — a problem in a non-deterministic system — the Writer builds on a weak foundation. The Critic checkpoint pauses the pipeline after the Critic runs and before the Writer starts. The user reviews the critique and decides whether it is good enough to proceed. This is a direct application of the enterprise principle: stop before wasting compute, not after.

**Why run two writers in parallel?** A single writer makes one interpretation of the evidence. That interpretation may be correct but not necessarily the strongest framing. Writer B challenges assumptions, leads with different implications, or structures the argument from a contrarian angle. The Debate Judge picks the stronger one. The total time cost is approximately one writer (both run simultaneously), not two. The quality benefit is real: the winning draft has survived a competitive selection that a single-writer pipeline cannot provide.

**Why add a Debate Judge instead of just picking Writer A?** Writer A is not always better than Writer B. The evidence favors one framing over the other — and the model that decides which framing wins should read both drafts holistically, not defer to the order in which they were generated. The Debate Judge makes that selection explicit and explainable. The user can read the verdict and understand why one draft won.

**Why fact-check the draft before the Judge runs?** The Judge scores argument quality and source integration — but it cannot verify whether specific claims are true. A draft can score well on argument structure while containing unsupported assertions. The Fact Checker runs before the Judge specifically to flag those unsupported claims before the final quality evaluation. This also gives the user a chance to re-draft if the winning draft contains factual gaps.

**Why skip Debate and Fact Check on re-draft?** A re-draft carries the user's explicit corrective instructions. Running the debate again would pit the re-draft against the original Writer B — which did not incorporate the user's feedback. Running fact check again is redundant if the user already reviewed the findings and chose to re-draft rather than proceed. Skipping both gates on re-draft avoids spending 2-3 extra LLM calls on a known-good loop.

**Why add a Judge checkpoint?** The Writer can produce a paper that looks complete but fails on measurable dimensions: too short, missing research questions, weak argument, superficial source use. A human reading 2,000 words to evaluate these properties is slow and inconsistent. The Judge automates this evaluation and surfaces specific findings. The Judge checkpoint gives the user one structured decision point: the paper meets the bar, or here are the specific problems and a suggested fix. The pre-filled re-draft note means the user does not need to translate the Judge's findings into instructions — the corrective text is ready to send.

**Why calibrate agents with scepticism bias?** Uncalibrated LLMs tend to rate generously. A Critic that rates most sources Strong, or a Judge that scores most dimensions 4+, does not help the user make a real decision. Calibration — concrete definitions, scoring anchors, and an explicit instruction to use the lower score when in doubt — produces ratings that reflect actual quality rather than AI optimism.

**Why show token counts and cost estimates?** A production agentic system has real cost. Each LLM call consumes tokens. A 5-agent pipeline with a quality gate and retries can make 8 to 12 LLM calls in a single run. Without visibility, neither the user nor the operator knows what is being consumed. Cost awareness is a core discipline in agentic systems — this feature makes that concrete with real numbers.

**Why show the prompt sent to each agent?** The prompt is the complete instruction set the AI receives. What the AI produces is determined almost entirely by what it was asked. Showing the prompt makes the system inspectable: the user can see exactly why an agent produced the output it did. This is important for debugging, for trust, and for anyone building their own prompts.

---

## Enterprise Patterns Demonstrated

The generalizable principles from this build are captured in the **Agentic Patterns Playbook** (`outputs/Agentic Patterns Playbook.md`). That document covers 20 patterns across cost economics, quality control, human-in-the-loop design, model strategy, pipeline architecture, auditability, and UX. This section summarizes the three most important ones from Module 1.

Module 1 deliberately shows three patterns that any enterprise deploying agentic AI must address:

**Human-in-the-loop control.** Four checkpoints stop the pipeline and put a human in charge: Planner approval, Researcher quality gate, Critic smart checkpoint, and Judge smart checkpoint. None of these are optional. In a system where agents can consume API budget, trigger downstream processes, or produce documents that go to clients, automatic progression without human review is a risk. The checkpoint design shows where and why to insert human judgment — and when a checkpoint can be skipped automatically (clean run) versus when it must pause (problems found).

**Cost visibility.** Every LLM call is tracked by agent name, model, and token count. The run summary shows total calls, total tokens (input and output separately), per-agent breakdown, and an estimated cost. Two non-obvious cost dynamics become visible here: first, input tokens are typically 3–10× larger than output tokens in research pipelines, because each agent receives all prior work in its prompt. Second, a 5-agent pipeline does not cost 5× a single call — it can cost 20–50× because context accumulates. In a production system, this data goes to a cost dashboard, is logged per user, and triggers alerts if a run exceeds budget thresholds.

**Auditability.** Every agent's prompt is stored and surfaced to the user. If a paper contains a claim the user did not expect, they can trace it back to the evidence the Researcher found, the critique the Critic produced, and the exact instruction the Writer received. This audit trail is the foundation of accountability in AI-assisted work.

---

## Why a Pipeline Instead of a Chat AI?

This is the most common question from non-technical audiences. The answer is worth understanding precisely.

**What a chat AI does.**
You type a question. The model generates an answer from its training data. The training data has a cutoff date — months or years in the past. The model cannot search the web. It cannot verify its claims against current sources. It produces confident-looking text whether its information is current or fabricated.

This works well for general knowledge. It fails for research tasks where recency, source quality, and factual accuracy matter.

**What this pipeline does differently.**

| Step | Chat AI | This pipeline |
|---|---|---|
| Source of information | Model memory (training data) | Live web search — current sources |
| Source verification | None | Critic rates every source before Writer sees it |
| Gap handling | Model fills gaps with plausible-sounding text | Writer is instructed to name gaps explicitly |
| Quality check | None | Judge scores draft on four dimensions before Editor runs |
| Word count | Model estimates (often wrong) | Python counts exactly; injected into Judge prompt |
| Traceability | No sources cited | Every claim traces to a URL found by the Researcher |

**The tradeoff.**
A chat AI answers in seconds. This pipeline takes several minutes because it is doing real work at each step: live searches, source evaluation, structured drafting, quality scoring, and a final edit pass.

The output is different in kind, not just quality. A chat AI paper sounds like a paper. This pipeline produces a paper with real sources attached, explicit gap disclosures, and a structured quality record.

**Why this matters for enterprise use.**
In a business context, confident-sounding text that cannot be verified is a liability. When a report goes to a board, a client, or a regulator, the question will be: where did this come from? This pipeline produces an answer to that question. A chat AI does not.

**The deeper pattern.**
This pipeline demonstrates a principle that applies to every agentic system: structure the work so that failures are caught at the cheapest possible point. The Planner approval catches a wrong direction before any searches run. The Critic checkpoint catches poor evidence before the Writer burns tokens. The Judge checkpoint catches a thin draft before the Editor runs. Each checkpoint is positioned right before a more expensive step.

A chat AI has no equivalent structure because it has no steps. Everything happens in one forward pass. The pipeline trades speed for accountability at every stage.

---

## Making Agents Reliable Despite Non-Determinism

### The core problem

Every agent in this pipeline calls a language model. Language models are non-deterministic. The same prompt, sent twice, can return different output. A score that was 4 on Tuesday may be 3 on Wednesday. A claim flagged as Unsupported in one run may be Weakly Supported in the next.

This is not a bug. It is a fundamental property of how LLMs work. The question for any agentic system is: given that every agent has this property, how do you build a pipeline that is reliable enough to trust?

There are two categories of controls. **Hard controls** do not involve the LLM at all. They produce the same result every time for the same input. **Soft judgment** uses the LLM but structures it carefully to reduce variance. Production-grade reliability requires both.

---

### Hard controls — deterministic, always consistent

**Domain check (Researcher quality gate — Pass 1)**
Pure Python. Checks whether a search result comes from a known low-authority domain (YouTube, Reddit, LinkedIn, etc.) or returns zero results. Runs before any LLM is involved. Cannot be fooled by a convincingly written but worthless source.

**Word count and section check (Judge — Pass 1)**
Python counts the exact number of words and heading levels in the draft. This count is injected into the Judge's LLM prompt as a stated fact. The LLM interprets the count but cannot override it. LLMs cannot count words accurately — this check ensures the measurement is objective even when the evaluation is not.

**Re-draft shortcut**
When the user triggers a re-draft, Debate Judge and Fact Checker are skipped by logic in the phase state machine. This is a deterministic decision: a code condition, not an LLM judgment. It prevents redundant LLM calls on a path that has already been evaluated.

**Failure defaults**
When the Fact Checker call fails with an exception, the handler returns `flagged=True, error=True` — not a clean pass. When the Judge fails to return a score for a dimension, it defaults to `score=1` (poor), not `score=3` (neutral). These defaults are hard-coded in Python. They guarantee that a broken agent surfaces as a problem, not as a false clean pass.

---

### Soft judgment — LLM-based, structured to reduce variance

**Relevance check (Researcher quality gate — Pass 2)**
Asks the LLM a binary question per source: does this content directly address the research question? YES or NO. Binary output has lower variance than a five-point scale. The LLM cannot produce a hedged intermediate answer. Threshold logic (count YES, flag if fewer than half) is applied in Python, not by the model.

**Critic ratings**
The Critic is calibrated with concrete definitions: Strong means multiple authoritative sources with specific data; Adequate means at least one relevant source; Weak means no direct coverage or all low-authority sources. An explicit scepticism bias instruction tells the Critic to use the lower rating when uncertain. This reduces the tendency toward optimistic ratings across runs.

**Judge scores**
The Judge receives a Python-computed word count and a pass/fail rule result before it evaluates quality dimensions. This grounds the evaluation in an objective measurement. Each dimension (Completeness, Argument Quality, Source Integration, Format Adherence) has a specific definition. Scepticism bias is built in: when between scores, use the lower one.

**Fact Checker verdicts**
The Fact Checker is given only the source evidence the Researcher collected. It does not search the web or rely on its own training data. This limits hallucination surface area: the model can only flag or support claims against sources it can actually read.

---

### What this means in practice

| Check | Type | Consistent across runs? | Notes |
|---|---|---|---|
| Domain check | Hard (Python) | Always | Social media domains, zero results — no LLM |
| Word count check | Hard (Python) | Always | Count injected into Judge prompt |
| Re-draft shortcut | Hard (code logic) | Always | Phase state machine condition |
| Fact Checker failure default | Hard (exception handler) | Always | Returns flagged=True — never a silent pass |
| Judge failure default | Hard (fallback dict) | Always | Score 1, not 3 — always flags |
| Relevance check | Soft (LLM, binary) | Mostly | Binary YES/NO narrows variance |
| Critic ratings | Soft (LLM, calibrated) | Mostly | Definitions and scepticism bias reduce drift |
| Judge scores | Soft (LLM, calibrated) | Mostly | Grounded by Python word count |
| Fact Checker verdicts | Soft (LLM, scoped) | Mostly | Scoped to Researcher evidence only |

The hard controls are the floor. The soft controls are calibrated judgment on top of that floor.

---

### What this platform does well

- **Catches failures at the cheapest point.** Each checkpoint is positioned right before the next expensive agent. A wrong direction caught after the Planner costs nothing downstream.
- **Hard defaults on failure.** The Fact Checker and Judge exception handlers produce conservative defaults. A broken evaluation flags for human review — never a silent pass.
- **Binary questions where binary answers suffice.** The relevance check uses YES/NO rather than a rating. The Critic checkpoint uses automatic progression (all Strong) rather than asking the LLM to decide.
- **Python measurement, LLM interpretation.** Word count, section count, domain authority — all Python. The LLM interprets the result but cannot change it.
- **Transparent risk labeling.** The "Proceed despite unsupported claims →" button makes the user's risk explicit. Soft judgment results are never hidden behind neutral-looking proceed buttons.

---

### What remains soft — and what production would add

Several evaluations remain LLM-based with no objective backup. This is acceptable for a learning platform. A production deployment would add the following.

**Fact Checker — JSON schema output**
Currently the Fact Checker returns prose parsed by code. Parsing prose is fragile — a slight change in output format can cause silent failures. Production would require structured JSON output validated against a schema before storing the result. Schema validation failure triggers a retry before falling through to the error default.

**Claim-to-source mapping**
Currently the Fact Checker checks claims against the research block but does not produce a machine-readable link from each claim to a specific source URL. Production would require this mapping. Each supported claim would carry the URL that supports it — enabling claim-level citations in the final document, not just a sources list at the end.

**URL verification**
The Researcher stores URLs but does not verify them. A URL that returns a 404 or redirect still appears as a source. Production would verify each URL before storing it and filter dead links before the Critic and Fact Checker see them.

**Double-run on borderline scores**
The Judge currently runs once. A score of 3 on any dimension is borderline. Production would re-run the Judge on any borderline result, compare the two scores, and show the user both. Disagreement between runs is itself a signal worth surfacing.

**Auto-reject on Fact Checker API failure**
Currently a Fact Checker failure shows a red error box but still offers a proceed option. Production would remove the proceed button on a known-broken fact check. Human review of a broken evaluation is a process risk, not a safety net.

---

### The learning

The reliability of an agentic system is not primarily determined by the quality of its prompts. It is determined by what the system does when things go wrong.

Prompts can always be improved. But a prompt cannot prevent an API timeout. It cannot guarantee a consistent score across runs. It cannot stop a model from returning malformed output. What prevents those events from becoming silent failures is code: exception handlers, hard defaults, objective measurements, schema validation.

The pattern: use LLMs for judgment where human-level language understanding is required. Use Python for everything that can be computed objectively. Design failure modes conservatively — a false negative (flagging something good) is recoverable; a false positive (passing something bad) is not.

The session 10 reliability fixes — Fact Checker failure defaults, Judge score defaults, model chain restart from index 0 — are all examples of this principle applied directly.

---

## Session 8 — Key Technical Learnings

These are the non-obvious decisions and failure modes from Session 8. They recur in agentic systems and are worth understanding once rather than rediscovering each time.

**Silent API hangs vs. 429 errors.**
When an API is rate-limited, it normally returns a 429 error immediately. The fallback chain catches this and moves to the next provider in under a second. gemini-2.5-pro behaved differently: it accepted connections but never responded. No error. No signal. The chain waited the full 120-second timeout before falling through. This caused every Writer run to take 3+ minutes on the first call of a session. The fix was to move gemini-2.5-pro to near-last position in the chain. The learning: silent hangs are harder to diagnose than explicit errors. When a step is slow and there is no error, check whether the underlying provider is hanging rather than responding slowly.

**LLMs cannot count words.**
The Judge scored Format Adherence at 3/5 ("the paper is short") on a paper Python counted at 2,236 words against a 2,000-word minimum. LLMs estimate length from token patterns — they do not actually count. The fix was to inject the Python word count and pass/fail status directly into the Judge's LLM prompt. The model cannot override an objective count when it is stated explicitly. The learning: any check that requires counting, measuring, or objective comparison belongs in Python, not in the LLM prompt alone. The LLM can interpret the result, but the measurement must be computed outside the model.

**"Approximately" means optional.**
The Writer prompt said "write approximately 2,000 words." Models treat "approximately" as permission to stop early. Changing to "minimum 2,000 words. This is a hard floor, not a suggestion" with section-depth instructions and a self-check rule produced substantially longer outputs. The learning: LLMs respond to linguistic cues about constraints. Hedge words (approximately, around, roughly) signal flexibility. Hard instructions (minimum, floor, required) signal constraints. Be precise when precision matters.

**Streamlit button ghost clicks.**
Buttons without explicit `key=` parameters are identified by their position in the DOM. When the page layout changes between reruns, Streamlit can misidentify which button was clicked. Clicking "Re-draft with feedback" fired "Clear / New topic" instead, resetting the entire app. The fix was to add explicit `key=` to all 16 buttons. The learning: in any Streamlit app with dynamic layouts, every interactive element must have an explicit key.

**Parallel search architecture.**
Running Tavily and Exa sequentially meant the Researcher waited for each to complete before starting the next. Running them via `ThreadPoolExecutor(max_workers=2)` cuts search time roughly in half per question. Results are merged and deduplicated by URL. Serper is retained as a fallback only when both primary providers return zero results — it is a safety net, not a parallel participant. The learning: whenever two independent I/O operations do not depend on each other's output, run them in parallel. Network latency dominates; thread overhead is negligible.

---

## The Phase State Machine — Full

The complete phase sequence:

```
idle
  │ Run Research clicked
  ▼
planner_done  ←──── (user edits and replans, loop until approved)
  │ Approved
  ▼
running
  │ Researcher completes (parallel Tavily + Exa) + enrichment + quality gate runs
  ├── >2 flagged after retries → quality_gate → user: Proceed or Stop
  └── ≤2 flagged → continue
  ▼
critic_running
  │ Critic completes
  ▼
critic_done  ← CHECKPOINT 3: smart — auto-proceed if all Strong; else verdict box + per-question table
  │ Approved
  ▼
writing_parallel  ← Writers A+B run simultaneously (ThreadPoolExecutor)
  │ Both writers complete + Debate Judge runs
  ▼
debate_done  ← CHECKPOINT 4: user reviews winner, reasoning, what to incorporate
  │ Approved
  ▼
fact_check_running
  │ Fact Checker completes
  ▼
fact_check_done  ← CHECKPOINT 5: smart — auto-proceed if all supported + first attempt; else verdict table
  │ Approved (or re-draft → writing_parallel, skipping debate + fact check)
  ▼
judge_running
  │ Judge completes (rule check + LLM eval)
  ▼
judge_done  ← CHECKPOINT 6: smart — single green button if all pass; else 3 buttons + notes + pre-filled re-draft
  │ Approved (or re-draft → writing_parallel, skipping debate + fact check)
  ▼
editor_running
  │ Editor completes
  ▼
complete  → run summary, sources, download button
```

Six human-in-the-loop stops. Each is a decision point that prevents wasted compute downstream. The re-draft shortcut (skipping Debate Judge and Fact Checker on re-draft attempts) prevents exponential cost growth when the user iterates.

---

## Glossary

| Term | Plain English |
|---|---|
| Agent | An AI function that does one specific job: plan, search, critique, write, or edit |
| Pipeline | A sequence of agents where each one builds on the previous |
| State | A shared dictionary that carries data between agents |
| LangGraph | The library that wires agents together and handles the pipeline execution |
| Node | An agent's function in the LangGraph graph |
| Edge | A connection between two agents — defines what runs next |
| Streaming | Yielding results as each step completes, instead of waiting for all steps |
| Fallback chain | A list of providers tried in order — moves to the next if the current one fails |
| Closure | A function that remembers a variable from the scope where it was defined |
| Session state | Streamlit's mechanism for preserving data across page re-renders |
| Token | The unit AI models use to measure text length. 1,000 tokens ≈ 750 words |
| max_tokens | The maximum response length the AI is allowed to produce |
| Rate limit | A cap on how many requests you can make per minute or per day |
| TPM / TPD | Tokens per minute / tokens per day — how rate limits are measured |
| Snippet | A short excerpt (300-400 characters) from a web page, returned by search |
| Normalise | Convert different data formats into one consistent format |
| Bytes | Raw binary data — how files are passed around in memory without writing to disk |
| Alias | A name that points to another target — not a fixed model, but a pointer |
| Closure | A function that captures and remembers variables from its surrounding scope |
| Quality gate | An automated checkpoint that measures output quality and pauses the pipeline if it falls below a threshold |
| Domain check | An objective check that flags sources based on their website domain, with no AI involved |
| Relevance check | An LLM-based check that reads source content and judges whether it answers the research question |
| Targeted re-search | Re-running the Researcher on flagged questions only, merging new results back without discarding good ones |
| Two-pass check | Running two independent quality checks in sequence — one objective, one LLM — and combining their results |
| Token | The unit AI models use to measure text length. 1,000 tokens ≈ 750 words |
| Input tokens | Tokens in the prompt sent to the AI — the question, context, and instructions |
| Output tokens | Tokens in the response the AI generated |
| Estimated cost | An approximation of what the LLM calls cost, based on public per-million-token pricing |
| Call log | A record of every LLM call made in a run: model used, input tokens, output tokens |
| Run summary | A panel shown after a completed run with total calls, total tokens, and estimated cost |
| Prompt viewer | A collapsible panel in each agent's UI that shows the exact system and user message sent to the AI |
| Critic checkpoint | A smart pause after the Critic runs: auto-proceeds on a clean run, pauses with a verdict box when issues are found |
| Debate Judge checkpoint | A human pause after Writers A+B complete: user reviews which draft won and why before Fact Checker runs |
| Fact Checker checkpoint | A smart pause after the Fact Checker runs: auto-proceeds if all claims supported on first attempt; otherwise shows verdict table |
| Judge checkpoint | A smart pause after the winning draft is evaluated: auto-proceeds if all dimensions score 4+, pauses with specific notes and a pre-filled re-draft suggestion when issues are found |
| Parallel writers | Writers A and B run simultaneously via ThreadPoolExecutor — primary and contrarian angles on the same evidence |
| Re-draft shortcut | On re-draft attempt 2+, Debate Judge and Fact Checker are skipped — Writers run directly to Judge to avoid redundant evaluation |
| Source enrichment | Calling Tavily Extract after search to replace short snippets with up to 2,000 characters of full article text |
| TOP_N_PER_PROVIDER | Maximum 3 results returned per provider per question (Tavily + Exa = up to 6 per question) |
| Scepticism bias | A calibration instruction telling the agent to use the lower rating when uncertain — prevents inflated scores |
| Pre-filled re-draft | A text box auto-populated with specific corrective instructions from the Judge's findings — ready to send to the Writer |
| Human-in-the-loop | A design pattern where a human reviews and approves before the pipeline continues |
| Auditability | The ability to trace any output back to the evidence and instructions that produced it |
| Context accumulation | Each agent in a pipeline inherits all prior agents' output as input. Costs grow with each step, not linearly — the later agents have far larger prompts than the earlier ones |
| Agent label | A tag stored with each LLM call identifying which agent made it. Enables per-agent cost attribution in the run summary |

---

*Multi-Agent Studio — Research Assistant*
*Architecture and Learning Guide*
*June 2026 — updated Session 10: model chain lock bug fixed (always starts from index 0; same fix applied to all apps); debate_done phase removed (auto-proceeds to Fact Checker); Fact Checker failure now returns flagged=True (was silent clean pass); Judge failure defaults to score=1 (was score=3); "Proceed despite unsupported claims" button labelling; new section: Making Agents Reliable Despite Non-Determinism. Session 9: 9-agent pipeline (Writers A+B parallel, Debate Judge, Fact Checker, Judge, Editor); re-draft shortcut (skip Debate + Fact Check on re-draft); Tavily Extract source enrichment; TOP_N_PER_PROVIDER=3; llama-4-scout-17b replaced with qwen/qwen3.6-27b (deprecated June 2026). Prior session updates: parallel search (Tavily + Exa simultaneous), gemini-2.5-pro moved to near-last (silent hang on free tier), LLM word count fix, hard floor Writer prompt, Streamlit button key fix. See also: Agentic Patterns Playbook.*
