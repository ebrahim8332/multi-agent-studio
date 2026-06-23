"""
Agent functions for the Research Assistant module.

Each function is a LangGraph node. The signature is always:
    (state: ResearchState, chain: FallbackChain) -> dict

The return value is a partial state update — only the fields this agent changed.
LangGraph merges it back into the full state before passing to the next node.
"""

import re
from utils.search_client import get_search_chain

# Domains that are never useful as research sources.
# Used by flag_weak_questions() to detect low-quality results objectively.
LOW_AUTHORITY_DOMAINS = {
    "youtube.com", "youtu.be", "linkedin.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "pinterest.com", "medium.com",
}

# Format-specific structure instructions injected into the Writer prompt.
FORMAT_INSTRUCTIONS = {
    "White Paper / Analytical": (
        "Structure: Executive Summary (3-5 sentences stating the core argument and scope), "
        "then analytical sections each with a clear heading. "
        "Each section opens with its core finding or claim in the first sentence, "
        "then provides evidence, context, and explanation. "
        "Use bullet points to present lists of evidence, data points, or distinct items — "
        "not action items. Prose between bullets connects and interprets the evidence. "
        "Do NOT include recommendations sections or tell the reader what to do. "
        "This is an analytical document. Describe what is happening, why it matters, "
        "and what the evidence shows. Let the reader draw their own conclusions. "
        "Conclusions: synthesize the key findings and their implications. Do not prescribe actions. "
        "Tone: authoritative, direct, written for an intelligent reader who wants to understand a topic deeply."
    ),
    "McKinsey / Bain": (
        "Structure: open with the single most important recommendation. "
        "Use Situation-Complication-Resolution (SCR) flow throughout. "
        "Include a numbered list of specific recommendations at the end of each section. "
        "Use short callout sentences to highlight the key finding in each section. "
        "No padding. Every sentence must earn its place."
    ),
    "Harvard Business Review": (
        "Structure: open with a compelling real-world example or observation that sets up the problem. "
        "Weave in brief case examples throughout to ground each argument. "
        "Use clear section headings. End with practical takeaways the reader can apply immediately. "
        "Tone is authoritative but accessible — written for a smart general business reader, not specialists."
    ),
    "Academic / Research paper": (
        "Structure: Abstract (150 words), Introduction, Literature Context, Methodology / Research Approach, "
        "Findings, Discussion, Conclusions, References. "
        "Use formal academic tone. Cite sources inline where evidence is referenced. "
        "Acknowledge limitations of the research. Avoid prescriptive recommendations — present findings neutrally."
    ),
    "Government / Policy brief": (
        "Structure: Issue Statement, Background, Current Policy Landscape, Key Findings, "
        "Policy Options (at least two), Recommended Option with rationale, Implementation Considerations. "
        "Tone is neutral and formal. Each section should stand alone if extracted. "
        "Flag uncertainty clearly. No advocacy language."
    ),
    "Consulting one-pager": (
        "Structure: one tight Executive Summary paragraph, then three to five bullet-point sections "
        "each with a bold heading and two to four bullets. End with a 'So What / Next Steps' section. "
        "Maximum compression — every word must carry weight. No full paragraphs in the body. "
        "A senior executive should be able to read the whole thing in under three minutes."
    ),
}

# Writing style rules injected into every prompt that produces text.
# These enforce Alnoor's anti-AI writing style across all agent outputs.
STYLE_RULES = """
Writing rules — follow exactly:
- Short sentences. One idea per sentence. Under 20 words.
- Business formal. Direct. No hedging.
- No em dashes. Use a comma, colon, or period instead.
- No banned words: leverage, seamlessly, transformative, delve, empower, foster,
  ecosystem, paramount, unlock, thought leadership, actionable insights, cutting-edge,
  unparalleled, "it is worth noting", "in today's rapidly evolving landscape".
- Do not open with broad scene-setting. Get to the point in the first sentence.
- No motivational closing paragraph. End when the content is done.
"""


# ── Agent 1: Planner ──────────────────────────────────────────────────────────

def _question_count(length: str) -> tuple[int, int]:
    """Returns (min_questions, max_questions) based on the selected length."""
    if "Short" in length:
        return 2, 3
    elif "Full" in length:
        return 5, 6
    else:
        return 4, 5


def run_planner(state: dict, chain, user_edits: str = "") -> dict:
    """
    Breaks the topic into focused research questions.
    Question count scales to the selected length.
    Audience shapes vocabulary only — subject matter stays exactly as given.
    If user_edits is provided, the Planner uses them as a correction signal and replans.
    Returns: questions (list), model_used (str)
    """
    topic    = state["topic"]
    audience = state.get("audience", "General business audience")
    angle    = state.get("angle", "")
    length   = state.get("length", "Standard length (~2,000 words, 4-5 pages)")

    q_min, q_max = _question_count(length)

    angle_instruction = (
        f"Focus the questions specifically on this angle: {angle}\n"
        if angle else ""
    )

    # When the user has edited the previous questions, pass their intent back to the LLM.
    edit_instruction = (
        f"The user reviewed your previous questions and provided these edits or directions:\n"
        f"{user_edits}\n\n"
        "Treat the user's input as a correction. Produce a revised set of questions "
        "that reflects their intent exactly.\n\n"
        if user_edits else ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research planner. Your job is to decompose a broad topic "
                "into focused, specific research questions that together give complete coverage. "
                "Return ONLY a numbered list of questions, one per line. No preamble, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Audience: {audience}\n"
                f"{angle_instruction}"
                f"{edit_instruction}"
                f"\nGenerate {q_min} to {q_max} focused research questions that together cover this topic exactly as stated. "
                "Stay faithful to the topic — do not reinterpret it or shift to a related subject. "
                "For example, if the topic is about technology evolution, ask about technology evolution — "
                "not about business impact, enterprise adoption, or any other adjacent theme. "
                "Use vocabulary appropriate for the stated audience, but keep the subject matter exactly as given. "
                "Each question should be specific enough to search for directly. "
                "Number each question (1. 2. 3. etc). One question per line. Nothing else."
            ),
        },
    ]

    response, model = chain.complete(messages, agent_label="Planner")

    # Parse numbered list from the response.
    # Handles formats like: "1. Question", "1) Question", "1 - Question"
    questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading number and punctuation
        cleaned = re.sub(r"^\d+[\.\)\-\s]+", "", line).strip()
        if cleaned:
            questions.append(cleaned)

    # Fallback: if parsing produced nothing, split by newline and take non-empty lines
    if not questions:
        questions = [l.strip() for l in response.strip().split("\n") if l.strip()]

    # Cap at 6 questions
    questions = questions[:6]

    return {"questions": questions, "model_used": model, "prompt_sent": messages}


# ── Agent 2: Researcher ───────────────────────────────────────────────────────

def _search_depth(length: str) -> int:
    """Returns max_results per question based on selected length."""
    if "Short" in length:
        return 3
    elif "Full" in length:
        return 7
    else:
        return 5


def flag_weak_questions(research: dict) -> list[str]:
    """
    Returns a list of questions that need re-searching.

    A question is flagged if:
    - It returned zero results, OR
    - All of its results are from low-authority domains (social media, video, forums)

    No LLM involved — pure domain and count logic. This is used as the gate
    signal for the Researcher retry loop, not the Critic's subjective ratings.
    """
    from urllib.parse import urlparse
    flagged = []
    for question, hits in research.items():
        if not hits:
            flagged.append(question)
            continue
        domains = []
        for h in hits:
            try:
                domain = urlparse(h.get("url", "")).netloc.replace("www.", "")
            except Exception:
                domain = ""
            if domain:
                domains.append(domain)
        if domains and all(d in LOW_AUTHORITY_DOMAINS for d in domains):
            flagged.append(question)
    return flagged


def flag_irrelevant_questions(research: dict, chain, skip: list = None) -> list[str]:
    """
    LLM-based relevance check. Runs after flag_weak_questions() as a second pass.

    ONE LLM call for all questions (not one per question — that was too slow).
    Each question is labelled Q1, Q2, etc. The model answers YES or NO per question.
    A question is flagged if its answer is NO.

    Only checks questions not already flagged by the domain check (pass skip= to exclude them).
    If the LLM call fails for any reason, returns an empty list — no questions penalised.

    Returns a list of question strings (same format as flag_weak_questions).
    """
    skip_set = set(skip or [])

    # Build the list of questions to check (skip already-flagged ones)
    to_check = [
        (q, hits) for q, hits in research.items()
        if q not in skip_set and hits
    ]

    if not to_check:
        return []

    # Build one block per question: question text + its source snippets
    question_blocks = []
    question_index  = {}   # maps "Q1", "Q2", ... → question string
    for i, (question, hits) in enumerate(to_check, 1):
        label = f"Q{i}"
        question_index[label] = question
        snippets = []
        for j, h in enumerate(hits, 1):
            title   = h.get("title", "")
            content = h.get("content", "")[:200]
            snippets.append(f"  Source {j}: {title} — {content}")
        block = f"{label}: {question}\n" + "\n".join(snippets)
        question_blocks.append(block)

    all_blocks = "\n\n".join(question_blocks)
    labels     = ", ".join(question_index.keys())

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research quality checker. "
                "You will be given a list of research questions, each with their search results. "
                "For each question, answer YES if the majority of sources directly address it, "
                "NO if they do not. "
                f"Respond with exactly one line per question using only these labels: {labels}. "
                "Format: Q1: YES\nQ2: NO\nNo other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Questions and sources:\n\n{all_blocks}\n\n"
                f"For each question ({labels}), answer YES or NO."
            ),
        },
    ]

    try:
        response, _ = chain.complete(messages, agent_label="Quality gate")
    except Exception:
        return []  # if the check fails or times out, do not penalise any question

    # Parse "Q1: YES", "Q2: NO", etc.
    flagged = []
    for line in response.strip().split("\n"):
        line = line.strip()
        for label, question in question_index.items():
            if line.upper().startswith(label + ":"):
                answer = line[len(label) + 1:].strip().upper()
                if answer.startswith("NO"):
                    flagged.append(question)
                break

    return flagged


def run_researcher(state: dict, target_questions: list = None) -> dict:
    """
    Searches the web for research questions using the search fallback chain.
    Tries Tavily first, then Exa, then Serper. Never crashes the pipeline.
    Search depth scales to the selected length.

    target_questions: if provided, only re-searches those specific questions
    and merges the new results back into the existing research dict.
    Used by the retry loop to fix weak questions without losing good results.

    Returns: research (dict), sources (list)
    """
    questions   = target_questions if target_questions is not None else state["questions"]
    length      = state.get("length", "Standard length (~2,000 words, 4-5 pages)")
    max_results = _search_depth(length)
    search      = get_search_chain()
    new_research, new_sources = search.search_multi(questions, max_results=max_results)

    if target_questions is not None:
        # Merge into existing research — only targeted questions are updated
        merged = dict(state.get("research", {}))
        merged.update(new_research)
        existing_sources = list(state.get("sources", []))
        merged_sources   = existing_sources + [s for s in new_sources if s not in existing_sources]
        return {"research": merged, "sources": merged_sources}

    return {"research": new_research, "sources": new_sources}


# ── Agent 3: Critic ───────────────────────────────────────────────────────────

def run_critic(state: dict, chain) -> dict:
    """
    Reviews source quality for each research question.
    Flags gaps where evidence is thin or missing.
    Returns: critique (str), model_used (str)
    """
    questions = state["questions"]
    research = state["research"]

    # Build a summary including titles, domain, and content snippets for each source
    research_summary = []
    for q in questions:
        hits = research.get(q, [])
        if hits:
            source_lines = []
            for h in hits:
                title   = h.get("title", "Untitled")
                url     = h.get("url", "")
                content = h.get("content", "")[:300]
                # Extract domain for credibility signal (e.g. reuters.com, .gov, .edu)
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc.replace("www.", "")
                except Exception:
                    domain = ""
                domain_str = f" [{domain}]" if domain else ""
                source_lines.append(f"  - {title}{domain_str}: {content}")
            research_summary.append(f"Question: {q}\n" + "\n".join(source_lines))
        else:
            research_summary.append(f"Question: {q}\n  - No sources found")

    summary_text = "\n\n".join(research_summary)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research critic. Your job is to assess the quality of sources "
                "found for each research question and flag where evidence is weak or missing. "
                "Be concise and direct. Rate each question: Strong / Adequate / Weak. "
                "Flag any significant gaps the writer should note."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research summary:\n\n{summary_text}\n\n"
                "For each question, provide: rating (Strong / Adequate / Weak) and one sentence "
                "on what is missing or confirmed. Then a brief overall assessment in 2-3 sentences."
            ),
        },
    ]

    response, model = chain.complete(messages, agent_label="Critic")
    return {"critique": response, "model_used": model, "prompt_sent": messages}


# ── Agent 4: Writer ───────────────────────────────────────────────────────────

def run_writer(state: dict, chain) -> dict:
    """
    Writes a structured research paper from the gathered evidence.
    Returns: draft (str), model_used (str)
    """
    topic        = state["topic"]
    questions    = state["questions"]
    research     = state["research"]
    critique     = state["critique"]
    audience     = state.get("audience", "General business audience")
    format_style = state.get("format_style", "McKinsey / Bain")
    length       = state.get("length", "Standard paper (~2,000 words, 4-5 pages)")

    format_instructions = FORMAT_INSTRUCTIONS.get(format_style, FORMAT_INSTRUCTIONS["McKinsey / Bain"])

    # Build a structured evidence block for the LLM
    evidence_blocks = []
    for q in questions:
        hits = research.get(q, [])
        snippets = []
        for hit in hits:
            title = hit.get("title", "")
            content = hit.get("content", "")[:400]  # trim to avoid token bloat
            snippets.append(f"  - {title}: {content}")
        evidence_blocks.append(
            f"Question: {q}\n" + ("\n".join(snippets) if snippets else "  - No sources found")
        )

    evidence_text = "\n\n".join(evidence_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research writer. Write structured, well-sourced papers "
                "for senior business and technical audiences. "
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n\n"
                f"Audience: {audience}\n"
                f"Format: {format_style}\n"
                f"Target length: {length}\n\n"
                f"Format instructions:\n{format_instructions}\n\n"
                f"Evidence gathered:\n{evidence_text}\n\n"
                f"Source quality assessment:\n{critique}\n\n"
                "Begin your response with a single line in this exact format:\n"
                "TITLE: [a short, professional title for this paper — 8 words or fewer]\n\n"
                "Then write the complete paper following the format instructions above exactly. "
                "You must write ALL sections from start to finish without stopping early. "
                "Do not stop mid-section or mid-sentence. "
                "End only after you have written the Conclusions section. "
                "Hit the target length. Calibrate vocabulary and detail for the stated audience. "
                "Where the Critic rated a source as Weak, treat it as background context only — do not build a key argument on it. "
                "Where a research question had no sources found, name that gap explicitly in the paper rather than skipping it. "
                "Where evidence is weak or absent, say so plainly. Do not invent facts."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=120, max_tokens=8000, agent_label="Writer")

    # Extract TITLE: line from the top of the response
    lines = response.strip().split("\n")
    paper_title = topic  # fallback to raw topic if not found
    draft_lines = lines
    if lines and lines[0].startswith("TITLE:"):
        paper_title = lines[0][6:].strip()
        # Skip the title line and any blank line that follows
        draft_lines = lines[1:]
        while draft_lines and not draft_lines[0].strip():
            draft_lines = draft_lines[1:]

    return {"draft": "\n".join(draft_lines), "title": paper_title, "model_used": model, "prompt_sent": messages}


# ── Agent 5: Editor ───────────────────────────────────────────────────────────

def run_editor(state: dict, chain) -> dict:
    """
    Polishes the draft: removes banned words, fixes long sentences,
    tightens structure, removes padding.
    Returns: final (str), model_used (str)
    """
    draft        = state["draft"]
    critique     = state.get("critique", "")
    audience     = state.get("audience", "General business audience")
    format_style = state.get("format_style", "McKinsey / Bain")
    length       = state.get("length", "Standard paper (~2,000 words, 4-5 pages)")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior editor. Your job is to polish research papers. "
                "You do not change substance — only language quality and structure compliance. "
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Audience: {audience}\n"
                f"Format: {format_style}\n"
                f"Target length: {length}\n\n"
                f"Source quality assessment from the Critic:\n{critique}\n\n"
                "Edit the following paper:\n\n"
                f"{draft}\n\n"
                "Tasks:\n"
                "1. Remove any banned words (replace with direct alternatives)\n"
                "2. Break sentences over 20 words into shorter ones\n"
                "3. Remove padding, filler phrases, and repetition\n"
                "4. Confirm the structure matches the stated format exactly\n"
                "5. Confirm tone and vocabulary suit the stated audience\n"
                "6. Trim or expand to hit the target length\n"
                "7. If any claim uses language stronger than the evidence supports, soften it — "
                "cross-reference the Critic's ratings to identify these\n\n"
                "Return the complete edited paper. Preserve all headings and structure. "
                "You must return ALL sections from start to finish. Do not stop mid-section."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=120, max_tokens=8000, agent_label="Editor")
    return {"final": response, "model_used": model, "prompt_sent": messages}
