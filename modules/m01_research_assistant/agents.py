"""
Agent functions for the Research Assistant module.

Each function is a LangGraph node. The signature is always:
    (state: ResearchState, chain: FallbackChain) -> dict

The return value is a partial state update — only the fields this agent changed.
LangGraph merges it back into the full state before passing to the next node.
"""

import re
from utils.search_client import get_search_chain

# Word count targets for Judge rule check: {length_string: (target_words, min_acceptable)}
LENGTH_WORD_TARGETS = {
    "Short brief (~800 words, 1-2 pages)":       (800,  400),
    "Standard length (~2,000 words, 4-5 pages)": (2000, 1000),
    "Full report (~4,500 words, 9-11 pages)":    (4500, 2000),
}

# Minimum section headings expected for each length
LENGTH_SECTION_TARGETS = {
    "Short brief (~800 words, 1-2 pages)":       2,
    "Standard length (~2,000 words, 4-5 pages)": 3,
    "Full report (~4,500 words, 9-11 pages)":    4,
}

# Domains that are never useful as research sources.
# Used by flag_weak_questions() to detect low-quality results objectively.
LOW_AUTHORITY_DOMAINS = {
    "youtube.com", "youtu.be", "linkedin.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "pinterest.com", "medium.com",
}

# Format-specific question guidance injected into the Planner prompt.
# Tells the Planner what KIND of questions suit each output format.
FORMAT_QUESTION_GUIDANCE = {
    "White Paper / Analytical": (
        "The output is an analytical white paper. Questions should uncover mechanisms, causes, "
        "trends, and implications. Avoid questions framed as 'what should we do' — focus on "
        "'what is happening', 'why', and 'what does the evidence show'."
    ),
    "McKinsey / Bain": (
        "The output is a consulting deliverable. Questions should cover: current state and scale "
        "of the problem, root causes, options or approaches being used, evidence on what works, "
        "and implementation considerations. Frame questions toward decisions and recommendations."
    ),
    "Harvard Business Review": (
        "The output is a business article. Questions should uncover real-world examples, "
        "practitioner perspectives, research findings, and practical lessons. Mix strategic "
        "framing with concrete case evidence."
    ),
    "Academic / Research paper": (
        "The output is an academic paper. Questions should cover: existing literature and prior "
        "research, methodology debates, empirical findings, limitations of current knowledge, "
        "and areas of scholarly consensus vs. disagreement."
    ),
    "Government / Policy brief": (
        "The output is a policy brief. Questions should cover: scope and scale of the issue, "
        "current policy landscape, evidence on interventions, trade-offs between options, "
        "and implementation or feasibility considerations."
    ),
    "Consulting one-pager": (
        "The output is a compressed executive summary. Questions should focus on the single "
        "most important facts, one key insight per area, and what a senior executive needs "
        "to decide or act. Prioritise signal over breadth."
    ),
}

# Format-specific source quality criteria injected into the Critic prompt.
# Tells the Critic what 'good sources' look like for each format.
FORMAT_SOURCE_CRITERIA = {
    "White Paper / Analytical": (
        "Good sources: research institutions, think tanks, government data, peer-reviewed studies, "
        "reputable journalism. Weak sources: opinion pieces without data, vendor marketing, social media."
    ),
    "McKinsey / Bain": (
        "Good sources: industry reports, consulting firm research, credible business press "
        "(FT, WSJ, HBR), company filings, analyst reports. Weak sources: blog posts, general news."
    ),
    "Harvard Business Review": (
        "Good sources: business press, academic research with practical findings, case studies, "
        "executive interviews, industry data. Weak sources: unsourced claims, pure opinion."
    ),
    "Academic / Research paper": (
        "Good sources: peer-reviewed journals, academic preprints, conference papers, "
        "institutional research. Weak sources: anything not peer-reviewed or not citing primary data."
    ),
    "Government / Policy brief": (
        "Good sources: government reports, international organisation data (UN, OECD, World Bank), "
        "academic policy research, legislative records. Weak sources: advocacy material, lobbying content."
    ),
    "Consulting one-pager": (
        "Good sources: current industry data (within 2 years), credible business press, "
        "proprietary research. Weak sources: outdated statistics, generic overviews."
    ),
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

def _parse_critic_ratings(critique: str, questions: list) -> dict:
    """
    Parses Critic output to extract per-question ratings.
    Splits the critique by 'Question:' blocks (in the order the Critic received them)
    and reads the 'Rating:' line from each block.
    Falls back to 'Adequate' for any question that cannot be parsed.
    Returns: {question_text: "Strong" | "Adequate" | "Weak"}
    """
    ratings = {}
    blocks = re.split(r"\nQuestion:", "\n" + critique)
    blocks = [b.strip() for b in blocks if b.strip()]

    for i, question in enumerate(questions):
        if i < len(blocks):
            m = re.search(r"Rating:\s*(Strong|Adequate|Weak)", blocks[i], re.IGNORECASE)
            rating = m.group(1).capitalize() if m else "Adequate"
        else:
            rating = "Adequate"
        ratings[question] = rating

    return ratings


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
    Format style shapes the type of questions — not just vocabulary.
    If user_edits is provided, the Planner uses them as a correction signal and replans.
    Returns: questions (list), model_used (str)
    """
    topic        = state["topic"]
    audience     = state.get("audience", "General business audience")
    angle        = state.get("angle", "")
    length       = state.get("length", "Standard length (~2,000 words, 4-5 pages)")
    format_style = state.get("format_style", "White Paper / Analytical")

    q_min, q_max = _question_count(length)

    format_guidance = FORMAT_QUESTION_GUIDANCE.get(format_style, "")

    angle_instruction = (
        f"Specific angle to focus on: {angle}\n"
        if angle else ""
    )

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
                "You are a research planner. Your job is to decompose a topic into focused, "
                "specific research questions that together give complete coverage. "
                "The questions must be tailored to both the topic and the intended output format. "
                "Return ONLY a numbered list of questions, one per line. No preamble, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Audience: {audience}\n"
                f"Output format: {format_style}\n"
                f"{angle_instruction}"
                f"\nFormat guidance — the questions should be shaped for this output type:\n{format_guidance}\n\n"
                f"{edit_instruction}"
                f"Generate {q_min} to {q_max} focused research questions that together cover this topic exactly as stated. "
                "Stay faithful to the topic — do not reinterpret it or shift to a related subject. "
                "Each question should be specific enough to search for directly on the web. "
                "Use vocabulary appropriate for the stated audience. "
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
        return 8
    else:
        return 6


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
    Searches the web for research questions using parallel search (Tavily + Exa simultaneously).
    Serper is used as fallback only if both primary providers return zero results for a query.
    Search depth scales to the selected length.

    target_questions: if provided, only re-searches those specific questions
    and merges the new results back into the existing research dict.
    Used by the retry loop to fix weak questions without losing good results.

    Returns: research (dict), sources (list), provider_stats (dict)
    """
    questions   = target_questions if target_questions is not None else state["questions"]
    length      = state.get("length", "Standard length (~2,000 words, 4-5 pages)")
    max_results = _search_depth(length)
    search      = get_search_chain()
    new_research, new_sources, provider_stats = search.search_parallel(questions, max_results=max_results)

    prompt_sent = [
        {"role": "system", "content": f"Search engines: Tavily + Exa (parallel)\nMax results per query per engine: {max_results}"},
        {"role": "user",   "content": "Search queries:\n\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))},
    ]

    if target_questions is not None:
        # Merge into existing research — only targeted questions are updated
        merged = dict(state.get("research", {}))
        merged.update(new_research)
        existing_sources = list(state.get("sources", []))
        merged_sources   = existing_sources + [s for s in new_sources if s not in existing_sources]
        existing_stats   = dict(state.get("provider_stats", {}))
        existing_stats.update(provider_stats)
        return {"research": merged, "sources": merged_sources, "provider_stats": existing_stats, "prompt_sent": prompt_sent}

    return {"research": new_research, "sources": new_sources, "provider_stats": provider_stats, "prompt_sent": prompt_sent}


# ── Agent 3: Critic ───────────────────────────────────────────────────────────

def run_critic(state: dict, chain) -> dict:
    """
    Reviews source quality for each research question.
    Has full context: topic, audience, format, length, and format-specific source criteria.
    Returns: critique (str), model_used (str)
    """
    topic        = state["topic"]
    audience     = state.get("audience", "General business audience")
    format_style = state.get("format_style", "White Paper / Analytical")
    length       = state.get("length", "Standard length (~2,000 words, 4-5 pages)")
    questions    = state["questions"]
    research     = state["research"]

    source_criteria = FORMAT_SOURCE_CRITERIA.get(format_style, "")

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
                "You are a research critic. Assess the quality of sources found for each "
                "research question. Your ratings directly inform the Writer and Judge — "
                "be specific, not generic.\n\n"
                "Rating definitions — apply these exactly:\n"
                "Strong: Multiple sources directly address the question with specific data, "
                "research, or evidence. At least one source clearly meets the quality standard "
                "for this output format.\n"
                "Adequate: At least one source is relevant, but coverage is partial, sources "
                "are borderline quality for this format, or evidence lacks specificity.\n"
                "Weak: No sources directly address the question, OR all sources are from "
                "low-authority or generic domains, OR sources address a related topic rather "
                "than the question itself.\n\n"
                "Calibration: default to scepticism. When in doubt between Strong and Adequate, "
                "rate Adequate. When in doubt between Adequate and Weak, rate Weak. "
                "Do not rate Strong unless the evidence is genuinely specific and authoritative.\n\n"
                "For each question provide exactly:\n"
                "Rating: [Strong / Adequate / Weak]\n"
                "Strongest source: [title and domain of the best source, or 'none']\n"
                "Gap: [one sentence on what is missing or thin, or 'none']\n\n"
                "End with an Overall Assessment (3-4 sentences) covering: "
                "breadth of coverage, most significant gaps, and any topic areas the Writer "
                "should treat cautiously due to weak evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Audience: {audience}\n"
                f"Output format: {format_style}\n"
                f"Target length: {length}\n\n"
                f"Source quality standard for this format:\n{source_criteria}\n\n"
                f"Research summary:\n\n{summary_text}\n\n"
                "Assess each question using the format specified. Apply the source quality "
                "standard for this output format — a source acceptable for a blog post may "
                "be Weak for an academic paper. Be direct. Name specific gaps."
            ),
        },
    ]

    response, model = chain.complete(messages, agent_label="Critic")
    return {"critique": response, "model_used": model, "prompt_sent": messages}


# ── Agent 4: Writer ───────────────────────────────────────────────────────────

def run_writer(state: dict, chain, user_feedback: str = "") -> dict:
    """
    Writes a structured research paper from the gathered evidence.
    If user_feedback is provided (re-draft), the feedback is injected as a correction note.
    Returns: draft (str), title (str), model_used (str)
    """
    topic        = state["topic"]
    questions    = state["questions"]
    research     = state["research"]
    critique     = state["critique"]
    audience     = state.get("audience", "General business audience")
    format_style = state.get("format_style", "McKinsey / Bain")
    length       = state.get("length", "Standard paper (~2,000 words, 4-5 pages)")
    angle        = state.get("angle", "")

    format_instructions = FORMAT_INSTRUCTIONS.get(format_style, FORMAT_INSTRUCTIONS["McKinsey / Bain"])
    target_words, _     = LENGTH_WORD_TARGETS.get(length, (2000, 1000))

    angle_instruction = (
        f"Specific angle to maintain throughout: {angle}\n"
        if angle else ""
    )

    # Parse Critic ratings so each evidence block can be annotated inline.
    # This lets the Writer see source quality for each question without cross-referencing.
    critic_ratings = _parse_critic_ratings(critique, questions)

    # Build a structured evidence block — include domain and Critic rating per question
    evidence_blocks = []
    for q in questions:
        rating = critic_ratings.get(q, "Adequate")
        hits = research.get(q, [])
        snippets = []
        for hit in hits:
            title   = hit.get("title", "")
            content = hit.get("content", "")[:700]
            url     = hit.get("url", "")
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.replace("www.", "")
            except Exception:
                domain = ""
            domain_str = f" [{domain}]" if domain else ""
            snippets.append(f"  - {title}{domain_str}: {content}")
        evidence_blocks.append(
            f"Question: {q} [Critic source rating: {rating}]\n"
            + ("\n".join(snippets) if snippets else "  - No sources found")
        )

    evidence_text = "\n\n".join(evidence_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research writer producing structured, well-sourced papers "
                "for senior business and technical audiences. "
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Audience: {audience}\n"
                f"Format: {format_style}\n"
                f"Target length: minimum {target_words:,} words\n"
                f"{angle_instruction}"
                f"\nFormat instructions:\n{format_instructions}\n\n"
                f"Evidence gathered:\n{evidence_text}\n\n"
                f"Source quality assessment from the Critic:\n{critique}\n\n"
                + (
                    f"IMPORTANT — RE-DRAFT: A previous draft was reviewed and found lacking. "
                    f"The reviewer's feedback:\n{user_feedback}\n\n"
                    f"Address all feedback points in this new draft.\n\n"
                    if user_feedback else ""
                ) +
                "Begin your response with a single line in this exact format:\n"
                "TITLE: [a short, professional title for this paper — 8 words or fewer]\n\n"
                "Then write the complete paper. Follow these rules exactly:\n"
                "1. SYNTHESISE — do not write one section per research question. "
                "Identify the key themes and arguments that run across the evidence, "
                "then organise the paper around those themes. The paper should read as "
                "integrated analysis, not as a sequential answer to each question.\n"
                "2. STRUCTURE — follow the format instructions above exactly.\n"
                "3. HEADINGS — use ## for all top-level section headings, ### for subheadings. "
                "Never use # (single hash) anywhere in the paper. "
                "Do NOT repeat the paper title as a heading — begin directly with the first section.\n"
                f"4. LENGTH — write a minimum of {target_words:,} words. This is a hard floor, not a suggestion. "
                "Every section must be fully developed: multiple paragraphs of analysis, evidence, and interpretation. "
                "Do not summarise a point in one sentence when two paragraphs of depth are warranted. "
                "Do not stop early. Before finishing, check that every section is substantive. "
                "If any section feels thin, expand it before submitting.\n"
                "5. SOURCES — where the Critic rated a source as Weak, use it as background "
                "context only. Do not build a key argument on it.\n"
                "6. GAPS — where a research question had no sources, name that gap explicitly "
                "rather than skipping it. Where evidence is weak, say so plainly.\n"
                "7. AUDIENCE — calibrate vocabulary, depth, and assumed knowledge for the stated audience."
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


# ── Agent 5: Judge ────────────────────────────────────────────────────────────

def run_judge(state: dict, chain) -> dict:
    """
    Evaluates the Writer's draft before it goes to the Editor.

    Two-pass:
      Pass 1 — rule check (free, instant): word count vs target, section heading count.
      Pass 2 — LLM evaluation (one call): scores four dimensions 1–5 with a one-line note each.

    Returns a structured result dict with rule_check, scores, flagged flag, and prompt_sent.
    flagged is True if any score < 3 or if the rule check failed critically.
    """
    draft        = state.get("draft", "")
    topic        = state.get("topic", "")
    questions    = state.get("questions", [])
    angle        = state.get("angle", "")
    format_style = state.get("format_style", "White Paper / Analytical")
    length       = state.get("length", "Standard length (~2,000 words, 4-5 pages)")
    critique     = state.get("critique", "")

    # ── Pass 1: Rule check ─────────────────────────────────────────────────────
    words          = len(draft.split())
    target_words, min_words = LENGTH_WORD_TARGETS.get(length, (2000, 1000))
    word_count_ok  = words >= min_words

    section_count  = sum(1 for line in draft.split("\n") if line.strip().startswith("#"))
    min_sections   = LENGTH_SECTION_TARGETS.get(length, 3)
    sections_ok    = section_count >= min_sections

    rule_check = {
        "word_count":        words,
        "word_count_target": target_words,
        "word_count_ok":     word_count_ok,
        "section_count":     section_count,
        "min_sections":      min_sections,
        "sections_ok":       sections_ok,
    }

    # ── Pass 2: LLM evaluation ─────────────────────────────────────────────────
    target_words, _ = LENGTH_WORD_TARGETS.get(length, (2000, 1000))
    format_instructions = FORMAT_INSTRUCTIONS.get(format_style, "")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict editorial quality evaluator. "
                "Score a research paper draft on exactly four dimensions using a 1–5 scale "
                "where 1 = very poor, 3 = acceptable, 5 = excellent. "
                "Be critical. Reserve 5 for genuinely strong work. A score of 3 means the "
                "dimension is acceptable but has clear room for improvement. "
                "You have full context: the topic, the research questions, the required format "
                "with its specific structural requirements, the source quality assessment, "
                "and the complete draft.\n\n"
                "Dimension definitions:\n"
                "COMPLETENESS: does the paper substantively address every research question listed? "
                "Flag any question that is skipped, underdeveloped, or mentioned only in passing.\n"
                "ARGUMENT_QUALITY: are claims well-reasoned and supported by evidence? "
                "Does the paper synthesise across sources rather than summarise them serially?\n"
                "SOURCE_INTEGRATION: are sources cited in context, with their quality accurately "
                "reflected? Are Weak-rated sources treated with appropriate caution?\n"
                "FORMAT_ADHERENCE: does the structure, tone, and organisation match the required "
                "format exactly? Check against the format instructions provided.\n\n"
                "Scoring anchors — apply to all dimensions:\n"
                "5 = Excellent. Genuinely strong. No significant weaknesses.\n"
                "4 = Good. One minor weakness that does not undermine the work.\n"
                "3 = Acceptable. Meets the minimum bar but has clear room for improvement.\n"
                "2 = Below standard. Multiple weaknesses that affect the value of the work.\n"
                "1 = Poor. Fails the basic requirement of this dimension.\n\n"
                "Calibration: reserve 4+ for work that is clearly good by the standards of the "
                "target format. When in doubt between two scores, use the lower one.\n\n"
                "Respond with exactly four lines — nothing else:\n"
                "COMPLETENESS: [1-5] | [one sentence note]\n"
                "ARGUMENT_QUALITY: [1-5] | [one sentence note]\n"
                "SOURCE_INTEGRATION: [1-5] | [one sentence note]\n"
                "FORMAT_ADHERENCE: [1-5] | [one sentence note]"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Target format: {format_style}\n"
                f"Target length: approximately {target_words:,} words\n"
                + (f"Required angle: {angle}\n" if angle else "")
                + f"\nResearch questions the paper must address:\n"
                + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))
                + f"\n\nWhat FORMAT_ADHERENCE means for this format:\n{format_instructions}\n\n"
                f"Critic's source quality assessment:\n{critique}\n\n"
                f"Complete draft to evaluate:\n{draft}"
            ),
        },
    ]

    scores = {}
    model  = ""
    try:
        response, model = chain.complete(messages, agent_label="Judge")
        for line in response.strip().split("\n"):
            line = line.strip()
            m = re.match(
                r"(COMPLETENESS|ARGUMENT_QUALITY|SOURCE_INTEGRATION|FORMAT_ADHERENCE)"
                r":\s*([1-5])\s*\|(.+)", line, re.IGNORECASE,
            )
            if m:
                key   = m.group(1).lower()
                score = int(m.group(2))
                note  = m.group(3).strip()
                scores[key] = {"score": score, "note": note}
    except Exception:
        pass

    # Fill any missing dimensions with a neutral placeholder
    for dim in ("completeness", "argument_quality", "source_integration", "format_adherence"):
        if dim not in scores:
            scores[dim] = {"score": 3, "note": "Could not evaluate."}

    flagged = (
        not rule_check["word_count_ok"]
        or not rule_check["sections_ok"]
        or any(v["score"] < 3 for v in scores.values())
    )

    return {
        "rule_check":  rule_check,
        "scores":      scores,
        "flagged":     flagged,
        "model_used":  model,
        "prompt_sent": messages,
    }


# ── Agent 6: Editor ───────────────────────────────────────────────────────────

def run_editor(state: dict, chain) -> dict:
    """
    Polishes the draft: removes banned words, fixes long sentences,
    tightens structure, removes padding. Has full context from the pipeline.
    Returns: final (str), model_used (str)
    """
    topic        = state["topic"]
    draft        = state["draft"]
    title        = state.get("title", "")
    critique     = state.get("critique", "")
    audience     = state.get("audience", "General business audience")
    format_style = state.get("format_style", "McKinsey / Bain")
    length       = state.get("length", "Standard paper (~2,000 words, 4-5 pages)")
    angle        = state.get("angle", "")

    format_instructions = FORMAT_INSTRUCTIONS.get(format_style, "")
    target_words, _     = LENGTH_WORD_TARGETS.get(length, (2000, 1000))

    angle_instruction = (
        f"The paper should maintain this specific angle throughout: {angle}\n"
        if angle else ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior editor. Your job is to polish research papers. "
                "You do not change substance or restructure arguments — only improve "
                "language quality, sentence clarity, and structure compliance. "
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                + (f"Paper title: {title}\n" if title else "")
                + f"Audience: {audience}\n"
                f"Format: {format_style}\n"
                f"Target length: approximately {target_words:,} words\n"
                f"{angle_instruction}"
                f"\nWhat this format requires:\n{format_instructions}\n\n"
                f"Source quality assessment from the Critic:\n{critique}\n\n"
                "Edit the following paper:\n\n"
                f"{draft}\n\n"
                "Editing tasks — complete all of them:\n"
                "1. Remove any banned words and replace with direct alternatives\n"
                "2. Break any sentence over 20 words into shorter sentences\n"
                "3. Remove padding, filler phrases, and repetition\n"
                "4. Confirm the structure exactly matches the format requirements above\n"
                "5. Confirm tone and vocabulary are appropriate for the stated audience\n"
                f"6. Trim or expand to reach approximately {target_words:,} words — "
                "do not reduce substantially below this target\n"
                "7. Soften any claim that uses language stronger than the evidence supports — "
                "use the Critic's source ratings to identify these\n"
                "8. Preserve all ## section headings and ### subheadings exactly as written — "
                "do not change heading levels or remove headings\n"
                "9. Confirm the topic and angle are maintained throughout — do not let the "
                "paper drift to adjacent subjects\n"
                + (f"10. Preserve the paper title exactly as given: \"{title}\" — "
                   "do not change, shorten, or paraphrase it. "
                   "Do NOT add the title as a heading at the top of your output — "
                   "begin directly with the Executive Summary or first section heading.\n" if title else "")
                + "\n"
                "Return the complete edited paper from start to finish. "
                "Do not stop mid-section. Do not summarise or skip sections."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=120, max_tokens=8000, agent_label="Editor")
    return {"final": response, "model_used": model, "prompt_sent": messages}
