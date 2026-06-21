"""
Agent functions for the Research Assistant module.

Each function is a LangGraph node. The signature is always:
    (state: ResearchState, chain: FallbackChain) -> dict

The return value is a partial state update — only the fields this agent changed.
LangGraph merges it back into the full state before passing to the next node.
"""

import os
import re
from tavily import TavilyClient

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

def run_planner(state: dict, chain) -> dict:
    """
    Breaks the topic into 4-6 focused research questions.
    Returns: questions (list), model_used (str)
    """
    topic = state["topic"]

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
                f"Topic: {topic}\n\n"
                "Generate 4 to 6 focused research questions that together cover this topic completely. "
                "Each question should be specific enough to search for directly. "
                "Number each question (1. 2. 3. etc). One question per line. Nothing else."
            ),
        },
    ]

    response, model = chain.complete(messages)

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

    return {"questions": questions, "model_used": model}


# ── Agent 2: Researcher ───────────────────────────────────────────────────────

def run_researcher(state: dict) -> dict:
    """
    Runs a Tavily web search for each research question.
    Collects results and source URLs.
    Returns: research (dict), sources (list)
    """
    questions = state["questions"]
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        # Graceful degradation: return empty research rather than crashing
        return {
            "research": {q: [] for q in questions},
            "sources": [],
        }

    client = TavilyClient(api_key=api_key)
    research = {}
    sources = []

    for question in questions:
        try:
            result = client.search(
                query=question,
                search_depth="advanced",
                max_results=3,
            )
            hits = result.get("results", [])
            research[question] = hits
            for hit in hits:
                url = hit.get("url", "")
                if url and url not in sources:
                    sources.append(url)
        except Exception:
            # If one question fails, record empty results and continue
            research[question] = []

    return {"research": research, "sources": sources}


# ── Agent 3: Critic ───────────────────────────────────────────────────────────

def run_critic(state: dict, chain) -> dict:
    """
    Reviews source quality for each research question.
    Flags gaps where evidence is thin or missing.
    Returns: critique (str), model_used (str)
    """
    questions = state["questions"]
    research = state["research"]

    # Build a compact summary of what was found — avoids sending full content to the LLM
    research_summary = []
    for q in questions:
        hits = research.get(q, [])
        if hits:
            titles = [h.get("title", "Untitled") for h in hits]
            research_summary.append(f"Question: {q}\nSources found: {', '.join(titles)}")
        else:
            research_summary.append(f"Question: {q}\nSources found: None")

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

    response, model = chain.complete(messages)
    return {"critique": response, "model_used": model}


# ── Agent 4: Writer ───────────────────────────────────────────────────────────

def run_writer(state: dict, chain) -> dict:
    """
    Writes a structured research paper from the gathered evidence.
    Returns: draft (str), model_used (str)
    """
    topic = state["topic"]
    questions = state["questions"]
    research = state["research"]
    critique = state["critique"]

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
                f"Evidence gathered:\n{evidence_text}\n\n"
                f"Source quality assessment:\n{critique}\n\n"
                "Write a complete research paper. Structure:\n"
                "1. Executive Summary (3-5 sentences — what the reader needs to know)\n"
                "2. Body sections — one section per research question, with a clear heading\n"
                "3. Conclusions — final synthesis only, no repetition of the body\n\n"
                "Where evidence is weak, say so plainly. Do not invent facts. "
                "Write in business formal style. Short sentences."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=120)
    return {"draft": response, "model_used": model}


# ── Agent 5: Editor ───────────────────────────────────────────────────────────

def run_editor(state: dict, chain) -> dict:
    """
    Polishes the draft: removes banned words, fixes long sentences,
    tightens structure, removes padding.
    Returns: final (str), model_used (str)
    """
    draft = state["draft"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior editor. Your job is to polish research papers "
                "for executive audiences. You do not change substance — only language quality. "
                f"{STYLE_RULES}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Edit the following research paper:\n\n"
                f"{draft}\n\n"
                "Tasks:\n"
                "1. Remove any banned words (replace with direct alternatives)\n"
                "2. Break sentences over 20 words into shorter ones\n"
                "3. Remove padding, filler phrases, and repetition\n"
                "4. Ensure the Executive Summary is sharp and standalone\n"
                "5. Ensure Conclusions add new synthesis, not repetition\n\n"
                "Return the complete edited paper. Preserve all headings and structure."
            ),
        },
    ]

    response, model = chain.complete(messages, timeout=120)
    return {"final": response, "model_used": model}
