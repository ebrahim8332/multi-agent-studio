"""
Multi-Agent Studio — main entry point.

Renders the sidebar navigation and routes to the selected module.
Modules not yet built show a Coming Soon placeholder automatically.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Studio",
    page_icon="🤖",
    layout="wide",
)

# Module registry. Key = sidebar label. Value = import path (None = not built yet).
MODULES = {
    "🏠 Welcome":                    None,
    "📝 Research Assistant":          "m01_research_assistant",
    "🔍 Competitive Intelligence":    None,
    "📄 Document Interrogator":       None,
    "📅 Meeting Prep":                None,
    "⚖️ Regulatory Watch":            None,
    "💼 Investment Diligence":        None,
    "📑 Contract Risk Reviewer":      None,
    "📊 Earnings Analyzer":           None,
}

st.sidebar.title("Multi-Agent Studio")
st.sidebar.caption("AI agent pipelines, live on screen")
st.sidebar.markdown("---")

selection = st.sidebar.radio(
    "Select Module",
    list(MODULES.keys()),
    label_visibility="collapsed",
)

# ── Main area ──────────────────────────────────────────────────────────────

if selection == "🏠 Welcome":
    st.title("Multi-Agent Studio")
    st.markdown(
        """
        This platform runs multi-agent AI pipelines and shows every agent step on screen.

        Each module sends a task through a chain of specialized agents.
        Each agent does one job, then passes its output to the next.
        You see each agent's status and output as it runs.

        **Modules**

        | Module | What it does |
        |--------|-------------|
        | 📝 Research Assistant | Enter a topic — five agents research, critique, write, and edit a structured paper |
        | 🔍 Competitive Intelligence | Enter a company — agents produce a structured competitive briefing |
        | 📄 Document Interrogator | Upload a document — agents extract claims, fact-check them, and return a verdict |
        | 📅 Meeting Prep | Enter a company and meeting date — agents build a one-page pre-meeting brief |
        | ⚖️ Regulatory Watch | Monitor defined topics for regulatory developments |
        | 💼 Investment Diligence | Enter a company — agents produce a structured diligence memo |
        | 📑 Contract Risk Reviewer | Upload a contract — agents identify obligations and flag risk clauses |
        | 📊 Earnings Analyzer | Paste a transcript — agents extract guidance, tone shifts, and risks |

        Select a module from the sidebar to begin.
        """
    )

elif MODULES.get(selection) is None:
    # Any module not yet built shows this placeholder
    st.title(selection)
    st.info("This module is coming soon. Check back after the next deploy.")

else:
    # When a module is built, its import path goes in MODULES above.
    # This block dynamically loads and renders it.
    import importlib
    module_path = MODULES[selection]
    try:
        mod = importlib.import_module(f"modules.{module_path}.ui")
        mod.render()
    except Exception as e:
        st.error(f"Module failed to load: {e}")
