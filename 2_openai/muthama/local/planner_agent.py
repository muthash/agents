# planner_agent.py
from dataclasses import dataclass, asdict
from typing import List, Optional
import re
from agents import Agent, register_agent_handler

# Heuristic dynamic search count
MIN_SEARCHES = 1
MAX_SEARCHES = 6
QUESTION_WORDS = {"what","how","why","when","where","which","who","whom","does","is","are","can","could","should","would"}

def determine_search_count(query: str, min_searches: int = MIN_SEARCHES, max_searches: int = MAX_SEARCHES) -> int:
    if not query or not query.strip():
        return min_searches
    q = query.strip()
    word_count = len(q.split())
    punctuation_count = len(re.findall(r'[?,;:-]', q))
    has_question_word = any(w in q.lower() for w in QUESTION_WORDS)
    topic_hint = q.count(",") + q.count("/") + len(re.findall(r'\band\b', q.lower()))
    base = 1
    if word_count >= 8: base += 1
    if punctuation_count >= 1: base += 1
    if has_question_word: base += 1
    if topic_hint >= 1: base += 1
    return max(min_searches, min(max_searches, base))

@dataclass
class WebSearchItem:
    query: str
    reason: Optional[str] = None
    priority: float = 0.5
    rank: Optional[int] = None
    tags: Optional[List[str]] = None

@dataclass
class WebSearchPlan:
    searches: List[WebSearchItem]

# The Agent object (exposed to other modules)
planner_agent = Agent(
    name="PlannerAgent",
    instructions="Produce a WebSearchPlan for the incoming query.",
    model="simulated-planner-model",
    output_type=WebSearchPlan,
)

# Basic validation/refinement
def validate_and_refine_searches(items, max_items):
    seen = set()
    refined = []
    for it in items:
        q = it.query.strip()
        q_norm = re.sub(r'\s+', ' ', q).strip(" .;,:")
        key = q_norm.lower()
        if len(q_norm) < 3: continue
        if key in seen: continue
        seen.add(key)
        refined.append(WebSearchItem(query=q_norm, reason=it.reason, priority=round(it.priority, 3)))
    refined.sort(key=lambda x: -x.priority)
    return refined[:max_items]

# Handler implementation (simulated LLM output)
def _planner_handler(agent: Agent, payload: str):
    query = payload if isinstance(payload, str) else str(payload)
    n = determine_search_count(query)
    # Simple strategy: break on commas or 'about'/'for' or split into keywords
    parts = re.split(r',|about|for| on | regarding | about ', query, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    candidates = []
    # construct candidate queries
    if parts:
        for p in parts:
            candidates.append(WebSearchItem(query=p + " overview", reason="General overview"))
            candidates.append(WebSearchItem(query=p + " tutorial", reason="How-to / hands-on"))
            candidates.append(WebSearchItem(query=p + " recent developments", reason="Latest updates"))
    # fallback: split into keywords
    if not candidates:
        for w in query.split()[:6]:
            candidates.append(WebSearchItem(query=w, reason="Keyword"))
    # trim and refine
    candidates = validate_and_refine_searches(candidates, max_items=n)
    # assign ranks
    for i, c in enumerate(candidates, start=1):
        c.rank = i
    return WebSearchPlan(searches=candidates)

register_agent_handler(planner_agent, _planner_handler)
