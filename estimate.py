"""
estimate.py — Web-search-enabled renovation cost estimating
Claude · ChatGPT · Grok all search the live web before estimating.
"""

import asyncio
import os
import json
import re
from typing import Optional, Dict, Any, Callable, Awaitable, List
import anthropic
import openai

# ── Models ──────────────────────────────────────────────────────────────────
CLAUDE_MODEL        = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
OPENAI_SEARCH_MODEL = "gpt-4o-search-preview"   # built-in live web search
GROK_MODEL          = os.getenv("GROK_MODEL",   "grok-3")

# ── Token pricing ────────────────────────────────────────────────────────────
PRICING = {
    "claude":  {"input": 15.00, "output": 75.00},
    "chatgpt": {"input":  2.50, "output": 10.00},
    "grok":    {"input":  3.00, "output": 15.00},
}

def _calc_cost(ai: str, inp: int, out: int) -> float:
    p = PRICING.get(ai, {"input": 5.0, "output": 15.0})
    return (inp * p["input"] + out * p["output"]) / 1_000_000

# ── Clients ───────────────────────────────────────────────────────────────────
def _get_anthropic():
    return anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

def _get_openai():
    return openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def _get_grok():
    return openai.AsyncOpenAI(
        api_key=os.getenv("GROK_API_KEY", ""),
        base_url="https://api.x.ai/v1",
    )

# ── Standard renovation line-item categories ─────────────────────────────────
CATEGORIES = [
    "Demolition & Disposal",
    "Rough Plumbing",
    "Finish Plumbing & Fixtures",
    "Rough Electrical",
    "Finish Electrical",
    "Framing & Carpentry",
    "Waterproofing & Substrate",
    "Tile & Flooring",
    "Drywall & Painting",
    "Cabinetry & Millwork",
    "Countertops",
    "Doors & Hardware",
    "Ventilation & HVAC",
    "Permits & Inspections",
    "Labor (General)",
    "Materials & Supplies",
    "Subcontractors",
    "Cleanup & Waste Removal",
]

# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(form: Dict) -> str:
    project_type = form.get("project_type", "renovation")
    city         = form.get("city", "")
    sqft         = form.get("sqft", "")
    scope        = form.get("scope", [])
    quality      = form.get("quality", "mid-range")
    description  = form.get("description", "")
    scope_text   = ", ".join(scope) if scope else "full renovation"

    cats_json = json.dumps([
        {"category": c, "conservative": 0, "market_rate": 0, "premium": 0, "applicable": False}
        for c in CATEGORIES
    ], indent=2)

    return f"""You are a professional renovation cost estimator. Search the web RIGHT NOW for current 2025 contractor pricing in {city}.

Search for these exact queries:
- "{city} {project_type} renovation cost 2025"
- "{city} general contractor rates {project_type}"
- "how much does {project_type} renovation cost in {city}"
- Current material costs {project_type}
- Homeadvisor angi {city} {project_type} contractor pricing

PROJECT SPECS:
- Type: {project_type}
- Location: {city}
- Size: {sqft} sq ft
- Quality tier: {quality}
- Scope: {scope_text}
- Details: {description}

After searching, return ONLY valid JSON — no markdown, no explanation, no code fences — in this exact structure:
{{
  "search_sources": ["list of actual URLs or sites you found pricing from"],
  "market_context": "2-3 sentences describing the current contractor market in {city}",
  "line_items": {cats_json},
  "project_total": {{"conservative": 0, "market_rate": 0, "premium": 0}},
  "recommended_bid": 0,
  "recommended_bid_rationale": "Why this price point is right for this market"
}}

Rules:
- Use REAL numbers from your web search. Do NOT use training-data guesses.
- Set "applicable": true only for categories relevant to this job scope.
- All amounts in USD, whole numbers.
- Do NOT include contractor markup/profit — that is the contractor's own addition.
- project_total must equal the sum of all applicable line items.
"""

# ── JSON extractor ────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try largest {...} block
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None

# ── Individual AI callers ─────────────────────────────────────────────────────
async def _ask_claude(prompt: str) -> tuple:
    """Claude with Anthropic's native web_search_20250305 tool."""
    try:
        client = _get_anthropic()
        resp = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        # Grab the final text block (after any tool-use blocks)
        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text = block.text
        usage = resp.usage
        return (text, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
    except Exception as e:
        return (f"[Claude error: {e}]", 0, 0)


async def _ask_chatgpt(prompt: str) -> tuple:
    """ChatGPT via gpt-4o-search-preview — live web search built in."""
    try:
        client = _get_openai()
        resp = await client.chat.completions.create(
            model=OPENAI_SEARCH_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system",
                 "content": "You are a professional renovation cost estimator. "
                             "Always search the live web before providing any numbers. "
                             "Return only valid JSON as instructed."},
                {"role": "user", "content": prompt},
            ],
        )
        usage = resp.usage
        return (
            resp.choices[0].message.content,
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )
    except Exception as e:
        return (f"[ChatGPT error: {e}]", 0, 0)


async def _ask_grok(prompt: str) -> tuple:
    """Grok with real-time X/web knowledge."""
    try:
        client = _get_grok()
        resp = await client.chat.completions.create(
            model=GROK_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system",
                 "content": "You are a professional renovation cost estimator with access to "
                             "real-time pricing data via the web and X/Twitter. "
                             "Search for current contractor rates before answering. "
                             "Return only valid JSON as instructed."},
                {"role": "user", "content": prompt},
            ],
        )
        usage = resp.usage
        return (
            resp.choices[0].message.content,
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )
    except Exception as e:
        return (f"[Grok error: {e}]", 0, 0)


# ── Synthesis ─────────────────────────────────────────────────────────────────
def _synthesize(
    claude_est: Optional[Dict],
    chatgpt_est: Optional[Dict],
    grok_est: Optional[Dict],
) -> Dict:
    valid = [e for e in [claude_est, chatgpt_est, grok_est] if e and "line_items" in e]
    if not valid:
        return {"error": "No valid estimates received from any AI."}

    # Average each category across valid estimates
    synth_items = []
    for cat in CATEGORIES:
        vals: Dict[str, List[float]] = {"conservative": [], "market_rate": [], "premium": []}
        applicable = False
        for est in valid:
            for item in est.get("line_items", []):
                if item.get("category") == cat:
                    if item.get("applicable", False):
                        applicable = True
                        for k in vals:
                            v = item.get(k, 0)
                            if isinstance(v, (int, float)) and v > 0:
                                vals[k].append(float(v))
                    break
        if applicable:
            synth_items.append({
                "category": cat,
                "conservative": round(sum(vals["conservative"]) / max(len(vals["conservative"]), 1)),
                "market_rate":  round(sum(vals["market_rate"])  / max(len(vals["market_rate"]),  1)),
                "premium":      round(sum(vals["premium"])      / max(len(vals["premium"]),      1)),
                "applicable": True,
            })

    totals = {
        "conservative": sum(i["conservative"] for i in synth_items),
        "market_rate":  sum(i["market_rate"]  for i in synth_items),
        "premium":      sum(i["premium"]       for i in synth_items),
    }

    # Recommended bid = average of market_rate recommendations
    recs = [e.get("recommended_bid", 0) for e in valid if isinstance(e.get("recommended_bid"), (int, float)) and e.get("recommended_bid", 0) > 0]
    recommended = round(sum(recs) / len(recs)) if recs else totals["market_rate"]

    contexts = [e.get("market_context", "") for e in valid if e.get("market_context")]
    sources  = []
    for e in valid:
        sources.extend(e.get("search_sources", []))

    ai_names = ["claude", "chatgpt", "grok"]
    ai_ests  = [claude_est, chatgpt_est, grok_est]
    ai_totals = {
        name: est.get("project_total")
        for name, est in zip(ai_names, ai_ests)
        if est and "project_total" in est
    }

    return {
        "line_items":       synth_items,
        "project_total":    totals,
        "recommended_bid":  recommended,
        "market_context":   " ".join(c for c in contexts if c),
        "search_sources":   list(dict.fromkeys(sources))[:12],
        "ai_totals":        ai_totals,
        "ai_rationale": {
            name: est.get("recommended_bid_rationale", "")
            for name, est in zip(ai_names, ai_ests)
            if est
        },
    }


# ── Main entry point ──────────────────────────────────────────────────────────
ProgressCallback = Callable[[str], Awaitable[None]]


async def run_estimate(
    form_data: Dict,
    progress: Optional[ProgressCallback] = None,
) -> Dict:
    """Run a web-search-powered three-AI estimate and return the synthesized result."""

    async def emit(msg: str):
        if progress:
            await progress(msg)

    tok = {ai: {"input": 0, "output": 0} for ai in ("claude", "chatgpt", "grok")}

    def _add(ai, inp, out):
        tok[ai]["input"]  += inp
        tok[ai]["output"] += out

    prompt = _build_prompt(form_data)

    await emit("Searching the web for current contractor pricing in your market…")

    (claude_txt, ci, co), (chatgpt_txt, gi, go), (grok_txt, xi, xo) = await asyncio.gather(
        _ask_claude(prompt),
        _ask_chatgpt(prompt),
        _ask_grok(prompt),
    )
    _add("claude",  ci, co)
    _add("chatgpt", gi, go)
    _add("grok",    xi, xo)

    await emit("Parsing estimates from all three AIs…")

    def _safe_parse(txt, ai_name):
        if txt.startswith(f"[{ai_name}"):
            return None
        return _extract_json(txt)

    claude_est  = _safe_parse(claude_txt,  "Claude")
    chatgpt_est = _safe_parse(chatgpt_txt, "ChatGPT")
    grok_est    = _safe_parse(grok_txt,    "Grok")

    await emit("Synthesizing final estimate across all three sources…")

    result = _synthesize(claude_est, chatgpt_est, grok_est)

    # Token usage
    result["token_usage"] = {
        ai: {
            "input_tokens":  counts["input"],
            "output_tokens": counts["output"],
            "cost_usd":      round(_calc_cost(ai, counts["input"], counts["output"]), 6),
        }
        for ai, counts in tok.items()
    }
    result["form_data"] = form_data

    return result
