"""
debate.py — Core AI debate orchestration for ZenZone AI Panel

Flow:
  1. Round 1: Claude (web search), ChatGPT (web search), Grok (real-time) answer independently
  2. Convergence check: measure pairwise similarity — stop at 90% or diminishing returns
  3. If not converged: each AI sees the others and refines (up to 5 rounds total)
  4. Synthesis: best answer distilled by Claude
  5. Summary: plain-English meta-analysis
"""

import asyncio
import os
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any, Callable, Awaitable
import anthropic
import openai

# ── Clients ───────────────────────────────────────────────────────────────────

def _get_anthropic():
    return anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

def _get_openai():
    return openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def _get_grok():
    return openai.AsyncOpenAI(api_key=os.getenv("GROK_API_KEY", ""), base_url="https://api.x.ai/v1")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GROK_MODEL   = os.getenv("GROK_MODEL",   "grok-3")

AI_TIMEOUT         = 60    # seconds per individual AI call
MAX_ROUNDS         = 5     # hard cap on debate rounds
CONSENSUS_THRESHOLD = 0.90  # pairwise similarity to declare consensus
DIMINISHING_THRESHOLD = 0.05  # if no AI changes more than 5%, stop

# ── Token pricing ─────────────────────────────────────────────────────────────

PRICING = {
    "claude":  {"input": 15.00, "output": 75.00},
    "chatgpt": {"input":  2.50, "output": 10.00},
    "grok":    {"input":  3.00, "output": 15.00},
}

def _calc_cost(ai: str, inp: int, out: int) -> float:
    p = PRICING.get(ai, {"input": 5.0, "output": 15.0})
    return (inp * p["input"] + out * p["output"]) / 1_000_000

# ── Message builders ──────────────────────────────────────────────────────────

def _build_claude_messages(history: List[Dict], question: str,
                            file_text: Optional[str] = None) -> List[Dict]:
    msgs = []
    for m in history:
        msgs.append({"role": "user" if m["role"] == "user" else "assistant",
                     "content": m["content"]})
    body = f"{question}\n\n[Attached file content]:\n{file_text}" if file_text else question
    msgs.append({"role": "user", "content": body})
    return msgs

def _build_oai_messages(system: str, history: List[Dict], question: str,
                         file_text: Optional[str] = None) -> List[Dict]:
    msgs = [{"role": "system", "content": system}]
    for m in history:
        msgs.append({"role": "user" if m["role"] == "user" else "assistant",
                     "content": m["content"]})
    body = f"{question}\n\n[Attached file content]:\n{file_text}" if file_text else question
    msgs.append({"role": "user", "content": body})
    return msgs

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful, honest, and thorough AI assistant specializing in renovation and construction. "
    "Format responses with natural bullet points (using - dashes) for lists, steps, and cost breakdowns. "
    "Do NOT use markdown headers (# ## ###), pipe tables, or blockquotes. "
    "Write clearly and directly, like a knowledgeable colleague. "
    "For cost ranges, write them inline: e.g. 'Demo and haul-away: $950–$1,800'."
)

# ── Convergence check ─────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower()[:3000], b.lower()[:3000]).ratio()

def _should_continue(current: Dict[str, str], previous: Optional[Dict[str, str]] = None) -> bool:
    """Return True if another debate round is warranted."""
    a, b, c = current["claude"], current["chatgpt"], current["grok"]

    # Stop if answers are 90%+ similar to each other
    avg_sim = (_similarity(a, b) + _similarity(b, c) + _similarity(a, c)) / 3
    if avg_sim >= CONSENSUS_THRESHOLD:
        return False

    # Stop if no AI changed more than 5% from previous round (diminishing returns)
    if previous:
        max_change = max(
            1 - _similarity(current[ai], previous[ai])
            for ai in ("claude", "chatgpt", "grok")
        )
        if max_change < DIMINISHING_THRESHOLD:
            return False

    return True

# ── Individual AI callers ─────────────────────────────────────────────────────

async def _ask_claude(messages: List[Dict], web_search: bool = False) -> tuple:
    """Claude — with optional live web search on Round 1."""
    client = _get_anthropic()

    if web_search:
        try:
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=2048,
                    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
                    messages=messages,
                ),
                timeout=AI_TIMEOUT,
            )
            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text = block.text
            if text:
                usage = resp.usage
                return (text, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        except Exception:
            pass  # fall through to standard call

    try:
        resp = await asyncio.wait_for(
            client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                messages=messages,
            ),
            timeout=AI_TIMEOUT,
        )
        usage = resp.usage
        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text = block.text
        return (text or "",
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0))
    except asyncio.TimeoutError:
        return ("[Claude timed out]", 0, 0)
    except Exception as e:
        return (f"[Claude unavailable: {e}]", 0, 0)


async def _ask_chatgpt(messages: List[Dict], web_search: bool = False) -> tuple:
    """ChatGPT — with optional live web search via Responses API on Round 1."""
    client = _get_openai()

    if web_search:
        try:
            # Build condensed context for Responses API
            context_parts = []
            for m in messages[1:]:  # skip system message
                role = m.get("role", "user")
                content = m.get("content", "")[:300]
                context_parts.append(f"{role.title()}: {content}")
            full_input = "\n".join(context_parts) if context_parts else messages[-1]["content"]

            resp = await asyncio.wait_for(
                client.responses.create(
                    model=OPENAI_MODEL,
                    tools=[{"type": "web_search_preview"}],
                    instructions=SYSTEM_PROMPT,
                    input=full_input,
                ),
                timeout=AI_TIMEOUT,
            )
            text = ""
            for item in resp.output:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", "") == "output_text":
                            text += getattr(c, "text", "")
            usage = getattr(resp, "usage", None)
            inp = getattr(usage, "input_tokens", 0) if usage else 0
            out = getattr(usage, "output_tokens", 0) if usage else 0
            if text:
                return (text, inp, out)
        except Exception:
            pass  # fall through to chat completions

    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=2048,
                messages=messages,
            ),
            timeout=AI_TIMEOUT,
        )
        usage = resp.usage
        return (resp.choices[0].message.content,
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0))
    except asyncio.TimeoutError:
        return ("[ChatGPT timed out]", 0, 0)
    except Exception as e:
        return (f"[ChatGPT unavailable: {e}]", 0, 0)


async def _ask_grok(messages: List[Dict]) -> tuple:
    """Grok — real-time training knowledge."""
    try:
        client = _get_grok()
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=GROK_MODEL,
                max_tokens=2048,
                messages=messages,
            ),
            timeout=AI_TIMEOUT,
        )
        usage = resp.usage
        return (resp.choices[0].message.content,
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0))
    except asyncio.TimeoutError:
        return ("[Grok timed out]", 0, 0)
    except Exception as e:
        return (f"[Grok unavailable: {e}]", 0, 0)

# ── Debate prompts ────────────────────────────────────────────────────────────

def _refine_prompt(question: str, own: str, other1_name: str, other1: str,
                   other2_name: str, other2: str) -> str:
    return (
        f'You previously answered: "{question}"\n\n'
        f"Your answer:\n{own}\n\n"
        f"{other1_name}'s answer:\n{other1}\n\n"
        f"{other2_name}'s answer:\n{other2}\n\n"
        "Review the other perspectives carefully. Incorporate valid points you missed, "
        "correct any errors, and stand firm where you're right. "
        "Give your refined answer using the same bullet-point format."
    )

def _detect_change(r1: str, r2: str) -> bool:
    indicators = [
        "however, after reviewing", "i've reconsidered", "on reflection",
        "you raise a good point", "i was wrong", "i now think", "revising my",
        "actually,", "i need to correct", "i missed", "good point",
        "i should clarify", "i agree with"
    ]
    return any(ind in r2.lower() for ind in indicators)

# ── Main entry point ──────────────────────────────────────────────────────────

ProgressCallback = Callable[[str], Awaitable[None]]


async def run_debate(
    question: str,
    history: List[Dict],
    file_text: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:

    async def emit(msg: str):
        if progress:
            await progress(msg)

    tok: Dict[str, Dict[str, int]] = {
        "claude":  {"input": 0, "output": 0},
        "chatgpt": {"input": 0, "output": 0},
        "grok":    {"input": 0, "output": 0},
    }

    def _add(ai: str, inp: int, out: int):
        tok[ai]["input"]  += inp
        tok[ai]["output"] += out

    # ── Round 1 — all three with live web search ──────────────────────────────
    await emit("Searching the web and consulting all three AIs simultaneously…")

    (c_txt, ci, co), (g_txt, gi, go), (x_txt, xi, xo) = await asyncio.gather(
        _ask_claude(_build_claude_messages(history, question, file_text), web_search=True),
        _ask_chatgpt(_build_oai_messages(SYSTEM_PROMPT, history, question, file_text), web_search=True),
        _ask_grok(_build_oai_messages(SYSTEM_PROMPT, history, question, file_text)),
    )
    _add("claude", ci, co); _add("chatgpt", gi, go); _add("grok", xi, xo)

    current  = {"claude": c_txt, "chatgpt": g_txt, "grok": x_txt}
    previous = None
    round_num = 1
    r1_answers = dict(current)

    # ── Convergence loop — up to MAX_ROUNDS ───────────────────────────────────
    while _should_continue(current, previous) and round_num < MAX_ROUNDS:
        round_num += 1

        if round_num == 2:
            await emit("Round 2: each AI is reviewing the others' answers…")
        else:
            await emit(f"⚡ Answers still diverging — pushing to Round {round_num} of {MAX_ROUNDS}…")

        previous = dict(current)

        claude_msg  = _build_claude_messages([], _refine_prompt(
            question, current["claude"], "ChatGPT", current["chatgpt"], "Grok", current["grok"]))
        chatgpt_msg = _build_oai_messages(SYSTEM_PROMPT, [], _refine_prompt(
            question, current["chatgpt"], "Claude", current["claude"], "Grok", current["grok"]))
        grok_msg    = _build_oai_messages(SYSTEM_PROMPT, [], _refine_prompt(
            question, current["grok"], "Claude", current["claude"], "ChatGPT", current["chatgpt"]))

        (c2, ci2, co2), (g2, gi2, go2), (x2, xi2, xo2) = await asyncio.gather(
            _ask_claude(claude_msg),
            _ask_chatgpt(chatgpt_msg),
            _ask_grok(grok_msg),
        )
        _add("claude", ci2, co2); _add("chatgpt", gi2, go2); _add("grok", xi2, xo2)

        current = {"claude": c2, "chatgpt": g2, "grok": x2}

    # ── Synthesis ─────────────────────────────────────────────────────────────
    await emit("Synthesizing the best possible answer…")

    synthesis_q = (
        f'Synthesize these three AI final answers into one authoritative response.\n\n'
        f'Original question: "{question}"\n\n'
        f"Claude:\n{current['claude']}\n\n"
        f"ChatGPT:\n{current['chatgpt']}\n\n"
        f"Grok:\n{current['grok']}\n\n"
        "Instructions:\n"
        "- Take the strongest, most accurate points from each\n"
        "- Resolve contradictions using your best judgment\n"
        "- Do NOT mention AIs, debate, or rounds — just answer the question directly\n"
        "- Use natural bullet points (- dashes) for any lists, costs, or steps\n"
        "- Do NOT use markdown headers (# ##), pipe tables, or blockquotes\n"
        "- Write like a knowledgeable colleague: clear, direct, easy to read\n"
        "- Lead with the key answer, then supporting bullets"
    )
    synthesis, si, so = await _ask_claude(_build_claude_messages([], synthesis_q))
    _add("claude", si, so)

    # ── Summary ───────────────────────────────────────────────────────────────
    await emit("Generating debate insights…")

    def _snip(s: str, n: int = 300) -> str:
        return s[:n] + "…" if len(s) > n else s

    rounds_note = f"The debate ran {round_num} round{'s' if round_num > 1 else ''}." if round_num > 1 else ""

    summary_q = (
        f'Analyze this multi-AI debate and write 3–4 sentences for the user.\n\n'
        f'Question: "{question}"\n'
        f'{rounds_note}\n\n'
        f"Round 1:\n"
        f"- Claude: {_snip(r1_answers['claude'])}\n"
        f"- ChatGPT: {_snip(r1_answers['chatgpt'])}\n"
        f"- Grok: {_snip(r1_answers['grok'])}\n\n"
        f"Final round answers:\n"
        f"- Claude: {_snip(current['claude'])}\n"
        f"- ChatGPT: {_snip(current['chatgpt'])}\n"
        f"- Grok: {_snip(current['grok'])}\n\n"
        "Write a specific, insightful summary: where they agreed, where they disagreed, "
        "who changed position and why. Be concrete — name which AI said what. "
        "Address the user directly. Use plain sentences, no bullet points."
    )
    summary, smi, smo = await _ask_claude(_build_claude_messages([], summary_q))
    _add("claude", smi, smo)

    changes = {
        "claude":  _detect_change(r1_answers["claude"],  current["claude"]),
        "chatgpt": _detect_change(r1_answers["chatgpt"], current["chatgpt"]),
        "grok":    _detect_change(r1_answers["grok"],    current["grok"]),
    }

    token_usage = {}
    for ai, counts in tok.items():
        inp, out = counts["input"], counts["output"]
        token_usage[ai] = {
            "input_tokens":  inp,
            "output_tokens": out,
            "cost_usd":      round(_calc_cost(ai, inp, out), 6),
        }

    return {
        "synthesis":   synthesis,
        "summary":     summary,
        "round1":      r1_answers,
        "round2":      current,   # final round answers (may be same as round1 if converged early)
        "rounds_run":  round_num,
        "changes":     changes,
        "token_usage": token_usage,
    }
