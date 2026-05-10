"""
debate.py — Core AI debate orchestration for ZenZone AI Panel

Flow:
  1. Round 1: Claude (with web search), ChatGPT, and Grok each answer independently
  2. Round 2: Each AI sees the others' answers and refines
  3. Synthesis: Best answer distilled by Claude
  4. Summary: Plain-English meta-analysis of what happened
"""

import asyncio
import os
from typing import Optional, List, Dict, Any, Callable, Awaitable
import anthropic
import openai

# ── Clients ───────────────────────────────────────────────────────────────────

def _get_anthropic():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return anthropic.AsyncAnthropic(api_key=key)

def _get_openai():
    key = os.getenv("OPENAI_API_KEY", "")
    return openai.AsyncOpenAI(api_key=key)

def _get_grok():
    key = os.getenv("GROK_API_KEY", "")
    return openai.AsyncOpenAI(api_key=key, base_url="https://api.x.ai/v1")

CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL",  "claude-opus-4-6")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",  "gpt-4o")
GROK_MODEL    = os.getenv("GROK_MODEL",    "grok-3")

AI_TIMEOUT = 60  # seconds per call

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


# ── Token pricing ─────────────────────────────────────────────────────────────

PRICING = {
    "claude":  {"input": 15.00, "output": 75.00},
    "chatgpt": {"input":  2.50, "output": 10.00},
    "grok":    {"input":  3.00, "output": 15.00},
}

def _calc_cost(ai: str, input_tok: int, output_tok: int) -> float:
    p = PRICING.get(ai, {"input": 5.0, "output": 15.0})
    return (input_tok * p["input"] + output_tok * p["output"]) / 1_000_000


# ── Individual AI callers ─────────────────────────────────────────────────────

async def _ask_claude(messages: List[Dict], web_search: bool = False) -> tuple:
    """Claude — optionally with live web search on Round 1."""
    client = _get_anthropic()

    if web_search:
        # Try with web search tool
        try:
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=2048,
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }],
                    messages=messages,
                ),
                timeout=AI_TIMEOUT,
            )
            # Extract final text block (after any tool-use blocks)
            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text = block.text
            if text:
                usage = resp.usage
                return (text,
                        getattr(usage, "input_tokens", 0),
                        getattr(usage, "output_tokens", 0))
        except Exception:
            pass  # fall through to no-search

    # Standard call (no web search, or web search failed)
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
        text = resp.content[0].text if resp.content else ""
        return (text,
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0))
    except asyncio.TimeoutError:
        return ("[Claude timed out]", 0, 0)
    except Exception as e:
        return (f"[Claude unavailable: {e}]", 0, 0)


async def _ask_chatgpt(messages: List[Dict]) -> tuple:
    try:
        client = _get_openai()
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


# ── Debate logic ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = ("You are a helpful, honest, and thorough AI assistant. "
                 "Give clear, well-reasoned, comprehensive answers.")

def _debate_prompt_claude(question: str, own: str, other1_name: str, other1: str,
                           other2_name: str, other2: str) -> str:
    return (
        f'You previously answered this question: "{question}"\n\n'
        f"Your answer:\n{own}\n\n"
        f"Here are two other AI systems' answers:\n"
        f"{other1_name}:\n{other1}\n\n"
        f"{other2_name}:\n{other2}\n\n"
        "Carefully review the other perspectives. Incorporate any valid points you may have missed, "
        "correct any errors in your original answer, and stand firm where you believe you are right. "
        "Provide your refined, final answer."
    )

def _debate_prompt_oai(question: str, own: str, other1_name: str, other1: str,
                        other2_name: str, other2: str) -> str:
    return _debate_prompt_claude(question, own, other1_name, other1, other2_name, other2)


def _detect_change(r1: str, r2: str) -> bool:
    indicators = [
        "however, after reviewing", "i've reconsidered", "on reflection",
        "you raise a good point", "i was wrong", "i now think", "revising my",
        "actually,", "i need to correct", "i missed", "good point",
        "i should clarify", "i agree with"
    ]
    r2_lower = r2.lower()
    return any(ind in r2_lower for ind in indicators)


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

    await emit("Consulting Claude, ChatGPT, and Grok simultaneously…")

    # Round 1 — Claude uses web search to pull current info
    (claude_r1,  ci1, co1), (chatgpt_r1, gi1, go1), (grok_r1, xri1, xro1) = \
        await asyncio.gather(
            _ask_claude(_build_claude_messages(history, question, file_text), web_search=True),
            _ask_chatgpt(_build_oai_messages(SYSTEM_PROMPT, history, question, file_text)),
            _ask_grok(   _build_oai_messages(SYSTEM_PROMPT, history, question, file_text)),
        )
    _add("claude", ci1, co1); _add("chatgpt", gi1, go1); _add("grok", xri1, xro1)

    await emit("Round 2: each AI is reviewing the others' answers…")

    claude_r2_msgs  = _build_claude_messages([], _debate_prompt_claude(
        question, claude_r1, "ChatGPT", chatgpt_r1, "Grok", grok_r1))
    chatgpt_r2_msgs = _build_oai_messages(SYSTEM_PROMPT, [], _debate_prompt_oai(
        question, chatgpt_r1, "Claude", claude_r1, "Grok", grok_r1))
    grok_r2_msgs    = _build_oai_messages(SYSTEM_PROMPT, [], _debate_prompt_oai(
        question, grok_r1, "Claude", claude_r1, "ChatGPT", chatgpt_r1))

    (claude_r2, ci2, co2), (chatgpt_r2, gi2, go2), (grok_r2, xri2, xro2) = \
        await asyncio.gather(
            _ask_claude( claude_r2_msgs),
            _ask_chatgpt(chatgpt_r2_msgs),
            _ask_grok(   grok_r2_msgs),
        )
    _add("claude", ci2, co2); _add("chatgpt", gi2, go2); _add("grok", xri2, xro2)

    await emit("Synthesizing the best possible answer…")

    synthesis_q = (
        f'Synthesize these three AI final answers into one authoritative, clear response.\n\n'
        f'Original question: "{question}"\n\n'
        f"Claude's final answer:\n{claude_r2}\n\n"
        f"ChatGPT's final answer:\n{chatgpt_r2}\n\n"
        f"Grok's final answer:\n{grok_r2}\n\n"
        "Take the strongest, most accurate points from each. Resolve contradictions using your best "
        "judgment. Do NOT mention AIs or debate in your answer — just provide the best possible "
        "response to the original question as if it were your own answer."
    )
    synthesis, si, so = await _ask_claude(_build_claude_messages([], synthesis_q))
    _add("claude", si, so)

    await emit("Generating debate insights…")

    def _snip(s: str, n: int = 400) -> str:
        return s[:n] + "…" if len(s) > n else s

    summary_q = (
        f'Analyze this multi-AI debate and write 3-5 sentences for the user.\n\n'
        f'Question: "{question}"\n\n'
        f"Round 1 answers:\n"
        f"• Claude: {_snip(claude_r1)}\n"
        f"• ChatGPT: {_snip(chatgpt_r1)}\n"
        f"• Grok: {_snip(grok_r1)}\n\n"
        f"Round 2 answers:\n"
        f"• Claude: {_snip(claude_r2)}\n"
        f"• ChatGPT: {_snip(chatgpt_r2)}\n"
        f"• Grok: {_snip(grok_r2)}\n\n"
        "Write a specific, insightful summary: where did they agree, where did they disagree, "
        "which ones changed position and why, any notable patterns. "
        "Be concrete — name which AI said what. Address the user directly (second person)."
    )
    summary, smi, smo = await _ask_claude(_build_claude_messages([], summary_q))
    _add("claude", smi, smo)

    changes = {
        "claude":  _detect_change(claude_r1,  claude_r2),
        "chatgpt": _detect_change(chatgpt_r1, chatgpt_r2),
        "grok":    _detect_change(grok_r1,    grok_r2),
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
        "round1":      {"claude": claude_r1,  "chatgpt": chatgpt_r1,  "grok": grok_r1},
        "round2":      {"claude": claude_r2,  "chatgpt": chatgpt_r2,  "grok": grok_r2},
        "changes":     changes,
        "token_usage": token_usage,
    }
