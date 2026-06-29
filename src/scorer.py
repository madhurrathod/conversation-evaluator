"""
Scoring pipeline — conversation-level mode.

Scores the ENTIRE conversation in 2 API calls (180 facets each).
One set of scores per facet represents how that facet manifests
across the whole conversation; those scores are attached to every turn
so the output format stays compatible with the UI.

Why not 1 call? 276 facets × ~20 output tokens = ~5,500 tokens output
alone, pushing past the per-request limit on free-tier accounts when
combined with input. 2 calls of 180 facets each (~4,500 tokens/call)
is safe on all accounts.

Token budget:
  2 calls × ~4,500 tokens × 50 convos = ~450,000 tokens total
  → fits in ONE free-tier account (500k/day)

Model: llama-3.1-8b-instant (open-weights ≤16B — satisfies assignment constraint)

Output: results/<conv_id>_scores.json

Usage:
    python -m src.scorer                     # score all conversations
    python -m src.scorer conv_001 conv_002   # score specific conversations
"""

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from groq import AsyncGroq

CONVERSATIONS_DIR = Path("conversations")
RESULTS_DIR       = Path("results")
INDEX_FILE        = CONVERSATIONS_DIR / "index.json"
REGISTRY_CSV      = Path("data/facets_registry.csv")

SCORING_MODEL    = "llama-3.1-8b-instant"   # open-weights, ≤16B
CALLS_PER_MINUTE = 28                        # Groq free tier cap = 30; stay under
FACETS_PER_CALL  = 138                       # ceil(276/138)=2 calls per conv
MAX_TOKENS       = 4000                      # 138 × ~12 tokens (1-word reason) ≈ 1,656; total/call ~3,300
TURNS_CONTEXT    = 5                         # max turns included in conversation snippet

# ---------------------------------------------------------------------------
# Load facet registry
# ---------------------------------------------------------------------------

def _load_facets() -> list[dict]:
    df = pd.read_csv(REGISTRY_CSV)
    return (
        df[df["inferrable"] == True][["facet_name", "category"]]
        .to_dict(orient="records")
    )

# ---------------------------------------------------------------------------
# Prompt — 2-step: OBSERVE the full conversation, then SCORE a facet chunk
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert conversation analyst and psychometrician. "
    "Analyse the provided conversation and score the listed behavioral facets. "
    "Follow the 2-step reasoning process. "
    "Respond ONLY with valid JSON — no markdown, no extra text."
)

_PROMPT = """\
=== CONVERSATION ===
{conversation_text}

=== TASK ===
Step 1 — OBSERVE : In 2 sentences, describe the dominant tone, behaviors, and patterns across this conversation.
Step 2 — SCORE   : For each facet below, give one score representing how it manifests across the WHOLE conversation.

Scale: 1=absent  2=slight  3=moderate  4=clear  5=dominant
Confidence: 1.0=certain  0.7=fairly sure  0.5=uncertain

=== FACETS TO SCORE ({n}) ===
{facet_list}

=== CRITICAL OUTPUT RULES ===
- "r" must be EXACTLY ONE word (e.g. "absent", "moderate", "strong", "dominant", "clear")
- No sentences, no phrases — one word only

=== OUTPUT (strict JSON only) ===
{{
  "_obs": "<2-sentence observation>",
  "ExampleFacet1": {{"s": 3, "r": "moderate", "c": 0.7}},
  "ExampleFacet2": {{"s": 1, "r": "absent", "c": 0.9}},
  "ExampleFacet3": {{"s": 5, "r": "dominant", "c": 0.8}},
  "<actual_facet_name>": {{"s": <integer 1-5>, "r": "<quoted_word>", "c": <float>}},
  ... one entry per facet ...
}}"""


def _build_conv_text(turns: list[dict]) -> str:
    lines = []
    for t in turns[-TURNS_CONTEXT:]:          # last N turns to keep input compact
        snippet = t["content"][:120] + ("…" if len(t["content"]) > 120 else "")
        lines.append(f'[{t["role"].upper()}]: {snippet}')
    prefix = f"(showing last {TURNS_CONTEXT} of {len(turns)} turns)\n" if len(turns) > TURNS_CONTEXT else ""
    return prefix + "\n".join(lines)


def _build_prompt(conversation_text: str, chunk: list[dict]) -> str:
    facet_list = "\n".join(
        f'- {f["facet_name"]} [{f["category"]}]'
        for f in chunk
    )
    return _PROMPT.format(
        conversation_text=conversation_text,
        n=len(chunk),
        facet_list=facet_list,
    )


def _parse_response(raw: str, chunk: list[dict]) -> dict:
    from json_repair import repair_json
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(repair_json(raw))
    # Strip any leading "N. " the model may echo from numbered lists
    data = {re.sub(r"^\d+\.\s+", "", k): v for k, v in data.items()}

    scores = {}
    for f in chunk:
        name  = f["facet_name"]
        entry = data.get(name, {})
        scores[name] = {
            "score":      entry.get("s"),
            "reasoning":  entry.get("r", ""),
            "confidence": entry.get("c", 0.0),
        }
    return scores


def _empty_scores(chunk: list[dict]) -> dict:
    return {
        f["facet_name"]: {"score": None, "reasoning": "scoring error", "confidence": 0.0}
        for f in chunk
    }


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self._interval = 60.0 / calls_per_minute
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_event_loop()
            now  = loop.time()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = loop.time()


# ---------------------------------------------------------------------------
# Core — 2 API calls per conversation
# ---------------------------------------------------------------------------

async def _score_chunk(
    conversation_text: str,
    chunk: list[dict],
    client: AsyncGroq,
    rate_limiter: RateLimiter,
) -> dict:
    prompt = _build_prompt(conversation_text, chunk)
    await rate_limiter.acquire()

    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=SCORING_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_TOKENS,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_response(raw, chunk)

        except json.JSONDecodeError as exc:
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
            else:
                print(f"        [warn] JSON parse failed: {exc}")

        except Exception as exc:
            err = str(exc)
            if attempt < 2:
                wait = 10 if ("rate_limit" in err or "429" in err) else 3
                await asyncio.sleep(wait * (attempt + 1))
            else:
                print(f"        [warn] chunk failed: {exc}")

    return _empty_scores(chunk)


async def score_conversation(
    conv: dict,
    facets: list[dict],
    client: AsyncGroq,
    rate_limiter: RateLimiter,
    output_dir: Path,
) -> dict:
    conv_id  = conv["conversation_id"]
    out_path = output_dir / f"{conv_id}_scores.json"

    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        total = sum(len(t.get("scores", {})) for t in cached.get("turns", []))
        none_count = sum(
            1 for t in cached.get("turns", [])
            for v in t.get("scores", {}).values()
            if v.get("score") is None
        )
        if total > 0 and (none_count / total) < 0.20:
            print(f"  {conv_id} — already scored ({none_count} errors / {total}), skipping.")
            return cached
        print(f"  {conv_id} — {none_count}/{total} None scores, re-scoring ...")

    turns = conv["turns"]
    import math
    chunks = [facets[i:i + FACETS_PER_CALL] for i in range(0, len(facets), FACETS_PER_CALL)]
    n_calls = len(chunks)
    print(f"  {conv_id} | {len(turns)} turns | {n_calls} calls (conversation-level)", end=" ... ", flush=True)

    conversation_text = _build_conv_text(turns)

    # Score all facet chunks
    all_scores: dict = {}
    for chunk in chunks:
        result = await _score_chunk(conversation_text, chunk, client, rate_limiter)
        all_scores.update(result)

    valid = sum(1 for v in all_scores.values() if v.get("score") is not None)
    print(f"{valid}/{len(all_scores)} facets scored.")

    # Apply same conversation-level scores to every turn
    scored_turns = [
        {
            "turn_id":     t["turn_id"],
            "turn_number": t["turn_number"],
            "role":        t["role"],
            "content":     t["content"],
            "scores":      all_scores,
        }
        for t in turns
    ]

    result = {
        "conversation_id":  conv_id,
        "scenario":         conv["scenario"],
        "scored_at":        datetime.now(timezone.utc).isoformat(),
        "model":            SCORING_MODEL,
        "scoring_mode":     "conversation-level",
        "total_turns":      len(scored_turns),
        "turns":            scored_turns,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    Saved → {out_path.name}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(api_key: str, target_ids: Optional[list[str]] = None) -> None:
    with open(INDEX_FILE) as f:
        index = json.load(f)

    facets = _load_facets()

    if target_ids:
        index = [c for c in index if c["conversation_id"] in target_ids]

    import math
    n_calls_per_conv = math.ceil(len(facets) / FACETS_PER_CALL)
    total_calls = len(index) * n_calls_per_conv
    total_tokens_est = total_calls * (MAX_TOKENS + 1200)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client       = AsyncGroq(api_key=api_key)
    rate_limiter = RateLimiter(CALLS_PER_MINUTE)

    print(f"Scoring pipeline  (conversation-level, {n_calls_per_conv} calls/conv)")
    print(f"  Model        : {SCORING_MODEL}  (open-weights ≤16B)")
    print(f"  Facets       : {len(facets)}")
    print(f"  Convs        : {len(index)}")
    print(f"  Total calls  : {total_calls}")
    print(f"  Tokens est.  : ~{total_tokens_est:,}  ({'fits' if total_tokens_est < 500_000 else 'exceeds'} 500k/day)")
    print(f"  ETA          : ~{total_calls / CALLS_PER_MINUTE:.0f} min at {CALLS_PER_MINUTE} req/min")
    print()

    for conv_info in index:
        conv_path = CONVERSATIONS_DIR / conv_info["file"]
        with open(conv_path) as f:
            conv = json.load(f)
        await score_conversation(conv, facets, client, rate_limiter, RESULTS_DIR)

    print("\nScoring complete.")


def _load_api_key() -> str:
    import os
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        import re as _re
        text = secrets_path.read_text()
        m = _re.search(r'GROQ_API_KEY\s*=\s*["\'](.+?)["\']', text)
        if m:
            return m.group(1)
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise SystemExit("GROQ_API_KEY not found in .streamlit/secrets.toml or environment.")
    return key


if __name__ == "__main__":
    ids = sys.argv[1:] or None
    asyncio.run(run(api_key=_load_api_key(), target_ids=ids))
