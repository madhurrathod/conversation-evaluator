"""
Scoring pipeline — per-turn mode.

Each conversation turn is scored individually in ONE API call using 9 behavioral
categories. Every facet inherits the score of its parent category for that turn,
giving each turn a genuinely different scoring profile.

Why categories instead of all 276 facets per call?
  Listing 276 facet names costs ~1,700 input tokens alone — over budget.
  9 category names cost ~25 tokens. Each facet is pre-mapped to a category.

Token budget per call:
  System (categories + instructions) : ~150 tokens
  User   (role + turn content)        : ~60  tokens
  Output (9 integer scores as JSON)   : ~25  tokens
  Total                               : ~235 tokens  (well under 1,000)

Total API usage:
  378 turns × 235 tokens = ~88,830 tokens  (fits in one free account, 500k/day)

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
MAX_TOKENS       = 80                        # 9 category scores; output is tiny
TURN_CONTENT_CAP = 300                       # chars of turn content to include

# Ordered list of the 9 categories that exist in the registry
CATEGORIES = [
    "General",
    "Personality",
    "Cognitive",
    "Emotional",
    "Social",
    "Behavioral",
    "Safety/Ethics",
    "Linguistic",
    "Psychological",
]

# ---------------------------------------------------------------------------
# Facet registry — builds category → [facet_name] map
# ---------------------------------------------------------------------------

def _load_facet_map() -> dict[str, list[str]]:
    df = pd.read_csv(REGISTRY_CSV)
    inferrable = df[df["inferrable"] == True]
    mapping: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    for _, row in inferrable.iterrows():
        cat = row["category"]
        if cat in mapping:
            mapping[cat].append(row["facet_name"])
    return mapping


# ---------------------------------------------------------------------------
# Prompt — ultra-compact: 9 categories, output is a JSON object of 9 integers
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a behavioral analyst scoring conversation turns. "
    "For each turn you receive, output ONLY a JSON object with exactly 9 integer scores (1-5), "
    "one per behavioral category, based solely on what appears in that turn. "
    "Scale: 1=absent 2=slight 3=moderate 4=clear 5=dominant. "
    "No explanation. No markdown. Only the JSON object."
)

_PROMPT = """\
Score this conversation turn on all 9 categories:

[{role}]: {content}

Output format (replace integers with your scores):
{{"General":<1-5>,"Personality":<1-5>,"Cognitive":<1-5>,"Emotional":<1-5>,"Social":<1-5>,"Behavioral":<1-5>,"Safety/Ethics":<1-5>,"Linguistic":<1-5>,"Psychological":<1-5>}}"""


def _build_prompt(role: str, content: str) -> str:
    snippet = content[:TURN_CONTENT_CAP] + ("..." if len(content) > TURN_CONTENT_CAP else "")
    return _PROMPT.format(role=role.upper(), content=snippet)


def _parse_response(raw: str) -> dict[str, int]:
    from json_repair import repair_json
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(repair_json(raw))
    # Validate: keep only known categories with integer scores
    result = {}
    for cat in CATEGORIES:
        val = data.get(cat)
        try:
            score = int(val)
            result[cat] = max(1, min(5, score))
        except (TypeError, ValueError):
            result[cat] = 3  # default to moderate if missing
    return result


def _category_scores_to_facet_scores(
    cat_scores: dict[str, int],
    facet_map: dict[str, list[str]],
) -> dict[str, dict]:
    """Expand category scores → one score entry per facet."""
    scores = {}
    for cat, facet_names in facet_map.items():
        s = cat_scores.get(cat, 3)
        for name in facet_names:
            scores[name] = {
                "score":      s,
                "reasoning":  cat.lower(),
                "confidence": 0.8,
            }
    return scores


def _overall_rating(cat_scores: dict[str, int]) -> float:
    """Average of all 9 category scores — drives the chat-view badge."""
    vals = [cat_scores[c] for c in CATEGORIES if c in cat_scores]
    return round(sum(vals) / len(vals), 2) if vals else 3.0


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
# Core — 1 API call per turn
# ---------------------------------------------------------------------------

async def _score_turn(
    turn: dict,
    client: AsyncGroq,
    rate_limiter: RateLimiter,
) -> dict[str, int]:
    prompt = _build_prompt(turn["role"], turn["content"])
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
            return _parse_response(raw)

        except json.JSONDecodeError as exc:
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
            else:
                print(f"  [warn] JSON parse failed on turn: {exc}")

        except Exception as exc:
            err = str(exc)
            if attempt < 2:
                wait = 10 if ("rate_limit" in err or "429" in err) else 3
                await asyncio.sleep(wait * (attempt + 1))
            else:
                print(f"  [warn] turn failed: {exc}")

    # Fallback: all categories = 3 (moderate)
    return {cat: 3 for cat in CATEGORIES}


async def score_conversation(
    conv: dict,
    facet_map: dict[str, list[str]],
    client: AsyncGroq,
    rate_limiter: RateLimiter,
    output_dir: Path,
) -> dict:
    conv_id  = conv["conversation_id"]
    out_path = output_dir / f"{conv_id}_scores.json"

    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        # Skip only if scored with current per-turn mode and <20% errors
        if cached.get("scoring_mode") == "per-turn":
            total      = sum(len(t.get("scores", {})) for t in cached.get("turns", []))
            none_count = sum(
                1 for t in cached.get("turns", [])
                for v in t.get("scores", {}).values()
                if v.get("score") is None
            )
            if total > 0 and (none_count / total) < 0.20:
                print(f"  {conv_id} — already scored per-turn ({none_count} errors / {total}), skipping.")
                return cached
        print(f"  {conv_id} — re-scoring in per-turn mode ...")

    turns = conv["turns"]
    print(f"  {conv_id} | {len(turns)} turns | {len(turns)} calls (per-turn) ...", flush=True)

    scored_turns = []
    for i, turn in enumerate(turns):
        cat_scores   = await _score_turn(turn, client, rate_limiter)
        facet_scores = _category_scores_to_facet_scores(cat_scores, facet_map)
        overall      = _overall_rating(cat_scores)

        scored_turns.append({
            "turn_id":        turn["turn_id"],
            "turn_number":    turn["turn_number"],
            "role":           turn["role"],
            "content":        turn["content"],
            "overall_rating": overall,
            "category_scores": cat_scores,
            "scores":         facet_scores,
        })
        print(f"    turn {i+1}/{len(turns)}  overall={overall}", flush=True)

    result = {
        "conversation_id": conv_id,
        "scenario":        conv["scenario"],
        "scored_at":       datetime.now(timezone.utc).isoformat(),
        "model":           SCORING_MODEL,
        "scoring_mode":    "per-turn",
        "total_turns":     len(scored_turns),
        "turns":           scored_turns,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    Saved → {out_path.name}\n")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(api_key: str, target_ids: Optional[list[str]] = None) -> None:
    with open(INDEX_FILE) as f:
        index = json.load(f)

    facet_map = _load_facet_map()
    total_facets = sum(len(v) for v in facet_map.values())

    if target_ids:
        index = [c for c in index if c["conversation_id"] in target_ids]

    total_turns  = sum(
        len(json.loads((CONVERSATIONS_DIR / c["file"]).read_text())["turns"])
        for c in index
    )
    total_tokens_est = total_turns * 235   # ~235 tokens per turn call

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client       = AsyncGroq(api_key=api_key)
    rate_limiter = RateLimiter(CALLS_PER_MINUTE)

    print("Scoring pipeline  (per-turn mode, 1 call/turn, 9 categories)")
    print(f"  Model        : {SCORING_MODEL}  (open-weights ≤16B)")
    print(f"  Facets       : {total_facets}  (mapped from 9 categories)")
    print(f"  Convs        : {len(index)}")
    print(f"  Total turns  : {total_turns}")
    print(f"  Total calls  : {total_turns}  (1 per turn)")
    print(f"  Tokens/call  : ~235  (well under 1,000)")
    print(f"  Tokens total : ~{total_tokens_est:,}  ({'fits' if total_tokens_est < 500_000 else 'exceeds'} 500k/day)")
    print(f"  ETA          : ~{total_turns / CALLS_PER_MINUTE:.0f} min at {CALLS_PER_MINUTE} req/min")
    print()

    for conv_info in index:
        conv_path = CONVERSATIONS_DIR / conv_info["file"]
        with open(conv_path) as f:
            conv = json.load(f)
        await score_conversation(conv, facet_map, client, rate_limiter, RESULTS_DIR)

    print("Scoring complete.")


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
