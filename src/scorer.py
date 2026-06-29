"""
Scoring pipeline.

For each conversation turn, scores it on all inferrable facets using
Groq llama-3.1-8b-instant (≤16B — satisfies open-weights constraint).

Architecture:
  - Facets are pre-clustered into 19 batches of ~15 (see clusterer.py)
  - Per turn: one async API call per cluster (chain-of-thought, NOT one-shot)
  - Global rate limiter respects Groq free tier (30 req/min)
  - Results saved per-conversation so scoring can be safely interrupted/resumed

Output: results/<conv_id>_scores.json

Usage:
    python -m src.scorer                     # score all conversations
    python -m src.scorer conv_001 conv_002   # score specific conversations
"""

import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from groq import AsyncGroq

CONVERSATIONS_DIR = Path("conversations")
RESULTS_DIR = Path("results")
INDEX_FILE = CONVERSATIONS_DIR / "index.json"
CLUSTERS_JSON = Path("data/facets_clusters.json")

SCORING_MODEL = "llama-3.1-8b-instant"   # open-weights, ≤16B
CALLS_PER_MINUTE = 28                     # Groq free tier = 30; stay under
CONTEXT_WINDOW = 2                        # prior turns passed as context

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert conversation analyst and psychometrician. "
    "Score the highlighted turn on the listed behavioral facets. "
    "Use structured, evidence-based reasoning before assigning each score. "
    "Respond ONLY with valid JSON — no markdown, no extra text."
)

_PROMPT = """\
=== CONVERSATION CONTEXT (for reference only) ===
{context_block}
=== TURN TO SCORE (your target) ===
Role   : {role}
Content: \"\"\"{content}\"\"\"

=== TASK ===
Score the above turn on {n} facets from the theme: **{cluster_label}**

Follow this 3-step process for EACH facet:
  1. OBSERVE  — identify specific words, tone, or content in the turn that signal this facet
  2. REASON   — explain what your observation implies about the level of this facet
  3. CONCLUDE — assign score 1–5 and confidence 0.0–1.0

Score scale: 1=absent/very low  2=slight  3=moderate  4=clear  5=dominant/very high
Confidence : 1.0=certain  0.7=fairly sure  0.5=uncertain  0.3=guessing

=== FACETS TO SCORE ===
{facet_block}

=== SCORE ANCHORS ===
{anchor_block}

=== OUTPUT (JSON only, no markdown) ===
{{
{output_skeleton}
}}"""


def _build_prompt(turn: dict, prior_turns: list[dict], cluster: dict) -> str:
    # Context block: last N turns (excluding the target)
    if prior_turns:
        ctx_lines = []
        for t in prior_turns:
            snippet = t["content"][:200] + ("…" if len(t["content"]) > 200 else "")
            ctx_lines.append(f'[{t["role"].upper()}]: {snippet}')
        context_block = "\n".join(ctx_lines)
    else:
        context_block = "(This is the first turn — no prior context.)"

    facets = cluster["facets"]
    facet_block = "\n".join(
        f'{i+1}. **{f["facet_name"]}**: {f.get("description") or "No description."}'
        for i, f in enumerate(facets)
    )
    anchor_block = "\n".join(
        f'- {f["facet_name"]}: '
        f'score 1 → {f.get("score_anchor_1") or "very low"} | '
        f'score 5 → {f.get("score_anchor_5") or "very high"}'
        for f in facets
    )
    output_skeleton = ",\n".join(
        f'  "{f["facet_name"]}": {{"score": <1-5>, "reasoning": "<one sentence>", "confidence": <0.0-1.0>}}'
        for f in facets
    )

    return _PROMPT.format(
        context_block=context_block,
        role=turn["role"],
        content=turn["content"][:800],   # cap to avoid token overflow
        n=len(facets),
        cluster_label=cluster["label"],
        facet_block=facet_block,
        anchor_block=anchor_block,
        output_skeleton=output_skeleton,
    )


# ---------------------------------------------------------------------------
# Rate limiter — token bucket, one-at-a-time acquisition
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self._interval = 60.0 / calls_per_minute
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = loop.time()


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

async def _score_cluster(
    turn: dict,
    prior_turns: list[dict],
    cluster: dict,
    client: AsyncGroq,
    rate_limiter: RateLimiter,
) -> dict:
    prompt = _build_prompt(turn, prior_turns, cluster)
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
                max_tokens=2500,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)

        except json.JSONDecodeError as exc:
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
            else:
                print(f"        [warn] JSON parse failed for cluster '{cluster['label']}': {exc}")
                return _empty_cluster_scores(cluster)

        except Exception as exc:
            err = str(exc)
            is_rate = "rate_limit" in err or "429" in err
            wait = (60 if is_rate else 5) * (attempt + 1)
            if attempt < 2:
                await asyncio.sleep(wait)
            else:
                print(f"        [warn] cluster '{cluster['label']}' failed: {exc}")
                return _empty_cluster_scores(cluster)

    return _empty_cluster_scores(cluster)


def _empty_cluster_scores(cluster: dict) -> dict:
    return {
        f["facet_name"]: {"score": None, "reasoning": "scoring error", "confidence": 0.0}
        for f in cluster["facets"]
    }


async def score_turn(
    turn: dict,
    prior_turns: list[dict],
    clusters: list[dict],
    client: AsyncGroq,
    rate_limiter: RateLimiter,
) -> dict:
    # Score all clusters for this turn — one at a time through the rate limiter
    all_scores: dict = {}
    for cluster in clusters:
        result = await _score_cluster(turn, prior_turns, cluster, client, rate_limiter)
        all_scores.update(result)

    return {
        "turn_id":      turn["turn_id"],
        "turn_number":  turn["turn_number"],
        "role":         turn["role"],
        "content":      turn["content"],
        "scores":       all_scores,
    }


async def score_conversation(
    conv: dict,
    clusters: list[dict],
    client: AsyncGroq,
    rate_limiter: RateLimiter,
    output_dir: Path,
) -> dict:
    conv_id = conv["conversation_id"]
    out_path = output_dir / f"{conv_id}_scores.json"

    if out_path.exists():
        print(f"  {conv_id} — already scored, skipping.")
        with open(out_path) as f:
            return json.load(f)

    turns = conv["turns"]
    n_turns = len(turns)
    n_clusters = len(clusters)
    print(f"  {conv_id} | {n_turns} turns × {n_clusters} clusters = {n_turns * n_clusters} calls")

    scored_turns = []
    for i, turn in enumerate(turns):
        prior = turns[max(0, i - CONTEXT_WINDOW):i]
        print(f"    [{i+1}/{n_turns}] {turn['role']} turn ...", end=" ", flush=True)
        scored = await score_turn(turn, prior, clusters, client, rate_limiter)
        scored_turns.append(scored)
        valid = sum(1 for v in scored["scores"].values() if v.get("score") is not None)
        print(f"{valid}/{len(scored['scores'])} facets scored.")

    result = {
        "conversation_id": conv_id,
        "scenario":        conv["scenario"],
        "scored_at":       datetime.now(timezone.utc).isoformat(),
        "model":           SCORING_MODEL,
        "total_turns":     len(scored_turns),
        "turns":           scored_turns,
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

    with open(CLUSTERS_JSON) as f:
        clusters = json.load(f)

    if target_ids:
        index = [c for c in index if c["conversation_id"] in target_ids]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncGroq(api_key=api_key)
    rate_limiter = RateLimiter(CALLS_PER_MINUTE)

    total_calls = sum(c["total_turns"] for c in index) * len(clusters)
    eta_min = total_calls / CALLS_PER_MINUTE
    print(f"Scoring pipeline")
    print(f"  Model      : {SCORING_MODEL}  (open-weights ≤16B)")
    print(f"  Facets     : {sum(c['size'] for c in clusters)} inferrable")
    print(f"  Clusters   : {len(clusters)}")
    print(f"  Convs      : {len(index)}")
    print(f"  Total calls: ~{total_calls:,}")
    print(f"  ETA        : ~{eta_min:.0f} min at {CALLS_PER_MINUTE} req/min")
    print()

    for conv_info in index:
        conv_path = CONVERSATIONS_DIR / conv_info["file"]
        with open(conv_path) as f:
            conv = json.load(f)
        await score_conversation(conv, clusters, client, rate_limiter, RESULTS_DIR)

    print("\nScoring complete.")


def _load_api_key() -> str:
    """Read GROQ_API_KEY from .streamlit/secrets.toml, then env var."""
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
    # argv[1:] are optional conv_ids to score; omit to score all
    ids = sys.argv[1:] or None
    asyncio.run(run(api_key=_load_api_key(), target_ids=ids))
