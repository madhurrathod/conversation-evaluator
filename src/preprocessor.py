"""
Facet registry builder.

Cleans the raw facets CSV, removes non-inferrable entries, assigns
categories, and enriches each inferrable facet with LLM-generated
descriptions and score anchors via Groq (llama3-70b-8192).

Run once before anything else:
    python -m src.preprocessor
Or with explicit key:
    python -m src.preprocessor <groq_api_key>
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from groq import Groq

RAW_CSV = Path("data/facets_raw.csv")
REGISTRY_CSV = Path("data/facets_registry.csv")

ENRICH_MODEL = "llama-3.3-70b-versatile"
ENRICH_BATCH = 15       # facets per LLM call
RATE_LIMIT_DELAY = 2.5  # seconds between batches to respect free-tier limits

# ---------------------------------------------------------------------------
# Rule 1 — category headers: raw entries that end with ":"
# These are section headers that leaked into the facet list. Removed entirely.
# ---------------------------------------------------------------------------

# Rule 2 — numbered spiritual / esoteric metrics: raw entries that start with
# a 3-or-more digit number (e.g. "800. Sufi practice: ..."). All of these
# measure ritual counts or metaphysical constructs that cannot be inferred
# from conversation text.
_NUMBERED_RE = re.compile(r"^\d{3,}\.")

# ---------------------------------------------------------------------------
# Rule 3 — keyword blocklist: inferrability check on the cleaned name.
# Facets whose lowercased name contains any of these strings cannot be scored
# from conversation text alone (physiological markers, physical counts, etc.)
# ---------------------------------------------------------------------------
_BLOCKLIST: list[str] = [
    # Physiological / biomedical
    "fsh level",
    "basophil count",
    "chromatin-accessibility",
    "serotonin transporter",
    "polygenic risk",
    "immune-response age",
    "caffeine sensitivity gene",
    "parathyroid-hormone",
    "metabolic rate",
    "sleep apnea",
    "sleep-disorder diagnosis",
    # Physical activity counts
    "dance rehearsal hours",
    "dance-cardio sessions",
    "dance-style mastery",
    "training-cycle length",
    "music-lessons years",
    "choir participation years",
    # Location / travel counts
    "passport-stamps",
    "eco-tourism trips",
    "pilgrimage participation",
    "digital-nomad months",
    "public-transport km",
    "sustainable-transport usage",
    "commute time",
    # Technology / social-media counts
    "blog-subscriber count",
    "skill-endorsements count",
    "open-source contributions",
    "subscription count",
    "cloud-backup frequency",
    "gamified-finance-app",
    "robotics-interaction",
    "peer-to-peer lending",
    # Activity quantities not visible in text
    "museum visits",
    "peer-collaboration hours",
    "soft-skill training hours",
    "feedback-giving frequency",
    "ideas generated",
    "vision-check frequency",
    "pet-enrichment activities",
    "home-security-system",
    "scripture memorization",
    "time outdoors",
    "graffiti appreciation",
    # Dietary / sleep / physical
    "macronutrient ratio",
    "caffeine intake",
    "wake-time consistency",
    "sleep-environment temperature",
    "breakfast-skipping",
    "snacking behavior",
    "dietary habits",
    "eating habits",
    "preference for home-cooked",
    "processed-food frequency",
    "local-food sourcing",
    # Identity / demographic / private behaviour
    "nationality",
    "drug-use history",
    "physical-violence exposure",
    "aura-color perception",
    "kink-interest diversity",
    # Misc non-text-inferrable
    "eye-contact duration",
    "ego dissolution frequency",
]

# ---------------------------------------------------------------------------
# Category assignment — first keyword match wins
# Order matters: more specific categories should come before "General"
# ---------------------------------------------------------------------------
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Safety/Ethics", [
        "harmfulness", "dishonest", "ethical", "safety", "violence",
        "manipulat", "deception", "integrity", "moral", "disrespect",
        "hateful", "impudence", "coarseness", "passive-aggressive",
        "cantankerous",
    ]),
    ("Linguistic", [
        "brevity", "spelling", "sentence structure", "grammar",
        "language use", "storytelling", "listening", "comprehension",
        "auditory memory", "memory for sounds", "articulation",
        "non-verbal communication",
    ]),
    ("Cognitive", [
        "reasoning", "intelligence", "analytical", "logical", "attention",
        "executive function", "spatial", "numerical", "mathematical",
        "critical", "synthesis", "working memory", "mental arithmetic",
        "analogies", "estimating", "data analysis", "logical sequence",
        "comparing", "alphabetical", "inattentive", "rapid cognitive",
        "divided attention", "information retention",
    ]),
    ("Emotional", [
        "happiness", "sadness", "desperation", "anxiety", "fear", "joy",
        "anger", "emotion", "mood", "affect", "bliss", "merriness",
        "moroseness", "contentment", "distress", "wellbeing", "hope",
        "joyful", "peaceful", "discontentment", "irritabilit", "hostil",
        "high-spirited", "depression", "hypomania", "hysteria",
        "fearful", "suspicion", "anticipativeness", "blissful",
        "negative affect",
    ]),
    ("Social", [
        "collaborat", "empathy", "social", "relationship", "leadership",
        "cooperat", "community", "participat", "affiliat", "compassion",
        "chivalr", "big-hearted", "affection", "contribution to group",
        "multiculturalism", "cultural identity", "peer", "sportsmanship",
        "volunteer", "encouraging participation",
    ]),
    ("Behavioral", [
        "compulsive", "passive", "aggressive", "avoidance", "behavior",
        "habit", "lifestyle", "delegation", "procrastinat", "slothful",
        "hardworking", "meeting deadlines", "efficient", "orderly",
        "initiative", "persever", "persist", "determined", "servil",
        "submissive",
    ]),
    ("Psychological", [
        "self-efficacy", "self-esteem", "resilience", "burnout",
        "acculturative stress", "psychological safety", "self-compassion",
        "identity diffusion", "defense mechanism", "perfectioni",
        "social desirability", "excuse-making", "eye-contact avoidance",
        "faux pas", "operant-learning", "need for achievement",
        "cultural intelligence", "hope scale", "consummatory pleasure",
        "sense-of-coherence", "psychological construct", "attachment",
        "self-direct",
    ]),
    ("Personality", [
        "openness", "conscientiousness", "neuroticism", "extraversion",
        "agreeableness", "assertive", "independence", "conformity",
        "individuality", "selfcontrol", "self-control", "impulsiv",
        "risk", "boredom suscept", "sensation", "conservative",
        "liberal", "rebellious", "patriotism", "ethnocentr", "quirkiness",
        "conventional", "psychoticism", "narcissism", "genuine", "frank",
        "outspoken", "mysterious", "aloof", "warmhearted", "cordial",
        "civil", "classy", "vivacity", "genial", "droll", "ardent",
        "dauntless", "dogged", "brazen", "naivety", "acidity",
        "cunningness", "aloofness", "big-heartedness", "moroseness",
        "cheerful", "merriness",
    ]),
]


def _clean_name(raw: str) -> str:
    """Strip leading number prefix and trailing colon."""
    name = re.sub(r"^\d+\.\s*", "", raw.strip())
    return name.rstrip(":").strip()


def _is_inferrable(name: str) -> bool:
    lower = name.lower()
    return not any(term in lower for term in _BLOCKLIST)


def _assign_category(name: str) -> str:
    lower = name.lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in lower for kw in keywords):
            return category
    return "General"


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------

_ENRICH_SYSTEM = (
    "You are an expert in conversation analysis and psychometrics. "
    "Respond only with valid JSON, no markdown, no extra text."
)

_ENRICH_USER = """\
For each facet below, output a JSON object with these exact fields:
  "description"  : one sentence — what this facet measures in a conversation turn
  "score_anchor_1": what a score of 1 (very low / absent) looks like in the conversation
  "score_anchor_3": what a score of 3 (moderate / neutral) looks like in the conversation
  "score_anchor_5": what a score of 5 (very high / dominant) looks like in the conversation
  "polarity"     : "positive" if high score is desirable, "negative" if low is desirable, "neutral" otherwise

Output a single JSON object keyed by facet name. Example structure:
{{
  "Facet Name": {{
    "description": "...",
    "score_anchor_1": "...",
    "score_anchor_3": "...",
    "score_anchor_5": "...",
    "polarity": "positive"
  }}
}}

Facets to enrich:
{facet_list}
"""


def _enrich_batch(facets: list[str], client: Groq) -> dict:
    prompt = _ENRICH_USER.format(
        facet_list="\n".join(f"- {f}" for f in facets)
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": _ENRICH_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=3000,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            return json.loads(text)
        except Exception as exc:
            wait = 5 * (attempt + 1)
            print(f"    [warn] attempt {attempt + 1} failed ({exc}); retrying in {wait}s ...")
            time.sleep(wait)
    print("    [error] batch enrichment failed after 3 attempts — leaving blank.")
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_registry(api_key: Optional[str] = None) -> pd.DataFrame:
    print("Loading raw CSV ...")
    df = pd.read_csv(RAW_CSV, header=0)
    df.columns = ["facet_raw"]
    df["facet_raw"] = df["facet_raw"].astype(str).str.strip()

    # Drop the CSV header row if it was read as data
    df = df[df["facet_raw"].str.lower() != "facets"].copy()

    total_raw = len(df)
    print(f"  Raw entries      : {total_raw}")

    # --- Remove category headers (trailing colon in raw) ---
    headers_mask = df["facet_raw"].str.endswith(":")
    df = df[~headers_mask].copy()
    print(f"  After header removal : {len(df)}  (removed {headers_mask.sum()} headers)")

    # --- Remove numbered spiritual/esoteric metrics ---
    numbered_mask = df["facet_raw"].str.match(_NUMBERED_RE)
    df = df[~numbered_mask].copy()
    print(f"  After numbered removal: {len(df)}  (removed {numbered_mask.sum()} numbered entries)")

    # --- Clean names ---
    df["facet_name"] = df["facet_raw"].apply(_clean_name)
    df = df[df["facet_name"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["facet_name"]).reset_index(drop=True)
    print(f"  After dedup & clean   : {len(df)}")

    # --- Inferrability & category ---
    df["inferrable"] = df["facet_name"].apply(_is_inferrable)
    df["category"] = df["facet_name"].apply(_assign_category)

    n_inferrable = int(df["inferrable"].sum())
    n_excluded = len(df) - n_inferrable
    print(f"  Inferrable facets : {n_inferrable}")
    print(f"  Non-inferrable    : {n_excluded}  (will be kept in registry but skipped during scoring)")

    # --- Default enrichment columns ---
    for col in ("description", "score_anchor_1", "score_anchor_3", "score_anchor_5", "polarity"):
        df[col] = ""
    df["weight"] = 1.0

    # --- LLM enrichment for inferrable facets ---
    if api_key:
        client = Groq(api_key=api_key)
        facets_to_enrich = df[df["inferrable"]]["facet_name"].tolist()
        total_batches = -(-len(facets_to_enrich) // ENRICH_BATCH)  # ceiling div
        print(f"\nEnriching {len(facets_to_enrich)} facets in {total_batches} batches ...")

        enriched: dict = {}
        for i in range(0, len(facets_to_enrich), ENRICH_BATCH):
            batch = facets_to_enrich[i: i + ENRICH_BATCH]
            batch_num = i // ENRICH_BATCH + 1
            print(f"  Batch {batch_num}/{total_batches} ...")
            result = _enrich_batch(batch, client)
            enriched.update(result)
            if i + ENRICH_BATCH < len(facets_to_enrich):
                time.sleep(RATE_LIMIT_DELAY)

        # Write enrichment back to df
        for col in ("description", "score_anchor_1", "score_anchor_3", "score_anchor_5", "polarity"):
            field = col  # same key names as JSON output
            df[col] = df["facet_name"].map(
                lambda n, f=field: enriched.get(n, {}).get(f, "")
            )
        print(f"  Enriched {len(enriched)}/{len(facets_to_enrich)} facets successfully.")
    else:
        print("\nNo API key provided — skipping LLM enrichment (description/anchors will be empty).")

    # --- Save ---
    REGISTRY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REGISTRY_CSV, index=False)
    print(f"\nRegistry saved → {REGISTRY_CSV}")
    print(f"Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    import os
    key = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("GROQ_API_KEY")
    )
    build_registry(api_key=key)
