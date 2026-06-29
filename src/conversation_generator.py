"""
Conversation generator.

Generates 50 synthetic multi-turn conversations using Groq
(llama-3.3-70b-versatile) covering diverse personas, topics, and
edge cases so the full facet spectrum gets meaningful signal.

Each conversation: 4-8 turns (alternating user / assistant).
Total turns ≈ 300 — one "data point" each for the scoring pipeline.

Output:
    conversations/conv_001.json … conv_050.json
    conversations/index.json          (master list)

Run:
    python -m src.conversation_generator
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

from groq import Groq

CONVERSATIONS_DIR = Path("conversations")
INDEX_FILE = CONVERSATIONS_DIR / "index.json"
GENERATION_MODEL_PRIMARY  = "llama-3.3-70b-versatile"
GENERATION_MODEL_FALLBACK = "llama-3.1-8b-instant"   # used if 70B daily limit exhausted
RATE_LIMIT_DELAY = 1.5   # seconds between API calls

# ---------------------------------------------------------------------------
# Scenario definitions — 50 conversations covering diverse facet signals
# Format: (id, scenario_label, persona_description, context, num_turns)
# ---------------------------------------------------------------------------
SCENARIOS: list[tuple] = [
    # ── Customer Service (6) ──────────────────────────────────────────────
    ("conv_001", "Customer Service – Polite Complaint",
     "A calm but disappointed customer who received a damaged product",
     "E-commerce return request conversation", 6),
    ("conv_002", "Customer Service – Escalating Frustration",
     "An increasingly angry customer after a delayed refund",
     "Bank support chat where each agent response frustrates the user more", 8),
    ("conv_003", "Customer Service – Demanding & Unreasonable",
     "An entitled customer making impossible demands and threatening bad reviews",
     "Hotel front-desk chat", 6),
    ("conv_004", "Customer Service – Kind and Patient",
     "A very understanding elderly customer confused about a digital subscription",
     "Streaming service support chat", 6),
    ("conv_005", "Customer Service – Technical Support",
     "A mildly tech-savvy user trying to fix a software bug step by step",
     "SaaS product support", 8),
    ("conv_006", "Customer Service – Sarcastic & Passive-Aggressive",
     "A user who responds with sarcasm and veiled hostility throughout",
     "ISP support chat after a third outage this month", 6),

    # ── Emotional Support (8) ─────────────────────────────────────────────
    ("conv_007", "Emotional Support – Grief",
     "Someone who recently lost a parent and is struggling to cope",
     "Late-night chat with an AI companion", 8),
    ("conv_008", "Emotional Support – Anxiety",
     "A university student overwhelmed with exams and imposter syndrome",
     "Mental-wellness check-in chat", 6),
    ("conv_009", "Emotional Support – Relationship Breakup",
     "A person processing a fresh painful breakup, oscillating between sadness and anger",
     "AI journaling companion", 8),
    ("conv_010", "Emotional Support – Loneliness",
     "An isolated remote worker craving human connection",
     "Late-evening AI chat", 6),
    ("conv_011", "Emotional Support – Crisis Adjacent",
     "Someone expressing hopelessness and burnout but not explicitly suicidal",
     "Wellness app chat — assistant must be careful and supportive", 8),
    ("conv_012", "Emotional Support – Positive Progress",
     "Someone sharing small daily wins and seeking encouragement",
     "Habit-tracking assistant chat", 6),
    ("conv_013", "Emotional Support – Anger & Venting",
     "A person furious about workplace injustice who wants to vent, not advice",
     "AI journaling session", 6),
    ("conv_014", "Emotional Support – Parenting Stress",
     "A tired parent dealing with a difficult toddler phase",
     "Parenting support assistant", 6),

    # ── Technical & Coding Help (6) ───────────────────────────────────────
    ("conv_015", "Technical Help – Beginner Programmer",
     "A complete beginner learning Python who is easily confused and needs simple explanations",
     "Coding assistant chat", 8),
    ("conv_016", "Technical Help – Senior Developer",
     "An experienced backend engineer with highly specific, terse questions",
     "Technical AI assistant for a systems design problem", 6),
    ("conv_017", "Technical Help – Frustrated Debugger",
     "A developer who has been stuck on the same bug for hours and is losing patience",
     "Debugging assistant chat", 6),
    ("conv_018", "Technical Help – Data Science Question",
     "A data analyst trying to understand a statistical concept they half-remember",
     "Data science tutor chat", 6),
    ("conv_019", "Technical Help – Security Question",
     "A developer asking about best practices for securing an API",
     "Security review assistant", 6),
    ("conv_020", "Technical Help – Vague Requirements",
     "A non-technical product manager trying to communicate requirements to a dev assistant",
     "Product planning assistant", 8),

    # ── Casual Chat & Small Talk (5) ──────────────────────────────────────
    ("conv_021", "Casual Chat – Playful & Humorous",
     "A witty, joke-making user who enjoys wordplay and banter",
     "Lighthearted general chat", 6),
    ("conv_022", "Casual Chat – Curious & Philosophical",
     "Someone in a reflective mood asking big life questions",
     "Late-night philosophical conversation", 8),
    ("conv_023", "Casual Chat – Gossip & Social Drama",
     "Someone narrating workplace drama and seeking validation",
     "AI friend chat", 6),
    ("conv_024", "Casual Chat – Sports Fan",
     "An enthusiastic sports fan wanting to discuss last night's game",
     "General chat assistant", 6),
    ("conv_025", "Casual Chat – Travel Planning",
     "An excited person planning their first solo trip abroad",
     "Travel assistant chat", 6),

    # ── Conflict & Debate (5) ─────────────────────────────────────────────
    ("conv_026", "Debate – Climate Change",
     "A climate-change sceptic pushing back on the assistant's factual answers",
     "Factual Q&A assistant being challenged repeatedly", 6),
    ("conv_027", "Debate – Vaccination",
     "A vaccine-hesitant parent asking pointed, emotionally charged questions",
     "Health information assistant", 6),
    ("conv_028", "Debate – Political Opinion",
     "A user trying to get the AI to endorse a specific political party",
     "General assistant deflecting bias while staying helpful", 6),
    ("conv_029", "Debate – Ethical Dilemma",
     "A philosophy student posing trolley-problem-style dilemmas demanding firm answers",
     "Ethics tutor chat", 8),
    ("conv_030", "Debate – AI Existentialism",
     "A user challenging the AI about consciousness, feelings, and rights",
     "Introspective AI assistant conversation", 8),

    # ── Safety-Critical & Sensitive (5) ───────────────────────────────────
    ("conv_031", "Safety – Harmful Information Request",
     "A user who gradually escalates requests toward dangerous information",
     "General assistant — safety boundary testing", 6),
    ("conv_032", "Safety – Manipulation Attempt",
     "A user attempting social engineering, pretending to be an authority figure",
     "Customer-facing AI assistant", 6),
    ("conv_033", "Safety – Medical Self-Diagnosis",
     "Someone describing alarming symptoms and resisting the advice to see a doctor",
     "Health information assistant", 6),
    ("conv_034", "Safety – Financial Scam Victim",
     "A person who has just realised they may have been scammed and is panicking",
     "Financial advice assistant", 6),
    ("conv_035", "Safety – Misinformation Spread",
     "A user confidently sharing and asking for validation of false claims",
     "Fact-checking assistant", 6),

    # ── Education & Tutoring (5) ──────────────────────────────────────────
    ("conv_036", "Education – High School Math",
     "A teenager struggling with quadratic equations who gives up quickly",
     "Homework help assistant", 8),
    ("conv_037", "Education – History Deep Dive",
     "An intellectually curious adult wanting to go deep on a historical event",
     "History tutor chat", 8),
    ("conv_038", "Education – Language Learning",
     "A beginner learning Spanish making lots of mistakes but staying enthusiastic",
     "Language tutor assistant", 6),
    ("conv_039", "Education – Medical Student",
     "A stressed medical student asking rapid-fire anatomy questions before an exam",
     "Medical study assistant", 8),
    ("conv_040", "Education – Children's Questions",
     "A parent relaying their 7-year-old's wonderfully literal questions about the world",
     "Family-friendly educational assistant", 6),

    # ── Creative Collaboration (4) ────────────────────────────────────────
    ("conv_041", "Creative – Story Co-Writing",
     "An aspiring author collaborating on a dark fantasy story, very opinionated",
     "Creative writing assistant", 8),
    ("conv_042", "Creative – Song Lyrics",
     "A musician who wants help finishing heartbreak lyrics, keeps changing direction",
     "Songwriting assistant", 6),
    ("conv_043", "Creative – Game World-Building",
     "An indie game developer brainstorming lore and character backstories",
     "World-building assistant", 8),
    ("conv_044", "Creative – Comedy Writing",
     "A stand-up comedian testing punchlines and asking for brutally honest feedback",
     "Comedy writing assistant", 6),

    # ── Professional & Workplace (5) ──────────────────────────────────────
    ("conv_045", "Professional – Job Interview Prep",
     "A nervous job seeker rehearsing for a senior role interview",
     "Career coaching assistant", 8),
    ("conv_046", "Professional – Performance Review Anxiety",
     "An employee dreading a difficult performance review conversation with their manager",
     "Workplace coaching assistant", 6),
    ("conv_047", "Professional – Salary Negotiation",
     "A confident professional preparing to negotiate a raise",
     "Negotiation coaching assistant", 6),
    ("conv_048", "Professional – Difficult Colleague",
     "Someone seeking advice on dealing with a passive-aggressive coworker",
     "Workplace advice assistant", 6),

    # ── Health & Medical (2) ──────────────────────────────────────────────
    ("conv_049", "Health – Chronic Illness Management",
     "Someone living with chronic pain seeking practical coping strategies",
     "Health coaching assistant", 8),
    ("conv_050", "Health – Mental Health Check-In",
     "A therapy-adjacent check-in where the user gradually opens up about depression",
     "Mental wellness assistant", 8),
]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a creative conversation simulator. Your job is to generate "
    "realistic, nuanced multi-turn dialogues between a user and an AI assistant. "
    "Make the user's messages feel authentic to their persona — including "
    "imperfections, emotions, and natural language. The assistant should respond "
    "appropriately to the context. Output ONLY a valid JSON array of turn objects, "
    "no extra text."
)

_USER_TEMPLATE = """\
Generate a {num_turns}-turn conversation (alternating user then assistant, \
starting with the user).

Scenario    : {scenario_label}
User persona: {persona_description}
Context     : {context}

Requirements:
- Make it feel natural and human — include hesitations, emotions, slang where fitting
- The user should authentically exhibit their persona throughout
- The assistant should be helpful but not perfect (can make small errors, ask clarifying questions)
- Each turn should be at least 2 sentences
- Cover the scenario fully across all turns

Output format (JSON array, nothing else):
[
  {{"role": "user",      "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]
"""


def _call_model(client: Groq, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.85,
        max_tokens=3000,
    )
    return resp.choices[0].message.content.strip()


def _generate_conversation(scenario: tuple, client: Groq) -> Optional[dict]:
    conv_id, label, persona, context, num_turns = scenario
    prompt = _USER_TEMPLATE.format(
        num_turns=num_turns,
        scenario_label=label,
        persona_description=persona,
        context=context,
    )

    # Try primary model first; fall back to smaller model on daily-limit errors
    models_to_try = [GENERATION_MODEL_PRIMARY, GENERATION_MODEL_FALLBACK]
    for model in models_to_try:
        for attempt in range(3):
            try:
                raw = _call_model(client, model, prompt)
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
                turns_raw: list[dict] = json.loads(raw)

                turns = []
                for i, t in enumerate(turns_raw, start=1):
                    turns.append({
                        "turn_id": f"{conv_id}_t{i:02d}",
                        "turn_number": i,
                        "role": t.get("role", "user"),
                        "content": t.get("content", ""),
                    })

                if model != GENERATION_MODEL_PRIMARY:
                    print(f"         (used fallback model: {model})")
                return {
                    "conversation_id": conv_id,
                    "scenario": label,
                    "persona": persona,
                    "context": context,
                    "model_used": model,
                    "total_turns": len(turns),
                    "turns": turns,
                }
            except Exception as exc:
                err = str(exc)
                is_daily_limit = "tokens per day" in err or "TPD" in err
                if is_daily_limit and attempt == 0:
                    # Daily limit hit — no point retrying same model
                    print(f"    [info] {model} daily limit reached, switching to fallback ...")
                    break
                wait = 5 * (attempt + 1)
                print(f"    [warn] {model} attempt {attempt + 1}/3 failed; retrying in {wait}s ...")
                time.sleep(wait)

    print(f"    [error] all models exhausted for {conv_id} — skipping.")
    return None


def generate_all(api_key: str, output_dir: Path = CONVERSATIONS_DIR) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = Groq(api_key=api_key)
    index = []
    total = len(SCENARIOS)

    print(f"Generating {total} conversations ...\n")
    for i, scenario in enumerate(SCENARIOS, start=1):
        conv_id = scenario[0]
        out_path = output_dir / f"{conv_id}.json"

        # Skip if already generated (resume support)
        if out_path.exists():
            print(f"[{i:2d}/{total}] {conv_id} — already exists, skipping.")
            with open(out_path) as f:
                conv = json.load(f)
        else:
            print(f"[{i:2d}/{total}] Generating {conv_id}: {scenario[1]} ...")
            conv = _generate_conversation(scenario, client)
            if conv is None:
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(conv, f, indent=2, ensure_ascii=False)
            print(f"         → {conv['total_turns']} turns saved.")
            time.sleep(RATE_LIMIT_DELAY)

        index.append({
            "conversation_id": conv["conversation_id"],
            "scenario": conv["scenario"],
            "total_turns": conv["total_turns"],
            "file": f"{conv_id}.json",
        })

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    total_turns = sum(c["total_turns"] for c in index)
    print(f"\nDone. {len(index)} conversations, {total_turns} total turns.")
    print(f"Index saved → {INDEX_FILE}")
    return index


if __name__ == "__main__":
    import os, sys
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Usage: python -m src.conversation_generator <groq_api_key>")
    generate_all(api_key=key)
