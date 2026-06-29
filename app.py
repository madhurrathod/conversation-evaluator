"""
Conversation Evaluation UI — Streamlit app.

Displays scored conversations with per-turn facet scores,
reasoning, confidence, and analytics charts.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONVERSATIONS_DIR = Path("conversations")
RESULTS_DIR = Path("results")
REGISTRY_CSV = Path("data/facets_registry.csv")
INDEX_FILE = CONVERSATIONS_DIR / "index.json"

# ---------------------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Conversation Evaluator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Chat bubbles */
.bubble-user {
    background: #e3f2fd;
    border-radius: 12px 12px 12px 0;
    padding: 10px 14px;
    margin: 6px 60px 6px 0;
    font-size: 0.92rem;
    line-height: 1.5;
}
.bubble-assistant {
    background: #f3e5f5;
    border-radius: 12px 12px 0 12px;
    padding: 10px 14px;
    margin: 6px 0 6px 60px;
    font-size: 0.92rem;
    line-height: 1.5;
}
.role-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.6;
    margin-bottom: 2px;
}
/* Score badge */
.score-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
    margin: 2px 3px;
}
/* Stat cards */
.stat-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-number { font-size: 2rem; font-weight: 800; }
.stat-label  { font-size: 0.78rem; opacity: 0.6; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    with open(INDEX_FILE) as f:
        return json.load(f)


@st.cache_data
def load_conversation(conv_id: str) -> dict | None:
    p = CONVERSATIONS_DIR / f"{conv_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_scores(conv_id: str) -> dict | None:
    p = RESULTS_DIR / f"{conv_id}_scores.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_registry() -> pd.DataFrame:
    df = pd.read_csv(REGISTRY_CSV)
    df["description"] = df["description"].fillna("")
    df["polarity"] = df["polarity"].fillna("neutral")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SCORE_COLORS = {1: "#d32f2f", 2: "#f57c00", 3: "#fbc02d", 4: "#388e3c", 5: "#1b5e20"}
SCORE_LABELS = {1: "Very Low", 2: "Slight", 3: "Moderate", 4: "Clear", 5: "Dominant"}


def score_color(s) -> str:
    if s is None:
        return "#bdbdbd"
    return SCORE_COLORS.get(int(s), "#bdbdbd")


def scores_to_df(turn_scores: dict, registry: pd.DataFrame) -> pd.DataFrame:
    """Flatten turn scores dict into a DataFrame joined with registry metadata."""
    rows = []
    for facet, data in turn_scores.items():
        row = {"facet_name": facet, **data}
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.merge(
        registry[["facet_name", "category", "polarity", "description"]],
        on="facet_name", how="left",
    )
    df["category"] = df["category"].fillna("General")
    return df


def avg_score_by_category(scores_df: pd.DataFrame) -> pd.DataFrame:
    return (
        scores_df[scores_df["score"].notna()]
        .groupby("category")["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "avg_score"})
        .sort_values("avg_score", ascending=False)
    )


# ---------------------------------------------------------------------------
# Sidebar — conversation selector
# ---------------------------------------------------------------------------
index = load_index()
registry = load_registry()

with st.sidebar:
    st.title("📊 Convo Evaluator")
    st.caption("Facet-level conversation scoring")
    st.divider()

    if not index:
        st.warning("No conversations found. Run `python -m src.conversation_generator` first.")
        st.stop()

    scored_ids = {p.stem.replace("_scores", "") for p in RESULTS_DIR.glob("*_scores.json")}

    options = {
        f"{'✅' if c['conversation_id'] in scored_ids else '⏳'} {c['conversation_id']} — {c['scenario'][:40]}": c["conversation_id"]
        for c in index
    }
    selected_label = st.selectbox("Select conversation", list(options.keys()))
    conv_id = options[selected_label]

    st.divider()
    st.markdown(f"**Scored:** {len(scored_ids)} / {len(index)}")
    st.markdown(f"**Facets:** 276 inferrable")
    st.caption("Run `python -m src.scorer` to score all conversations.")

# ---------------------------------------------------------------------------
# Load selected conversation
# ---------------------------------------------------------------------------
conv = load_conversation(conv_id)
scores = load_scores(conv_id)

if conv is None:
    st.error(f"Conversation file not found for {conv_id}.")
    st.stop()

st.markdown(f"## {conv['scenario']}")
st.caption(f"`{conv_id}` · {conv['total_turns']} turns · {conv.get('context', '')}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_scores, tab_analytics = st.tabs(["💬 Chat View", "📋 Facet Scores", "📈 Analytics"])

# ── Tab 1: Chat View ─────────────────────────────────────────────────────────
with tab_chat:
    if scores is None:
        st.info("⏳ This conversation hasn't been scored yet. Run `python -m src.scorer` to score it.")
        for turn in conv["turns"]:
            role = turn["role"]
            content = turn["content"]
            css = "bubble-user" if role == "user" else "bubble-assistant"
            st.markdown(
                f'<div class="role-label">{role.upper()}</div>'
                f'<div class="{css}">{content}</div>',
                unsafe_allow_html=True,
            )
    else:
        # Build a score lookup by turn_id
        scored_by_id = {t["turn_id"]: t for t in scores["turns"]}

        for turn in conv["turns"]:
            role = turn["role"]
            content = turn["content"]
            turn_id = turn["turn_id"]
            css = "bubble-user" if role == "user" else "bubble-assistant"
            scored = scored_by_id.get(turn_id, {})
            turn_scores = scored.get("scores", {})

            # Compute top 3 scoring facets (valid scores only)
            valid = {k: v for k, v in turn_scores.items() if v.get("score") is not None}
            top3 = sorted(valid.items(), key=lambda x: x[1]["score"], reverse=True)[:3]
            badges = " ".join(
                f'<span class="score-pill" style="background:{score_color(v["score"])}">'
                f'{k[:18]} {v["score"]}</span>'
                for k, v in top3
            )

            st.markdown(
                f'<div class="role-label">{role.upper()} · Turn {turn["turn_number"]}</div>'
                f'<div class="{css}">{content}<br><br>{badges}</div>',
                unsafe_allow_html=True,
            )
            st.write("")

# ── Tab 2: Facet Scores ───────────────────────────────────────────────────────
with tab_scores:
    if scores is None:
        st.info("⏳ Score this conversation first via `python -m src.scorer`.")
    else:
        scored_turns = scores["turns"]
        turn_options = {
            f"Turn {t['turn_number']} ({t['role']}): {t['content'][:60]}…": i
            for i, t in enumerate(scored_turns)
        }
        selected_turn_label = st.selectbox("Select turn", list(turn_options.keys()))
        sel_idx = turn_options[selected_turn_label]
        sel_turn = scored_turns[sel_idx]

        col_filter, col_sort, col_conf = st.columns(3)
        categories = ["All"] + sorted(registry["category"].dropna().unique().tolist())
        cat_filter = col_filter.selectbox("Category", categories)
        sort_by = col_sort.selectbox("Sort by", ["Score ↓", "Score ↑", "Confidence ↓", "Facet name"])
        min_conf = col_conf.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

        df = scores_to_df(sel_turn["scores"], registry)
        df = df[df["score"].notna()]
        if cat_filter != "All":
            df = df[df["category"] == cat_filter]
        df = df[df["confidence"] >= min_conf]

        sort_map = {
            "Score ↓": ("score", False),
            "Score ↑": ("score", True),
            "Confidence ↓": ("confidence", False),
            "Facet name": ("facet_name", True),
        }
        scol, sasc = sort_map[sort_by]
        df = df.sort_values(scol, ascending=sasc).reset_index(drop=True)

        st.caption(f"Showing {len(df)} facets for this turn")

        for _, row in df.iterrows():
            score = int(row["score"])
            conf = float(row["confidence"])
            color = score_color(score)
            with st.expander(
                f"**{row['facet_name']}**  ·  "
                f"Score {score}/5 ({SCORE_LABELS[score]})  ·  "
                f"Confidence {conf:.0%}  ·  `{row['category']}`"
            ):
                scol1, scol2 = st.columns([1, 3])
                with scol1:
                    st.markdown(
                        f'<div style="background:{color};color:white;border-radius:8px;'
                        f'padding:16px;text-align:center;font-size:2.5rem;font-weight:800;">'
                        f'{score}</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(conf, text=f"Confidence: {conf:.0%}")
                with scol2:
                    if row.get("description"):
                        st.caption(f"**What this measures:** {row['description']}")
                    st.markdown(f"**Reasoning:** {row.get('reasoning', '—')}")
                    pol = row.get("polarity", "neutral")
                    pol_icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(pol, "⚪")
                    st.caption(f"Polarity: {pol_icon} {pol}")

# ── Tab 3: Analytics ─────────────────────────────────────────────────────────
with tab_analytics:
    if scores is None:
        st.info("⏳ Score this conversation first via `python -m src.scorer`.")
    else:
        scored_turns = scores["turns"]

        # Build full dataframe: all turns × all facets
        all_rows = []
        for t in scored_turns:
            for facet, data in t["scores"].items():
                if data.get("score") is not None:
                    all_rows.append({
                        "turn_id": t["turn_id"],
                        "turn_number": t["turn_number"],
                        "role": t["role"],
                        "facet_name": facet,
                        "score": data["score"],
                        "confidence": data["confidence"],
                    })
        full_df = pd.DataFrame(all_rows)
        full_df = full_df.merge(
            registry[["facet_name", "category", "polarity"]],
            on="facet_name", how="left",
        )
        full_df["category"] = full_df["category"].fillna("General")

        col_a, col_b = st.columns(2)

        # --- Chart 1: Average score by category (radar) ---
        with col_a:
            st.subheader("Score by Category")
            cat_avg = (
                full_df.groupby("category")["score"].mean().reset_index()
                .rename(columns={"score": "avg_score"})
                .sort_values("avg_score", ascending=False)
            )
            fig_bar = px.bar(
                cat_avg, x="category", y="avg_score",
                color="avg_score",
                color_continuous_scale=["#d32f2f", "#fbc02d", "#1b5e20"],
                range_color=[1, 5],
                labels={"avg_score": "Avg Score", "category": ""},
                height=320,
            )
            fig_bar.update_layout(
                showlegend=False, coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Chart 2: Confidence distribution ---
        with col_b:
            st.subheader("Confidence Distribution")
            fig_hist = px.histogram(
                full_df, x="confidence", nbins=20,
                color_discrete_sequence=["#7c4dff"],
                labels={"confidence": "Confidence", "count": "# Facets"},
                height=320,
            )
            fig_hist.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # --- Chart 3: Turn × Category heatmap ---
        st.subheader("Score Heatmap — Turn × Category")
        pivot = (
            full_df.groupby(["turn_number", "category"])["score"]
            .mean()
            .reset_index()
            .pivot(index="category", columns="turn_number", values="score")
        )
        pivot.columns = [f"T{c}" for c in pivot.columns]
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale=["#d32f2f", "#fbc02d", "#1b5e20"],
            range_color=[1, 5],
            aspect="auto",
            labels={"x": "Turn", "y": "Category", "color": "Avg Score"},
            height=350,
        )
        fig_heat.update_layout(margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        # --- Chart 4: User vs Assistant score comparison ---
        st.subheader("User vs Assistant — Average Score per Category")
        role_cat = (
            full_df.groupby(["role", "category"])["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "avg_score"})
        )
        fig_role = px.bar(
            role_cat, x="category", y="avg_score", color="role",
            barmode="group",
            color_discrete_map={"user": "#1976d2", "assistant": "#7b1fa2"},
            labels={"avg_score": "Avg Score", "category": "", "role": "Role"},
            height=320,
        )
        fig_role.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_role, use_container_width=True)
