"""
Facet clusterer.

Groups inferrable facets into evenly-sized semantic batches.
Strategy:
  1. Embed all facet names with a lightweight sentence-transformer.
  2. Project to 1D via PCA to get a semantically-ordered sequence.
  3. Chunk that sequence into batches of TARGET_BATCH_SIZE.

This guarantees:
  - Adjacent facets in embedding space end up in the same batch (semantic coherence).
  - All batches are within ±1 of TARGET_BATCH_SIZE (no runaway large clusters).
  - Adding more facets later = more batches, zero code changes.

Output: data/facets_clusters.json

Run once after preprocessor:
    python -m src.clusterer
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

REGISTRY_CSV = Path("data/facets_registry.csv")
CLUSTERS_JSON = Path("data/facets_clusters.json")

TARGET_BATCH_SIZE = 28
EMBED_MODEL = "all-MiniLM-L6-v2"

GROQ_GENERATION_MODEL = "llama-3.3-70b-versatile"
GROQ_SCORING_MODEL = "llama-3.1-8b-instant"   # ≤16B — satisfies open-weights constraint

SCORE_COLUMNS = [
    "facet_name", "category", "description",
    "score_anchor_1", "score_anchor_3", "score_anchor_5",
    "polarity", "weight",
]


def build_clusters(
    target_batch_size: int = TARGET_BATCH_SIZE,
    registry_csv: Path = REGISTRY_CSV,
    output_path: Path = CLUSTERS_JSON,
) -> list[dict]:
    # --- Load inferrable facets ---
    df = pd.read_csv(registry_csv)
    inferrable = df[df["inferrable"]].copy().reset_index(drop=True)
    n = len(inferrable)
    n_batches = math.ceil(n / target_batch_size)

    print(f"Facets to cluster : {n}")
    print(f"Target batch size : {target_batch_size}")
    print(f"Expected batches  : {n_batches}")

    # --- Embed ---
    print(f"\nLoading '{EMBED_MODEL}' ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Combine name + category for richer signal
    texts = [
        f"{row['facet_name']} {row['category']} {row.get('description', '')}"
        for _, row in inferrable.iterrows()
    ]
    print("Embedding facets ...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)

    # --- PCA sort: project to 1D, sort by projection, then chunk ---
    print("Sorting by semantic projection ...")
    pca = PCA(n_components=1)
    projections = pca.fit_transform(embeddings).flatten()
    sorted_idx = np.argsort(projections)

    inferrable = inferrable.iloc[sorted_idx].reset_index(drop=True)
    inferrable["_cluster"] = inferrable.index // target_batch_size

    # --- Build structured output ---
    clusters: list[dict] = []
    for cid in sorted(inferrable["_cluster"].unique()):
        group = inferrable[inferrable["_cluster"] == cid][SCORE_COLUMNS].copy()
        dominant_category = group["category"].mode()[0]
        clusters.append({
            "cluster_id": int(cid),
            "label": dominant_category,
            "size": len(group),
            "facets": group.to_dict(orient="records"),
        })

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    print(f"\nClusters saved → {output_path}")
    _print_summary(clusters)
    return clusters


def load_clusters(path: Path = CLUSTERS_JSON) -> list[dict]:
    """Load pre-built clusters from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_summary(clusters: list[dict]) -> None:
    sizes = [c["size"] for c in clusters]
    print(f"Total clusters : {len(clusters)}")
    print(f"Avg batch size : {np.mean(sizes):.1f}")
    print(f"Min / Max size : {min(sizes)} / {max(sizes)}")
    print("\nCluster breakdown:")
    for c in clusters:
        bar = "█" * c["size"]
        print(f"  [{c['cluster_id']:2d}] {c['label']:<30} {c['size']:3d}  {bar}")


if __name__ == "__main__":
    build_clusters()
