"""
UMAP dimensionality reduction and visualization for music prompts vs descriptions.

Panel A: Prompts vs Descriptions — do the two registers occupy different regions?
Panel B: Same UMAP space, prompts colored by alignment quartile (top 25% vs bottom 25%).

GPU acceleration:
  - SBERT encoding: auto-uses CUDA if available (torch)
  - UMAP: uses cuML (RAPIDS) GPU UMAP if available, else falls back to umap-learn (CPU)
    Install cuML: https://docs.rapids.ai/install  (requires CUDA 11.x / 12.x)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sentence_transformers import SentenceTransformer

# ── GPU detection ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch device: {DEVICE}", flush=True)
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

# cuML UMAP (GPU) with fallback to umap-learn (CPU)
try:
    from cuml.manifold import UMAP as cumlUMAP
    USE_GPU_UMAP = True
    print("cuML detected → using GPU UMAP", flush=True)
except ImportError:
    import umap as umap_cpu
    USE_GPU_UMAP = False
    print("cuML not found → using CPU UMAP (pip install cuml for GPU support)", flush=True)

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = "../latent/prompt_description_similarity.csv"
MODEL_NAME = "all-mpnet-base-v2"   # fast, good quality SBERT model
UMAP_SEED  = 42
OUTPUT_PATH = "umap_panels.png"
# Larger batch size on GPU (fits more into VRAM)
ENCODE_BATCH = 512 if DEVICE == "cuda" else 64

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["prompt", "description", "similarity"])

# One prompt text per song_id (deduplicate)
prompt_df = df.groupby("song_id").agg(
    prompt=("prompt", "first"),
    mean_similarity=("similarity", "mean"),
).reset_index()

# All individual descriptions
desc_texts = df["description"].tolist()

# ── Alignment quartiles (per song_id) ────────────────────────────────────────
q25 = prompt_df["mean_similarity"].quantile(0.25)
q75 = prompt_df["mean_similarity"].quantile(0.75)

def quartile_label(sim):
    if sim >= q75:
        return "high"
    elif sim <= q25:
        return "low"
    return "mid"

prompt_df["quartile"] = prompt_df["mean_similarity"].apply(quartile_label)

# ── Encode all texts with SBERT ───────────────────────────────────────────────
print(f"Loading SBERT model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

prompt_texts = prompt_df["prompt"].tolist()

print(f"Encoding {len(prompt_texts)} prompts …")
prompt_embs = model.encode(prompt_texts, show_progress_bar=True, batch_size=ENCODE_BATCH)

print(f"Encoding {len(desc_texts)} descriptions …")
desc_embs = model.encode(desc_texts, show_progress_bar=True, batch_size=ENCODE_BATCH)

# ── Joint UMAP fit ────────────────────────────────────────────────────────────
all_embs = np.vstack([prompt_embs, desc_embs])

print(f"Fitting UMAP ({'GPU/cuML' if USE_GPU_UMAP else 'CPU'}) …")
umap_kwargs = dict(n_neighbors=15, min_dist=0.1, n_components=2)
if USE_GPU_UMAP:
    # cuML UMAP: cosine metric via precomputed or direct (use euclidean on L2-normed vecs)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    all_embs_normed = all_embs / np.clip(norms, 1e-8, None)
    reducer = cumlUMAP(**umap_kwargs, random_state=UMAP_SEED)
    all_2d = reducer.fit_transform(all_embs_normed)
    all_2d = np.array(all_2d)   # cuDF → numpy
else:
    reducer = umap_cpu.UMAP(**umap_kwargs, metric="cosine", random_state=UMAP_SEED)
    all_2d = reducer.fit_transform(all_embs)

n_prompts = len(prompt_texts)
prompt_2d = all_2d[:n_prompts]
desc_2d   = all_2d[n_prompts:]

# ── Color maps ────────────────────────────────────────────────────────────────
PROMPT_COLOR = "#2563EB"   # blue
DESC_COLOR   = "#F59E0B"   # amber
HIGH_COLOR   = "#16A34A"   # green  (top 25%)
LOW_COLOR    = "#DC2626"   # red    (bottom 25%)
MID_COLOR    = "#9CA3AF"   # gray   (middle 50%)
DESC_BG_COLOR = "#D1D5DB"  # light gray for background descriptions in Panel B

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#F8FAFC")

# ── Panel A: Prompts vs Descriptions ─────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#F1F5F9")

ax.scatter(
    desc_2d[:, 0], desc_2d[:, 1],
    c=DESC_COLOR, alpha=0.35, s=12, linewidths=0,
    label=f"Descriptions (n={len(desc_texts)})",
    zorder=2,
)
ax.scatter(
    prompt_2d[:, 0], prompt_2d[:, 1],
    c=PROMPT_COLOR, alpha=0.85, s=60, marker="D", linewidths=0.4,
    edgecolors="white",
    label=f"Prompts (n={n_prompts})",
    zorder=3,
)

ax.set_title("Panel A\nPrompts vs Descriptions", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("UMAP 1", fontsize=10)
ax.set_ylabel("UMAP 2", fontsize=10)
ax.legend(loc="best", fontsize=9, framealpha=0.8)
ax.tick_params(labelsize=8)

# ── Panel B: Prompts colored by alignment quartile ───────────────────────────
ax = axes[1]
ax.set_facecolor("#F1F5F9")

# Description cloud as muted background
ax.scatter(
    desc_2d[:, 0], desc_2d[:, 1],
    c=DESC_BG_COLOR, alpha=0.25, s=10, linewidths=0,
    zorder=1,
)

quartile_styles = {
    "high": (HIGH_COLOR, "Top 25% (high alignment)"),
    "mid":  (MID_COLOR,  "Middle 50%"),
    "low":  (LOW_COLOR,  "Bottom 25% (low alignment)"),
}
z_order = {"high": 4, "mid": 2, "low": 3}
alpha   = {"high": 0.90, "mid": 0.40, "low": 0.90}
sizes   = {"high": 80,   "mid": 40,   "low": 80}

for q, (color, label) in quartile_styles.items():
    mask = prompt_df["quartile"] == q
    ax.scatter(
        prompt_2d[mask.values, 0], prompt_2d[mask.values, 1],
        c=color, alpha=alpha[q], s=sizes[q],
        marker="D", linewidths=0.4, edgecolors="white",
        label=f"{label} (n={mask.sum()})",
        zorder=z_order[q],
    )

ax.set_title("Panel B\nPrompts Colored by Alignment Quartile", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("UMAP 1", fontsize=10)
ax.set_ylabel("UMAP 2", fontsize=10)

legend_patches = [
    mpatches.Patch(color=HIGH_COLOR, label=f"Top 25% — high alignment (≥{q75:.3f})"),
    mpatches.Patch(color=LOW_COLOR,  label=f"Bottom 25% — low alignment (≤{q25:.3f})"),
    mpatches.Patch(color=MID_COLOR,  label="Middle 50%"),
    mpatches.Patch(color=DESC_BG_COLOR, label="Descriptions (background)"),
]
ax.legend(handles=legend_patches, loc="best", fontsize=8.5, framealpha=0.85)
ax.tick_params(labelsize=8)

# ── Shared summary ────────────────────────────────────────────────────────────
fig.suptitle(
    "UMAP of Music Prompts and Listener Descriptions\n"
    f"(SBERT: {MODEL_NAME}  |  cosine metric  |  n_neighbors=15)",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUTPUT_PATH}")
plt.show()

# ── Quick distance summary (Panel B insight) ─────────────────────────────────
desc_centroid = desc_2d.mean(axis=0)

for q in ["high", "low"]:
    mask = (prompt_df["quartile"] == q).values
    pts  = prompt_2d[mask]
    mean_dist = np.linalg.norm(pts - desc_centroid, axis=1).mean()
    print(f"{q.upper():4s} alignment prompts → mean distance from description centroid: {mean_dist:.4f}")
