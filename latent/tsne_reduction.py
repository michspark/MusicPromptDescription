"""
t-SNE dimensionality reduction and visualization for music prompts vs descriptions.

Panel A: Prompts vs Descriptions — do the two registers occupy different regions?
Panel B: Same t-SNE space, prompts colored by alignment quartile (top 25% vs bottom 25%).

GPU acceleration:
  - SBERT encoding: auto-uses CUDA if available (torch)
  - t-SNE: uses cuML (RAPIDS) GPU t-SNE if available, else falls back to sklearn (CPU)
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

# cuML t-SNE (GPU) with fallback to sklearn (CPU)
try:
    from cuml.manifold import TSNE as cumlTSNE
    USE_GPU_TSNE = True
    print("cuML detected → using GPU t-SNE", flush=True)
except ImportError:
    from sklearn.manifold import TSNE as skTSNE
    USE_GPU_TSNE = False
    print("cuML not found → using sklearn CPU t-SNE (pip install cuml for GPU support)", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "../latent/prompt_description_similarity.csv"
MODEL_NAME  = "all-mpnet-base-v2"
TSNE_SEED   = 42
OUTPUT_PATH = "tsne_panels.png"
ENCODE_BATCH = 512 if DEVICE == "cuda" else 64

# t-SNE hyperparams
PERPLEXITY   = 40    # increase if dataset is large (rule of thumb: sqrt(N))
N_ITER       = 1000
LEARNING_RATE = "auto"   # sklearn ≥1.2 / cuML ignores this; good default

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["prompt", "description", "similarity"])

# One prompt text per song_id
prompt_df = df.groupby("song_id").agg(
    prompt=("prompt", "first"),
    mean_similarity=("similarity", "mean"),
).reset_index()

desc_texts = df["description"].tolist()

# ── Alignment quartiles ───────────────────────────────────────────────────────
q25 = prompt_df["mean_similarity"].quantile(0.25)
q75 = prompt_df["mean_similarity"].quantile(0.75)

def quartile_label(sim):
    if sim >= q75:
        return "high"
    elif sim <= q25:
        return "low"
    return "mid"

prompt_df["quartile"] = prompt_df["mean_similarity"].apply(quartile_label)

# ── SBERT encoding ────────────────────────────────────────────────────────────
print(f"Loading SBERT model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

prompt_texts = prompt_df["prompt"].tolist()

print(f"Encoding {len(prompt_texts)} prompts …")
prompt_embs = model.encode(prompt_texts, show_progress_bar=True, batch_size=ENCODE_BATCH)

print(f"Encoding {len(desc_texts)} descriptions …")
desc_embs = model.encode(desc_texts, show_progress_bar=True, batch_size=ENCODE_BATCH)

# ── Joint t-SNE fit ───────────────────────────────────────────────────────────
all_embs = np.vstack([prompt_embs, desc_embs])
n_prompts = len(prompt_texts)

# L2-normalize so cosine distance ≈ euclidean distance on the unit sphere.
# Both cuML and sklearn benefit from this when metric="cosine" isn't supported.
norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
all_embs_normed = all_embs / np.clip(norms, 1e-8, None)

print(f"Fitting t-SNE ({'GPU/cuML' if USE_GPU_TSNE else 'CPU/sklearn'}) "
      f"on {len(all_embs_normed)} samples …")

if USE_GPU_TSNE:
    tsne = cumlTSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        n_iter=N_ITER,
        metric="euclidean",   # on L2-normed vecs → cosine equivalent
        random_state=TSNE_SEED,
        verbose=1,
    )
    all_2d = np.array(tsne.fit_transform(all_embs_normed))
else:
    import sklearn
    iter_kwarg = (
        {"max_iter": N_ITER}
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5)
        else {"n_iter": N_ITER}
    )
    tsne = skTSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        **iter_kwarg,
        metric="cosine",
        learning_rate=LEARNING_RATE,
        init="pca",           # more stable than random init
        random_state=TSNE_SEED,
        verbose=1,
        n_jobs=-1,
    )
    all_2d = tsne.fit_transform(all_embs_normed)

prompt_2d = all_2d[:n_prompts]
desc_2d   = all_2d[n_prompts:]

# ── Colors ────────────────────────────────────────────────────────────────────
PROMPT_COLOR  = "#2563EB"
DESC_COLOR    = "#F59E0B"
HIGH_COLOR    = "#16A34A"
LOW_COLOR     = "#DC2626"
MID_COLOR     = "#9CA3AF"
DESC_BG_COLOR = "#D1D5DB"

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
ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.legend(loc="best", fontsize=9, framealpha=0.8)
ax.tick_params(labelsize=8)

# ── Panel B: Prompts colored by alignment quartile ───────────────────────────
ax = axes[1]
ax.set_facecolor("#F1F5F9")

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
ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)

legend_patches = [
    mpatches.Patch(color=HIGH_COLOR,  label=f"Top 25% — high alignment (≥{q75:.3f})"),
    mpatches.Patch(color=LOW_COLOR,   label=f"Bottom 25% — low alignment (≤{q25:.3f})"),
    mpatches.Patch(color=MID_COLOR,   label="Middle 50%"),
    mpatches.Patch(color=DESC_BG_COLOR, label="Descriptions (background)"),
]
ax.legend(handles=legend_patches, loc="best", fontsize=8.5, framealpha=0.85)
ax.tick_params(labelsize=8)

# ── Shared title ──────────────────────────────────────────────────────────────
backend = "GPU/cuML" if USE_GPU_TSNE else "CPU/sklearn"
fig.suptitle(
    "t-SNE of Music Prompts and Listener Descriptions\n"
    f"(SBERT: {MODEL_NAME}  |  perplexity={PERPLEXITY}  |  {backend})",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUTPUT_PATH}")
plt.show()

# ── Distance summary ──────────────────────────────────────────────────────────
desc_centroid = desc_2d.mean(axis=0)
for q in ["high", "low"]:
    mask = (prompt_df["quartile"] == q).values
    pts  = prompt_2d[mask]
    mean_dist = np.linalg.norm(pts - desc_centroid, axis=1).mean()
    print(f"{q.upper():4s} alignment prompts → mean distance from description centroid: {mean_dist:.4f}")
