import numpy as np
import pandas as pd

CATEGORIES = [
    "genre",
    "mood/emotion",
    "instrumentation",
    "timbre",
    "function",
    "music theory",
    "story/narrative/lyrics",
]

# ── Load data ──────────────────────────────────────────────────────────────────
ranked_df = pd.read_csv("song_similarity_ranked.csv")
cat_df = pd.read_csv("../data/udio_sample_categories.csv")          # text labels → percentage
density_df = pd.read_csv("../data/udio_sample_categories_density.csv")  # float 0-1 → density

# ── Match truncated song_id to full UUID ───────────────────────────────────────
def match_id(short_id, full_ids):
    for fid in full_ids:
        if fid.startswith(short_id):
            return fid
    return None

full_ids = cat_df["song_id"].tolist()
ranked_df["full_id"] = ranked_df["song_id"].apply(lambda x: match_id(x, full_ids))

unmatched = ranked_df["full_id"].isna().sum()
if unmatched > 0:
    print(f"[Warning] {unmatched} songs could not be matched to categorization data.")

ranked_df = ranked_df.dropna(subset=["full_id"])

# ── Split into high / low 25% by mean similarity ──────────────────────────────
n = len(ranked_df)
n_quarter = int(np.ceil(n * 0.25))

# rank is 1-based and already sorted descending by mean → rank 1 = highest mean
high_ids = set(ranked_df[ranked_df["rank"] <= n_quarter]["full_id"])
low_ids  = set(ranked_df[ranked_df["rank"] > n - n_quarter]["full_id"])

print(f"Total songs: {n}")
print(f"High 25% (top {n_quarter}): ranks 1–{n_quarter}")
print(f"Low  25% (bottom {n_quarter}): ranks {n - n_quarter + 1}–{n}")

# ── Percentage (binary: is category filled?) ──────────────────────────────────
def is_filled(val):
    if pd.isna(val):
        return 0
    return 0 if str(val).strip() == "" else 1

pct_df = cat_df.copy()
for cat in CATEGORIES:
    pct_df[cat] = cat_df[cat].apply(is_filled)

high_pct = pct_df[pct_df["song_id"].isin(high_ids)][CATEGORIES]
low_pct  = pct_df[pct_df["song_id"].isin(low_ids)][CATEGORIES]

# ── Density ────────────────────────────────────────────────────────────────────
high_den = density_df[density_df["song_id"].isin(high_ids)][CATEGORIES]
low_den  = density_df[density_df["song_id"].isin(low_ids)][CATEGORIES]

# ── Print results ──────────────────────────────────────────────────────────────
def print_stats(label, pct_group, den_group):
    print(f"\n{'='*65}")
    print(f"  {label}  (n={len(pct_group)})")
    print(f"{'='*65}")
    print(f"{'Category':<28} {'Pct Mean':>9} {'Pct Std':>9} {'Den Mean':>9} {'Den Std':>9}")
    print(f"{'-'*65}")
    for cat in CATEGORIES:
        pm = pct_group[cat].mean()
        ps = pct_group[cat].std(ddof=1)
        dm = den_group[cat].mean()
        ds = den_group[cat].std(ddof=1)
        print(f"  {cat:<26} {pm:>9.4f} {ps:>9.4f} {dm:>9.4f} {ds:>9.4f}")

print_stats("HIGH 25% (highest mean similarity)", high_pct, high_den)
print_stats("LOW  25% (lowest  mean similarity)", low_pct,  low_den)

# ── Difference (High - Low) ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  DIFFERENCE: High 25% - Low 25%")
print(f"{'='*65}")
print(f"{'Category':<28} {'dPct Mean':>10} {'dDen Mean':>10}")
print(f"{'-'*65}")
for cat in CATEGORIES:
    d_pct = high_pct[cat].mean() - low_pct[cat].mean()
    d_den = high_den[cat].mean() - low_den[cat].mean()
    print(f"  {cat:<26} {d_pct:>+10.4f} {d_den:>+10.4f}")
