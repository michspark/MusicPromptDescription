import pandas as pd

sim_path = r"C:\Users\MICHA\Codes\MusicPromptDescription\latent\prompt_description_similarity.csv"
df = pd.read_csv(sim_path)

# Attach full URL from udio_sample_data so we can display it alongside the hash
prompt_df = pd.read_csv(r"C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_data.csv")
prompt_df["song_id"] = prompt_df["url"].str.extract(r"/songs/([a-f0-9-]+)").iloc[:, 0].str[:18]
id_to_url = prompt_df.set_index("song_id")["url"].to_dict()

# ── Per-song stats ────────────────────────────────────────────────────────────
stats = (
    df.groupby("song_id")["similarity"]
    .agg(count="count", mean="mean", std="std")
    .reset_index()
)
stats["std"] = stats["std"].fillna(0)           # single-response songs → NaN → 0
stats["mean"] = stats["mean"].round(4)
stats["std"]  = stats["std"].round(4)
stats["url"]  = stats["song_id"].map(id_to_url)
stats["prompt"] = stats["song_id"].map(df.groupby("song_id")["prompt"].first())

# ── Rank by mean (descending) ─────────────────────────────────────────────────
stats = stats.sort_values("mean", ascending=False).reset_index(drop=True)
stats.index += 1                                # 1-based rank
stats.index.name = "rank"

# ── Print ─────────────────────────────────────────────────────────────────────
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 160)
print(stats[["song_id", "url", "mean", "std", "count", "prompt"]].to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = r"C:\Users\MICHA\Codes\MusicPromptDescription\latent\song_similarity_ranked.csv"
stats.to_csv(out_path)
print(f"\nSaved -> {out_path}")
