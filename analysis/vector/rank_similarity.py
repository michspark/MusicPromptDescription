import pandas as pd

sim_path = r"C:\Users\MICHA\Codes\MusicPromptDescription\analysis\vector\prompt_description_similarity.csv"
df = pd.read_csv(sim_path)

prompt_df = pd.read_csv(r"C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_data.csv")
prompt_df["song_id"] = prompt_df["url"].str.extract(r"/songs/([a-f0-9-]+)").iloc[:, 0].str[:18]
id_to_url = prompt_df.set_index("song_id")["url"].to_dict()

stats = (
    df.groupby("song_id")["similarity"]
    .agg(count="count", mean="mean", std="std")
    .reset_index()
)
stats["std"] = stats["std"].fillna(0)
stats["mean"] = stats["mean"].round(4)
stats["std"]  = stats["std"].round(4)
stats["url"]  = stats["song_id"].map(id_to_url)
stats["prompt"] = stats["song_id"].map(df.groupby("song_id")["prompt"].first())

stats = stats.sort_values("mean", ascending=False).reset_index(drop=True)
stats.index += 1
stats.index.name = "rank"

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 160)
print(stats[["song_id", "url", "mean", "std", "count", "prompt"]].to_string())

out_path = r"C:\Users\MICHA\Codes\MusicPromptDescription\analysis\vector\song_similarity_ranked.csv"
stats.to_csv(out_path)
print(f"\nSaved -> {out_path}")
