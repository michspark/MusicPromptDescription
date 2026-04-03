from sentence_transformers import SentenceTransformer
import pandas as pd

# Load model (same as sbert_test.py)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Load data ────────────────────────────────────────────────────────────────
# Description CSV: row 0 = col headers (truncated UUIDs), row 1 = question text,
#                  row 2 = Qualtrics import IDs, row 3+ = actual responses
desc_raw = pd.read_csv("data/survey_description_data.csv", header=None)
desc_headers = desc_raw.iloc[0].tolist()          # column names
desc_data    = desc_raw.iloc[3:].reset_index(drop=True)  # actual responses
desc_data.columns = desc_headers

# Prompt CSV: url contains full UUID, prompt column has the text
prompt_df = pd.read_csv("data/survey_prompt_data.csv")
prompt_df["uuid_full"] = prompt_df["url"].str.extract(r"/songs/([a-f0-9-]+)")

# ── Song columns only (skip metadata columns 0‥17) ──────────────────────────
song_cols = desc_headers[18:]   # everything after 'Practice Ex'

# Build a lookup: truncated_uuid → prompt text
# Column headers are the first N chars of the full UUID — match by prefix
prompt_lookup = {}
for _, row in prompt_df.iterrows():
    uuid = str(row["uuid_full"])
    for col in song_cols:
        if isinstance(col, str) and uuid.startswith(col):
            prompt_lookup[col] = row["prompt"]
            break

# ── Compute similarities ─────────────────────────────────────────────────────
records = []

for col in song_cols:
    prompt_text = prompt_lookup.get(col)
    if prompt_text is None or pd.isna(prompt_text):
        continue  # no matching prompt – skip

    for row_idx, description in desc_data[col].items():
        if pd.isna(description) or str(description).strip() == "":
            continue  # question not answered – skip

        prompt_emb = model.encode(str(prompt_text), convert_to_tensor=True)
        desc_emb   = model.encode(str(description),  convert_to_tensor=True)
        similarity = float(model.similarity(prompt_emb.unsqueeze(0),
                                            desc_emb.unsqueeze(0))[0][0])

        records.append({
            "song_id":     col,
            "respondent":  row_idx,
            "prompt":      prompt_text,
            "description": description,
            "similarity":  round(similarity, 4),
        })

# ── Save ──────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(records)
out_path = "latent/prompt_description_similarity.csv"
out_df.to_csv(out_path, index=False)

print(f"Saved {len(out_df)} rows -> {out_path}")
print(out_df.describe())