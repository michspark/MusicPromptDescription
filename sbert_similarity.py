import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load data
survey = pd.read_csv("data/0403_eng.csv")
udio = pd.read_csv("data/udio_sample_data.csv")

# Skip first two metadata rows (question label row + importId row)
survey_data = survey.iloc[2:].reset_index(drop=True)

# Build mapping: hash prefix (first 20 chars) -> udio row
udio["hash_prefix"] = udio["url"].str.extract(r"songs/(.+)")[0].str[:20]
hash_to_udio = udio.set_index("hash_prefix")[["url", "prompt"]].to_dict("index")

# Identify hash columns in survey (column names are 20-char hash prefixes)
hash_cols = [c for c in survey.columns if c in hash_to_udio]
print(f"Matched {len(hash_cols)} song columns between survey and udio data")

# Load SBERT model
print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Build rows: person_id, url, original_udio_prompt, description, similarity
records = []
for _, row in survey_data.iterrows():
    person_id = row["ResponseId"]
    for col in hash_cols:
        description = row[col]
        if pd.isna(description) or str(description).strip() == "":
            continue
        udio_row = hash_to_udio[col]
        prompt = udio_row["prompt"]
        url = udio_row["url"]
        if pd.isna(prompt) or str(prompt).strip() == "":
            continue
        records.append({
            "person_id": person_id,
            "url": url,
            "original_udio_prompt": prompt,
            "description": description,
        })

print(f"Computing SBERT similarity for {len(records)} pairs...")
prompts = [r["original_udio_prompt"] for r in records]
descriptions = [r["description"] for r in records]

prompt_embeddings = model.encode(prompts, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
desc_embeddings = model.encode(descriptions, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

similarities = util.cos_sim(prompt_embeddings, desc_embeddings).diagonal().cpu().numpy()

for i, rec in enumerate(records):
    rec["similarity"] = round(float(similarities[i]), 4)

result_df = pd.DataFrame(records, columns=["person_id", "url", "original_udio_prompt", "description", "similarity"])
output_path = "data/sbert_similarity_results.csv"
result_df.to_csv(output_path, index=False)
print(f"Saved {len(result_df)} rows to {output_path}")
print(result_df.head())
