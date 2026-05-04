import os
import spacy
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

BASE = r"C:\Users\MICHA\Codes\MusicPromptDescription"
DATA_PATH = os.path.join(BASE, "analysis", "vector", "prompt_description_similarity.csv")
CONC_PATH = os.path.join(BASE, "analysis", "word_level", "Concreteness_ratings_Brysbaert_et_al_BRM.txt")
OUT_CSV   = os.path.join(BASE, "analysis", "word_level", "concreteness_scores.csv")

nlp = spacy.load("en_core_web_sm")
for w in {"song", "music", "track", "sound", "audio", "feel", "make", "hear", "listen", "like", "vibe"}: # domain-specific stop words to ignore in concreteness scoring
    nlp.vocab[w].is_stop = True

def lemmatize(text):
    if pd.isna(text) or not str(text).strip():
        return []
    doc = nlp(str(text).lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    merged, i = [], 0
    while i < len(tokens):
        if tokens[i] == "hip" and i + 1 < len(tokens) and tokens[i + 1] == "hop":
            merged.append("hip-hop"); i += 2
        else:
            merged.append(tokens[i]); i += 1
    return merged

conc_dict = dict(zip(
    pd.read_csv(CONC_PATH, sep="\t")["Word"].str.lower(),
    pd.read_csv(CONC_PATH, sep="\t")["Conc.M"],
))

def score(text):
    tokens = lemmatize(text)
    rated = [conc_dict[w] for w in tokens if w in conc_dict]
    return float(np.mean(rated)) if rated else None, (len(rated)/len(tokens) if tokens else 0.0)

df = pd.read_csv(DATA_PATH).dropna(subset=["prompt", "description", "similarity"])

prompt_scores = df.drop_duplicates("song_id")[["song_id", "prompt"]].copy()
prompt_scores[["prompt_conc", "prompt_cov"]] = prompt_scores["prompt"].apply(lambda t: pd.Series(score(t)))

df[["desc_conc", "desc_cov"]] = df["description"].apply(lambda t: pd.Series(score(t)))
df = df.merge(prompt_scores[["song_id", "prompt_conc", "prompt_cov"]], on="song_id")
df = df.dropna(subset=["prompt_conc", "desc_conc"])

df[["song_id", "respondent", "prompt_conc", "prompt_cov", "desc_conc", "desc_cov", "similarity"]].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

song = df.groupby("song_id").agg(
    prompt_conc=("prompt_conc", "first"),
    desc_conc=("desc_conc", "mean"),
    similarity=("similarity", "mean"),
).reset_index()

W, p_w = wilcoxon(song["prompt_conc"], song["desc_conc"])
diff    = song["desc_conc"] - song["prompt_conc"]
r_pa, p_pa = spearmanr(song["prompt_conc"], song["similarity"])
r_da, p_da = spearmanr(song["desc_conc"],   song["similarity"])

print(f"n = {len(song)} songs,  {len(df)} rows\n")
print(f"Prompt concreteness    : M={song['prompt_conc'].mean():.3f}  SD={song['prompt_conc'].std():.3f}")
print(f"Description concreteness: M={song['desc_conc'].mean():.3f}  SD={song['desc_conc'].std():.3f}")
print(f"Δ (desc − prompt)      : {diff.mean():+.3f}")
print(f"Wilcoxon               : W={W:.0f}, p={p_w:.4f}\n")
print(f"Prompt conc  × alignment: r={r_pa:+.3f}, p={p_pa:.4f}")
print(f"Desc conc    × alignment: r={r_da:+.3f}, p={p_da:.4f}")
print(f"\nSaved → {OUT_CSV}")
