from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent / "data"

PROMPT_DENSITY   = BASE / "udio_sample_categories_density.csv"
DESC_DENSITY     = BASE / "0428_eng_categories_density.csv"
KOREAN_DENSITY   = BASE / "0428_korean_categories_density.csv"

CATEGORIES = [
    "genre", "mood/emotion", "instrumentation",
    "timbre", "function", "music theory", "story/narrative/lyrics",
]
CAT_LABELS = ["genre", "mood", "instrumentation", "timbre", "function", "theory", "story"]


def to_long(df: pd.DataFrame, corpus: str, song_id_col: str, prefix: str, truncate_id: int = 0) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        song_id = str(row[song_id_col]).strip()
        if truncate_id:
            song_id = song_id[:truncate_id]
        text_id = f"{prefix}{i:05d}"
        for cat, label in zip(CATEGORIES, CAT_LABELS):
            d = float(row.get(cat, 0.0))
            if np.isnan(d):
                d = 0.0
            rows.append({
                "text_id" : text_id,
                "song_id" : song_id,
                "corpus"  : corpus,
                "category": label,
                "presence": int(d > 0),
                "density" : d,
            })
    return pd.DataFrame(rows)


prompt_df  = pd.read_csv(PROMPT_DENSITY)
desc_df    = pd.read_csv(DESC_DENSITY)
korean_df  = pd.read_csv(KOREAN_DENSITY)

long_df = pd.concat([
    to_long(prompt_df,  "prompt",      "song_id",      "p", truncate_id=20),
    to_long(desc_df,    "description", "problem_hash", "d"),
], ignore_index=True)

out_path = BASE / "long_combined.csv"
long_df.to_csv(out_path, index=False)
print(f"Saved {len(long_df):,} rows -> {out_path}")
print(f"  prompt rows     : {(long_df.corpus == 'prompt').sum():,}")
print(f"  description rows: {(long_df.corpus == 'description').sum():,}")
