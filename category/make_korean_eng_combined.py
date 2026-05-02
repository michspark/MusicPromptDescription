from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent / "data"

KOREAN_DENSITY = BASE / "0428_korean_categories_density.csv"
ENG_DENSITY    = BASE / "0428_eng_categories_density.csv"

CATEGORIES = [
    "genre", "mood/emotion", "instrumentation",
    "timbre", "function", "music theory", "story/narrative/lyrics",
]
CAT_LABELS = ["genre", "mood", "instrumentation", "timbre", "function", "theory", "story"]


def to_long(df: pd.DataFrame, corpus: str, prefix: str) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        song_id = str(row["problem_hash"]).strip()
        text_id = f"{prefix}{i:05d}"
        for cat, label in zip(CATEGORIES, CAT_LABELS):
            d = float(row.get(cat, 0.0))
            if np.isnan(d):
                d = 0.0
            rows.append({
                "text_id":  text_id,
                "song_id":  song_id,
                "corpus":   corpus,
                "category": label,
                "presence": int(d > 0),
                "density":  d,
            })
    return pd.DataFrame(rows)


korean_df = pd.read_csv(KOREAN_DENSITY)
eng_df    = pd.read_csv(ENG_DENSITY)

long_df = pd.concat([
    to_long(korean_df, "korean",  "k"),
    to_long(eng_df,    "english", "e"),
], ignore_index=True)

out_path = BASE / "long_korean_eng.csv"
long_df.to_csv(out_path, index=False)
print(f"Saved {len(long_df):,} rows -> {out_path}")
print(f"  korean rows : {(long_df.corpus == 'korean').sum():,}")
print(f"  english rows: {(long_df.corpus == 'english').sum():,}")
