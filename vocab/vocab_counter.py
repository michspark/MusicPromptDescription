import csv
import re
from collections import Counter

DATA_DIR = r"C:\Users\MICHA\Codes\MusicPromptDescription\data"
OUT_DIR = r"C:\Users\MICHA\Codes\MusicPromptDescription\vocab"
TOP_N = 300


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alpha characters, drop empty/single-char tokens."""
    text = text.lower()
    return [w for w in re.split(r"[^a-z']+", text) if len(w) > 1]


# ── 1. PROMPT word count ─ from udio_sample_data.csv, "prompt" column ─────────

prompt_counter: Counter = Counter()

with open(f"{DATA_DIR}/udio_sample_data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cell = row.get("prompt", "")
        if cell:
            prompt_counter.update(tokenize(cell))

with open(f"{OUT_DIR}/prompt_word_count.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "count"])
    for word, count in prompt_counter.most_common(TOP_N):
        writer.writerow([word, count])

print(f"prompt_word_count.csv written — top {TOP_N} words from udio_sample_data.csv (prompt column)")


# ── 2. DESCRIPTION word count ─ from 0403_eng.csv, description columns ────────
# Row 0: machine IDs (header)
# Row 1: human-readable question text
# Row 2: Qualtrics import metadata
# Row 3+: actual responses
# Description columns start at index 18 (col 17 is the practice block, skip it)

DESCRIPTION_START_COL = 18

desc_counter: Counter = Counter()

with open(f"{DATA_DIR}/0403_eng.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)

# Identify description column indices from human-readable row (row index 1)
human_headers = rows[1]
desc_col_indices = [
    i for i in range(DESCRIPTION_START_COL, len(human_headers))
    if "describe" in human_headers[i].lower()
]

# Process response rows (skip the 3 header rows: 0, 1, 2)
for row in rows[3:]:
    for idx in desc_col_indices:
        if idx < len(row):
            cell = row[idx].strip()
            if cell:
                desc_counter.update(tokenize(cell))

with open(f"{OUT_DIR}/description_word_count.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "count"])
    for word, count in desc_counter.most_common(TOP_N):
        writer.writerow([word, count])

print(f"description_word_count.csv written — top {TOP_N} words from 0403_eng.csv (description columns)")
print(f"  Description columns used: {len(desc_col_indices)} columns (indices {desc_col_indices[0]}–{desc_col_indices[-1]})")
