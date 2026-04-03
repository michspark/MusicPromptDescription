"""
Builds vocab_frequency.csv from prompt_category.csv.

Input:  ../data/prompt_category.csv   (output of prompt_vocab_classifier.py)
Output: ../data/vocab_frequency.csv

Also prints summary statistics to console:
  - Category coverage  (% of prompts using each category)
  - Term density       (share of total classified terms per category)
  - Diversity          (how many categories each prompt uses, on average)

Usage:
    python vocab_frequency.py
"""

import csv
import os
from collections import defaultdict

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
INPUT_CSV  = os.path.join(BASE_DIR, "..", "data", "prompt_category.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "data", "vocab_frequency.csv")

CATEGORIES = [
    "mood/emotion",
    "genre",
    "music theory",
    "timbre",
    "instrumentation",
    "function",
    "story/narrative/lyrics",
]


# ── Build frequency table ──────────────────────────────────────────────────────
def build_frequency_table(wide: pd.DataFrame) -> pd.DataFrame:
    """
    From wide-format prompt_category.csv, produce long-format frequency table.
    Each row = unique (term, category) with count and pipe-joined hashes.
    """
    # (term, category) → list of hashes that contain it
    term_hashes: dict[tuple, list] = defaultdict(list)

    for _, row in wide.iterrows():
        h = row["hash"]
        for cat in CATEGORIES:
            cell = str(row.get(cat, "") or "").strip()
            if not cell:
                continue
            for term in cell.split("|"):
                term = term.strip().lower()
                if term:
                    term_hashes[(term, cat)].append(h)

    rows = [
        {
            "term":     term,
            "category": cat,
            "count":    len(hashes),
            "hashes":   "|".join(hashes),
        }
        for (term, cat), hashes in sorted(
            term_hashes.items(),
            key=lambda x: (-len(x[1]), x[0][1], x[0][0]),   # sort: freq desc → category → term
        )
    ]
    return pd.DataFrame(rows)


# ── Summary statistics ─────────────────────────────────────────────────────────
def print_summary(wide: pd.DataFrame, freq: pd.DataFrame):
    n_prompts = len(wide)
    print(f"\n{'='*60}")
    print(f"  TOTAL PROMPTS: {n_prompts}")
    print(f"{'='*60}\n")

    # ── Category coverage ──────────────────────────────────────────
    print("CATEGORY COVERAGE  (% of prompts using each category)")
    print(f"{'Category':<25} {'Prompts':>8}  {'Coverage':>10}")
    print("-" * 47)
    for cat in CATEGORIES:
        used = wide[cat].fillna("").str.strip().str.len() > 0
        pct  = used.mean() * 100
        print(f"  {cat:<23} {used.sum():>8}  {pct:>9.1f}%")

    # ── Term density ───────────────────────────────────────────────
    print(f"\nTERM DENSITY  (share of all classified terms per category)")
    cat_totals = freq.groupby("category")["count"].sum()
    grand_total = cat_totals.sum()
    print(f"{'Category':<25} {'Total terms':>12}  {'Density':>10}")
    print("-" * 51)
    for cat in CATEGORIES:
        total = cat_totals.get(cat, 0)
        pct   = (total / grand_total * 100) if grand_total else 0
        print(f"  {cat:<23} {total:>12}  {pct:>9.1f}%")
    print(f"  {'TOTAL':<23} {grand_total:>12}  {'100.0%':>10}")

    # ── Prompt diversity ───────────────────────────────────────────
    wide["_cat_count"] = wide[CATEGORIES].apply(
        lambda r: sum(bool(str(v).strip()) for v in r), axis=1
    )
    print(f"\nPROMPT DIVERSITY  (categories used per prompt)")
    print(f"  Average categories per prompt : {wide['_cat_count'].mean():.2f}")
    print(f"  Max categories in one prompt  : {wide['_cat_count'].max()}")
    print(f"  Min categories in one prompt  : {wide['_cat_count'].min()}")
    print(f"  Prompts using all 7 categories: {(wide['_cat_count'] == 7).sum()}")
    print(f"  Prompts using ≥3 categories   : {(wide['_cat_count'] >= 3).sum()}")

    # ── Top 5 terms per category ───────────────────────────────────
    print(f"\nTOP 5 TERMS PER CATEGORY")
    for cat in CATEGORIES:
        subset = freq[freq["category"] == cat].head(5)
        if subset.empty:
            continue
        print(f"\n  {cat.upper()}")
        for _, r in subset.iterrows():
            print(f"    {r['term']:<25}  count={r['count']}")

    print(f"\n{'='*60}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input not found: {INPUT_CSV}\n"
            "Run prompt_vocab_classifier.py first."
        )

    wide = pd.read_csv(INPUT_CSV, dtype=str)
    print(f"Loaded {len(wide)} rows from {os.path.abspath(INPUT_CSV)}")

    freq = build_frequency_table(wide)

    freq.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved → {os.path.abspath(OUTPUT_CSV)}  ({len(freq)} unique term-category pairs)")

    print_summary(wide, freq)


if __name__ == "__main__":
    main()
