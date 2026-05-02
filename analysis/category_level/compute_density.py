"""
compute_density.py

Reads a categorized CSV (with pipe-separated values in category columns) and
writes a density CSV where each category value is replaced with the fraction of
category items that belong to that category for that row.

Usage (CLI):
    python compute_density.py path/to/input.csv
    python compute_density.py path/to/input.csv --output path/to/output.csv

"""

import argparse
import os
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


def _count_items(cell) -> int:
    """Count pipe-separated items in a cell. Empty / NaN cells count as 0."""
    if pd.isna(cell) or str(cell).strip() == "":
        return 0
    return len([item for item in str(cell).split("|") if item.strip()])


def compute_density(input_path: str, output_path: str | None = None) -> str:
    """
    Convert a categorized CSV to a density CSV.

    For each row, the density of a category is:
        count_of_items_in_category / total_items_across_all_categories

    Parameters
    ----------
    input_path : str
        Path to the input CSV file (e.g. "data/0414_eng_categories.csv").
    output_path : str | None
        Path for the output CSV.  If None, the output is placed next to the
        input file with "_density" appended before the extension
        (e.g. "data/0414_eng_categories_density.csv").

    Returns
    -------
    str
        The path where the density CSV was saved.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_density{ext}"

    df = pd.read_csv(input_path)

    # Only process category columns that actually exist in this file
    present_cats = [c for c in CATEGORIES if c in df.columns]

    # Count items per category per row
    counts = df[present_cats].map(_count_items)

    # Total items per row
    totals = counts.sum(axis=1)

    # Replace category columns with density values (0 when total == 0)
    df_out = df.copy()
    for cat in present_cats:
        df_out[cat] = counts[cat].where(totals == 0, counts[cat] / totals).where(
            totals != 0, 0.0
        )

    df_out.to_csv(output_path, index=False)
    print(f"Density CSV saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute category density CSV.")
    parser.add_argument("input", help="Path to the input categorized CSV")
    parser.add_argument(
        "--output", "-o", default=None, help="Output path (default: <input>_density.csv)"
    )
    args = parser.parse_args()
    compute_density(args.input, args.output)
