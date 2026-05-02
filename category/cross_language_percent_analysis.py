import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

CATEGORIES = [
    "genre",
    "mood/emotion",
    "instrumentation",
    "timbre",
    "function",
    "music theory",
    "story/narrative/lyrics",
]

korean_path = "../data/0414_korean_categories.csv"
english_path = "../data/0414_eng_categories.csv"

korean_df = pd.read_csv(korean_path)
english_df = pd.read_csv(english_path)


def is_filled(series: pd.Series) -> pd.Series:
    """Return boolean Series: True if cell is non-empty, False otherwise."""
    return series.notna() & (series.astype(str).str.strip() != "")


output_path = "../data/cross_language_percent_analysis_results.txt"
lines = []

lines.append(f"Korean respondents:  {len(korean_df)}")
lines.append(f"English respondents: {len(english_df)}")
lines.append(f"Bonferroni alpha:    0.05 / {len(CATEGORIES)} = {0.05 / len(CATEGORIES):.4f}")

# Per-category analysis
alpha = 0.05 / len(CATEGORIES)

summary_rows = []

for cat in CATEGORIES:
    kor_filled = is_filled(korean_df[cat]).sum()
    kor_empty = len(korean_df) - kor_filled

    eng_filled = is_filled(english_df[cat]).sum()
    eng_empty = len(english_df) - eng_filled

    table = pd.DataFrame(
        {
            "filled": [kor_filled, eng_filled],
            "not filled": [kor_empty, eng_empty],
        },
        index=["korean", "english"],
    )

    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    significant = p < alpha

    kor_pct = kor_filled / len(korean_df) * 100
    eng_pct = eng_filled / len(english_df) * 100
    diff = eng_pct - kor_pct

    lines.append(f"\n=== {cat} ===")
    lines.append(table.to_string())
    lines.append(f"  korean:  {kor_filled}/{len(korean_df)} ({kor_pct:.1f}%)")
    lines.append(f"  english: {eng_filled}/{len(english_df)} ({eng_pct:.1f}%)")
    lines.append(f"  diff (eng - kor): {diff:+.1f} pp")
    lines.append(f"  chi2={chi2:.4f}, p={p:.4f}, Cramer's V={cramers_v:.4f}")
    lines.append(f"  Bonferroni alpha={alpha:.4f} -> {'SIGNIFICANT' if significant else 'not significant'}")

    summary_rows.append({
        "category": cat,
        "korean_%": round(kor_pct, 1),
        "english_%": round(eng_pct, 1),
        "diff_pp": round(diff, 1),
        "chi2": round(chi2, 4),
        "p_value": round(p, 4),
        "cramers_v": round(cramers_v, 4),
        "significant": significant,
    })

# Summary table
summary_df = pd.DataFrame(summary_rows).set_index("category")
lines.append("\n\n=== SUMMARY TABLE ===")
lines.append(summary_df.to_string())

output = "\n".join(lines)
print(output)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output + "\n")

print(f"\nResults saved to {output_path}")
