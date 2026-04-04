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

survey_path = "../data/0403_eng_categories.csv"
udio_path = "../data/udio_categorized_gpt.csv"

survey_df = pd.read_csv(survey_path)
udio_df = pd.read_csv(udio_path)


def is_filled(series: pd.Series) -> pd.Series:
    """Return boolean Series: True if cell is non-empty, False otherwise."""
    return series.notna() & (series.astype(str).str.strip() != "")


output_path = "../data/percent_analysis_results.txt"
contingency_tables = {}
lines = []

for cat in CATEGORIES:
    survey_filled = is_filled(survey_df[cat]).sum()
    survey_empty = len(survey_df) - survey_filled

    udio_filled = is_filled(udio_df[cat]).sum()
    udio_empty = len(udio_df) - udio_filled

    table = pd.DataFrame(
        {
            "filled": [survey_filled, udio_filled],
            "not filled": [survey_empty, udio_empty],
        },
        index=["survey", "udio"],
    )
    contingency_tables[cat] = table

    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

    alpha = 0.05 / len(CATEGORIES)  # Bonferroni correction
    significant = p < alpha

    lines.append(f"\n=== {cat} ===")
    lines.append(table.to_string())
    lines.append(f"  survey: {survey_filled}/{len(survey_df)} ({survey_filled/len(survey_df)*100:.1f}%)")
    lines.append(f"  udio:   {udio_filled}/{len(udio_df)} ({udio_filled/len(udio_df)*100:.1f}%)")
    lines.append(f"  chi2={chi2:.4f}, p={p:.4f}, Cramer's V={cramers_v:.4f}")
    lines.append(f"  Bonferroni alpha={alpha:.4f} -> {'SIGNIFICANT' if significant else 'not significant'}")

output = "\n".join(lines)
print(output)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output + "\n")
