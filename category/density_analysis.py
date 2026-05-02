import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from compute_density import compute_density

CATEGORIES = [
    "genre",
    "mood/emotion",
    "instrumentation",
    "timbre",
    "function",
    "music theory",
    "story/narrative/lyrics",
]

survey_path = compute_density("../data/0428_eng_categories.csv")
udio_path = "../data/udio_categorized_gpt_density.csv"
output_path = "../data/density_test_results.txt"

survey_df = pd.read_csv(survey_path)
udio_df = pd.read_csv(udio_path)

# Bonferroni-corrected alpha
alpha = 0.05 / len(CATEGORIES)

lines = []
lines.append("=" * 65)
lines.append("DENSITY ANALYSIS: Welch's t-test + Glass's Delta per Category")
lines.append(f"Bonferroni-corrected alpha = 0.05 / {len(CATEGORIES)} = {alpha:.4f}")
lines.append(f"Control group (denominator for Glass's delta): survey")
lines.append("=" * 65)

for cat in CATEGORIES:
    survey_vals = survey_df[cat].dropna().values
    udio_vals = udio_df[cat].dropna().values

    survey_mean = np.mean(survey_vals)
    udio_mean = np.mean(udio_vals)
    survey_std = np.std(survey_vals, ddof=1)

    # Welch's t-test (equal_var=False)
    t_stat, p_val = ttest_ind(survey_vals, udio_vals, equal_var=False)

    # Glass's delta = (mean_udio - mean_survey) / SD_survey
    glass_delta = (udio_mean - survey_mean) / survey_std if survey_std > 0 else float("nan")

    significant = p_val < alpha

    lines.append(f"\n--- {cat} ---")
    lines.append(f"  Survey : mean={survey_mean:.4f}, std={survey_std:.4f}, n={len(survey_vals)}")
    lines.append(f"  Udio   : mean={udio_mean:.4f}, std={np.std(udio_vals, ddof=1):.4f}, n={len(udio_vals)}")
    lines.append(f"  Welch's t={t_stat:.4f}, p={p_val:.4f}")
    lines.append(f"  Glass's delta={glass_delta:.4f}")
    lines.append(f"  Bonferroni alpha={alpha:.4f} -> {'SIGNIFICANT' if significant else 'not significant'}")

output = "\n".join(lines)
print(output)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output + "\n")
