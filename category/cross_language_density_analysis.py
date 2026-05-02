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

korean_path = compute_density("../data/0428_korean_categories.csv")
english_path = compute_density("../data/0428_eng_categories.csv")
output_path = "../data/cross_language_density_results.txt"

korean_df = pd.read_csv(korean_path)
english_df = pd.read_csv(english_path)

# Bonferroni-corrected alpha
alpha = 0.05 / len(CATEGORIES)

lines = []
lines.append("=" * 65)
lines.append("CROSS-LANGUAGE DENSITY ANALYSIS: Welch's t-test + Glass's Delta")
lines.append(f"Korean respondents:  {len(korean_df)}")
lines.append(f"English respondents: {len(english_df)}")
lines.append(f"Bonferroni-corrected alpha = 0.05 / {len(CATEGORIES)} = {alpha:.4f}")
lines.append(f"Control group (denominator for Glass's delta): Korean")
lines.append("=" * 65)

summary_rows = []

for cat in CATEGORIES:
    kor_vals = korean_df[cat].dropna().values
    eng_vals = english_df[cat].dropna().values

    kor_mean = np.mean(kor_vals)
    eng_mean = np.mean(eng_vals)
    kor_std = np.std(kor_vals, ddof=1)
    eng_std = np.std(eng_vals, ddof=1)

    # Welch's t-test (unequal variances, unequal sample sizes)
    t_stat, p_val = ttest_ind(kor_vals, eng_vals, equal_var=False)

    # Glass's delta = (mean_english - mean_korean) / SD_korean
    glass_delta = (eng_mean - kor_mean) / kor_std if kor_std > 0 else float("nan")

    significant = p_val < alpha

    lines.append(f"\n--- {cat} ---")
    lines.append(f"  Korean:  mean={kor_mean:.4f}, std={kor_std:.4f}, n={len(kor_vals)}")
    lines.append(f"  English: mean={eng_mean:.4f}, std={eng_std:.4f}, n={len(eng_vals)}")
    lines.append(f"  diff (eng - kor): {eng_mean - kor_mean:+.4f}")
    lines.append(f"  Welch's t={t_stat:.4f}, p={p_val:.4f}")
    lines.append(f"  Glass's delta={glass_delta:.4f}")
    lines.append(f"  Bonferroni alpha={alpha:.4f} -> {'SIGNIFICANT' if significant else 'not significant'}")

    summary_rows.append({
        "category": cat,
        "korean_mean": round(kor_mean, 4),
        "english_mean": round(eng_mean, 4),
        "diff": round(eng_mean - kor_mean, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "glass_delta": round(glass_delta, 4),
        "significant": significant,
    })

summary_df = pd.DataFrame(summary_rows).set_index("category")
lines.append("\n\n=== SUMMARY TABLE ===")
lines.append(summary_df.to_string())

output = "\n".join(lines)
print(output)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output + "\n")

print(f"\nResults saved to {output_path}")
