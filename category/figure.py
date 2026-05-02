"""
figure.py — Grouped bar charts for two comparisons:
  Figure 1: Description (Survey/English) vs. Prompt (Udio)  — density & presence
  Figure 2: Korean prompts vs. English prompts              — density & presence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind, chi2_contingency

from compute_density import compute_density

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORIES = [
    "genre",
    "mood/emotion",
    "instrumentation",
    "timbre",
    "function",
    "music theory",
    "story/narrative/lyrics",
]

CAT_LABELS = [
    "Genre",
    "Mood/\nEmotion",
    "Instrumentation",
    "Timbre",
    "Function",
    "Music\nTheory",
    "Story/\nNarrative",
]

ALPHA_BONF = 0.05 / len(CATEGORIES)


def is_filled(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


# ── Load data ──────────────────────────────────────────────────────────────────
eng_cat_path = "../data/0414_eng_categories.csv"
kor_cat_path = "../data/0414_korean_categories.csv"

eng_den = pd.read_csv(compute_density(eng_cat_path))
kor_den = pd.read_csv(compute_density(kor_cat_path))
eng_cat = pd.read_csv(eng_cat_path)
kor_cat = pd.read_csv(kor_cat_path)

udio_den = pd.read_csv("../data/udio_categorized_gpt_density.csv")
udio_cat = pd.read_csv("../data/udio_categorized_gpt.csv")


# ── Stat helpers ───────────────────────────────────────────────────────────────
def density_stats(df_a, df_b):
    means_a = [df_a[c].dropna().mean() for c in CATEGORIES]
    means_b = [df_b[c].dropna().mean() for c in CATEGORIES]
    sems_a  = [df_a[c].dropna().sem()  for c in CATEGORIES]
    sems_b  = [df_b[c].dropna().sem()  for c in CATEGORIES]
    sig = []
    for c in CATEGORIES:
        _, p = ttest_ind(df_a[c].dropna(), df_b[c].dropna(), equal_var=False)
        sig.append(p < ALPHA_BONF)
    return means_a, means_b, sems_a, sems_b, sig


def presence_stats(cat_a, cat_b):
    pct_a = [is_filled(cat_a[c]).mean() * 100 for c in CATEGORIES]
    pct_b = [is_filled(cat_b[c]).mean() * 100 for c in CATEGORIES]
    sig = []
    for c in CATEGORIES:
        fa = is_filled(cat_a[c]).sum()
        fb = is_filled(cat_b[c]).sum()
        table = [[fa, len(cat_a) - fa], [fb, len(cat_b) - fb]]
        _, p, _, _ = chi2_contingency(table)
        sig.append(p < ALPHA_BONF)
    return pct_a, pct_b, sig


# ── Drawing helper ─────────────────────────────────────────────────────────────
def draw_bars(ax, vals_a, vals_b, errs_a, errs_b,
              sig_flags, color_a, color_b, label_a, label_b,
              ylabel, title, footnote_test):
    x = np.arange(len(CATEGORIES))
    w = 0.35

    ax.bar(x - w / 2, vals_a, w, yerr=errs_a, color=color_a, capsize=4,
           error_kw={"elinewidth": 1.2}, label=label_a)
    ax.bar(x + w / 2, vals_b, w, yerr=errs_b, color=color_b, capsize=4,
           error_kw={"elinewidth": 1.2}, label=label_b)

    zeros = [0] * len(CATEGORIES)
    top_vals = [
        max(a + (ea or 0), b + (eb or 0))
        for a, ea, b, eb in zip(vals_a, errs_a or zeros,
                                vals_b, errs_b or zeros)
    ]
    y_max = max(max(vals_a), max(vals_b))
    for i, (sig, top) in enumerate(zip(sig_flags, top_vals)):
        if sig:
            ax.text(x[i], top + y_max * 0.03, "*",
                    ha="center", va="bottom", fontsize=14,
                    fontweight="bold", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, y_max * 1.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(1.0, -0.18,
            f"* p < Bonferroni α ({ALPHA_BONF:.4f}), {footnote_test}",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="gray")


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        path = f"../data/{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Description (Survey/English) vs. Prompt (Udio)
# ══════════════════════════════════════════════════════════════════════════════
C_DESC   = "#4C72B0"
C_PROMPT = "#DD8452"

d_means_eng, d_means_udio, d_sems_eng, d_sems_udio, d_sig = density_stats(eng_den, udio_den)
p_pct_eng, p_pct_udio, p_sig = presence_stats(eng_cat, udio_cat)

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 5))

draw_bars(ax1a, p_pct_eng, p_pct_udio, None, None, p_sig,
          C_DESC, C_PROMPT, "Description (Survey)", "Prompt (Udio)",
          "Presence (%)", "(a) Category Presence Rate", "χ² test")

draw_bars(ax1b, d_means_eng, d_means_udio, d_sems_eng, d_sems_udio, d_sig,
          C_DESC, C_PROMPT, "Description (Survey)", "Prompt (Udio)",
          "Mean Category Density", "(b) Category Density", "Welch's t-test")

fig1.legend(
    handles=[mpatches.Patch(color=C_DESC,   label="Description (Survey)"),
             mpatches.Patch(color=C_PROMPT, label="Prompt (Udio)")],
    loc="upper center", ncol=2, fontsize=10, frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
fig1.suptitle("Category Distribution: Description vs. Prompt",
              fontsize=13, fontweight="bold", y=1.05)
fig1.tight_layout()
save_figure(fig1, "figure_prompt_vs_description")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Korean prompts vs. English prompts
# ══════════════════════════════════════════════════════════════════════════════
C_KOR = "#2ca02c"
C_ENG = "#9467bd"

d_means_kor, d_means_eng2, d_sems_kor, d_sems_eng2, d_sig2 = density_stats(kor_den, eng_den)
p_pct_kor, p_pct_eng2, p_sig2 = presence_stats(kor_cat, eng_cat)

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 5))

draw_bars(ax2a, p_pct_kor, p_pct_eng2, None, None, p_sig2,
          C_KOR, C_ENG, "Korean Prompts", "English Prompts",
          "Presence (%)", "(a) Category Presence Rate", "χ² test")

draw_bars(ax2b, d_means_kor, d_means_eng2, d_sems_kor, d_sems_eng2, d_sig2,
          C_KOR, C_ENG, "Korean Prompts", "English Prompts",
          "Mean Category Density", "(b) Category Density", "Welch's t-test")

fig2.legend(
    handles=[mpatches.Patch(color=C_KOR, label="Korean Prompts"),
             mpatches.Patch(color=C_ENG, label="English Prompts")],
    loc="upper center", ncol=2, fontsize=10, frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
fig2.suptitle("Category Distribution: Korean vs. English Prompts",
              fontsize=13, fontweight="bold", y=1.05)
fig2.tight_layout()
save_figure(fig2, "figure_korean_vs_english")

plt.show()
