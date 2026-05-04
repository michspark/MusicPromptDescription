library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(dplyr)
library(patchwork)

# Load Data
df <- read.csv("C:/Users/MICHA/Codes/MusicPromptDescription/data/long_combined.csv")
df$corpus   <- relevel(factor(df$corpus),   ref = "description")
df$category <- relevel(factor(df$category), ref = "genre")

m1 <- glmer(
  presence ~ corpus * category + (1 | text_id) + (1 | song_id),
  data = df, family = binomial,
  control = glmerControl(optimizer = "bobyqa",
                         optCtrl = list(maxfun = 100000))
)

m2 <- lmer(
  density ~ corpus * category + (1 | text_id) + (1 | song_id),
  data = df
)

standardize_emm <- function(emm_df) {
  # Identify estimate column
  est_col <- if ("prob" %in% names(emm_df)) "prob" else "emmean"

  # Identify CI columns (try multiple naming conventions)
  if (all(c("asymp.LCL", "asymp.UCL") %in% names(emm_df))) {
    lcl_col <- "asymp.LCL"; ucl_col <- "asymp.UCL"
  } else if (all(c("lower.CL", "upper.CL") %in% names(emm_df))) {
    lcl_col <- "lower.CL"; ucl_col <- "upper.CL"
  } else if (all(c("LCL", "UCL") %in% names(emm_df))) {
    lcl_col <- "LCL"; ucl_col <- "UCL"
  } else {
    stop("Could not find CI columns. Available: ",
         paste(names(emm_df), collapse = ", "))
  }

  data.frame(
    Category = tools::toTitleCase(as.character(emm_df$category)),
    Corpus   = tools::toTitleCase(as.character(emm_df$corpus)),
    Value    = emm_df[[est_col]] * 100,
    LCL      = emm_df[[lcl_col]] * 100,
    UCL      = emm_df[[ucl_col]] * 100
  )
}

emm_m1 <- emmeans(m1, ~ corpus * category, type = "response")
emm_m2 <- emmeans(m2, ~ corpus * category)

df_plot_m1 <- standardize_emm(as.data.frame(emm_m1))
df_plot_m2 <- standardize_emm(as.data.frame(emm_m2))


cat_order <- df_plot_m1 %>%
  filter(Corpus == "Prompt") %>%
  arrange(desc(Value)) %>%
  pull(Category)

df_plot_m1$Category <- factor(df_plot_m1$Category, levels = cat_order)
df_plot_m2$Category <- factor(df_plot_m2$Category, levels = cat_order)

df_plot_m1$Corpus <- factor(df_plot_m1$Corpus, levels = c("Prompt", "Description"))
df_plot_m2$Corpus <- factor(df_plot_m2$Corpus, levels = c("Prompt", "Description"))

theme_paper <- theme_classic(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0.5),
    axis.text.x      = element_text(angle = 35, hjust = 1, face = "bold", color = "black"),
    axis.text.y      = element_text(color = "black"),
    legend.title     = element_text(face = "bold"),
    legend.text      = element_text(size = 12),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3)
  )

corpus_colors <- c("Prompt" = "#E69F00", "Description" = "#56B4E9")

# ---------- Plot (a): Presence ----------
plot_presence <- ggplot(df_plot_m1,
                        aes(x = Category, y = Value, fill = Corpus)) +
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.85),
           width = 0.78,
           color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = LCL, ymax = UCL),
                width = 0.22,
                position = position_dodge(width = 0.85),
                alpha = 0.75) +
  geom_text(aes(y = UCL + 4, label = sprintf("%.0f%%", Value)),
            position = position_dodge(width = 0.85),
            size = 3.3, fontface = "bold") +
  scale_y_continuous(limits = c(0, 110),
                     breaks = seq(0, 100, 25),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_fill_manual(values = corpus_colors) +
  labs(title = "(a) Category Presence Rate",
       y = "Predicted Probability (%)",
       x = NULL,
       fill = "Corpus:") +
  theme_paper

# ---------- Plot (b): Density ----------
plot_density <- ggplot(df_plot_m2,
                       aes(x = Category, y = Value, fill = Corpus)) +
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.85),
           width = 0.78,
           color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = LCL, ymax = UCL),
                width = 0.22,
                position = position_dodge(width = 0.85),
                alpha = 0.75) +
  geom_text(aes(y = UCL + 1.5, label = sprintf("%.1f%%", Value)),
            position = position_dodge(width = 0.85),
            size = 3.3, fontface = "bold") +
  scale_y_continuous(limits = c(0, 50),
                     breaks = seq(0, 50, 10),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_fill_manual(values = corpus_colors) +
  labs(title = "(b) Mean Category Density",
       y = "Predicted Density (%)",
       x = NULL,
       fill = "Corpus:") +
  theme_paper

# ---------- Combine side by side ----------
final_plot <- (plot_presence | plot_density) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

# ---------- Display + save ----------
print(final_plot)

ggsave("C:/Users/MICHA/Codes/MusicPromptDescription/plots/prompt_desc.png", plot = final_plot,
       width = 15, height = 7, dpi = 300)
