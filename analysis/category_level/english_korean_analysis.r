library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(dplyr)
library(patchwork)

# Load Data
df <- read.csv("C:/Users/MICHA/Codes/MusicPromptDescription/data/long_korean_eng.csv")

# Fit Models
df$corpus <- relevel(factor(df$corpus), ref = "english")
df$category <- relevel(factor(df$category), ref = "genre")

# GLMM - Presence
m1 <- glmer(presence ~ corpus * category + (1 | text_id) + (1 | song_id),
            data = df, family = binomial,
            control = glmerControl(optimizer = "bobyqa",
                                   optCtrl = list(maxfun = 100000)))

# LMM - Density
m2 <- lmer(density ~ corpus * category + (1 | text_id) + (1 | song_id), data = df)

# Extract emmeans & build plot dataframes
levels_order <- c("Genre", "Story", "Instrumentation", "Mood", "Theory", "Timbre", "Function")


# --- m1: Presence ---
emm_m1 <- as.data.frame(emmeans(m1, ~ corpus * category, type = "response"))
colnames(emm_m1)[grepl("LCL|lower.CL", colnames(emm_m1))] <- "LCL"
colnames(emm_m1)[grepl("UCL|upper.CL", colnames(emm_m1))] <- "UCL"

df_plot_m1 <- emm_m1 %>%
  mutate(
    Value    = prob * 100,
    LCL      = LCL  * 100,
    UCL      = UCL  * 100,
    Category = factor(tools::toTitleCase(as.character(category)), levels = levels_order),
    Corpus   = factor(tools::toTitleCase(as.character(corpus)),   levels = c("English", "Korean"))
  )

# Verify AFTER mutating
print(unique(df_plot_m1$Corpus))   # Should be: English Korean
print(levels(df_plot_m1$Corpus))   # Should be: "English" "Korean"

# --- m2: Density ---
emm_m2 <- as.data.frame(emmeans(m2, ~ corpus * category))
colnames(emm_m2)[grepl("LCL|lower.CL", colnames(emm_m2))] <- "LCL"
colnames(emm_m2)[grepl("UCL|upper.CL", colnames(emm_m2))] <- "UCL"

df_plot_m2 <- emm_m2 %>%
  mutate(
    Value    = emmean * 100,
    LCL      = LCL   * 100,
    UCL      = UCL   * 100,
    Category = factor(tools::toTitleCase(as.character(category)), levels = levels_order),
    Corpus   = factor(tools::toTitleCase(as.character(corpus)),   levels = c("English", "Korean"))
  )

# Plot
fill_colors <- c("English" = "#60318C", "Korean" = "#318C57")

plot_presence <- ggplot(df_plot_m1, aes(x = Category, y = Value, fill = Corpus)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.85),
           width = 0.8, color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = LCL, ymax = UCL),
                width = 0.25, position = position_dodge(width = 0.85), alpha = 0.7) +
  geom_text(aes(y = UCL + 5, label = sprintf("%.1f%%", Value)),
            position = position_dodge(width = 0.85), size = 3.5, fontface = "bold") +
  scale_y_continuous(limits = c(0, 115), expand = expansion(mult = c(0, 0))) +
  scale_fill_manual(values = fill_colors) +
  labs(title = "(a) Category Presence Rate",
       y = "Predicted Probability (%)", x = "Taxonomy Category", fill = "Corpus:") +
  theme_classic() +
  theme(text = element_text(size = 14),
        plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        axis.text.x = element_text(angle = 40, hjust = 1, face = "bold", color = "black"),
        axis.text.y = element_text(color = "black"))

plot_density <- ggplot(df_plot_m2, aes(x = Category, y = Value, fill = Corpus)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.85),
           width = 0.8, color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = LCL, ymax = UCL),
                width = 0.25, position = position_dodge(width = 0.85), alpha = 0.7) +
  geom_text(aes(y = UCL + 3, label = sprintf("%.1f%%", Value)),
            position = position_dodge(width = 0.85), size = 3.5, fontface = "bold") +
  scale_y_continuous(limits = c(0, 55), expand = expansion(mult = c(0, 0))) +
  scale_fill_manual(values = fill_colors) +
  labs(title = "(b) Mean Category Density",
       y = "Predicted Density (%)", x = "Taxonomy Category", fill = "Corpus:") +
  theme_classic() +
  theme(text = element_text(size = 14),
        plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        axis.text.x = element_text(angle = 40, hjust = 1, face = "bold", color = "black"),
        axis.text.y = element_text(color = "black"))

final_plot <- (plot_presence | plot_density) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom",
        legend.title = element_text(face = "bold"),
        legend.text = element_text(size = 13))


ggsave("C:/Users/MICHA/Codes/MusicPromptDescription/plots/korean_english.png", plot = final_plot, width = 15, height = 7, dpi = 300)
