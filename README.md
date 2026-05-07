# From Prompting to Describing: A Cross-Cultural Study of Language for AI-Generated Music

The paper has been submitted for review to the International Society of Music Information Retrieval


## Abstract
Text-to-music (TTM) generation systems allow users to create music through natural language prompts, yet it is unclear whether the descriptive language used to prompt aligns with descriptive language used to summarize or describe heard music. We pair 200 real-world Udio prompts with their generated audio and free-form descriptions collected from English (n=62) and Korean-speaking (n=33) listeners, and contribute a human-derived taxonomy of musical prompting vocabulary grounded in real user data. Using this framework, alongside word- and vector-level analysis, we find a consistent structural asymmetry: prompts are dominated by Genre and Story/Narrative language. Genre terms propagate most reliably from prompt to perception, while narrative-heavy prompts are the strongest predictor of semantic misalignment. A preliminary cross-cultural comparison further suggests that description profiles vary across listener populations along narrative, functional, and affective dimensions, raising questions about whether current TTM systems, trained on aggregated English-centric corpora, can accommodate the full diversity of how people naturally express musical ideas.

participant numbers are updated


## Repository Structure
```

MusicPromptDescription/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── english.csv
│   ├── english_categories.csv
│   ├── korean.csv
│   ├── korean_categories.csv
│   └── combined.csv
├── src/
│   ├── classification/
│   │   └── gpt_classifier.py
│   ├── analysis/
│   │   ├── word_level/
│   │   ├── category_level/
│   │   └── vector/
│   └── utils/
│
├── results/
│   ├── figures/
│   │   ├── prompt_desc.png
│   │   └── korean_english.png
│   └── tables/
│       ├── similarity_scores.csv
│       ├── concreteness_scores.csv
│       └── chi2_results.csv

```