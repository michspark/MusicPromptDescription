import csv
import json
import os
import time
from typing import Dict, List
from openai import OpenAI

INPUT_CSV  = r"C:\Users\MICHA\Codes\MusicPromptDescription\data\0403_eng.csv"
OUTPUT_CSV = r"C:\Users\MICHA\Codes\MusicPromptDescription\data\0403_eng_categories.csv"

CATEGORIES = [
    "genre", "mood/emotion", "instrumentation", "timbre",
    "function", "music theory", "story/narrative/lyrics",
]

DEMO_FIELDS = [
    ("age",                    218),
    ("gender",                 219),
    ("english_proficiency",    220),
    ("english_native",         221),
    ("native_language",        222),
    ("years_music_making",     223),
    ("years_formal_instruc",   224),
    ("hours_per_week_music",   225),
    ("ai_text_tools_usage",    226),
    ("ai_music_tools_used",    227),
    ("ai_music_tools_freq",    228),
]

FEW_SHOT_MESSAGES = [
    {"role": "system", "content": f"""You are a musicology assistant that classifies music description vocabulary.
Given a music generation prompt, extract and classify every meaningful term or phrase into exactly these categories:

- genre               : musical style or subgenre (e.g., country, R&B, baroque pop, grime, punk)
- mood/emotion        : emotional qualities or atmosphere (e.g., sad, uplifting, melancholic, depressed, romantic)
- instrumentation     : specific instruments or voice types (e.g., acoustic guitar, drums, female vocalist, piano)
- timbre              : sound texture or tone quality — how it sounds, not how it feels emotionally (e.g., smooth, gritty, raw, dreamy, lush)
- function            : practical role, setting, or activity the music serves (e.g., background, dance, workout, cinematic, sci-fi)
- music theory        : tempo, rhythm, structure, dynamics (e.g., 70 bpm, slow, melodic, rhythmic, anthemic)
- story/narrative/lyrics : concrete scenes, characters, events, or lyrical themes (e.g., "story about a student", "a lonely traveler", "power of love", "summer road trip")

Rules:
1. Each term goes into at most ONE category — pick the best fit.
2. Normalize to lowercase, strip punctuation artefacts.
3. For narrative prompts, extract key concepts only.
4. If a category has no matching terms, return an empty list for it.
5. Return ONLY valid JSON — no explanation, no markdown, no extra text.
6. The JSON must exactly match this structure: {json.dumps({cat: [] for cat in CATEGORIES})}"""},

    {"role": "user", "content": "The song lyrics is in Spanish and has a salsa rhythm that makes you want to have fun and dance"},
    {"role": "assistant", "content": json.dumps({
        "genre": [], "mood/emotion": ["makes you want to have fun and dance"],
        "instrumentation": [], "timbre": [], "function": [],
        "music theory": ["salsa rhythm"], "story/narrative/lyrics": ["the song lyrics is in spanish"]
    })},
    {"role": "user", "content": "Male vocalist with a raspy voice singing over melancholic piano chords and drums increasing in intensity, with a slightly dissonant chorus featuring distorted guitars."},
    {"role": "assistant", "content": json.dumps({
        "genre": [], "mood/emotion": ["melancholic"],
        "instrumentation": ["male vocalist", "piano chords", "drums", "guitars"], "timbre": ["raspy", "distorted"],
        "function": [], "music theory": ["increasing in intensity", "slightly dissonant chorus"], "story/narrative/lyrics": []
    })},
    {"role": "user", "content": "This track is Hispanic genre, with Spanish lyrics, it has a sort of romantic, love theme, with trumpets, guitar and drums."},
    {"role": "assistant", "content": json.dumps({
        "genre": ["hispanic"], "mood/emotion": ["romantic", "love theme"],
        "instrumentation": ["trumpets", "guitar", "drums"], "timbre": [], "function": [],
        "music theory": [], "story/narrative/lyrics": ["spanish lyrics"]
    })},
]


def load_qualtrics_csv(path: str) -> tuple[List[Dict], List[str]]:
    """
    Parses the Qualtrics export (0403_eng.csv).
    Returns:
      - records: list of dicts, one per (respondent, non-empty prompt) pair
      - demo_keys: ordered list of demographic field names
    Qualtrics format: row 0 = headers, row 1 = descriptions, row 2 = importIds, row 3+ = data
    """
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return [], []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if len(all_rows) < 4:
        print("Error: CSV has fewer than 4 rows")
        return [], []

    headers  = all_rows[0]
    desc_row = all_rows[1]
    data_rows = all_rows[3:]  # skip importId row

    # Prompt columns: UUID-named columns whose description mentions "Describe the music"
    prompt_col_indices = [
        i for i, d in enumerate(desc_row)
        if "Describe the music" in d
    ]

    demo_keys = [name for name, _ in DEMO_FIELDS]

    records = []
    for user_num, row in enumerate(data_rows, start=1):
        # Pad short rows
        while len(row) < len(headers):
            row.append("")

        demo_values = {name: row[idx] for name, idx in DEMO_FIELDS}

        for col_idx in prompt_col_indices:
            prompt = row[col_idx].strip()
            if not prompt:
                continue
            problem_hash = headers[col_idx].strip()
            record = {
                "user_number":    user_num,
                "problem_hash":   problem_hash,
                "prompt":         prompt,
            }
            record.update(demo_values)
            records.append(record)

    return records, demo_keys


def classify_prompt(client: OpenAI, prompt: str) -> Dict:
    messages = FEW_SHOT_MESSAGES + [
        {"role": "user", "content": f'Classify this music prompt:\n\n"{prompt}"'}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)


def process_and_save(client: OpenAI, records: List[Dict], demo_keys: List[str], output_path: str):
    fieldnames = ["user_number", "problem_hash", "prompt"] + CATEGORIES + demo_keys

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Processing {len(records)} (user, problem) pairs\n")

        for i, rec in enumerate(records):
            print(
                f"[{i+1}/{len(records)}] user={rec['user_number']} "
                f"hash={rec['problem_hash'][:12]}... {rec['prompt'][:50]}...",
                end=" ", flush=True
            )
            try:
                parsed = classify_prompt(client, rec["prompt"])

                row = {
                    "user_number":  rec["user_number"],
                    "problem_hash": rec["problem_hash"],
                    "prompt":       rec["prompt"],
                }
                for cat in CATEGORIES:
                    terms = parsed.get(cat, [])
                    if isinstance(terms, str):
                        terms = [terms]
                    row[cat] = "|".join(str(t).lower().strip() for t in terms if str(t).strip())
                for key in demo_keys:
                    row[key] = rec[key]

                writer.writerow(row)
                f.flush()
                print("v")

            except Exception as e:
                print(f"x Failed: {e}")
                row = {
                    "user_number":  rec["user_number"],
                    "problem_hash": rec["problem_hash"],
                    "prompt":       rec["prompt"],
                }
                for cat in CATEGORIES:
                    row[cat] = ""
                for key in demo_keys:
                    row[key] = rec[key]
                writer.writerow(row)

            if i < len(records) - 1:
                time.sleep(0.3)


def interactive_mode(client: OpenAI):
    print("Music Prompt Classifier — Interactive Mode")
    print("   Type 'quit' to exit\n")
    while True:
        user_input = input("Enter prompt: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        try:
            parsed = classify_prompt(client, user_input)
            print("\nResult:")
            for cat, terms in parsed.items():
                if terms:
                    print(f"  {cat:<30} {terms}")
            print()
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("OPENAI_API_KEY: ").strip()
    client = OpenAI(api_key=api_key)

    print("Select mode:")
    print("  [1] Batch — process 0403_eng.csv -> CSV")
    print("  [2] Interactive — test prompts one by one")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        records, demo_keys = load_qualtrics_csv(INPUT_CSV)
        if records:
            process_and_save(client, records, demo_keys, OUTPUT_CSV)
            print(f"\nFinished! Saved to {os.path.abspath(OUTPUT_CSV)}")
        else:
            print("No records found.")
    elif mode == "2":
        interactive_mode(client)
    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
