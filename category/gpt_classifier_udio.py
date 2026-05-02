import csv
import json
import os
import time
from typing import Dict, List, Tuple
from openai import OpenAI

INPUT_CSV  = r"C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_data.csv"
OUTPUT_CSV = r"C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_categories.csv"

CATEGORIES = [
    "genre", "mood/emotion", "instrumentation", "timbre",
    "function", "music theory", "story/narrative/lyrics",
]

FEW_SHOT_MESSAGES = [
    {"role": "system", "content": f"""You are a musicology assistant that classifies music generation prompts.
Given a music generation prompt, extract and classify every meaningful term or phrase into exactly these categories:

- genre               : musical style or subgenre (e.g., country, R&B, baroque pop, grime, punk)
- mood/emotion        : emotional qualities or atmosphere (e.g., sad, uplifting, melancholic, depressed, romantic)
- instrumentation     : specific instruments or voice types (e.g., acoustic guitar, drums, female vocalist, piano)
- timbre              : sound texture or tone quality — how it sounds, not how it feels emotionally (e.g., smooth, gritty, raw, dreamy, lush)
- function            : practical role, setting, or activity the music serves (e.g., background, dance, workout, cinematic, sci-fi)
- music theory        : tempo, rhythm, structure, dynamics (e.g., 70 bpm, slow, melodic, rhythmic, anthemic)
- story/narrative/lyrics : concrete scenes, characters, events, or lyrical themes (e.g., "story about a student", "losing my keys", "power of love")

Rules:
1. Each term goes into at most ONE category — pick the best fit.
2. Normalize to lowercase, strip punctuation artefacts.
3. For narrative prompts, extract key concepts only.
4. If a category has no matching terms, return an empty list for it.
5. Return ONLY valid JSON — no explanation, no markdown, no extra text.
6. The JSON must exactly match this structure: {json.dumps({cat: [] for cat in CATEGORIES})}"""},

    {"role": "user", "content": 'a dark and sad country instrumental, acoustic guitar, slow, light drums, downtempo melancholic, (70 bpm), depressed,'},
    {"role": "assistant", "content": json.dumps({
        "genre": ["country instrumental"], "mood/emotion": ["dark", "sad", "melancholic", "depressed"],
        "instrumentation": ["acoustic guitar", "light drums"], "timbre": [],
        "function": [], "music theory": ["slow", "downtempo", "70 bpm"], "story/narrative/lyrics": []
    })},
    {"role": "user", "content": 'a song about losing my keys in the style of amy winehouse'},
    {"role": "assistant", "content": json.dumps({
        "genre": ["amy winehouse style"], "mood/emotion": [],
        "instrumentation": [], "timbre": [], "function": [],
        "music theory": [], "story/narrative/lyrics": ["losing my keys"]
    })},
    {"role": "user", "content": 'baroque pop, dreamy pop, boy band, romantic, piano jazz, electric cello, deep house, chillout,'},
    {"role": "assistant", "content": json.dumps({
        "genre": ["baroque pop", "dreamy pop", "boy band", "piano jazz", "deep house"],
        "mood/emotion": ["romantic", "chillout"],
        "instrumentation": ["electric cello"], "timbre": [], "function": [],
        "music theory": [], "story/narrative/lyrics": []
    })},
]


def load_udio_csv(path: str) -> List[Dict]:
    """
    Reads udio_sample_data.csv (columns: url, title, duration, prompt, tags, lyrics, status).
    Returns a list of dicts with url, song_id, title, prompt for rows with non-empty prompts.
    """
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return []

    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "").strip()
            if not prompt:
                continue
            url = row.get("url", "").strip()
            song_id = url.split("/songs/")[-1] if "/songs/" in url else url
            records.append({
                "url":     url,
                "song_id": song_id,
                "title":   row.get("title", "").strip(),
                "prompt":  prompt,
            })

    return records


def classify_prompt(client: OpenAI, prompt: str) -> Dict:
    messages = FEW_SHOT_MESSAGES + [
        {"role": "user", "content": f'Classify this music generation prompt:\n\n"{prompt}"'}
    ]
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)


def process_and_save(client: OpenAI, records: List[Dict], output_path: str):
    fieldnames = ["url", "song_id", "title", "prompt"] + CATEGORIES

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Processing {len(records)} prompts\n")

        for i, rec in enumerate(records):
            print(
                f"[{i+1}/{len(records)}] {rec['song_id'][:12]}... {rec['prompt'][:60]}...",
                end=" ", flush=True
            )
            try:
                parsed = classify_prompt(client, rec["prompt"])

                row = {
                    "url":     rec["url"],
                    "song_id": rec["song_id"],
                    "title":   rec["title"],
                    "prompt":  rec["prompt"],
                }
                for cat in CATEGORIES:
                    terms = parsed.get(cat, [])
                    if isinstance(terms, str):
                        terms = [terms]
                    row[cat] = "|".join(str(t).lower().strip() for t in terms if str(t).strip())

                writer.writerow(row)
                f.flush()
                print("v")

            except Exception as e:
                print(f"x Failed: {e}")
                row = {
                    "url":     rec["url"],
                    "song_id": rec["song_id"],
                    "title":   rec["title"],
                    "prompt":  rec["prompt"],
                }
                for cat in CATEGORIES:
                    row[cat] = ""
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
    print("  [1] Batch — process udio_sample_data.csv -> categories CSV")
    print("  [2] Interactive — test prompts one by one")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        records = load_udio_csv(INPUT_CSV)
        if records:
            process_and_save(client, records, OUTPUT_CSV)
            print(f"\nFinished! Saved to {os.path.abspath(OUTPUT_CSV)}")
            print(f"\nTo compute category densities, run:")
            print(f"  python compute_density.py {OUTPUT_CSV}")
        else:
            print("No records found.")
    elif mode == "2":
        interactive_mode(client)
    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
