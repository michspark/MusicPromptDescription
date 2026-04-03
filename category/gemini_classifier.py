import csv
import json
import os
import re
import time
from typing import Dict, List
from google import genai
from google.genai import types
from google.genai.errors import APIError
import hashlib

BASE_DIR   = os.path.dirname(__file__)
INPUT_text  = r"C:\Users\MICHA\Codes\PromptDescriptionPrj\category\sample_prompts.txt"
OUTPUT_CSV = r"C:\Users\MICHA\Codes\PromptDescriptionPrj\category\prompt_category_gemini_2.csv"

CATEGORIES = [
    "genre",
    "mood/emotion",
    "instrumentation",
    "timbre",
    "function",
    "music theory",
    "story/narrative/lyrics",
]

SYSTEM_PROMPT = """
You are a musicology assistant that classifies music description vocabulary.

Given a music generation prompt, extract and classify every meaningful term or phrase into exactly these categories:

- genre               : musical style or subgenre (e.g., country, R&B, baroque pop, grime, punk)
- mood/emotion        : emotional qualities or atmosphere (e.g., sad, uplifting, melancholic, depressed, romantic)
- instrumentation     : specific instruments or voice types (e.g., acoustic guitar, drums, female vocalist, piano)
- timbre              : sound texture or tone quality — how it sounds, not how it feels emotionally (e.g., smooth, gritty, raw, dreamy, lush)
- function            : practical role, setting, or activity the music serves (e.g., background, dance, workout, cinematic, chillout)
- music theory        : tempo, rhythm, structure, dynamics (e.g., 70 bpm, slow, melodic, rhythmic, anthemic)
- story/narrative/lyrics : concrete scenes, characters, events, or lyrical themes (e.g., "losing keys", "a lonely traveler", "power of love", "summer road trip")

Rules:
1. Each term goes into at most ONE category — pick the best fit.
2. Normalize to lowercase, strip punctuation artefacts.
3. For narrative prompts (e.g. "a song about X in the style of Y"), extract key concepts rather than copy full sentences.
4. If a category has no matching terms, return an empty list for it.
5. Return ONLY valid json — no markdown, no explanation.

"""

JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {cat: {"type": "ARRAY", "items": {"type": "STRING"}} for cat in CATEGORIES},
    "required": CATEGORIES,
}

# ── Few-shot examples (SDK 형식에 맞게 구성) ───────────────────────────────────
# 시스템 프롬프트는 별도로 관리하고, 예시들만 리스트로 만듭니다.
FEW_SHOT_EXAMPLES = [
    types.Content(role="user", parts=[types.Part(text="The song lyrics is in Spanish and has a salsa rhythm that makes you want to have fun and dance")]),
    types.Content(role="model", parts=[types.Part(text=json.dumps({
        "genre": [], "mood/emotion": ["makes you want to have fun and dance"],
        "instrumentation": [], "timbre": [], "function": [],
        "music theory": ["salsa rhythm"], "story/narrative/lyrics": ["the song lyrics is in spanish"]
    }))]),
    types.Content(role="user", parts=[types.Part(text="Male vocalist with a raspy voice singing over melancholic piano chords and drums increasing in intensity, with a slightly dissonant chorus featuring distorted guitars.")]),
    types.Content(role="model", parts=[types.Part(text=json.dumps({
        "genre": [], "mood/emotion": ["melancholic"],
        "instrumentation": ["male vocalist", "piano chords", "drums", "guitars"], "timbre": ["raspy", "distorted"],
        "function": [], "music theory": ["increasing in intensity", "slightly dissonant chorus"], "story/narrative/lyrics": []
    }))]),
    types.Content(role="user", parts=[types.Part(text="This track is Hispanic genre, with Spanish lyrics, it has a sort of romantic, love theme, with trumpets, guitar and drums.")]),
    types.Content(role="model", parts=[types.Part(text=json.dumps({
        "genre": ["hispanic"], "mood/emotion": ["romantic", "love theme"],
        "instrumentation": ["trumpets", "guitar", "drums"], "timbre": [], "function": [],
        "music theory": [], "story/narrative/lyrics": ["spanish lyrics"]
    }))]),
]
def process_and_save_prompts(client: genai.Client, prompts: List[dict], output_path: str):
    fieldnames = ['id', 'prompt'] + CATEGORIES
    file_exists = os.path.exists(output_path)

    with open(output_path, 'a' if file_exists else 'w', newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type='application/json',
            response_schema=JSON_SCHEMA,
            temperature=0.1,
        )

        for i, p in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] ID: {p['id']}...", end=" ", flush=True)

            current_user_content = types.Content(
                role="user",
                parts=[types.Part(text=f"Classify this music prompt:\n\n\"{p['prompt']}\"")]
            )
            full_contents = FEW_SHOT_EXAMPLES + [current_user_content]

            success = False
            retries = 0

            while not success and retries < 3:
                try:
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=full_contents,
                        config=config
                    )

                    parsed = json.loads(response.text)
                    row = {"id": p["id"], "prompt": p["prompt"]}
                    for cat in CATEGORIES:
                        terms = parsed.get(cat, [])
                        row[cat] = "|".join([t.lower().strip() for t in terms if t.strip()])

                    writer.writerow(row)
                    f.flush()
                    print("Done.")
                    success = True

                except Exception as e:
                    if "429" in str(e):
                        print("\n[!] Rate limit hit. Sleeping for 30s...")
                        time.sleep(30)
                        retries += 1
                    else:
                        print(f"Failed. Error: {e}")
                        row = {"id": p["id"], "prompt": p["prompt"]}
                        for cat in CATEGORIES: row[cat] = ""
                        writer.writerow(row)
                        success = True # Move to next prompt

            # Mandatory wait between requests for Free Tier
            if i < len(prompts) - 1:
                time.sleep(12.5)

def load_prompts(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        print(f'Error {path} cannot find a file')

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            prompt = line.strip()
            rows.append({"id": i +1, "prompt": prompt})

    return rows


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError("Set the GEMINI_API_KEY environment variable first.")

    client  = genai.Client()


    # Load all prompts, but slice the list to ONLY keep the first 2
    all_prompts = load_prompts(INPUT_text)
    prompts = all_prompts[10:]

    process_and_save_prompts(client, prompts, OUTPUT_CSV)

    print(f"\nFinished! All data safely saved to {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    main()

