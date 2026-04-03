import csv
import json
import os
import time
from typing import Dict, List
from openai import OpenAI

INPUT_CSV  = r"C:\Users\MICHA\Codes\0401_sample_data.csv"
OUTPUT_CSV = r"C:\Users\MICHA\Codes\PromptDescriptionPrj\category\prompt_category_survey.csv"

CATEGORIES = [
    "genre", "mood/emotion", "instrumentation", "timbre",
    "function", "music theory", "story/narrative/lyrics",
]

# 고정 few-shot — 매 요청마다 재사용, 누적 안 됨
FEW_SHOT_MESSAGES = [
    {"role": "system", "content": f"""You are a musicology assistant that deeply understands the overall intent and context of music descriptions.

First, read the entire prompt holistically to understand what kind of music is being described.
Then, identify and classify terms or phrases that best capture the musical meaning —
considering how words relate to each other in context, not just in isolation.

For example:
- "punchy" modifying "vocals" → timbre, not mood
- "funky" modifying "bassline" → music theory (rhythmic feel), not genre
- "dreamy" in "dreamy synth-pop" → could be timbre or mood depending on context

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


def load_prompts_from_csv(path: str) -> List[Dict]:
    """survey_prompt_data.csv에서 url UUID와 prompt만 추출"""
    rows = []
    if not os.path.exists(path):
        print(f"Error: {path} cannot find the file")
        return rows
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            prompt = row.get("prompt", "").strip()
            if not prompt:
                continue
            # URL 마지막 "/" 뒤의 UUID 추출
            song_id = url.rstrip("/").split("/")[-1]
            rows.append({"id": song_id, "prompt": prompt})
    return rows


def classify_prompt(client: OpenAI, prompt: str) -> Dict:
    """Few-shot messages + 새 prompt만 전송 — 이전 결과 누적 없음"""
    messages = FEW_SHOT_MESSAGES + [
        {"role": "user", "content": f'Classify this music prompt:\n\n"{prompt}"'}
    ]
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)


def process_and_save_prompts(client: OpenAI, prompts: List[Dict], output_path: str):
    fieldnames = ['id', 'prompt'] + CATEGORIES

    with open(output_path, 'w', newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Processing {len(prompts)} prompts\n")

        for i, p in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] {p['id']}: {p['prompt'][:60]}...", end=" ", flush=True)
            try:
                parsed = classify_prompt(client, p["prompt"])

                row = {"id": p["id"], "prompt": p["prompt"]}
                for cat in CATEGORIES:
                    terms = parsed.get(cat, [])
                    if isinstance(terms, str):
                        terms = [terms]
                    row[cat] = "|".join([str(t).lower().strip() for t in terms if str(t).strip()])

                writer.writerow(row)
                f.flush()
                print("v")

            except Exception as e:
                print(f"x Failed: {e}")
                row = {"id": p["id"], "prompt": p["prompt"]}
                for cat in CATEGORIES:
                    row[cat] = ""
                writer.writerow(row)

            if i < len(prompts) - 1:
                time.sleep(0.5)


def interactive_mode(client: OpenAI):
    """터미널에서 한 개씩 테스트하는 인터랙티브 모드"""
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
    print("  [1] Batch — process survey_prompt_data.csv -> CSV")
    print("  [2] Interactive — test prompts one by one")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        prompts = load_prompts_from_csv(INPUT_CSV)
        if prompts:
            process_and_save_prompts(client, prompts, OUTPUT_CSV)
            print(f"\nFinished! Saved to {os.path.abspath(OUTPUT_CSV)}")
    elif mode == "2":
        interactive_mode(client)
    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
