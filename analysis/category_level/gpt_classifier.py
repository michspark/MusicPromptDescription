"""
python gpt_classifier.py --mode prompt      [--input CSV] [--output CSV]
python gpt_classifier.py --mode description [--input CSV] [--output CSV]
python gpt_classifier.py --mode korean      [--input CSV] [--output CSV]
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Tuple
from openai import OpenAI

CATEGORIES = [
    "genre", "mood/emotion", "instrumentation", "timbre",
    "function", "music theory", "story/narrative/lyrics",
]

DEFAULTS = {
    "prompt": {
        "input":  r"C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_data.csv",
        "output": r"C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_categories.csv",
    },
    "description": {
        "input":  r"C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_data.csv",
        "output": r"C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_categories_desc.csv",
    },
    "korean": {
        "input":  r"C:\Users\MICHA\Codes\MusicPromptDescription\data\0428_korean.csv",
        "output": r"C:\Users\MICHA\Codes\MusicPromptDescription\data\0428_korean_categories.csv",
    },
}

DEMO_FIELDS_ENG = [
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

DEMO_FIELDS_KR = [
    ("age",                    218),
    ("gender",                 219),
    ("years_music_making",     220),
    ("years_formal_instruc",   221),
    ("hours_per_week_music",   222),
    ("ai_text_tools_usage",    223),
    ("ai_music_tools_used",    224),
    ("ai_music_tools_freq",    225),
]

FEW_SHOT_PROMPT = [
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

FEW_SHOT_DESC = [
    {"role": "system", "content": f"""You are a musicology assistant that classifies music description vocabulary.
Given a music description, extract and classify every meaningful term or phrase into exactly these categories:

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

FEW_SHOT_KOREAN = [
    {"role": "system", "content": f"""당신은 음악 설명 어휘를 분류하는 보조 AI입니다.
주어진 음악 생성 프롬프트에서 의미 있는 모든 단어나 구절을 추출하여 다음 카테고리로 정확히 분류하세요:
- genre               : 음악 스타일 또는 하위 장르 (예: 컨트리, R&B, 바로크 팝, 그라임, 펑크)
- mood/emotion        : 감정적 특성 또는 분위기 (예: 슬픈, 고양되는, 우울한, 침울한, 로맨틱한)
- instrumentation     : 특정 악기 또는 보컬 유형 (예: 어쿠스틱 기타, 드럼, 여성 보컬, 피아노)
- timbre              : 음향 질감 또는 음색 — 감정이 아닌 소리 자체의 특성 (예: 부드러운, 거친, 날것의, 몽환적인, 풍성한)
- function            : 음악의 실용적 역할, 배경 또는 활동 (예: 배경음악, 댄스, 운동, 영화적, SF)
- music theory        : 템포, 리듬, 구조, 다이나믹스 (예: 70 bpm, 느린, 선율적인, 리드미컬한, 웅장한)
- story/narrative/lyrics : 구체적인 장면, 인물, 사건 또는 가사 주제 (예: "학생에 관한 이야기", "외로운 여행자", "사랑의 힘", "여름 로드트립")
규칙:
1. 각 용어는 최대 하나의 카테고리에만 속합니다 — 가장 적합한 카테고리를 선택하세요.
2. 소문자로 정규화하고 불필요한 구두점을 제거하세요.
3. 서사적 프롬프트의 경우 핵심 개념만 추출하세요.
4. 해당 카테고리에 맞는 용어가 없으면 빈 리스트를 반환하세요.
5. 유효한 JSON만 반환하세요 — 설명, 마크다운, 추가 텍스트 없이.
6. JSON은 다음 구조와 정확히 일치해야 합니다: {json.dumps({cat: [] for cat in CATEGORIES})}"""},
    {"role": "user", "content": "노래 가사는 스페인어이며 신나게 즐기고 춤추고 싶게 만드는 살사 리듬이 있습니다"},
    {"role": "assistant", "content": json.dumps({
        "genre": [], "mood/emotion": ["신나고 춤추고 싶게 만드는"],
        "instrumentation": [], "timbre": [], "function": [],
        "music theory": ["살사 리듬"], "story/narrative/lyrics": ["스페인어 가사"]
    }, ensure_ascii=False)},
    {"role": "user", "content": "거친 목소리의 남성 보컬이 우울한 피아노 코드와 점점 강렬해지는 드럼 위에서 노래하며, 왜곡된 기타가 등장하는 약간 불협화음의 코러스가 포함됩니다."},
    {"role": "assistant", "content": json.dumps({
        "genre": [], "mood/emotion": ["우울한"],
        "instrumentation": ["남성 보컬", "피아노 코드", "드럼", "기타"], "timbre": ["거친", "왜곡된"],
        "function": [], "music theory": ["점점 강렬해지는", "약간 불협화음의 코러스"], "story/narrative/lyrics": []
    }, ensure_ascii=False)},
    {"role": "user", "content": "이 트랙은 히스패닉 장르로, 스페인어 가사에 로맨틱하고 사랑을 주제로 한 분위기이며, 트럼펫, 기타, 드럼이 사용됩니다."},
    {"role": "assistant", "content": json.dumps({
        "genre": ["히스패닉"], "mood/emotion": ["로맨틱한", "사랑을 주제로 한"],
        "instrumentation": ["트럼펫", "기타", "드럼"], "timbre": [], "function": [],
        "music theory": [], "story/narrative/lyrics": ["스페인어 가사"]
    }, ensure_ascii=False)},
]

FEW_SHOT = {
    "prompt":      FEW_SHOT_PROMPT,
    "description": FEW_SHOT_DESC,
    "korean":      FEW_SHOT_KOREAN,
}


def load_udio_csv(path: str) -> Tuple[List[Dict], List[str]]:
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return [], []
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
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
    return records, []

def load_qualtrics_csv(path: str, demo_fields: list, prompt_filter) -> Tuple[List[Dict], List[str]]:
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return [], []

    with open(path, newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.reader(f))

    if len(all_rows) < 4:
        print("Error: CSV has fewer than 4 rows")
        return [], []

    headers  = all_rows[0]
    desc_row = all_rows[1]
    data_rows = all_rows[3:]

    prompt_col_indices = [i for i, d in enumerate(desc_row) if prompt_filter(d)]
    demo_keys = [name for name, _ in demo_fields]

    records = []
    for user_num, row in enumerate(data_rows, start=1):
        while len(row) < len(headers):
            row.append("")
        demo_values = {name: row[idx] for name, idx in demo_fields}
        for col_idx in prompt_col_indices:
            prompt = row[col_idx].strip()
            if not prompt:
                continue
            record = {
                "user_number":  user_num,
                "problem_hash": headers[col_idx].strip(),
                "prompt":       prompt,
            }
            record.update(demo_values)
            records.append(record)

    return records, demo_keys

def classify(client: OpenAI, prompt: str, few_shot: list) -> Dict:
    messages = few_shot + [
        {"role": "user", "content": f'Classify this music prompt:\n\n"{prompt}"'}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)



def process_and_save(client: OpenAI, records: List[Dict], demo_keys: List[str],
                     output_path: str, few_shot: list, is_udio: bool):
    if is_udio:
        fieldnames = ["url", "song_id", "title", "prompt"] + CATEGORIES
    else:
        fieldnames = ["user_number", "problem_hash", "prompt"] + CATEGORIES + demo_keys

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Processing {len(records)} records\n")

        for i, rec in enumerate(records):
            if is_udio:
                label = f"{rec['song_id'][:12]}... {rec['prompt'][:60]}"
            else:
                label = f"user={rec['user_number']} hash={rec['problem_hash'][:12]}... {rec['prompt'][:50]}"
            print(f"[{i+1}/{len(records)}] {label}", end=" ", flush=True)

            try:
                parsed = classify(client, rec["prompt"], few_shot)
                row = {k: rec[k] for k in (["url", "song_id", "title"] if is_udio else ["user_number", "problem_hash"])}
                row["prompt"] = rec["prompt"]
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
                row = {k: rec[k] for k in (["url", "song_id", "title"] if is_udio else ["user_number", "problem_hash"])}
                row["prompt"] = rec["prompt"]
                for cat in CATEGORIES:
                    row[cat] = ""
                for key in demo_keys:
                    row[key] = rec.get(key, "")
                writer.writerow(row)

            if i < len(records) - 1:
                time.sleep(0.3)

def interactive_mode(client: OpenAI, few_shot: list):
    print("Music Classifier — Interactive Mode  (type 'quit' to exit)\n")
    while True:
        user_input = input("Enter prompt: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        try:
            parsed = classify(client, user_input, few_shot)
            print("\nResult:")
            for cat, terms in parsed.items():
                if terms:
                    print(f"  {cat:<30} {terms}")
            print()
        except Exception as e:
            print(f"Error: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Music prompt/description GPT classifier")
    parser.add_argument("--mode", choices=["prompt", "description", "korean"], required=True,
                        help="prompt: Udio prompts (EN)  |  description: Qualtrics descriptions (EN)  |  korean: Qualtrics descriptions (KR)")
    parser.add_argument("--input",       default=None, help="Input CSV path (overrides default)")
    parser.add_argument("--output",      default=None, help="Output CSV path (overrides default)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode instead of batch")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or input("OPENAI_API_KEY: ").strip()
    client = OpenAI(api_key=api_key)

    few_shot  = FEW_SHOT[args.mode]
    is_udio   = args.mode == "prompt"
    input_csv = args.input  or DEFAULTS[args.mode]["input"]
    output_csv= args.output or DEFAULTS[args.mode]["output"]

    if args.interactive:
        interactive_mode(client, few_shot)
        return

    if is_udio:
        records, demo_keys = load_udio_csv(input_csv)
    elif args.mode == "description":
        records, demo_keys = load_qualtrics_csv(
            input_csv, DEMO_FIELDS_ENG,
            prompt_filter=lambda d: "Describe the music" in d,
        )
    else:  # korean
        records, demo_keys = load_qualtrics_csv(
            input_csv, DEMO_FIELDS_KR,
            prompt_filter=lambda d: "해당 음악을" in d and not d.strip().startswith("연습"),
        )

    if not records:
        print("No records found.")
        return

    process_and_save(client, records, demo_keys, output_csv, few_shot, is_udio)
    print(f"\nFinished! Saved to {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    main()
