import csv
import json
import os
import time
from typing import Dict, List
from openai import OpenAI

INPUT_CSV  = r"C:\Users\MICHA\Codes\MusicPromptDescription\data\0428_korean.csv"
OUTPUT_CSV = r"C:\Users\MICHA\Codes\MusicPromptDescription\data\0428_korean_categories.csv"

CATEGORIES = [
    "genre", "mood/emotion", "instrumentation", "timbre",
    "function", "music theory", "story/narrative/lyrics",
]

DEMO_FIELDS = [
    ("age",                    218),
    ("gender",                 219),
    ("years_music_making",     220),
    ("years_formal_instruc",   221),
    ("hours_per_week_music",   222),
    ("ai_text_tools_usage",    223),
    ("ai_music_tools_used",    224),
    ("ai_music_tools_freq",    225),
]

FEW_SHOT_MESSAGES = [
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

    # Prompt columns: Korean survey descriptions contain "해당 음악을"
    # Exclude the practice column whose description starts with "연습 세션"
    prompt_col_indices = [
        i for i, d in enumerate(desc_row)
        if "해당 음악을" in d and not d.strip().startswith("연습")
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
        model="gpt-5.4",
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
