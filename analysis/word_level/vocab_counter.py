import re
import spacy
import pandas as pd
from collections import Counter

# ==========================================
# 1. spaCy 모델 로드 및 불용어 설정
# ==========================================
print("언어 모델을 불러오는 중입니다...")
nlp = spacy.load("en_core_web_sm")

# 도메인 특화 불용어(블랙리스트) - 음악 묘사에서 의미 없이 자주 쓰이는 단어
custom_stopwords = {'song', 'music', 'track', 'sound', 'audio', 'feel', 'make', 'hear', 'listen', 'like', 'vibe'}
for word in custom_stopwords:
    nlp.vocab[word].is_stop = True


# ==========================================
# 2. 텍스트 전처리 함수
# ==========================================
def extract_keywords(text):
    if pd.isna(text):
        return []
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    # "hip" 바로 다음에 "hop"이 오면 "hip-hop"으로 합치기
    merged = []
    i = 0
    while i < len(tokens):
        if tokens[i] == 'hip' and i + 1 < len(tokens) and tokens[i + 1] == 'hop':
            merged.append('hip-hop')
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def get_top_n_words(word_list, n=20):
    word_freq = Counter(word_list)
    return word_freq.most_common(n)


# ==========================================
# 3. 데이터 로드
# ==========================================

# --- (A) Udio prompt 데이터 ---
print("\nUdio 데이터를 불러오는 중...")
udio_df = pd.read_csv(
    r'C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_data.csv'
)
udio_prompts = udio_df['prompt'].dropna().tolist()
print(f"  Udio prompt 수: {len(udio_prompts)}")

# --- (B) English description 데이터 (Qualtrics 형식: 헤더 3줄) ---
print("English description 데이터를 불러오는 중...")
desc_df = pd.read_csv(
    r'C:\Users\MICHA\Codes\MusicPromptDescription\data\0428_eng.csv',
    skiprows=[1, 2]   # Qualtrics: row1=설명, row2=ImportId -> 스킵
)

# UUID 패턴 컬럼 = 각 곡(udio song ID)에 대한 참가자 설명
uuid_pattern = re.compile(r'^[0-9a-f]{8}-')
uuid_cols = [c for c in desc_df.columns if uuid_pattern.match(c)]

# 모든 UUID 컬럼의 값을 하나의 리스트로 flatten
eng_descriptions = []
for col in uuid_cols:
    eng_descriptions.extend(desc_df[col].dropna().tolist())
print(f"  English description 수 (응답 셀): {len(eng_descriptions)}")


# ==========================================
# 4. 키워드 추출 및 빈도 계산
# ==========================================
print("\n키워드 추출 중 (Udio prompts)...")
udio_keywords = []
for text in udio_prompts:
    udio_keywords.extend(extract_keywords(text))

print("키워드 추출 중 (English descriptions)...")
desc_keywords = []
for text in eng_descriptions:
    desc_keywords.extend(extract_keywords(text))


# ==========================================
# 5. 결과 출력
# ==========================================
TOP_N = 30

print(f"\n{'='*55}")
print(f"  Udio Prompt  - Top {TOP_N} 단어")
print(f"{'='*55}")
for rank, (word, count) in enumerate(get_top_n_words(udio_keywords, TOP_N), 1):
    print(f"  {rank:>2}. {word:<20} {count:>4}회")

print(f"\n{'='*55}")
print(f"  English Description  - Top {TOP_N} 단어")
print(f"{'='*55}")
for rank, (word, count) in enumerate(get_top_n_words(desc_keywords, TOP_N), 1):
    print(f"  {rank:>2}. {word:<20} {count:>4}회")

# ==========================================
# 6. CSV로 저장
# ==========================================
import os
output_dir = r'C:\Users\MICHA\Codes\MusicPromptDescription\vocab'

udio_result = pd.DataFrame(get_top_n_words(udio_keywords, n=len(Counter(udio_keywords))),
                           columns=['word', 'count'])
desc_result = pd.DataFrame(get_top_n_words(desc_keywords, n=len(Counter(desc_keywords))),
                           columns=['word', 'count'])

udio_out = os.path.join(output_dir, 'vocab_udio_prompt.csv')
desc_out = os.path.join(output_dir, 'vocab_eng_description.csv')

udio_result.to_csv(udio_out, index=False, encoding='utf-8-sig')
desc_result.to_csv(desc_out, index=False, encoding='utf-8-sig')

print(f"\n결과 저장 완료:")
print(f"  - {udio_out}")
print(f"  - {desc_out}")