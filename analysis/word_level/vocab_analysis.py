"""
vocab_analysis.py

vocab_udio_prompt.csv 에서 count >= 5 인 단어들에 대해 두 가지 분석:

[Analysis 1] Survival Rate
  - 해당 단어를 포함한 prompt의 대응 description에서
    그 단어가 얼마나 살아남는지 비율 계산

[Analysis 2] Chi-Squared
  - 해당 단어를 포함한 prompt 그룹 vs 포함하지 않은 그룹의
    description을 비교하여 가장 특징적인 단어와 chi2 점수 출력
"""

import re
import spacy
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. spaCy 로드 및 불용어 설정
# ============================================================
print("언어 모델을 불러오는 중...")
nlp = spacy.load("en_core_web_sm")

custom_stopwords = {
    'song', 'music', 'track', 'sound', 'audio',
    'feel', 'make', 'hear', 'listen', 'like', 'vibe'
}

for word in custom_stopwords:
    nlp.vocab[word].is_stop = True


def extract_keywords(text):
    """텍스트에서 lemmatized 키워드 추출 (모든 품사, 불용어/구두점/공백 제외)"""
    if pd.isna(text) or str(text).strip() == '':
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


# ============================================================
# 2. 데이터 로드
# ============================================================
print("데이터를 불러오는 중...")

# Udio prompt
udio_df = pd.read_csv(
    r'C:\Users\MICHA\Codes\MusicPromptDescription\data\udio_sample_data.csv'
)
udio_df['uuid'] = udio_df['url'].apply(
    lambda x: re.search(r'/songs/(.+)$', str(x)).group(1)
    if re.search(r'/songs/(.+)$', str(x)) else ''
)

# English description (Qualtrics: 헤더 3줄 -> 1,2행 스킵)
desc_df = pd.read_csv(
    r'C:\Users\MICHA\Codes\MusicPromptDescription\data\0502_english.csv',
    skiprows=[1, 2]
)
uuid_pattern = re.compile(r'^[0-9a-f]{8}-')
uuid_cols = [c for c in desc_df.columns if uuid_pattern.match(c)]

# vocab_udio_prompt: count >= 5 단어 목록
vocab_df = pd.read_csv(
    r'C:\Users\MICHA\Codes\MusicPromptDescription\analysis\word_level\vocab_udio_prompt.csv'
)
target_words = set(vocab_df[vocab_df['count'] >= 5]['word'].tolist())
print(f"  분석 대상 단어 수 (count >= 5): {len(target_words)}")


# ============================================================
# 3. 곡별 prompt 키워드 / description 키워드 매핑 구축
# ============================================================
print("키워드 추출 중 (prompt & descriptions)...")

# uuid -> (raw_prompt, prompt_keywords_set)
song_prompt = {}
for _, row in udio_df.iterrows():
    uid = row['uuid']
    kws = set(extract_keywords(row['prompt']))
    song_prompt[uid] = kws

# uuid -> list of description keyword sets (참가자별)
song_descs = defaultdict(list)
for col in uuid_cols:
    # col 앞 8자리로 udio uuid 매칭
    prefix = col[:8]
    matched_uid = next(
        (uid for uid in song_prompt if uid.startswith(prefix)), None
    )
    if matched_uid is None:
        continue
    for text in desc_df[col].dropna():
        kws = set(extract_keywords(text))
        if kws:  # 빈 결과 제외
            song_descs[matched_uid].append(kws)

total_songs_with_desc = sum(1 for v in song_descs.values() if v)
print(f"  Description이 있는 곡 수: {total_songs_with_desc}")


# ============================================================
# 4. Analysis 1 - Survival Rate
# ============================================================
print("\n[Analysis 1] Survival Rate 계산 중...")

survival_results = []

for word in sorted(target_words):
    # 해당 단어가 prompt에 있는 곡 / 없는 곡 분류
    songs_with = [uid for uid, kws in song_prompt.items() if word in kws]
    songs_without = [uid for uid, kws in song_prompt.items() if word not in kws]

    # 해당 곡들의 description 수집
    descs_with = []
    for uid in songs_with:
        descs_with.extend(song_descs.get(uid, []))

    if not descs_with:
        continue

    # 단어가 description에도 등장하는 경우
    survived = sum(1 for kw_set in descs_with if word in kw_set)
    total = len(descs_with)
    survival_rate = survived / total * 100

    survival_results.append({
        'prompt_word': word,
        'prompt_count': vocab_df.loc[vocab_df['word'] == word, 'count'].values[0],
        'songs_with_word': len(songs_with),
        'total_descriptions': total,
        'survived_count': survived,
        'survival_rate_pct': round(survival_rate, 1)
    })

survival_df = pd.DataFrame(survival_results).sort_values(
    'survival_rate_pct', ascending=False
)


# ============================================================
# 5. Analysis 2 - Chi-Squared
# ============================================================
print("[Analysis 2] Chi-Squared 계산 중...")

chi2_results = []

for word in sorted(target_words):
    songs_with = [uid for uid, kws in song_prompt.items() if word in kws]
    songs_without = [uid for uid, kws in song_prompt.items() if word not in kws]

    descs_with = []
    for uid in songs_with:
        descs_with.extend(song_descs.get(uid, []))

    descs_without = []
    for uid in songs_without:
        descs_without.extend(song_descs.get(uid, []))

    if len(descs_with) < 2 or len(descs_without) < 2:
        continue

    n_with = len(descs_with)
    n_without = len(descs_without)

    # with / without 그룹의 description에 나오는 모든 단어 수집
    vocab_with = Counter(w for kw_set in descs_with for w in kw_set)
    vocab_without = Counter(w for kw_set in descs_without for w in kw_set)
    all_vocab = set(vocab_with) | set(vocab_without)

    word_chi2 = []
    for v in all_vocab:
        # 2x2 contingency table
        # [v in with, v not in with]
        # [v in without, v not in without]
        a = sum(1 for kw_set in descs_with if v in kw_set)    # with & contains v
        b = sum(1 for kw_set in descs_without if v in kw_set)  # without & contains v
        c = n_with - a                                          # with & no v
        d = n_without - b                                       # without & no v

        if a + b == 0 or c + d == 0:
            continue

        try:
            chi2, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
            # with 그룹에서 더 자주 등장하는 단어만 (비율 비교)
            rate_with = a / n_with
            rate_without = b / n_without if n_without > 0 else 0
            if rate_with > rate_without and v != word and a >= 3:
                word_chi2.append((v, round(chi2, 2), round(p, 4), a, b))
        except Exception:
            continue

    # chi2 기준 상위 10개
    top10 = sorted(word_chi2, key=lambda x: x[1], reverse=True)[:10]

    chi2_results.append({
        'prompt_word': word,
        'prompt_count': vocab_df.loc[vocab_df['word'] == word, 'count'].values[0],
        'songs_with_word': len(songs_with),
        'desc_with': n_with,
        'desc_without': n_without,
        'top_chi2_words': ', '.join(
            f"{w}(chi2={c2}, p={p})" for w, c2, p, _, _ in top10
        )
    })

chi2_df = pd.DataFrame(chi2_results).sort_values('prompt_count', ascending=False)

# top_chi2_words를 행별로 펼쳐서 (prompt_word, desc_word, chi2, p) 형태로 flatten
# 전체 문자열에서 패턴을 findall로 한번에 추출 (split 대신)
entry_pattern = re.compile(r'([^\s,]+)\(chi2=([\d.]+),\s*p=([\d.]+)\)')
flat_rows = []
for _, row in chi2_df.iterrows():
    for m in entry_pattern.finditer(str(row['top_chi2_words'])):
        flat_rows.append({
            'prompt_word':  row['prompt_word'],
            'desc_word':    m.group(1),
            'chi2':         float(m.group(2)),
            'p_value':      float(m.group(3)),
            'prompt_count': row['prompt_count'],
            'desc_with':    row['desc_with'],
            'desc_without': row['desc_without'],
        })

chi2_flat_df = pd.DataFrame(flat_rows).sort_values('chi2', ascending=False)
chi2_flat_df = chi2_flat_df[chi2_flat_df['prompt_word'] != chi2_flat_df['desc_word']]
chi2_flat_df_10 = chi2_flat_df[
    (chi2_flat_df['desc_with'] >= 20) & (chi2_flat_df['prompt_count'] > 10)
]


# ============================================================
# 6. 결과 출력 및 저장
# ============================================================
import os
output_dir = r'C:\Users\MICHA\Codes\MusicPromptDescription\analysis\word_level'

# --- Survival Rate 출력 ---
print(f"\n{'='*65}")
print(f"  [Analysis 1] Survival Rate  (prompt word count >= 5)")
print(f"{'='*65}")
print(f"{'단어':<18} {'prompt':<7} {'곡수':<5} {'desc수':<6} {'생존':<5} {'생존율':>7}")
print(f"{'-'*65}")
for _, row in survival_df.iterrows():
    print(
        f"  {row['prompt_word']:<16} {int(row['prompt_count']):<7} "
        f"{int(row['songs_with_word']):<5} {int(row['total_descriptions']):<6} "
        f"{int(row['survived_count']):<5} {row['survival_rate_pct']:>6}%"
    )

# --- Chi-Squared 출력 ---
print(f"\n{'='*65}")
print(f"  [Analysis 2] Chi-Squared Top Words per Prompt Word")
print(f"{'='*65}")
for _, row in chi2_df.iterrows():
    print(f"\n  [{row['prompt_word']}] (prompt={int(row['prompt_count'])}, "
          f"desc_with={int(row['desc_with'])}, desc_without={int(row['desc_without'])})")
    print(f"    => {row['top_chi2_words']}")

# --- Chi-Squared flat 출력 (desc_with >= 10) ---
print(f"\n{'='*60}")
print(f"  [Analysis 2b] Chi-Squared  (desc_with >= 10, chi2 높은 순)")
print(f"{'='*60}")
print(f"{'prompt_word':<16} {'desc_word':<20} {'chi2':>8}  {'p':>8}  {'desc_with':>9}")
print(f"{'-'*60}")
for _, row in chi2_flat_df_10.iterrows():
    print(f"  {row['prompt_word']:<14} {row['desc_word']:<20} {row['chi2']:>8.2f}  {row['p_value']:>8.4f}  {int(row['desc_with']):>9}")

# --- CSV 저장 ---
survival_out = os.path.join(output_dir, 'analysis_survival_rate.csv')
chi2_out = os.path.join(output_dir, 'analysis_chi2.csv')
chi2_flat_out = os.path.join(output_dir, 'analysis_chi2_sorted_by_chi2.csv')

survival_df.to_csv(survival_out, index=False, encoding='utf-8-sig')
chi2_df.to_csv(chi2_out, index=False, encoding='utf-8-sig')
chi2_flat_df_10.to_csv(chi2_flat_out, index=False, encoding='utf-8-sig')

print(f"\n결과 저장 완료:")
print(f"  - {survival_out}")
print(f"  - {chi2_out}")
print(f"  - {chi2_flat_out}")
