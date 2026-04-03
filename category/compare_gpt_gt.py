import csv
import re

def normalize(text):
    if not text:
        return set()
    text = text.lower().strip()
    words = re.split(r'[|,]+', text)
    result = set()
    for w in words:
        w = w.strip()
        if w:
            result.add(w)
    return result

def token_overlap(gt_set, pred_set):
    if not gt_set:
        return None
    matched = 0
    for gt_word in gt_set:
        for pred_word in pred_set:
            if gt_word in pred_word or pred_word in gt_word:
                matched += 1
                break
    return matched, len(gt_set)

def read_csv_by_id(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_ = int(row['id'].strip())
            data[id_] = row
    return data

gt = read_csv_by_id('gt_category.csv')
gpt1 = read_csv_by_id('prompt_category_gpt.csv')
gpt2 = read_csv_by_id('prompt_category_gpt_2.csv')

gt_cols = {
    'genre': ' genre',
    'mood': ' mood',
    'inst': ' inst',
    'timbre': ' timbre',
    'function': ' function',
    'music_theory': ' music theory',
    'lyrics': ' lyrics'
}

gpt_cols = {
    'genre': 'genre',
    'mood': 'mood/emotion',
    'inst': 'instrumentation',
    'timbre': 'timbre',
    'function': 'function',
    'music_theory': 'music theory',
    'lyrics': 'story/narrative/lyrics'
}

col_names = ['genre', 'mood', 'inst', 'timbre', 'function', 'music_theory', 'lyrics']

print("=" * 110)
print(f"{'ID':<4} {'Category':<14} {'GT':<30} {'GPT-1':<30} {'GPT-2':<30} {'G1':<8} {'G2':<8}")
print("=" * 110)

total_g1_matched = 0
total_g1_total = 0
total_g2_matched = 0
total_g2_total = 0

col_g1 = {c: [0, 0] for c in col_names}
col_g2 = {c: [0, 0] for c in col_names}

for id_ in range(6, 24):
    gt_row = gt.get(id_, {})
    g1_row = gpt1.get(id_, {})
    g2_row = gpt2.get(id_, {})

    for col in col_names:
        gt_val = gt_row.get(gt_cols[col], '')
        g1_val = g1_row.get(gpt_cols[col], '')
        g2_val = g2_row.get(gpt_cols[col], '')

        gt_set = normalize(gt_val)
        g1_set = normalize(g1_val)
        g2_set = normalize(g2_val)

        r1 = token_overlap(gt_set, g1_set)
        r2 = token_overlap(gt_set, g2_set)

        gt_str = '|'.join(sorted(gt_set)) if gt_set else '-'
        g1_str = '|'.join(sorted(g1_set)) if g1_set else '-'
        g2_str = '|'.join(sorted(g2_set)) if g2_set else '-'

        g1_match_str = f"{r1[0]}/{r1[1]}" if r1 else "N/A"
        g2_match_str = f"{r2[0]}/{r2[1]}" if r2 else "N/A"

        if r1:
            total_g1_matched += r1[0]
            total_g1_total += r1[1]
            col_g1[col][0] += r1[0]
            col_g1[col][1] += r1[1]
        if r2:
            total_g2_matched += r2[0]
            total_g2_total += r2[1]
            col_g2[col][0] += r2[0]
            col_g2[col][1] += r2[1]

        if gt_set:
            print(f"{id_:<4} {col:<14} {gt_str[:28]:<30} {g1_str[:28]:<30} {g2_str[:28]:<30} {g1_match_str:<8} {g2_match_str:<8}")

    print("-" * 110)

print()
print("=" * 70)
print("Category Match Rate Summary")
print("=" * 70)
print(f"{'Category':<14} {'GPT-1':<20} {'GPT-1 %':<12} {'GPT-2':<20} {'GPT-2 %':<12}")
print("-" * 70)
for col in col_names:
    m1, t1 = col_g1[col]
    m2, t2 = col_g2[col]
    p1 = f"{m1/t1*100:.1f}%" if t1 else "N/A"
    p2 = f"{m2/t2*100:.1f}%" if t2 else "N/A"
    print(f"{col:<14} {f'{m1}/{t1}':<20} {p1:<12} {f'{m2}/{t2}':<20} {p2:<12}")

print("-" * 70)
p1_total = f"{total_g1_matched/total_g1_total*100:.1f}%"
p2_total = f"{total_g2_matched/total_g2_total*100:.1f}%"
print(f"{'TOTAL':<14} {f'{total_g1_matched}/{total_g1_total}':<20} {p1_total:<12} {f'{total_g2_matched}/{total_g2_total}':<20} {p2_total:<12}")
