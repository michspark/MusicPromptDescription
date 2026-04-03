"""
Compare gt_category.csv vs prompt_category_gpt.csv to evaluate category prediction accuracy.

Columns compared (gt → gpt):
  genre          → genre
  mood           → mood/emotion
  inst           → instrumentation
  timbre         → timbre
  function       → function
  music theory   → music theory
  lyrics         → story/narrative/lyrics

Each cell may contain pipe-separated multi-labels.
Matching is done at the token level with optional fuzzy matching.
"""

import csv
import re
from pathlib import Path
from difflib import SequenceMatcher

# ── Config ──────────────────────────────────────────────────────────────────
GT_FILE   = Path(__file__).parent / "gt_category.csv"
PRED_FIELDNAMES = ["id", "prompt", "genre", "mood/emotion", "instrumentation",
                   "timbre", "function", "music theory", "story/narrative/lyrics"]

MODEL_FILES = {
    "GPT-1":      Path(__file__).parent / "prompt_category_gpt.csv",
    "GPT-2":      Path(__file__).parent / "prompt_category_gpt_2.csv",
    "GPT-3":      Path(__file__).parent / "prompt_category_gpt_3.csv",
    "GPT-4":      Path(__file__).parent / "prompt_category_gpt_4.csv",
    "GPT-5.4":    Path(__file__).parent / "prompt_category_gpt_5_4.csv",
    "Gemini-2":   Path(__file__).parent / "prompt_category_gemini_2.csv",
}
EVAL_IDS = set(range(6, 24))   # 6~23 only
FUZZY_THRESHOLD = 0.5   # token-level string similarity cutoff
HALF_MATCH_THRESHOLD = 0.5  # cell is "correct" if recall >= this value
# ────────────────────────────────────────────────────────────────────────────

COLUMN_MAP = {
    # gt_col       : gpt_col
    "genre"        : "genre",
    "mood"         : "mood/emotion",
    "inst"         : "instrumentation",
    "timbre"       : "timbre",
    "function"     : "function",
    "music theory" : "music theory",
    "lyrics"       : "story/narrative/lyrics",
}


def normalize(text: str) -> str:
    """Lowercase, strip extra whitespace, remove trailing punctuation."""
    return re.sub(r"\s+", " ", text.strip().lower()).strip(".,;:")


def tokenize(cell: str) -> list[str]:
    """Split pipe-separated cell into normalized tokens, dropping empty."""
    return [t for tok in cell.split("|") if (t := normalize(tok))]


def fuzzy_match(a: str, b: str) -> bool:
    """True if the two strings are similar enough."""
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= FUZZY_THRESHOLD


def tokens_overlap(gt_tokens: list[str], pred_tokens: list[str]) -> tuple[int, int, int]:
    """
    Returns (tp, fn, fp) for a single cell comparison.
    tp = # GT tokens matched in pred
    fn = # GT tokens not found in pred
    fp = # pred tokens not matching any GT token
    """
    matched_pred = set()
    tp = 0
    for gt_tok in gt_tokens:
        hit = False
        for i, pred_tok in enumerate(pred_tokens):
            if i in matched_pred:
                continue
            if gt_tok == pred_tok or fuzzy_match(gt_tok, pred_tok):
                tp += 1
                matched_pred.add(i)
                hit = True
                break
        if not hit:
            pass  # counts as fn below
    fn = len(gt_tokens) - tp
    fp = len(pred_tokens) - len(matched_pred)
    return tp, fn, fp


def f1(tp: int, fn: int, fp: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        # If first cell looks like an integer id, file has no header row
        first_cell = first.split(",")[0].strip()
        if first_cell.isdigit():
            reader = csv.DictReader(f, fieldnames=PRED_FIELDNAMES)
        else:
            reader = csv.DictReader(f)
            reader.fieldnames = [k.strip() for k in reader.fieldnames]
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def evaluate_model(gt_rows: list[dict], pred_rows: list[dict], gt_id_col: str) -> dict:
    """Run evaluation for one model. Returns cat_stats dict."""
    pred_by_id = {row["id"]: row for row in pred_rows}

    cat_stats: dict[str, dict] = {
        col: {"tp": 0, "fn": 0, "fp": 0, "rows_with_gt": 0, "exact_hits": 0, "half_hits": 0}
        for col in COLUMN_MAP
    }

    for gt_row in gt_rows:
        row_id = gt_row[gt_id_col]
        if row_id not in EVAL_IDS and str(row_id) not in {str(i) for i in EVAL_IDS}:
            continue
        pred_row = pred_by_id.get(row_id) or pred_by_id.get(str(row_id))
        if pred_row is None:
            continue

        for gt_col, pred_col in COLUMN_MAP.items():
            gt_val   = gt_row.get(gt_col, "")
            pred_val = pred_row.get(pred_col, "")
            gt_toks   = tokenize(gt_val)
            pred_toks = tokenize(pred_val)

            if not gt_toks:
                cat_stats[gt_col]["fp"] += len(pred_toks)
                continue

            tp, fn, fp = tokens_overlap(gt_toks, pred_toks)
            exact_hit = (tp == len(gt_toks) and fp == 0)
            cell_recall = tp / len(gt_toks) if gt_toks else 0.0
            half_hit  = cell_recall >= HALF_MATCH_THRESHOLD
            cat_stats[gt_col]["tp"]           += tp
            cat_stats[gt_col]["fn"]           += fn
            cat_stats[gt_col]["fp"]           += fp
            cat_stats[gt_col]["rows_with_gt"] += 1
            if exact_hit:
                cat_stats[gt_col]["exact_hits"] += 1
            if half_hit:
                cat_stats[gt_col]["half_hits"]  += 1

    return cat_stats


def summarize(cat_stats: dict) -> tuple[float, float, float, float, float]:
    """Return (precision, recall, f1, exact_pct, half_pct) totals."""
    total_tp = total_fn = total_fp = total_rows = total_exact = total_half = 0
    for s in cat_stats.values():
        total_tp    += s["tp"]; total_fn += s["fn"]; total_fp += s["fp"]
        total_rows  += s["rows_with_gt"]
        total_exact += s["exact_hits"]
        total_half  += s["half_hits"]
    prec    = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec     = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1v     = f1(total_tp, total_fn, total_fp)
    ex_pc   = total_exact / total_rows * 100 if total_rows > 0 else 0.0
    half_pc = total_half  / total_rows * 100 if total_rows > 0 else 0.0
    return prec, rec, f1v, ex_pc, half_pc


def main():
    gt_rows  = load_csv(GT_FILE)
    gt_id_col = list(gt_rows[0].keys())[0]

    all_stats = {}
    for model_name, model_path in MODEL_FILES.items():
        pred_rows = load_csv(model_path)
        all_stats[model_name] = evaluate_model(gt_rows, pred_rows, gt_id_col)

    # ── Per-category comparison table ────────────────────────────────────────
    print(f"\nEval range: IDs {min(EVAL_IDS)}~{max(EVAL_IDS)}  |  Fuzzy threshold: {FUZZY_THRESHOLD}\n")

    col_w = 14
    model_w = 38  # Prec / Rec / F1 / Exact% / Half%

    header_models = "".join(f"  {m:^{model_w}}" for m in MODEL_FILES)
    print(f"{'Category':<{col_w}}{header_models}")
    sub = "".join(f"  {'Prec':>6} {'Rec':>6} {'F1':>6} {'Exact%':>8} {'Half%':>7}" for _ in MODEL_FILES)
    print(f"{'': <{col_w}}{sub}")
    sep = "-" * (col_w + len(MODEL_FILES) * (model_w + 2))
    print(sep)

    for col in COLUMN_MAP:
        row = f"{col:<{col_w}}"
        for model_name, cat_stats in all_stats.items():
            s = cat_stats[col]
            tp, fn, fp = s["tp"], s["fn"], s["fp"]
            n    = s["rows_with_gt"]
            ex   = s["exact_hits"]
            half = s["half_hits"]
            prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1v     = f1(tp, fn, fp)
            ex_pc   = ex   / n * 100 if n > 0 else 0.0
            half_pc = half / n * 100 if n > 0 else 0.0
            row += f"  {prec:>6.3f} {rec:>6.3f} {f1v:>6.3f} {ex_pc:>7.1f}% {half_pc:>6.1f}%"
        print(row)

    # ── Overall summary ───────────────────────────────────────────────────────
    print(sep)
    overall_row = f"{'OVERALL':<{col_w}}"
    for model_name, cat_stats in all_stats.items():
        prec, rec, f1v, ex_pc, half_pc = summarize(cat_stats)
        overall_row += f"  {prec:>6.3f} {rec:>6.3f} {f1v:>6.3f} {ex_pc:>7.1f}% {half_pc:>6.1f}%"
    print(overall_row)

    # ── Clean winner line ─────────────────────────────────────────────────────
    print()
    print(f"{'Model':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Exact%':>8} {'Half%':>7}")
    print("-" * 52)
    for model_name, cat_stats in all_stats.items():
        prec, rec, f1v, ex_pc, half_pc = summarize(cat_stats)
        print(f"{model_name:<10} {prec:>10.3f} {rec:>8.3f} {f1v:>8.3f} {ex_pc:>7.1f}% {half_pc:>6.1f}%")


if __name__ == "__main__":
    main()
