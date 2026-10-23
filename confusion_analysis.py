"""
Confusion analysis for zero-shot VLM eval results.

Produces two outputs per model:
  1. Top confused pairs — ranked list of (true_class, predicted_as, count)
  2. Macro-category confusion matrix — 16×16 grid aggregating all 300 classes
     into broad dish categories

Usage:
    # analyse all runs in the runs/ directory
    python confusion_analysis.py

    # specify files explicitly
    python confusion_analysis.py --runs runs/qwen_eval_*.json runs/gemini_eval_*.json

    # save full output to JSON
    python confusion_analysis.py --out confusion_results.json

    # show top N confused pairs per model (default 20)
    python confusion_analysis.py --top_n 30
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ── Macro-category mapping ────────────────────────────────────────────────────
# Rules are checked in order; first match wins.
# Each rule is (category_label, [substrings to match in class name]).

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Chicken",         ["chicken", "butter-chicken"]),
    ("Seafood",         ["fish", "prawn", "crab", "green-chromide"]),
    ("Meat",            ["mutton", "keema", "haleem", "nihari", "rogan-josh",
                         "vindaloo", "beef", "chorizo", "goa-sausage", "wazwan"]),
    ("Egg",             ["egg-bhurji", "egg-curry", "egg-roast", "tomato-omelette"]),
    ("Kebab/Tandoor",   ["kebab", "tandoori", "tandoor", "tikka", "chicken-65",
                         "chicken-lollipop", "chicken-ghee-roast", "chicken-razala",
                         "chicken-malai-kebab"]),
    ("Biryani/Rice",    ["biryani", "pulao", "yakhni-pulao", "akhni", "fried-rice",
                         "chitranna", "coconut-rice", "saffron-rice", "tamarind-rice",
                         "tehri", "khichdi", "pongal", "pilaf"]),
    ("Dal/Lentil",      ["dal", "daal", "sambar", "rasam", "kadhi"]),
    ("Paneer",          ["paneer", "palak-paneer"]),
    ("Veg Curry/Sabji", ["aloo", "gobi", "bhindi", "baingan", "karela", "dum-aloo",
                         "gatte", "gutti", "jackfruit", "ker-sangri", "kofta",
                         "navratan", "undhiyu", "usal", "poriyal", "thoran",
                         "raw-banana", "yam-fry", "mushroom", "corn"]),
    ("Bread/Flatbread", ["paratha", "naan", "bhatura", "chapati", "roti", "puri",
                         "kulcha", "bhakri", "thalipeeth", "makki-di-roti",
                         "litti", "missi-roti", "daal-puri"]),
    ("Breakfast/Dosa",  ["dosa", "idli", "upma", "poha", "uttapam", "pesarattu",
                         "paddu", "kuzhi-paniyaram", "idiyappam", "rava-idli",
                         "rava-upma", "bread-upma", "french-toast", "appam",
                         "neer-dosa", "set-dosa", "chitranna"]),
    ("Snack/Street",    ["chaat", "samosa", "pakora", "vada", "bhel", "pani-puri",
                         "momo", "kachori", "dabeli", "dhokla", "ganthiya",
                         "khandvi", "mathri", "murukku", "namak-para", "sev",
                         "banana-chips", "bread-pakora", "chow-mein", "noodles",
                         "manchurian", "misal-pav", "jhalmuri", "chotpoti",
                         "pizza", "shankarpali", "batata-vada", "kathi-roll",
                         "locho", "chikki", "papad", "manchow", "hot-and-sour",
                         "aloo-tikki", "aloo-chaat", "aloo-chap", "gulgula"]),
    ("Sweets/Dessert",  ["halwa", "kheer", "laddu", "jalebi", "gulab-jamun",
                         "rasgulla", "ras-malai", "sandesh", "kalakand",
                         "kaju-katli", "phirni", "rabri", "shrikhand", "falooda",
                         "basundi", "kulfi", "mysore-pak", "modak", "imarti",
                         "cham-cham", "sheera", "balushahi", "malapua", "doodhpak",
                         "mishti-doi", "ledikeni", "lyangcha", "crème-caramel",
                         "fruit-custard", "chak-hao-kheer", "sohan-halwa",
                         "sohan-papdi", "adhirasam", "anarsa", "ariselu",
                         "bandar-laddu", "dharwad-pedha", "dodol", "double-ka-meetha",
                         "ghevar", "kajjikaya", "kakinada-khaja", "kesari-bat",
                         "murabba", "poornalu", "pootharekulu", "puran-poli",
                         "qubani-ka-meetha", "shahi-tukra", "sheer-khurma",
                         "sheer-korma", "sutar-feni", "gavvalu", "enduri-pitha",
                         "kakara-pitha", "chakuli-pitha", "pitha", "pithe",
                         "achappam", "unni-appam", "boondi", "sel-roti",
                         "malai", "barfi", "payasam", "shahi-paneer",
                         "double-ka-meetha", "aam-papad"]),
    ("Beverage",        ["chai", "coffee", "lassi", "sharbat", "thandai",
                         "nimbu-pani", "smoothie", "masala-chai", "filter-coffee",
                         "ginger-tea", "solkadhi", "aam-panna", "dahi"]),
    ("Condiment/Side",  ["chutney", "raita", "pickle", "mango-pickle", "idli-podi",
                         "ghee", "jaggery", "rice-flour", "rava", "sattu",
                         "kosambari", "salad", "cucumber-raita", "coconut-chutney",
                         "tomato-chutney"]),
    ("Soup/Thali/Other", []),  # catch-all
]

CATEGORIES = [r[0] for r in CATEGORY_RULES]


def categorise(class_name: str) -> str:
    for cat, keywords in CATEGORY_RULES:
        if not keywords:
            return cat  # catch-all
        if any(kw in class_name for kw in keywords):
            return cat
    return "Soup/Thali/Other"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_run(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data


def short_model_name(model_str: str) -> str:
    """'Qwen/Qwen2.5-VL-7B-Instruct' → 'Qwen2.5-VL-7B'"""
    name = model_str.split("/")[-1]
    for suffix in ("-Instruct", "-instruct", "-hf", "-HF"):
        name = name.replace(suffix, "")
    return name


def analyse_run(data: dict, top_n: int) -> dict:
    records = [r for r in data.get("records", []) if not r.get("error")]
    if not records:
        return {}

    # top-1 predictions only (first matched class)
    # When model has no top-1 match, it counts as a miss but we skip it in
    # pair analysis (no prediction to confuse with).
    cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
    n_cats  = len(CATEGORIES)
    matrix  = [[0] * n_cats for _ in range(n_cats)]
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for r in records:
        true_cls = r["true"]
        matched  = r.get("matched", [])
        true_cat = categorise(true_cls)

        if not matched:
            # model produced no matchable guess → confusion with self (miss)
            continue

        pred_cls = matched[0]
        pred_cat = categorise(pred_cls)

        # macro matrix
        matrix[cat_idx[true_cat]][cat_idx[pred_cat]] += 1

        # fine-grained pairs (only actual confusions, not correct predictions)
        if pred_cls != true_cls:
            pair_counts[(true_cls, pred_cls)] += 1

    # top confused pairs
    top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:top_n]

    return {
        "model":       data.get("model", "?"),
        "top1_acc":    data.get("top1_acc"),
        "top5_acc":    data.get("top5_acc"),
        "total":       len(records),
        "top_confused_pairs": [
            {"true": t, "predicted_as": p, "count": c}
            for (t, p), c in top_pairs
        ],
        "macro_matrix": {
            "categories": CATEGORIES,
            "matrix":     matrix,
        },
    }


def print_analysis(result: dict, top_n: int):
    model = short_model_name(result["model"])
    print(f"\n{'='*70}")
    print(f"  {model}   top1={result['top1_acc']}%  top5={result['top5_acc']}%  ({result['total']} images)")
    print(f"{'='*70}")

    print(f"\n── Top {top_n} confused pairs ──")
    print(f"  {'True class':<35} {'Predicted as':<35} {'Count':>5}")
    print(f"  {'-'*35} {'-'*35} {'-'*5}")
    for p in result["top_confused_pairs"]:
        print(f"  {p['true']:<35} {p['predicted_as']:<35} {p['count']:>5}")

    print(f"\n── Macro-category confusion matrix (rows=true, cols=predicted) ──")
    cats   = result["macro_matrix"]["categories"]
    matrix = result["macro_matrix"]["matrix"]
    col_w  = 8
    label_w = 20

    # header
    header = f"  {'':>{label_w}}"
    for c in cats:
        header += f"  {c[:col_w-1]:>{col_w-1}}"
    print(header)
    print(f"  {'-'*(label_w + (col_w+2)*len(cats))}")

    for i, row_cat in enumerate(cats):
        row_total = sum(matrix[i])
        if row_total == 0:
            continue
        row_str = f"  {row_cat:>{label_w}}"
        for j in range(len(cats)):
            val = matrix[i][j]
            pct = 100 * val / row_total if row_total else 0
            # highlight diagonal (correct) vs off-diagonal (confused)
            cell = f"{pct:.0f}%" if val else "  ·"
            row_str += f"  {cell:>{col_w-1}}"
        row_str += f"  (n={row_total})"
        print(row_str)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",   nargs="*", default=None,
                   help="Run JSON files to analyse. Default: all runs/*.json")
    p.add_argument("--out",    default=None,
                   help="Optional path to save full results as JSON")
    p.add_argument("--top_n", type=int, default=20,
                   help="Number of top confused pairs to show per model (default 20)")
    p.add_argument("--min_top1", type=float, default=0.0,
                   help="Skip models with top-1 below this %% (useful to filter out CLIP etc.)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.runs:
        run_files = [Path(f) for pattern in args.runs for f in Path(".").glob(pattern)
                     if Path(f).exists()] or [Path(f) for f in args.runs if Path(f).exists()]
    else:
        run_files = sorted(Path("runs").glob("*.json"))

    if not run_files:
        print("No run files found. Run an eval script first, or pass --runs path/to/*.json")
        return

    print(f"Analysing {len(run_files)} run file(s) …")

    all_results = []
    for f in run_files:
        try:
            data = load_run(f)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
            continue

        top1 = data.get("top1_acc", 0) or 0
        if top1 < args.min_top1:
            continue

        result = analyse_run(data, args.top_n)
        if not result:
            print(f"  [skip] {f.name}: no valid records")
            continue

        result["source_file"] = str(f)
        all_results.append(result)
        print_analysis(result, args.top_n)

    # sort by top-1 acc descending for the summary
    all_results.sort(key=lambda r: -(r.get("top1_acc") or 0))

    print(f"\n\n{'='*70}")
    print("  SUMMARY (sorted by top-1)")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'Top-1':>7} {'Top-5':>7}  {'Images':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*7}  {'-'*7}")
    for r in all_results:
        print(f"  {short_model_name(r['model']):<35} {r['top1_acc']:>6.2f}%  {r['top5_acc']:>6.2f}%  {r['total']:>7}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
