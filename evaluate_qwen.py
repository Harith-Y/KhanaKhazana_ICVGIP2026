"""
Zero-shot evaluation of Qwen2-VL-7B-Instruct on the food test set.

Asks the model for its top-5 dish guesses (open-ended), then fuzzy-matches
each guess against the known class list to compute top-1 and top-5 accuracy.

Usage:
    python evaluate_qwen.py --manifest manifests/300cls
    python evaluate_qwen.py --manifest manifests/300cls --max_images 500  # quick smoke-test
"""

import argparse
import csv
import json
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise SystemExit("pip install qwen-vl-utils")


# ── helpers ──────────────────────────────────────────────────────────────────

def normalise(s: str) -> str:
    """Lowercase, strip accents, collapse hyphens/underscores/spaces."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace("-", " ").replace("_", " ").strip()


def best_match(pred: str, class_norms: dict[str, str]) -> str | None:
    """Return the class key whose normalised name best matches pred, or None."""
    pn = normalise(pred)
    # exact match first
    for key, norm in class_norms.items():
        if norm == pn:
            return key
    # fuzzy fallback
    norms = list(class_norms.values())
    keys  = list(class_norms.keys())
    close = get_close_matches(pn, norms, n=1, cutoff=0.72)
    if close:
        return keys[norms.index(close[0])]
    return None


def parse_top5(response: str) -> list[str]:
    """Extract up to 5 dish names from the model's numbered list response."""
    lines, dishes = response.strip().splitlines(), []
    for line in lines:
        line = line.strip()
        # strip leading "1. " / "1) " / "- "
        for prefix in ("1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "5)", "-"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            dishes.append(line)
        if len(dishes) == 5:
            break
    return dishes


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   default="manifests/300cls")
    p.add_argument("--output",     default=None, help="Path for results JSON (auto if omitted)")
    p.add_argument("--max_images", type=int, default=None, help="Cap for quick testing")
    p.add_argument("--model",      default="Qwen/Qwen2-VL-2B-Instruct")
    return p.parse_args()


def main():
    args   = parse_args()
    mdir   = Path(args.manifest)
    model_tag = args.model.split("/")[-1].lower().replace("-instruct", "")
    output = Path(args.output) if args.output else \
             Path("runs") / f"qwen_eval_{model_tag}_{mdir.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── load class list ──
    classes     = json.loads((mdir / "classes.json").read_text())   # name → idx
    idx_to_cls  = {v: k for k, v in classes.items()}
    class_norms = {k: normalise(k) for k in classes}               # name → normalised
    class_list  = sorted(classes.keys())
    num_classes = len(class_list)

    # ── load test samples ──
    samples = []
    with open(mdir / "test.csv") as f:
        for row in csv.DictReader(f):
            samples.append((row["path"], int(row["label"])))
    if args.max_images:
        samples = samples[: args.max_images]

    print(f"Classes : {num_classes}")
    print(f"Images  : {len(samples)}")

    # ── build class display string for prompt ──
    display = [k.replace("-", " ") for k in class_list]
    class_block = ", ".join(display)   # single line — keeps prompt compact

    PROMPT = (
        "You are an expert in Indian cuisine. "
        "Look at this food image and identify the dish.\n\n"
        f"The dish belongs to one of these {num_classes} Indian food categories:\n"
        f"{class_block}\n\n"
        "List your top 5 guesses in order, one per line, numbered 1-5. "
        "Use the exact category name from the list above. No explanation."
    )

    # ── load model ──
    print(f"\nLoading {args.model} (4-bit) …")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map="auto",
    )
    # cap image resolution: 256*28*28 ≈ 448×448 — enough for food classification,
    # reduces visual token count by ~75% vs the default 1280*28*28 max
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    print("Model ready.\n")

    # ── evaluate ──
    top1_correct = top5_correct = 0
    unmatched    = 0
    records      = []

    for img_path, label in tqdm(samples, desc="Evaluating"):
        true_cls = idx_to_cls[label]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": PROMPT},
            ],
        }]

        if not Path(img_path).exists():
            continue

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        response = processor.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        del inputs, out
        torch.cuda.empty_cache()

        guesses     = parse_top5(response)
        matched     = [best_match(g, class_norms) for g in guesses]
        matched     = [m for m in matched if m is not None]

        top1_hit = len(matched) > 0 and matched[0] == true_cls
        top5_hit = true_cls in matched

        if len(matched) == 0:
            unmatched += 1

        top1_correct += int(top1_hit)
        top5_correct += int(top5_hit)

        records.append({
            "path":      img_path,
            "true":      true_cls,
            "guesses":   guesses,
            "matched":   matched,
            "top1":      top1_hit,
            "top5":      top5_hit,
            "response":  response,
        })

    total   = len(records)
    top1acc = 100 * top1_correct / total if total else 0
    top5acc = 100 * top5_correct / total if total else 0

    summary = {
        "model":        args.model,
        "manifest":     str(mdir),
        "num_classes":  num_classes,
        "total_images": total,
        "top1_acc":     round(top1acc, 2),
        "top5_acc":     round(top5acc, 2),
        "unmatched":    unmatched,
        "records":      records,
    }
    output.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"Classes      : {num_classes}")
    print(f"Images       : {total}")
    print(f"Top-1 Acc    : {top1acc:.2f}%")
    print(f"Top-5 Acc    : {top5acc:.2f}%")
    print(f"Unmatched    : {unmatched}  ({100*unmatched/total:.1f}% of predictions)")
    print(f"Results      : {output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
