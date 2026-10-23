"""
Zero-shot evaluation of Llama-3.2-11B-Vision-Instruct on the food test set.

Same protocol as evaluate_qwen.py: top-5 dish guesses, fuzzy-matched
against the 300 class names.  Saves incrementally so overnight runs survive
interruption.

Prerequisite: accept Meta's license at
  https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
then `huggingface-cli login` before running.

Usage:
    python evaluate_llama_vision.py --manifest manifests/300cls
    python evaluate_llama_vision.py --manifest manifests/300cls --max_images 50
    python evaluate_llama_vision.py --manifest manifests/300cls --model meta-llama/Llama-3.2-90B-Vision-Instruct
"""

import argparse
import csv
import json
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import torch
from PIL import Image
from transformers import BitsAndBytesConfig, MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm


# ── text helpers (same as evaluate_qwen.py) ───────────────────────────────────

def normalise(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace("-", " ").replace("_", " ").strip()


def best_match(pred: str, class_norms: dict) -> str | None:
    pn = normalise(pred)
    for key, norm in class_norms.items():
        if norm == pn:
            return key
    norms = list(class_norms.values())
    keys  = list(class_norms.keys())
    close = get_close_matches(pn, norms, n=1, cutoff=0.72)
    if close:
        return keys[norms.index(close[0])]
    return None


def parse_top5(response: str) -> list[str]:
    lines, dishes = response.strip().splitlines(), []
    for line in lines:
        line = line.strip()
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
    p.add_argument("--output",     default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--model",      default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    p.add_argument("--img_size",   type=int, default=560,
                   help="Resize images to this before processing (560=one tile, avoids OOM).")
    return p.parse_args()


def main():
    args     = parse_args()
    mdir     = Path(args.manifest)
    model_tag = args.model.split("/")[-1].lower().replace("-instruct", "")
    output   = Path(args.output) if args.output else \
               Path("runs") / f"llama_vision_eval_{model_tag}_{mdir.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── class list ──
    classes     = json.loads((mdir / "classes.json").read_text())
    idx_to_cls  = {v: k for k, v in classes.items()}
    class_norms = {k: normalise(k) for k in classes}
    class_list  = sorted(classes.keys())
    num_classes = len(class_list)

    # ── test samples ──
    samples = []
    with open(mdir / "test.csv") as f:
        for row in csv.DictReader(f):
            samples.append((row["path"], int(row["label"])))
    if args.max_images:
        samples = samples[: args.max_images]

    # ── resume ──
    done: dict[str, dict] = {}
    if output.exists():
        try:
            prior = json.loads(output.read_text())
            for r in prior.get("records", []):
                done[r["path"]] = r
            print(f"Resuming — {len(done)} already evaluated, skipping")
        except Exception:
            pass

    todo = [(p, l) for p, l in samples if p not in done and Path(p).exists()]
    print(f"Classes  : {num_classes}")
    print(f"Total    : {len(samples)}  |  todo: {len(todo)}  |  done: {len(done)}")

    # ── prompt ──
    class_block = ", ".join(k.replace("-", " ") for k in class_list)
    PROMPT = (
        "You are an expert in Indian cuisine. "
        "Look at this food image and identify the dish.\n\n"
        f"The dish belongs to one of these {num_classes} Indian food categories:\n"
        f"{class_block}\n\n"
        "List your top 5 guesses in order, one per line, numbered 1-5. "
        "Use the exact category name from the list above. No explanation."
    )

    # Llama-3.2-Vision uses the <|image|> token in the message content
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }]

    # ── load model ──
    print(f"\nLoading {args.model} (4-bit) …")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    text_template = processor.apply_chat_template(messages, add_generation_prompt=True)
    print("Model ready.\n")

    # ── evaluate ──
    records = list(done.values())

    for img_path, label in tqdm(todo, desc="Evaluating"):
        true_cls = idx_to_cls[label]
        try:
            image = Image.open(img_path).convert("RGB")
            # Resize to exactly one tile to cap VRAM; 560×560 is Llama-Vision's native tile size
            if args.img_size:
                image = image.resize((args.img_size, args.img_size))

            inputs = processor(
                images=image,
                text=text_template,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            response = processor.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            del image, inputs, out
            torch.cuda.empty_cache()
        except Exception as e:
            torch.cuda.empty_cache()
            records.append({
                "path": img_path, "true": true_cls,
                "guesses": [], "matched": [], "top1": False, "top5": False,
                "response": f"ERROR: {e}", "error": True,
            })
            continue

        guesses = parse_top5(response)
        matched = [m for m in (best_match(g, class_norms) for g in guesses) if m]
        records.append({
            "path":     img_path,
            "true":     true_cls,
            "guesses":  guesses,
            "matched":  matched,
            "top1":     len(matched) > 0 and matched[0] == true_cls,
            "top5":     true_cls in matched,
            "response": response,
        })

        if len(records) % 100 == 0:
            _save(output, args.model, mdir, num_classes, records)

    _save(output, args.model, mdir, num_classes, records)

    valid = [r for r in records if not r.get("error")]
    n    = len(valid)
    top1 = sum(r["top1"] for r in valid)
    top5 = sum(r["top5"] for r in valid)
    print(f"\n{'='*50}")
    print(f"Model        : {args.model}")
    print(f"Classes      : {num_classes}")
    print(f"Images       : {n}")
    print(f"Top-1 Acc    : {100*top1/n:.2f}%")
    print(f"Top-5 Acc    : {100*top5/n:.2f}%")
    print(f"Errors       : {sum(1 for r in records if r.get('error'))}")
    print(f"Results      : {output}")
    print(f"{'='*50}")


def _save(output: Path, model: str, mdir: Path, num_classes: int, records: list):
    valid = [r for r in records if not r.get("error")]
    n = len(valid)
    output.write_text(json.dumps({
        "model":        model,
        "manifest":     str(mdir),
        "num_classes":  num_classes,
        "total_images": n,
        "top1_acc":     round(100 * sum(r["top1"] for r in valid) / n, 2) if n else 0,
        "top5_acc":     round(100 * sum(r["top5"] for r in valid) / n, 2) if n else 0,
        "errors":       sum(1 for r in records if r.get("error")),
        "records":      records,
    }, indent=2))


if __name__ == "__main__":
    main()
