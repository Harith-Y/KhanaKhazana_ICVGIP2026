"""
Zero-shot evaluation of LLaVA-1.5 models on the food test set.

Covers llava-hf/llava-1.5-7b-hf and llava-hf/llava-1.5-13b-hf.

Usage:
    python evaluate_llava.py --manifest manifests/300cls
    python evaluate_llava.py --manifest manifests/300cls --model llava-hf/llava-1.5-13b-hf
    python evaluate_llava.py --manifest manifests/300cls --max_images 50
"""

import argparse
import csv
import json
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   default="manifests/300cls")
    p.add_argument("--output",     default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--model",      default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--load_in_8bit", action="store_true",
                   help="Load with bitsandbytes 8-bit quantization (~half the VRAM, avoids CPU offload/OOM)")
    return p.parse_args()


def main():
    args      = parse_args()
    mdir      = Path(args.manifest)
    model_tag = args.model.split("/")[-1].lower()
    if args.load_in_8bit:
        model_tag += "-8bit"
    output    = Path(args.output) if args.output else \
                Path("runs") / f"llava_eval_{model_tag}_{mdir.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    classes     = json.loads((mdir / "classes.json").read_text())
    idx_to_cls  = {v: k for k, v in classes.items()}
    class_norms = {k: normalise(k) for k in classes}
    class_list  = sorted(classes.keys())
    num_classes = len(class_list)

    samples = []
    with open(mdir / "test.csv") as f:
        for row in csv.DictReader(f):
            samples.append((row["path"], int(row["label"])))
    if args.max_images:
        samples = samples[: args.max_images]

    # resume
    done: dict[str, dict] = {}
    if output.exists():
        try:
            prior = json.loads(output.read_text())
            for r in prior.get("records", []):
                done[r["path"]] = r
            print(f"Resuming — {len(done)} already done")
        except Exception:
            pass
    todo = [(p, l) for p, l in samples if p not in done and Path(p).exists()]
    print(f"Model   : {args.model}")
    print(f"Classes : {num_classes}  |  Total: {len(samples)}  |  Todo: {len(todo)}")

    class_block = ", ".join(k.replace("-", " ") for k in class_list)
    PROMPT = (
        "You are an expert in Indian cuisine. "
        "Look at this food image and identify the dish.\n\n"
        f"The dish belongs to one of these {num_classes} Indian food categories:\n"
        f"{class_block}\n\n"
        "List your top 5 guesses in order, one per line, numbered 1-5. "
        "Use the exact category name from the list above. No explanation."
    )

    load_kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if args.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print(f"\nLoading {args.model} (8-bit) …")
    else:
        print(f"\nLoading {args.model} (float16) …")
    model = LlavaForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print("Model ready.\n")

    records = list(done.values())

    for img_path, label in tqdm(todo, desc="Evaluating"):
        true_cls = idx_to_cls[label]
        try:
            image = Image.open(img_path).convert("RGB")
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            }]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            response = processor.decode(
                gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            del inputs, gen_ids
            torch.cuda.empty_cache()

            guesses = parse_top5(response)
            matched = [m for m in (best_match(g, class_norms) for g in guesses) if m]
            records.append({
                "path": img_path, "true": true_cls,
                "guesses": guesses, "matched": matched,
                "top1": len(matched) > 0 and matched[0] == true_cls,
                "top5": true_cls in matched,
                "response": response,
            })
        except Exception as e:
            torch.cuda.empty_cache()
            records.append({
                "path": img_path, "true": true_cls,
                "guesses": [], "matched": [], "top1": False, "top5": False,
                "response": f"ERROR: {e}", "error": True,
            })

        if len(records) % 100 == 0:
            _save(output, args.model, mdir, num_classes, records)

    _save(output, args.model, mdir, num_classes, records)

    valid = [r for r in records if not r.get("error")]
    n = len(valid)
    print(f"\n{'='*50}")
    print(f"Model        : {args.model}")
    print(f"Images       : {n}")
    if n:
        print(f"Top-1 Acc    : {100*sum(r['top1'] for r in valid)/n:.2f}%")
        print(f"Top-5 Acc    : {100*sum(r['top5'] for r in valid)/n:.2f}%")
    print(f"Errors       : {sum(1 for r in records if r.get('error'))}")
    print(f"Results      : {output}")
    print(f"{'='*50}")


def _save(output, model, mdir, num_classes, records):
    valid = [r for r in records if not r.get("error")]
    n = len(valid)
    output.write_text(json.dumps({
        "model": model, "manifest": str(mdir), "num_classes": num_classes,
        "total_images": n,
        "top1_acc": round(100 * sum(r["top1"] for r in valid) / n, 2) if n else 0,
        "top5_acc": round(100 * sum(r["top5"] for r in valid) / n, 2) if n else 0,
        "errors": sum(1 for r in records if r.get("error")),
        "records": records,
    }, indent=2))


if __name__ == "__main__":
    main()
