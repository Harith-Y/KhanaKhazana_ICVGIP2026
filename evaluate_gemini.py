"""
Zero-shot evaluation of Gemini (via Vertex AI) on the food test set.

Same protocol as the other eval scripts: top-5 dish guesses, fuzzy-matched
against the 300 class names. Saves incrementally so runs survive interruption.

Usage:
    python evaluate_gemini.py --manifest manifests/300cls
    python evaluate_gemini.py --manifest manifests/300cls --max_images 20
    python evaluate_gemini.py --manifest manifests/300cls --model gemini-2.5-flash

Requires:
    pip install google-genai python-dotenv
    GEMINI_API_KEY in .env
"""

import argparse
import csv
import json
import time
import unicodedata
from difflib import get_close_matches
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import os
from google import genai
from google.genai import types


# ── text helpers ──────────────────────────────────────────────────────────────

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


def parse_top5(text: str) -> list[str]:
    lines, dishes = text.strip().splitlines(), []
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


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   default="manifests/300cls")
    p.add_argument("--output",     default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--model",      default="gemini-2.5-flash-lite")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set — check your .env file.")

    mdir      = Path(args.manifest)
    model_tag = args.model.replace("-", "_")
    output    = Path(args.output) if args.output else \
                Path("runs") / f"gemini_eval_{model_tag}_{mdir.name}.json"
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
    print(f"Model    : {args.model}")
    print(f"Classes  : {num_classes}")
    print(f"Total    : {len(samples)}  |  todo: {len(todo)}  |  done: {len(done)}")

    if not todo:
        print("Nothing to do.")
        return

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

    # ── client ──
    client = genai.Client(vertexai=True, api_key=api_key)

    # ── evaluate ──
    records = list(done.values())
    t0      = time.time()

    for i, (img_path, label) in enumerate(todo):
        true_cls = idx_to_cls[label]

        try:
            img_bytes = Path(img_path).read_bytes()
            suffix    = Path(img_path).suffix.lower()
            mime      = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                         "png": "image/png", "webp": "image/webp"}.get(suffix.lstrip("."), "image/jpeg")

            response = client.models.generate_content(
                model=args.model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(max_output_tokens=100),
            )
            text    = response.text or ""
            guesses = parse_top5(text)
            matched = [m for m in (best_match(g, class_norms) for g in guesses) if m]
            records.append({
                "path":    img_path,
                "true":    true_cls,
                "guesses": guesses,
                "matched": matched,
                "top1":    len(matched) > 0 and matched[0] == true_cls,
                "top5":    true_cls in matched,
                "response": text,
            })

        except Exception as e:
            print(f"  [error] {type(e).__name__}: {str(e)[:120]}")
            records.append({
                "path": img_path, "true": true_cls,
                "guesses": [], "matched": [],
                "top1": False, "top5": False,
                "response": f"ERROR: {e}", "error": True,
            })

        # progress + incremental save every 50 images
        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            valid   = [r for r in records if not r.get("error")]
            n       = len(valid)
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            acc_str = f"top1={100*sum(r['top1'] for r in valid)/n:.2f}%  top5={100*sum(r['top5'] for r in valid)/n:.2f}%" if n else "no valid results yet"
            print(f"[{i+1}/{len(todo)}]  {acc_str}  ({rate:.2f} img/s)")
            _save(output, args.model, mdir, num_classes, records)

    # ── final ──
    valid = [r for r in records if not r.get("error")]
    n     = len(valid)
    top1  = sum(r["top1"] for r in valid)
    top5  = sum(r["top5"] for r in valid)
    _save(output, args.model, mdir, num_classes, records)

    print(f"\n{'='*50}")
    print(f"Model        : {args.model}")
    print(f"Classes      : {num_classes}")
    print(f"Images       : {n}")
    if n:
        print(f"Top-1 Acc    : {100*top1/n:.2f}%")
        print(f"Top-5 Acc    : {100*top5/n:.2f}%")
    print(f"Errors       : {sum(1 for r in records if r.get('error'))}")
    print(f"Results      : {output}")
    print(f"{'='*50}")


def _save(output, model, mdir, num_classes, records):
    valid = [r for r in records if not r.get("error")]
    n     = len(valid)
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
