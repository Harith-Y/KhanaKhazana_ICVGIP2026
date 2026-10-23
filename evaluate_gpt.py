"""
Zero-shot evaluation of GPT-4o on the food test set.

Same protocol as evaluate_qwen.py: asks the model for top-5 dish guesses
(open-ended), fuzzy-matches each guess against the 300 class names.

Usage:
    python evaluate_gpt.py --manifest manifests/300cls
    python evaluate_gpt.py --manifest manifests/300cls --max_images 50
    python evaluate_gpt.py --manifest manifests/300cls --model gpt-4o-mini --concurrency 10

Requires OPENAI_API_KEY env var.
"""

import argparse
import asyncio
import base64
import csv
import json
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import unicodedata
from difflib import get_close_matches
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError


# ── helpers ──────────────────────────────────────────────────────────────────

def normalise(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace("-", " ").replace("_", " ").strip()


def best_match(pred: str, class_norms: dict[str, str]) -> str | None:
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


def encode_image(path: str) -> tuple[str, str]:
    p = Path(path)
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
    }.get(p.suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), media_type


# ── async eval ───────────────────────────────────────────────────────────────

async def eval_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    img_path: str,
    true_cls: str,
    prompt: str,
    class_norms: dict[str, str],
    model: str,
    retries: int = 6,
) -> dict:
    async with sem:
        for attempt in range(retries):
            try:
                b64, mtype = encode_image(img_path)
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mtype};base64,{b64}",
                                    "detail": "low",   # ~85 tokens/image, sufficient for classification
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    max_tokens=150,
                    timeout=30,
                )
                text    = resp.choices[0].message.content or ""
                guesses = parse_top5(text)
                matched = [best_match(g, class_norms) for g in guesses]
                matched = [m for m in matched if m is not None]
                return {
                    "path":     img_path,
                    "true":     true_cls,
                    "guesses":  guesses,
                    "matched":  matched,
                    "top1":     len(matched) > 0 and matched[0] == true_cls,
                    "top5":     true_cls in matched,
                    "response": text,
                }
            except RateLimitError:
                if attempt < retries - 1:
                    await asyncio.sleep(60)
                else:
                    return {
                        "path": img_path, "true": true_cls,
                        "guesses": [], "matched": [],
                        "top1": False, "top5": False,
                        "response": "ERROR: RateLimitError after all retries",
                        "error": True,
                    }
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {
                        "path":     img_path,
                        "true":     true_cls,
                        "guesses":  [],
                        "matched":  [],
                        "top1":     False,
                        "top5":     False,
                        "response": f"ERROR: {e}",
                        "error":    True,
                    }


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    default="manifests/300cls")
    p.add_argument("--output",      default=None, help="Output JSON path (auto if omitted)")
    p.add_argument("--max_images",  type=int, default=None, help="Cap for quick testing")
    p.add_argument("--model",       default="gpt-4o")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Max parallel API calls (default 5)")
    return p.parse_args()


async def main():
    args      = parse_args()
    mdir      = Path(args.manifest)
    model_tag = args.model.replace("-", "").replace(".", "")
    output    = Path(args.output) if args.output else \
                Path("runs") / f"gpt_eval_{model_tag}_{mdir.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── load class list ──
    classes     = json.loads((mdir / "classes.json").read_text())
    idx_to_cls  = {v: k for k, v in classes.items()}
    class_norms = {k: normalise(k) for k in classes}
    class_list  = sorted(classes.keys())
    num_classes = len(class_list)

    # ── load test samples ──
    samples = []
    with open(mdir / "test.csv") as f:
        for row in csv.DictReader(f):
            samples.append((row["path"], int(row["label"])))
    if args.max_images:
        samples = samples[: args.max_images]

    # ── resume: reload already-done results ──
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
    display     = [k.replace("-", " ") for k in class_list]
    class_block = ", ".join(display)
    PROMPT = (
        "You are an expert in Indian cuisine. "
        "Look at this food image and identify the dish.\n\n"
        f"The dish belongs to one of these {num_classes} Indian food categories:\n"
        f"{class_block}\n\n"
        "List your top 5 guesses in order, one per line, numbered 1-5. "
        "Use the exact category name from the list above. No explanation."
    )

    # ── async evaluation ──
    client  = AsyncOpenAI()
    sem     = asyncio.Semaphore(args.concurrency)
    tasks   = [
        eval_one(client, sem, img_path, idx_to_cls[label], PROMPT, class_norms, args.model)
        for img_path, label in todo
    ]

    records   = list(done.values())
    completed = 0
    t0        = time.time()

    for coro in asyncio.as_completed(tasks):
        rec = await coro
        records.append(rec)
        completed += 1

        # save incrementally every 50 completions
        if completed % 50 == 0 or completed == len(tasks):
            valid   = [r for r in records if not r.get("error")]
            top1    = sum(r["top1"] for r in valid)
            top5    = sum(r["top5"] for r in valid)
            n       = len(valid)
            elapsed = time.time() - t0
            rate    = completed / elapsed if elapsed else 0
            print(
                f"[{completed}/{len(tasks)}]  "
                f"top1={100*top1/n:.2f}%  top5={100*top5/n:.2f}%  "
                f"({rate:.1f} img/s)"
            )
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
    print(f"Top-1 Acc    : {100*top1/n:.2f}%")
    print(f"Top-5 Acc    : {100*top5/n:.2f}%")
    print(f"Errors       : {sum(1 for r in records if r.get('error'))}")
    print(f"Results      : {output}")
    print(f"{'='*50}")


def _save(output: Path, model: str, mdir: Path, num_classes: int, records: list[dict]):
    valid = [r for r in records if not r.get("error")]
    n     = len(valid)
    summary = {
        "model":        model,
        "manifest":     str(mdir),
        "num_classes":  num_classes,
        "total_images": n,
        "top1_acc":     round(100 * sum(r["top1"] for r in valid) / n, 2) if n else 0,
        "top5_acc":     round(100 * sum(r["top5"] for r in valid) / n, 2) if n else 0,
        "errors":       sum(1 for r in records if r.get("error")),
        "records":      records,
    }
    output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
