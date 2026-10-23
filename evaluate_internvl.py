"""
Zero-shot evaluation of InternVL2-8B on the food test set.

Same protocol as evaluate_qwen.py: top-5 dish guesses, fuzzy-matched
against the 300 class names.  Saves incrementally so overnight runs survive
interruption.

Usage:
    python evaluate_internvl.py --manifest manifests/300cls
    python evaluate_internvl.py --manifest manifests/300cls --max_images 50
    python evaluate_internvl.py --manifest manifests/300cls --model OpenGVLab/InternVL2-8B --max_tiles 4
"""

import argparse
import csv
import json
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


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


# ── InternVL2 image loading ───────────────────────────────────────────────────

def _build_transform(size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _closest_ratio(aspect: float, ratios: list, w: int, h: int, size: int) -> tuple:
    best_diff, best = float("inf"), (1, 1)
    for r in ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff or (diff == best_diff and w * h > 0.5 * size * size * r[0] * r[1]):
            best_diff, best = diff, r
    return best


def _tile_image(image: Image.Image, max_tiles: int = 4, size: int = 448) -> torch.Tensor:
    """Dynamic-resolution tiling used by InternVL2 + thumbnail."""
    w, h = image.size
    ratios = sorted(
        {(i, j) for n in range(1, max_tiles + 1)
                 for i in range(1, n + 1)
                 for j in range(1, n + 1)
                 if 1 <= i * j <= max_tiles},
        key=lambda x: x[0] * x[1],
    )
    cols, rows = _closest_ratio(w / h, ratios, w, h, size)
    resized = image.resize((size * cols, size * rows))
    tiles = [
        resized.crop((
            (i % cols) * size, (i // cols) * size,
            (i % cols + 1) * size, (i // cols + 1) * size,
        ))
        for i in range(cols * rows)
    ]
    tiles.append(image.resize((size, size)))    # thumbnail always appended
    tf = _build_transform(size)
    return torch.stack([tf(t) for t in tiles])  # [N, 3, 448, 448]


def load_image(path: str, max_tiles: int = 4) -> torch.Tensor:
    return _tile_image(Image.open(path).convert("RGB"), max_tiles=max_tiles)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   default="manifests/300cls")
    p.add_argument("--output",     default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--model",      default="OpenGVLab/InternVL2-8B")
    p.add_argument("--max_tiles",  type=int, default=4,
                   help="Max image tiles (1=448×448 only, 4=default). Lower = less VRAM.")
    return p.parse_args()


def main():
    args     = parse_args()
    mdir     = Path(args.manifest)
    model_tag = args.model.split("/")[-1].lower()
    output   = Path(args.output) if args.output else \
               Path("runs") / f"internvl_eval_{model_tag}_{mdir.name}.json"
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
                if not r.get("error"):   # skip previously errored — re-evaluate them
                    done[r["path"]] = r
            n_err = sum(1 for r in prior.get("records", []) if r.get("error"))
            print(f"Resuming — {len(done)} done, {n_err} errored (will retry)")
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
    question = f"<image>\n{PROMPT}"

    # ── load model ──
    # InternVL's custom model class lacks `all_tied_weights_keys` added in
    # transformers 5.x.  Patch PreTrainedModel.__init__ to set it before any
    # accelerate device-map call touches it.
    from transformers import modeling_utils as _mu, BitsAndBytesConfig
    _orig_init = _mu.PreTrainedModel.__init__
    def _new_init(self, config, *a, **kw):
        _orig_init(self, config, *a, **kw)
        if not hasattr(self, 'all_tied_weights_keys') or self.all_tied_weights_keys is None:
            self.all_tied_weights_keys = {}
    _mu.PreTrainedModel.__init__ = _new_init

    # Use 4-bit quantization: InternVL2.5-8B (LLM 8B + ViT ~3B) in bfloat16 is
    # ~22 GB total — well beyond a 12 GB GPU.  8-bit still OOMs (ViT stays fp16,
    # leaving <500 MB for activations).  4-bit brings the LLM to ~4 GB + ViT ~6 GB
    # ≈ 10 GB total, leaving headroom for the long-prompt attention computation.
    # pixel_values must be cast to bfloat16 to match the ViT bias dtype.
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"\nLoading {args.model} (4-bit, ~6 GB VRAM) …")
    model = AutoModel.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # transformers ≥5.x: PreTrainedModel no longer inherits GenerationMixin, so
    # trust_remote_code models (InternLM2, InternVL3, etc.) lose .generate().
    # Patch it onto the inner language model class after loading.
    from transformers import GenerationConfig, GenerationMixin as _GM
    lm = getattr(model, "language_model", None)
    if lm is not None and not isinstance(lm, _GM):
        lm.__class__.__bases__ = lm.__class__.__bases__ + (_GM,)
    if lm is not None and not hasattr(lm, "generation_config"):
        lm.generation_config = GenerationConfig()
    gen_config = dict(max_new_tokens=150, do_sample=False)
    print("Model ready.\n")

    # ── evaluate ──
    records = list(done.values())

    for img_path, label in tqdm(todo, desc="Evaluating"):
        true_cls = idx_to_cls[label]
        try:
            pv = load_image(img_path, max_tiles=args.max_tiles).to(torch.bfloat16).cuda()
            with torch.no_grad():
                response = model.chat(tokenizer, pv, question, gen_config)
            del pv
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

        # incremental save every 100 images
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
    if n:
        print(f"Top-1 Acc    : {100*top1/n:.2f}%")
        print(f"Top-5 Acc    : {100*top5/n:.2f}%")
    else:
        print("Top-1 Acc    : N/A (all errors)")
        print("Top-5 Acc    : N/A (all errors)")
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
