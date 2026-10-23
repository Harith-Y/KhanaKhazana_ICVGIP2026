"""
Zero-shot evaluation of CLIP (openai/clip-vit-base-patch32) on the food test set.

For each image, computes cosine similarity against text embeddings of all class
names and picks the top-5. No training, no fine-tuning.

Usage:
    python evaluate_clip.py --manifest manifests/300cls
    python evaluate_clip.py --manifest manifests/300cls --max_images 500
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


PROMPTS_PER_CLASS = [
    "a photo of {name}",
    "a photo of {name}, an Indian dish",
    "a close-up photo of {name} food",
    "Indian food called {name}",
]


def build_text_embeddings(model, processor, class_names, device):
    """Average embeddings over multiple prompt templates per class."""
    all_embs = []
    for cls in tqdm(class_names, desc="Encoding class names"):
        display = cls.replace("-", " ")
        texts   = [t.format(name=display) for t in PROMPTS_PER_CLASS]
        inputs  = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_out = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            pooled = text_out.pooler_output          # (batch, hidden)
            embs   = model.text_projection(pooled)   # (batch, 512)
            embs   = F.normalize(embs, dim=-1).mean(dim=0)
            embs   = F.normalize(embs, dim=-1)
        all_embs.append(embs)
    return torch.stack(all_embs)          # (num_classes, 512)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   default="manifests/300cls")
    p.add_argument("--output",     default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model",      default="openai/clip-vit-base-patch32")
    return p.parse_args()


def main():
    args   = parse_args()
    mdir   = Path(args.manifest)
    output = Path(args.output) if args.output else \
             Path("runs") / f"clip_eval_{mdir.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    classes    = json.loads((mdir / "classes.json").read_text())
    idx_to_cls = {v: k for k, v in classes.items()}
    class_list = sorted(classes.keys())

    samples = []
    with open(mdir / "test.csv") as f:
        for row in csv.DictReader(f):
            samples.append((row["path"], int(row["label"])))
    if args.max_images:
        samples = samples[: args.max_images]

    print(f"Classes : {len(class_list)}")
    print(f"Images  : {len(samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Loading {args.model} …")

    model     = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    text_embs = build_text_embeddings(model, processor, class_list, device)
    # text_embs: (num_classes, 512) — on CPU to save VRAM during image encoding
    text_embs = text_embs.to(device)

    top1_correct = top5_correct = 0
    records = []

    bs = args.batch_size
    for start in tqdm(range(0, len(samples), bs), desc="Evaluating"):
        batch    = samples[start: start + bs]
        paths    = [p for p, _ in batch]
        labels   = [l for _, l in batch]

        images = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224)))

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled     = vision_out.pooler_output
            img_embs   = model.visual_projection(pooled)
            img_embs   = F.normalize(img_embs, dim=-1)        # (bs, 512)

        sims    = img_embs @ text_embs.T                      # (bs, num_classes)
        top5    = sims.topk(5, dim=-1).indices.cpu().tolist() # (bs, 5)

        for i, (label, top5_idx) in enumerate(zip(labels, top5)):
            true_cls  = idx_to_cls[label]
            pred_cls  = class_list[top5_idx[0]]
            top5_cls  = [class_list[j] for j in top5_idx]

            top1_hit  = pred_cls == true_cls
            top5_hit  = true_cls in top5_cls
            top1_correct += int(top1_hit)
            top5_correct += int(top5_hit)

            records.append({
                "path":  paths[i],
                "true":  true_cls,
                "top1_pred": pred_cls,
                "top5_pred": top5_cls,
                "top1": top1_hit,
                "top5": top5_hit,
            })

    total   = len(samples)
    top1acc = 100 * top1_correct / total
    top5acc = 100 * top5_correct / total

    summary = {
        "model":        args.model,
        "manifest":     str(mdir),
        "num_classes":  len(class_list),
        "total_images": total,
        "top1_acc":     round(top1acc, 2),
        "top5_acc":     round(top5acc, 2),
        "records":      records,
    }
    output.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"Model        : {args.model}")
    print(f"Classes      : {len(class_list)}")
    print(f"Images       : {total}")
    print(f"Top-1 Acc    : {top1acc:.2f}%")
    print(f"Top-5 Acc    : {top5acc:.2f}%")
    print(f"Results      : {output}")
    print(f"{'='*50}")
    print(f"\nEfficientNet-B3 baseline (300cls): 73.49% top-1 / 90.06% top-5")


if __name__ == "__main__":
    main()
