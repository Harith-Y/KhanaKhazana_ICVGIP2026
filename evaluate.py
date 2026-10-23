"""
Evaluate a trained checkpoint on the test set.

Usage:
    python evaluate.py --checkpoint runs/<run>/best.pth
    python evaluate.py --checkpoint runs/<run>/best.pth --split val
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import FoodDataset
from data.transforms import val_transforms
from models.classifier import build_model
from utils.metrics import AverageMeter, topk_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--per_class",  action="store_true", help="Print per-class accuracy")
    p.add_argument("--confusion",  action="store_true", help="Save full confusion matrix JSON")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes  = ckpt["classes"]          # {class_name: label_idx}
    backbone = ckpt.get("backbone", cfg["model"]["backbone"])
    num_classes = len(classes)

    idx_to_cls = {v: k for k, v in classes.items()}

    manifest_dir = Path(cfg["data"]["manifest_dir"])
    csv_path     = manifest_dir / f"{args.split}.csv"

    # If the manifest was regenerated after training, its label indices differ from
    # the checkpoint's.  Build a remap: manifest_idx → checkpoint_idx so we can
    # evaluate the checkpoint against any manifest revision.
    remap = None
    manifest_cls_path = manifest_dir / "classes.json"
    if manifest_cls_path.exists():
        manifest_cls = json.loads(manifest_cls_path.read_text())
        if manifest_cls != classes:
            remap = {m_idx: classes[name]
                     for name, m_idx in manifest_cls.items()
                     if name in classes}
            print(f"[warn] Manifest classes differ from checkpoint — "
                  f"remapping {len(remap)}/{len(manifest_cls)} labels")

    ds = FoodDataset(csv_path, val_transforms(cfg))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(backbone, num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loss_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()

    # Per-class tracking
    cls_correct = defaultdict(int)
    cls_total   = defaultdict(int)

    # Confusion matrix (only allocated when --confusion is requested)
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)] if args.confusion else None

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if remap is not None:
                mapped = torch.tensor([remap.get(l.item(), -1) for l in labels])
                keep = mapped >= 0
                images, mapped = images[keep], mapped[keep]
                if mapped.numel() == 0:
                    continue
                labels = mapped
            labels = labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            acc1, acc5 = topk_accuracy(logits, labels, topk=(1, 5))
            n = images.size(0)
            loss_m.update(loss.item(), n)
            acc1_m.update(acc1, n)
            acc5_m.update(acc5, n)

            preds = logits.argmax(dim=1)
            for pred, gt in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                cls_total[gt]   += 1
                cls_correct[gt] += int(pred == gt)
                if confusion_matrix is not None:
                    confusion_matrix[gt][pred] += 1

    print(f"\n{'='*50}")
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Classes    : {num_classes}")
    print(f"Images     : {len(ds)}")
    print(f"{'='*50}")
    print(f"Loss       : {loss_m.avg:.4f}")
    print(f"Top-1 Acc  : {acc1_m.avg:.2f}%")
    print(f"Top-5 Acc  : {acc5_m.avg:.2f}%")

    if args.per_class:
        print(f"\n{'Class':<40} {'Correct':>8} {'Total':>8} {'Acc%':>8}")
        print("-" * 70)
        rows = sorted(cls_total.items(), key=lambda x: cls_correct[x[0]] / x[1])
        for idx, total in rows:
            correct = cls_correct[idx]
            name    = idx_to_cls.get(idx, str(idx))
            print(f"{name:<40} {correct:>8} {total:>8} {correct/total*100:>7.1f}%")

    # Save results
    ckpt_path = Path(args.checkpoint)
    results = {
        "split":     args.split,
        "num_classes": num_classes,
        "num_images": len(ds),
        "loss":      round(loss_m.avg, 4),
        "top1_acc":  round(acc1_m.avg, 4),
        "top5_acc":  round(acc5_m.avg, 4),
        "per_class": {
            idx_to_cls.get(idx, str(idx)): {
                "correct": cls_correct[idx],
                "total":   cls_total[idx],
                "acc":     round(cls_correct[idx] / cls_total[idx] * 100, 2),
            }
            for idx in cls_total
        } if args.per_class else {},
    }
    out_path = ckpt_path.parent / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")

    if confusion_matrix is not None:
        # class list ordered by label index
        cls_list = [idx_to_cls[i] for i in range(num_classes)]
        confusion_out = {
            "split":      args.split,
            "top1_acc":   round(acc1_m.avg, 4),
            "top5_acc":   round(acc5_m.avg, 4),
            "classes":    cls_list,
            "matrix":     confusion_matrix,   # matrix[gt_idx][pred_idx] = count
            "per_class_acc": {
                idx_to_cls[i]: round(cls_correct[i] / cls_total[i] * 100, 2)
                if cls_total[i] else 0.0
                for i in range(num_classes)
            },
        }
        cm_path = ckpt_path.parent / f"eval_{args.split}_confusion.json"
        cm_path.write_text(json.dumps(confusion_out))
        print(f"Confusion matrix → {cm_path}")


if __name__ == "__main__":
    main()
