"""
Train the food classifier.

Usage:
    python train.py                          # uses config.yaml
    python train.py --config config.yaml --run_name my_run
    python train.py --backbone resnet18 --epochs 30 --batch_size 128
"""

import argparse
import csv
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml

# Reduces fragmentation OOMs on cards with limited VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Avoid SIGILL from cuDNN auto-tuning selecting AVX-512 kernels not supported by this CPU
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import FoodDataset
from data.transforms import train_transforms, val_transforms
from models.classifier import build_model
from utils.metrics import AverageMeter, topk_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--run_name",   default=None)
    p.add_argument("--seed",       type=int, default=None, help="Seed python/numpy/torch (default: unseeded)")
    p.add_argument("--resume",     default=None, help="Path to checkpoint to resume from")
    p.add_argument("--no_wandb",   action="store_true", help="Disable wandb logging")
    p.add_argument("--wandb_project", default="nutrition")
    # optional overrides
    p.add_argument("--backbone",   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=None,
                    help="Freeze backbone, train only the classification head (vit_s_16 only)")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.backbone:   cfg["model"]["backbone"]        = args.backbone
    if args.epochs:     cfg["training"]["epochs"]       = args.epochs
    if args.batch_size: cfg["training"]["batch_size"]   = args.batch_size
    if args.lr:         cfg["training"]["learning_rate"]= args.lr
    if args.freeze_backbone is not None:
        cfg["model"]["freeze_backbone"] = args.freeze_backbone
    return cfg


def run_epoch(model, loader, criterion, optimizer, scaler, device, training, desc="", amp_dtype=torch.float16):
    model.train() if training else model.eval()
    loss_m  = AverageMeter()
    acc1_m  = AverageMeter()
    acc5_m  = AverageMeter()

    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in bar:
            images = images.to(device, non_blocking=True)
            # PyTorch 2.x DataLoader workers can produce inference tensors
            # (e.g. from T.RandomErasing clone); clone() outside inference_mode
            # escapes it and prevents "Inference tensors do not track version
            # counter" errors in backward().
            if images.is_inference():
                images = images.clone()
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()
                with autocast("cuda", dtype=amp_dtype, enabled=scaler is not None):
                    logits = model(images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

            n = images.size(0)
            acc1, acc5 = topk_accuracy(logits, labels, topk=(1, 5))
            loss_m.update(loss.item(), n)
            acc1_m.update(acc1, n)
            acc5_m.update(acc5, n)

            bar.set_postfix(loss=f"{loss_m.avg:.4f}", acc1=f"{acc1_m.avg:.1f}%")

    return loss_m.avg, acc1_m.avg, acc5_m.avg


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    cfg  = apply_overrides(cfg, args)

    manifest_dir = Path(cfg["data"]["manifest_dir"])
    if not (manifest_dir / "train.csv").exists():
        print("ERROR: manifests not found. Run prepare_data.py first.")
        sys.exit(1)

    classes    = json.loads((manifest_dir / "classes.json").read_text())
    num_classes = len(classes)

    t_tfm = train_transforms(cfg)
    v_tfm = val_transforms(cfg)
    train_ds = FoodDataset(manifest_dir / "train.csv", t_tfm)
    val_ds   = FoodDataset(manifest_dir / "val.csv",   v_tfm)

    nw = cfg["training"]["num_workers"]
    bs = cfg["training"]["batch_size"]
    from collections import Counter
    label_counts = Counter(lbl for _, lbl in train_ds.samples)
    max_ratio = cfg["training"].get("sampler_max_ratio", None)
    if max_ratio is not None:
        # Cap upsampling: no class gets more than max_ratio × the majority class rate.
        # w_i = min(1/count_i, max_ratio / max_count)  then normalise.
        max_count = max(label_counts.values())
        cap = max_ratio / max_count
        raw_weights = {lbl: min(1.0 / cnt, cap) for lbl, cnt in label_counts.items()}
    else:
        raw_weights = {lbl: 1.0 / cnt for lbl, cnt in label_counts.items()}
    sample_weights = [raw_weights[lbl] for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=nw, pin_memory=True, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(
        cfg["model"]["backbone"], num_classes,
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=cfg["model"].get("freeze_backbone", False),
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    if cfg["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    use_amp   = cfg["training"]["mixed_precision"] and device.type == "cuda"
    amp_dtype = getattr(torch, cfg["training"].get("amp_dtype", "float16"))
    # BF16 has FP32's exponent range so no loss scaling needed
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler     = GradScaler("cuda", enabled=use_scaler)

    start_epoch  = 1
    best_val_acc = 0.0

    if args.resume:
        ckpt_path = Path(args.resume)
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("val_acc1", 0.0)
        # fast-forward scheduler to match completed epochs
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        # use the existing run directory
        run_path = ckpt_path.parent
        log_path = run_path / "log.csv"
        print(f"  Resumed at epoch {start_epoch}, best val acc1={best_val_acc:.2f}%")
    else:
        run_name = args.run_name or (
            f"{cfg['model']['backbone']}_{num_classes}cls_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        run_path = Path(cfg["output"]["run_dir"]) / run_name
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "config.yaml").write_text(open(args.config).read())
        log_path = run_path / "log.csv"
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc1", "train_acc5",
                 "val_loss",   "val_acc1",   "val_acc5",   "lr"]
            )

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_path.name,
            config={
                **cfg["model"],
                **cfg["training"],
                **cfg["augmentation"],
                "num_classes": num_classes,
                "train_size":  len(train_ds),
                "val_size":    len(val_ds),
                "resume":      args.resume,
            },
            resume="allow" if args.resume else False,
            id=run_path.name,
        )

    print(f"\n{'='*60}")
    print(f"Backbone    : {cfg['model']['backbone']}")
    print(f"Params      : {n_trainable:,} trainable / {n_total:,} total")
    print(f"Classes     : {num_classes}")
    print(f"Train       : {len(train_ds):,}  Val: {len(val_ds):,}")
    print(f"Device      : {device}  AMP: {use_amp}")
    print(f"Run dir     : {run_path}")
    print(f"W&B         : {'enabled' if use_wandb else 'disabled'}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        tl, ta1, ta5 = run_epoch(model, train_loader, criterion,
                                  optimizer, scaler, device, training=True,
                                  desc=f"Ep {epoch:03d}/{epochs} train", amp_dtype=amp_dtype)
        vl, va1, va5 = run_epoch(model, val_loader, criterion,
                                  None, None, device, training=False,
                                  desc=f"Ep {epoch:03d}/{epochs} val  ", amp_dtype=amp_dtype)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Ep {epoch:03d}/{epochs}  "
            f"train loss={tl:.4f} acc1={ta1:.1f}% acc5={ta5:.1f}%  "
            f"val loss={vl:.4f} acc1={va1:.1f}% acc5={va5:.1f}%  "
            f"lr={lr:.2e}  {elapsed:.0f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{tl:.4f}", f"{ta1:.4f}", f"{ta5:.4f}",
                f"{vl:.4f}", f"{va1:.4f}", f"{va5:.4f}",
                f"{lr:.6f}",
            ])

        if use_wandb:
            wandb.log({
                "train/loss": tl, "train/acc1": ta1, "train/acc5": ta5,
                "val/loss":   vl, "val/acc1":   va1, "val/acc5":   va5,
                "lr": lr,
            }, step=epoch)

        if va1 > best_val_acc:
            best_val_acc = va1
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "val_acc1":  va1,
                "val_acc5":  va5,
                "classes":   classes,
                "backbone":  cfg["model"]["backbone"],
            }, run_path / "best.pth")
            print(f"  ✓ New best val acc1={va1:.2f}%  saved → {run_path/'best.pth'}")

    # Save final checkpoint too
    torch.save({
        "epoch":     epochs,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "val_acc1":  best_val_acc,
        "classes":   classes,
        "backbone":  cfg["model"]["backbone"],
    }, run_path / "last.pth")

    if use_wandb:
        wandb.finish()

    print(f"\nBest val acc1 : {best_val_acc:.2f}%")
    print(f"Checkpoints   : {run_path}/")


def _cleanup(signum=None, frame=None):
    """Free GPU memory on exit / Ctrl-C so the process doesn't leak VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if signum is not None:
        print("\nInterrupted — GPU memory released.")
        sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    try:
        main()
    finally:
        _cleanup()
