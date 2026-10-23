"""
Prepare train/val/test manifest CSVs from proto_crawled_all.

Exclusion logic:
  - Images flagged as doubtful (in outliers.json) that have NOT yet been
    reviewed in the UI are excluded from training.
  - Images reviewed as 'ok' or 'move' are kept (user confirmed them).
  - Images reviewed as 'delete' are already gone from disk.

Class selection:
  - Count images per class after exclusion.
  - Keep classes with >= min_samples_per_class images.
  - Select top max_classes by image count.

Output (manifest_dir/):
  classes.json   — {class_name: label_idx} mapping
  train.csv      — path,label
  val.csv
  test.csv
  stats.txt      — summary
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

GOOD_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_cfg(path):
    return yaml.safe_load(open(path))


def build_exclude_set(outliers_path, actions_path):
    """Images that are flagged but not yet reviewed → exclude."""
    if not Path(outliers_path).exists():
        print(f"[warn] outliers.json not found at {outliers_path} — no images excluded")
        return set()

    flagged = {
        item["path"]
        for item in json.loads(Path(outliers_path).read_text())["suspicious"]
    }

    reviewed = set()
    if Path(actions_path).exists():
        reviewed = set(json.loads(Path(actions_path).read_text()).keys())

    exclude = flagged - reviewed
    print(f"Doubtful images  : {len(flagged):,}")
    print(f"Reviewed so far  : {len(reviewed):,}")
    print(f"Excluded (pending): {len(exclude):,}")
    return exclude


def collect_images(proto_dir, exclude, blacklist=None):
    """Return dict: class_name → [abs_path_str, ...]"""
    proto = Path(proto_dir)
    blacklist = set(blacklist or [])
    by_class = defaultdict(list)
    for cls_dir in sorted(proto.iterdir()):
        if not cls_dir.is_dir():
            continue
        if cls_dir.name in blacklist:
            print(f"  [blacklist] skipping class: {cls_dir.name}")
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() not in GOOD_EXT:
                continue
            if str(img) in exclude:
                continue
            by_class[cls_dir.name].append(str(img))
    return by_class


def select_classes(by_class, min_samples, max_classes, max_per_class=None):
    """Keep classes with enough images; take top-N by count; optionally cap per class."""
    eligible = {c: imgs for c, imgs in by_class.items() if len(imgs) >= min_samples}
    ranked = sorted(eligible.items(), key=lambda x: len(x[1]), reverse=True)
    selected = dict(ranked[:max_classes])
    if max_per_class:
        import random as _r
        rng = _r.Random(42)
        capped = 0
        for cls in selected:
            if len(selected[cls]) > max_per_class:
                selected[cls] = rng.sample(selected[cls], max_per_class)
                capped += 1
        if capped:
            print(f"Capped {capped} classes at {max_per_class} images each")
    print(f"Eligible classes (>= {min_samples} imgs): {len(eligible)}")
    print(f"Selected classes (top {max_classes})    : {len(selected)}")
    return selected


def stratified_split(by_class, val_frac, test_frac, seed):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls, paths in by_class.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * test_frac))
        n_val  = max(1, round(n * val_frac))
        test  += [(p, cls) for p in shuffled[:n_test]]
        val   += [(p, cls) for p in shuffled[n_test:n_test + n_val]]
        train += [(p, cls) for p in shuffled[n_test + n_val:]]
    return train, val, test


def write_csv(rows, label_map, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for img_path, cls in rows:
            w.writerow([img_path, label_map[cls]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    dc  = cfg["data"]

    print("=== Step 1: Build exclusion set ===")
    exclude = build_exclude_set(dc["outliers_json"], dc["actions_json"])

    print("\n=== Step 2: Collect images ===")
    by_class = collect_images(dc["proto_dir"], exclude, dc.get("class_blacklist"))
    total_raw = sum(len(v) for v in by_class.items())
    print(f"Total classes found : {len(by_class)}")
    print(f"Total images (clean): {sum(len(v) for v in by_class.values()):,}")

    print("\n=== Step 3: Select classes ===")
    selected = select_classes(
        by_class, dc["min_samples_per_class"], dc["max_classes"],
        dc.get("max_per_class")
    )

    print("\n=== Step 4: Stratified split ===")
    train, val, test = stratified_split(
        selected, dc["val_frac"], dc["test_frac"], dc["seed"]
    )
    print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    print("\n=== Step 5: Write manifests ===")
    out = Path(dc["manifest_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # Class → index mapping (alphabetical for reproducibility)
    classes_sorted = sorted(selected.keys())
    label_map = {cls: idx for idx, cls in enumerate(classes_sorted)}

    (out / "classes.json").write_text(json.dumps(label_map, indent=2))
    write_csv(train, label_map, out / "train.csv")
    write_csv(val,   label_map, out / "val.csv")
    write_csv(test,  label_map, out / "test.csv")

    stats = (
        f"Classes         : {len(selected)}\n"
        f"Train images    : {len(train)}\n"
        f"Val images      : {len(val)}\n"
        f"Test images     : {len(test)}\n"
        f"Total (selected): {len(train)+len(val)+len(test)}\n"
        f"Excluded (doubtful): {len(exclude)}\n"
        f"Min class size  : {min(len(v) for v in selected.values())}\n"
        f"Max class size  : {max(len(v) for v in selected.values())}\n"
        f"Avg class size  : {sum(len(v) for v in selected.values())//len(selected)}\n"
    )
    (out / "stats.txt").write_text(stats)
    print(stats)
    print(f"Manifests written to {out}/")


if __name__ == "__main__":
    main()
