"""
Manifest restricted to an explicit list of class names (as opposed to
prepare_data.py's "top N by image count" selection). Used for the
popularity-ranked 100/300-class sets, where class selection came from
real-world popularity research, not scraped-image volume.

Reuses prepare_data.py's own collect_images/stratified_split so the split
logic matches every other run exactly.

Usage:
  python3 prepare_data_by_class_list.py --classes_json /path/to/top100.json --manifest_dir manifests/popular_100cls
"""
import argparse
import json
from pathlib import Path

from prepare_data import collect_images, stratified_split, write_csv

PROTO = Path("data/proto_crawled_all")   # class folders of the (non-redistributed) image corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes_json", required=True, help="JSON list of class slugs to include")
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--max_per_class", type=int, default=1000)
    ap.add_argument("--proto", default=None, help="Override dataset root (default: proto_crawled_all)")
    args = ap.parse_args()
    manifest_dir = Path(args.manifest_dir)
    proto = Path(args.proto) if args.proto else PROTO

    wanted = set(json.loads(Path(args.classes_json).read_text()))
    print(f"Requested classes: {len(wanted)}")

    by_class = collect_images(str(proto), exclude=set(), blacklist=["unknown"])
    by_class = {c: imgs for c, imgs in by_class.items() if c in wanted}
    missing = wanted - set(by_class.keys())
    if missing:
        print(f"[warn] {len(missing)} requested classes had no images on disk: {sorted(missing)}")

    if args.max_per_class:
        import random
        rng = random.Random(42)
        for cls in by_class:
            if len(by_class[cls]) > args.max_per_class:
                by_class[cls] = rng.sample(by_class[cls], args.max_per_class)

    train, val, test = stratified_split(by_class, val_frac=0.15, test_frac=0.15, seed=42)
    print(f"Classes: {len(by_class)}  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    classes_sorted = sorted(by_class.keys())
    label_map = {cls: idx for idx, cls in enumerate(classes_sorted)}
    (manifest_dir / "classes.json").write_text(json.dumps(label_map, indent=2))
    write_csv(train, label_map, manifest_dir / "train.csv")
    write_csv(val, label_map, manifest_dir / "val.csv")
    write_csv(test, label_map, manifest_dir / "test.csv")

    sizes = [len(v) for v in by_class.values()]
    stats = (
        f"Classes         : {len(by_class)}\n"
        f"Train images    : {len(train)}\n"
        f"Val images      : {len(val)}\n"
        f"Test images     : {len(test)}\n"
        f"Total (selected): {len(train)+len(val)+len(test)}\n"
        f"Min class size  : {min(sizes)}\n"
        f"Max class size  : {max(sizes)}\n"
        f"Avg class size  : {sum(sizes)//len(sizes)}\n"
    )
    (manifest_dir / "stats.txt").write_text(stats)
    print(stats)


if __name__ == "__main__":
    main()
