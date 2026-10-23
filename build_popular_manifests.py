"""
Build popularity-ranked class lists and manifests for popular_200cls through popular_1700cls.

Ranking logic:
  1. Classes in dish_popularity.json, sorted by score DESC, then image-count DESC.
  2. Remaining classes (not in popularity file), sorted by image-count DESC.

This produces a single authoritative ordered list of all ~1762 classes.
Manifests are built at sizes: 200, 400, 500, 600, 700, 800, 900, 1000,
                               1100, 1200, 1300, 1400, 1500, 1600, 1700
(100 and 300 already exist as popular_100cls / popular_300cls).

Usage:
  # Show ranked list + what would be generated (no files written):
  python build_popular_manifests.py --dry-run

  # Generate class-list JSONs only (no train/val/test CSVs):
  python build_popular_manifests.py --lists-only

  # Generate everything (class lists + full manifests):
  python build_popular_manifests.py --all

  # Generate only one specific size:
  python build_popular_manifests.py --size 200
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

FOOD_BASE     = Path(__file__).parent
PROTO         = Path("data/proto_crawled_all")
POP_JSON      = FOOD_BASE / "data" / "dish_popularity.json"
MANIFESTS_DIR = FOOD_BASE / "manifests"
LISTS_DIR     = MANIFESTS_DIR / "_popular_class_lists"   # intermediate JSONs
# prepare_data.py imports yaml — use the env that has it
PYTHON        = sys.executable

SIZES = [200, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
GOOD_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def get_image_counts() -> dict[str, int]:
    counts = {}
    for d in PROTO.iterdir():
        if d.is_dir() and d.name != "unknown" and not d.name.startswith("_"):
            counts[d.name] = sum(
                1 for f in d.iterdir() if f.suffix.lower() in GOOD_EXT
            )
    return counts


def build_ranked_list(pop_data: list, img_counts: dict) -> list[str]:
    """Return all classes sorted by (popularity_score DESC, img_count DESC)."""
    pop_map = {entry["slug"]: entry["popularity_score"] for entry in pop_data}

    # Scored classes — sort by score desc, then count desc
    scored = sorted(
        [(cls, pop_map[cls], img_counts.get(cls, 0)) for cls in pop_map if cls in img_counts],
        key=lambda x: (-x[1], -x[2]),
    )
    scored_names = {cls for cls, _, _ in scored}

    # Unscored classes — sort by image count desc
    unscored = sorted(
        [(cls, img_counts[cls]) for cls in img_counts if cls not in scored_names],
        key=lambda x: -x[1],
    )

    ranked = [cls for cls, _, _ in scored] + [cls for cls, _ in unscored]
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",    action="store_true", help="Print ranked list preview only")
    ap.add_argument("--lists-only", action="store_true", help="Write class-list JSONs but not manifests")
    ap.add_argument("--all",        action="store_true", help="Write class lists + generate all manifests")
    ap.add_argument("--size",       type=int, default=None, help="Generate only this one size")
    args = ap.parse_args()

    if not any([args.dry_run, args.lists_only, args.all, args.size]):
        ap.print_help()
        sys.exit(0)

    print("Loading popularity scores …")
    pop_data = json.loads(POP_JSON.read_text())
    print(f"  {len(pop_data)} scored classes (scores 0–5)")

    print("Counting images per class …")
    img_counts = get_image_counts()
    print(f"  {len(img_counts)} class folders found")

    ranked = build_ranked_list(pop_data, img_counts)
    total  = len(ranked)
    print(f"\nFull ranked list: {total} classes")

    sizes = [args.size] if args.size else SIZES
    sizes = [s for s in sizes if s <= total]

    if args.dry_run:
        print("\n── Top 20 ranked classes ──")
        pop_map = {e["slug"]: e["popularity_score"] for e in pop_data}
        for i, cls in enumerate(ranked[:20], 1):
            score = pop_map.get(cls, "—")
            count = img_counts.get(cls, 0)
            print(f"  {i:3d}. {cls:<35s}  score={score}  imgs={count}")
        print("\n── Sizes to generate ──")
        for s in sizes:
            already = (MANIFESTS_DIR / f"popular_{s}cls").exists()
            print(f"  popular_{s}cls  ({'SKIP — exists' if already else 'to create'})")
        return

    # Write class-list JSONs
    LISTS_DIR.mkdir(parents=True, exist_ok=True)
    written_lists = []
    for size in sizes:
        out = LISTS_DIR / f"popular_{size}cls.json"
        cls_list = ranked[:size]
        out.write_text(json.dumps(cls_list, indent=2))
        written_lists.append((size, out, cls_list))
        print(f"  Wrote {out.name}  ({size} classes, "
              f"#{size} = {cls_list[-1] if cls_list else '—'})")

    if args.lists_only:
        print("\nClass lists written. To generate full manifests run:")
        for size, out, _ in written_lists:
            mdir = MANIFESTS_DIR / f"popular_{size}cls"
            print(f"  python prepare_data_by_class_list.py "
                  f"--classes_json {out} --manifest_dir {mdir}")
        return

    # Generate manifests
    print("\n── Generating manifests ──")
    for size, lists_json, cls_list in written_lists:
        mdir = MANIFESTS_DIR / f"popular_{size}cls"
        if mdir.exists():
            print(f"  SKIP popular_{size}cls — already exists")
            continue
        print(f"\n  ── popular_{size}cls ({size} classes) ──")
        result = subprocess.run(
            [
                PYTHON, str(FOOD_BASE / "prepare_data_by_class_list.py"),
                "--classes_json", str(lists_json),
                "--manifest_dir", str(mdir),
            ],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  ERROR: prepare_data_by_class_list.py failed for size {size}")

    print("\nDone.")


if __name__ == "__main__":
    main()
