"""
Head / mid / tail accuracy breakdown (camera-ready B6).

Groups classes by their number of training images and reports test accuracy per group.
Needs an eval_test.json written with `evaluate.py --per_class`.

Default grouping = terciles of classes ranked by training-image count (equal class counts).
Fixed thresholds: --head_min N --tail_max M  (head: >=N, tail: <=M, mid: otherwise).

Usage:
    python head_mid_tail.py --eval runs/<run>/eval_test.json --manifest manifests/popular_1700cls \
        [--head_min 100 --tail_max 49] [--out head_mid_tail.csv]
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True, help="eval_test.json produced with --per_class")
    p.add_argument("--manifest", required=True, help="manifest dir with train.csv + classes.json")
    p.add_argument("--head_min", type=int, default=None)
    p.add_argument("--tail_max", type=int, default=None)
    p.add_argument("--out", default="head_mid_tail.csv")
    a = p.parse_args()

    ev = json.loads(Path(a.eval).read_text())
    per_class = ev["per_class"]
    if not per_class:
        raise SystemExit("eval json has no per_class entries; rerun evaluate.py with --per_class")

    man = Path(a.manifest)
    name_to_idx = json.loads((man / "classes.json").read_text())
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    train_n = Counter(int(r["label"]) for r in csv.DictReader(open(man / "train.csv")))
    train_by_name = {idx_to_name[i]: train_n.get(i, 0) for i in idx_to_name}

    names = [n for n in per_class if n in train_by_name]
    if len(names) != len(per_class):
        print(f"[warn] {len(per_class) - len(names)} evaluated classes missing from manifest")

    if a.head_min is not None and a.tail_max is not None:
        grp = {n: "head" if train_by_name[n] >= a.head_min else
                  "tail" if train_by_name[n] <= a.tail_max else "mid" for n in names}
        rule = f"head: >={a.head_min} train imgs, tail: <={a.tail_max}"
    else:
        order = sorted(names, key=lambda n: -train_by_name[n])
        k = len(order) / 3
        grp = {n: ("head" if i < k else "mid" if i < 2 * k else "tail") for i, n in enumerate(order)}
        rule = "terciles of classes ranked by train count"

    rows = []
    for n in names:
        pc = per_class[n]
        rows.append({"class": n, "train_images": train_by_name[n], "test_images": pc["total"],
                     "correct": pc["correct"], "acc": pc["acc"], "group": grp[n]})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["train_images"]))

    print(f"Grouping: {rule}.  Overall top-1 (eval json): {ev['top1_acc']:.2f}%  on {ev['num_images']} images\n")
    print(f"{'group':<6} {'classes':>8} {'train imgs/class (min-max)':>28} {'test imgs':>10} {'top-1 %':>9} {'mean class acc %':>17}")
    for g in ("head", "mid", "tail"):
        r = [x for x in rows if x["group"] == g]
        if not r:
            continue
        tn = sum(x["test_images"] for x in r)
        top1 = 100 * sum(x["correct"] for x in r) / tn
        mca = sum(x["acc"] for x in r) / len(r)
        lo, hi = min(x["train_images"] for x in r), max(x["train_images"] for x in r)
        print(f"{g:<6} {len(r):>8} {f'{lo}-{hi}':>28} {tn:>10} {top1:>9.2f} {mca:>17.2f}")
    print(f"\nper-class CSV -> {a.out}")


if __name__ == "__main__":
    main()
